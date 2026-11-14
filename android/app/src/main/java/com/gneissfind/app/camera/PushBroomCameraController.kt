package com.gneissfind.app.camera

import android.content.Context
import android.graphics.Rect
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Surface
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.tan

/**
 * Live exposure, zoom, and calibrated vertical focal-length telemetry reported from `TotalCaptureResult`.
 */
data class CameraExposureTelemetry(
    val exposureTimeNs: Long = PushBroomCameraController.MAX_SHUTTER_EXPOSURE_NS,
    val sensorIso: Int = 400,
    val activeZoomRatio: Float = 2.0f,
    val verticalFocalLengthPx: Float = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX,
    val streamWidth: Int = PushBroomCameraController.TARGET_WIDTH_1080P,
    val streamHeight: Int = PushBroomCameraController.TARGET_HEIGHT_1080P
)

/**
 * Manages the CameraX + `Camera2Interop` reversible-landscape `1920x1080` rear camera session:
 *  - Defaults `2.0x` lossless center sensor crop (`setZoomRatio(2.0f)`) AND `<= 1/500 s`
 *    (`<= 2_000_000 ns`) shutter clamp to **ON** (`true`) at startup.
 *  - Synchronizes `Preview.targetRotation` and `ImageAnalysis.targetRotation` across
 *    `Surface.ROTATION_90` and `Surface.ROTATION_270` reversible landscape flips.
 *  - Reads `SENSOR_EXPOSURE_TIME`, `SENSOR_SENSITIVITY`, `LENS_INTRINSIC_CALIBRATION`, and
 *    `SCALER_CROP_REGION` via `Camera2Interop.Extender.setSessionCaptureCallback` to feed accurate
 *    telemetry (`1/500s`, `ISO`, `fy`, and live `GSD mm/px`) to the LCARS HUD.
 */
@OptIn(ExperimentalCamera2Interop::class)
class PushBroomCameraController(
    private val appContext: Context,
    private val onFrameAvailable: (ImageProxy, CameraExposureTelemetry) -> Unit
) : AutoCloseable {

    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "GneissFind-Analyzer").apply { priority = Thread.NORM_PRIORITY + 1 }
    }

    private var cameraProvider: ProcessCameraProvider? = null
    private var boundCamera: Camera? = null
    private var previewUseCase: Preview? = null
    private var imageAnalysisUseCase: ImageAnalysis? = null

    private var sensorIsoRange: Range<Int> = Range(100, 6400)
    private var sensorExposureRangeNs: Range<Long> = Range(100_000L, 100_000_000L)
    private var activeArraySize: Rect? = null

    @Volatile
    var targetSurfaceRotation: Int = Surface.ROTATION_90
        private set

    @Volatile
    var shutterClampEnabled: Boolean = true
        private set

    @Volatile
    var exposureBiasEv: Float = 0.0f
        private set

    @Volatile
    var twoXCropEnabled: Boolean = true
        private set

    @Volatile
    var latestTelemetry: CameraExposureTelemetry = CameraExposureTelemetry()
        private set

    // Adaptive floating ISO estimate when CONTROL_AE_MODE_OFF clamps shutter at <= 1/500 s
    @Volatile
    private var clampedFloatingIso: Int = 400

    fun bindToLifecycle(
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView,
        initialRotation: Int = targetSurfaceRotation
    ) {
        targetSurfaceRotation = normalizeLandscapeRotation(initialRotation)
        val providerFuture = ProcessCameraProvider.getInstance(appContext)
        providerFuture.addListener(
            {
                try {
                    val provider = providerFuture.get()
                    cameraProvider = provider
                    bindUseCases(provider, lifecycleOwner, previewView)
                } catch (t: Throwable) {
                    Log.e(TAG, "Failed binding CameraX pipeline", t)
                }
            },
            ContextCompat.getMainExecutor(appContext)
        )
    }

    /**
     * Dynamically updates CameraX `Preview` and `ImageAnalysis` targetRotation whenever the user
     * flips the phone 180 degrees between `Surface.ROTATION_90` and `Surface.ROTATION_270`.
     */
    fun updateTargetRotation(surfaceRotation: Int) {
        val normalized = normalizeLandscapeRotation(surfaceRotation)
        targetSurfaceRotation = normalized
        previewUseCase?.targetRotation = normalized
        imageAnalysisUseCase?.targetRotation = normalized
    }

    private fun bindUseCases(
        provider: ProcessCameraProvider,
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView
    ) {
        provider.unbindAll()

        val rotation = targetSurfaceRotation
        val landscape1080pSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(TARGET_WIDTH_1080P, TARGET_HEIGHT_1080P),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()

        val previewBuilder = Preview.Builder()
            .setTargetRotation(rotation)
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                    .build()
            )

        val analysisBuilder = ImageAnalysis.Builder()
            .setTargetRotation(rotation)
            .setResolutionSelector(landscape1080pSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)

        // Attach Camera2Interop session callback + default ON <= 1/500s shutter clamp
        val analysisExtender = Camera2Interop.Extender(analysisBuilder)
        if (shutterClampEnabled) {
            analysisExtender.setCaptureRequestOption(
                CaptureRequest.CONTROL_AE_MODE,
                CameraMetadata.CONTROL_AE_MODE_OFF
            )
            analysisExtender.setCaptureRequestOption(
                CaptureRequest.SENSOR_EXPOSURE_TIME,
                MAX_SHUTTER_EXPOSURE_NS
            )
            analysisExtender.setCaptureRequestOption(
                CaptureRequest.SENSOR_SENSITIVITY,
                clampedFloatingIso
            )
        }

        analysisExtender.setSessionCaptureCallback(
            object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(
                    session: CameraCaptureSession,
                    request: CaptureRequest,
                    result: TotalCaptureResult
                ) {
                    val expNs = result.get(TotalCaptureResult.SENSOR_EXPOSURE_TIME) ?: MAX_SHUTTER_EXPOSURE_NS
                    val iso = result.get(TotalCaptureResult.SENSOR_SENSITIVITY) ?: clampedFloatingIso
                    val intrinsics = result.get(TotalCaptureResult.LENS_INTRINSIC_CALIBRATION)
                    val cropRegion = result.get(TotalCaptureResult.SCALER_CROP_REGION)

                    val zoom = latestTelemetry.activeZoomRatio
                    val fyPx = computeStreamVerticalFocalLengthPx(
                        intrinsics = intrinsics,
                        cropRegion = cropRegion,
                        activeArray = activeArraySize,
                        activeZoomRatio = zoom,
                        outputStreamHeightPx = TARGET_HEIGHT_1080P
                    )

                    latestTelemetry = latestTelemetry.copy(
                        exposureTimeNs = expNs,
                        sensorIso = iso,
                        verticalFocalLengthPx = fyPx
                    )
                }
            }
        )

        val preview = previewBuilder.build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }
        previewUseCase = preview

        val imageAnalysis = analysisBuilder.build().also { analysis ->
            analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                val curTelemetry = latestTelemetry.copy(
                    streamWidth = imageProxy.width,
                    streamHeight = imageProxy.height
                )
                latestTelemetry = curTelemetry

                if (shutterClampEnabled) {
                    maybeServoFloatingIsoFromCenterLuma(imageProxy)
                }

                onFrameAvailable(imageProxy, curTelemetry)
            }
        }
        imageAnalysisUseCase = imageAnalysis

        val camera = provider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            imageAnalysis
        )
        boundCamera = camera

        // Query hardware ISO, exposure & sensor array dimensions via Camera2CameraInfo
        runCatching {
            val chars = Camera2CameraInfo.from(camera.cameraInfo)
            chars.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)?.let {
                sensorIsoRange = it
            }
            chars.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)?.let {
                sensorExposureRangeNs = it
            }
            activeArraySize = chars.getCameraCharacteristic(
                CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE
            )
        }

        // Immediately apply default 2.0x lossless sensor crop + <= 1/500s shutter clamp
        applyZoomAndExposureControls()
    }

    fun setShutterClamp(enabled: Boolean) {
        shutterClampEnabled = enabled
        applyZoomAndExposureControls()
    }

    fun setExposureBiasEv(ev: Float) {
        exposureBiasEv = ev.coerceIn(MIN_CANOPY_EV_BIAS, MAX_CANOPY_EV_BIAS)
        applyZoomAndExposureControls()
    }

    fun setTwoXSensorCrop(enabled: Boolean) {
        twoXCropEnabled = enabled
        applyZoomAndExposureControls()
    }

    private fun applyZoomAndExposureControls() {
        val camera = boundCamera ?: return
        val targetZoom = if (twoXCropEnabled) 2.0f else 1.0f
        val zoomState = camera.cameraInfo.zoomState.value
        val clampedZoom = if (zoomState != null) {
            targetZoom.coerceIn(zoomState.minZoomRatio, zoomState.maxZoomRatio)
        } else {
            targetZoom
        }
        camera.cameraControl.setZoomRatio(clampedZoom)
        latestTelemetry = latestTelemetry.copy(
            activeZoomRatio = clampedZoom,
            verticalFocalLengthPx = fallbackVerticalFocalLengthPx(clampedZoom)
        )

        val c2Control = Camera2CameraControl.from(camera.cameraControl)
        if (shutterClampEnabled) {
            val clampedExpNs = MAX_SHUTTER_EXPOSURE_NS.coerceIn(
                sensorExposureRangeNs.lower,
                sensorExposureRangeNs.upper
            )
            // Apply Canopy EV bias multiplier onto floating ISO while keeping shutter <= 1/500s
            val evGain = 2.0.pow(exposureBiasEv.toDouble()).toFloat()
            val targetIso = (clampedFloatingIso * evGain).roundToInt()
                .coerceIn(sensorIsoRange.lower, sensorIsoRange.upper)

            val options = CaptureRequestOptions.Builder()
                .setCaptureRequestOption(
                    CaptureRequest.CONTROL_AE_MODE,
                    CameraMetadata.CONTROL_AE_MODE_OFF
                )
                .setCaptureRequestOption(
                    CaptureRequest.SENSOR_EXPOSURE_TIME,
                    clampedExpNs
                )
                .setCaptureRequestOption(
                    CaptureRequest.SENSOR_SENSITIVITY,
                    targetIso
                )
                .build()
            c2Control.setCaptureRequestOptions(options)
        } else {
            val clearOptions = CaptureRequestOptions.Builder()
                .setCaptureRequestOption(
                    CaptureRequest.CONTROL_AE_MODE,
                    CameraMetadata.CONTROL_AE_MODE_ON
                )
                .build()
            c2Control.setCaptureRequestOptions(clearOptions)

            val evState = camera.cameraInfo.exposureState
            if (evState.isExposureCompensationSupported) {
                val step = evState.exposureCompensationStep.toFloat().coerceAtLeast(0.1f)
                val index = (exposureBiasEv / step).roundToInt()
                    .coerceIn(evState.exposureCompensationRange.lower, evState.exposureCompensationRange.upper)
                camera.cameraControl.setExposureCompensationIndex(index)
            }
        }
    }

    /**
     * Closed-loop auto-ISO servo sampling 16 points in the Y-plane strip when `CONTROL_AE_MODE_OFF`
     * clamps shutter to `<= 1/500 s`, allowing ISO to float smoothly from bright gravel bars (ISO 100)
     * into Western Cascade tree canopy (ISO 3200+) without walking motion blur.
     */
    private var lastIsoServoNs = 0L
    private fun maybeServoFloatingIsoFromCenterLuma(imageProxy: ImageProxy) {
        val nowNs = imageProxy.imageInfo.timestamp
        if (nowNs - lastIsoServoNs < 350_000_000L) return
        lastIsoServoNs = nowNs

        val yPlane = imageProxy.planes.firstOrNull() ?: return
        val buf = yPlane.buffer
        val limit = buf.limit()
        if (limit < 64) return

        var lumaSum = 0
        val step = (limit / 32).coerceAtLeast(1)
        var count = 0
        var idx = limit / 4
        while (idx < (limit * 3) / 4 && count < 16) {
            lumaSum += (buf.get(idx).toInt() and 0xFF)
            idx += step
            count++
        }
        if (count == 0) return
        val meanLuma = lumaSum.toFloat() / count
        val targetLuma = 118f
        val ratio = (targetLuma / meanLuma.coerceAtLeast(12f)).coerceIn(0.65f, 1.55f)
        if (kotlin.math.abs(ratio - 1.0f) > 0.12f) {
            clampedFloatingIso = (clampedFloatingIso * ratio).roundToInt()
                .coerceIn(sensorIsoRange.lower, sensorIsoRange.upper)
            applyZoomAndExposureControls()
        }
    }

    override fun close() {
        runCatching { cameraProvider?.unbindAll() }
        analysisExecutor.shutdown()
    }

    companion object {
        private const val TAG = "PushBroomCamera"
        const val TARGET_WIDTH_1080P = 1920
        const val TARGET_HEIGHT_1080P = 1080
        const val MAX_SHUTTER_EXPOSURE_NS = 2_000_000L // 1/500 s = 2,000,000 ns
        const val MIN_CANOPY_EV_BIAS = -2.0f
        const val MAX_CANOPY_EV_BIAS = 1.0f

        // Optical vertical focal length at 2.0x crop on 1920x1080 (~20.5 deg VFOV -> 540 / tan(10.25 deg) = 2985 px)
        val DEFAULT_FY_2X_1080P_PX: Float =
            ((TARGET_HEIGHT_1080P / 2.0) / tan(Math.toRadians(10.25))).toFloat()

        fun fallbackVerticalFocalLengthPx(zoomRatio: Float): Float {
            val safeZoom = zoomRatio.coerceIn(1.0f, 8.0f)
            return DEFAULT_FY_2X_1080P_PX * (safeZoom / 2.0f)
        }

        fun normalizeLandscapeRotation(rotation: Int): Int {
            return if (rotation == Surface.ROTATION_270) {
                Surface.ROTATION_270
            } else {
                Surface.ROTATION_90
            }
        }

        fun computeStreamVerticalFocalLengthPx(
            intrinsics: FloatArray?,
            cropRegion: Rect?,
            activeArray: Rect?,
            activeZoomRatio: Float,
            outputStreamHeightPx: Int = TARGET_HEIGHT_1080P
        ): Float {
            if (intrinsics != null && intrinsics.size >= 2) {
                val fySensorPx = intrinsics[1]
                val cropHeightPx = cropRegion?.height()?.takeIf { it > 0 }
                    ?: activeArray?.height()?.let { (it / activeZoomRatio.coerceAtLeast(1f)).roundToInt() }
                    ?: 0
                if (fySensorPx > 100f && cropHeightPx > 100) {
                    // Account for 16:9 vertical letterbox crop from 4:3 sensor active array
                    val cropWidthPx = cropRegion?.width()?.takeIf { it > 0 } ?: (cropHeightPx * 4 / 3)
                    val effective16x9CropHeight = (cropWidthPx * 9f / 16f).coerceAtMost(cropHeightPx.toFloat())
                    val scaledFy = fySensorPx * (outputStreamHeightPx.toFloat() / effective16x9CropHeight)
                    if (scaledFy in 800f..6000f) {
                        return scaledFy
                    }
                }
            }
            return fallbackVerticalFocalLengthPx(activeZoomRatio)
        }
    }
}
