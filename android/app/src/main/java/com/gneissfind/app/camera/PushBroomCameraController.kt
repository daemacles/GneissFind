package com.gneissfind.app.camera

import android.content.Context
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.util.Log
import android.util.Range
import android.util.Size
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

/**
 * Live exposure and sensor telemetry reported from `TotalCaptureResult`.
 */
data class CameraExposureTelemetry(
    val exposureTimeNs: Long = 0L,
    val sensorIso: Int = 0,
    val activeZoomRatio: Float = 2.0f,
    val streamWidth: Int = 3840,
    val streamHeight: Int = 2160
)

/**
 * Manages the CameraX + `Camera2Interop` 4K (`3840x2160`) rear camera session:
 *  - Applies `2.0x` lossless sensor crop (`setZoomRatio(2.0f)`) so the `1088x384` strip achieves
 *    `~0.85 mm/px` GSD at `1.50 m` slant range.
 *  - Enforces the mandatory `<= 1/500 s` (`<= 2_000_000 ns`) shutter clamp via `Camera2Interop` /
 *    `Camera2CameraControl` while allowing ISO to float up to hardware sensor limits scaled by
 *    the user's Exposure Bias (`-2.0 EV .. +2.0 EV`) slider.
 *  - Configures `ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST` on a dedicated single-thread executor.
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
    private var sensorIsoRange: Range<Int> = Range(100, 6400)
    private var sensorExposureRangeNs: Range<Long> = Range(100_000L, 100_000_000L)

    @Volatile
    var shutterClampEnabled: Boolean = true
        private set

    @Volatile
    var exposureBiasEv: Float = -0.5f
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
        previewView: PreviewView
    ) {
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

    private fun bindUseCases(
        provider: ProcessCameraProvider,
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView
    ) {
        provider.unbindAll()

        val uhd4kSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(TARGET_WIDTH_4K, TARGET_HEIGHT_4K),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()

        val previewBuilder = Preview.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                    .build()
            )

        val analysisBuilder = ImageAnalysis.Builder()
            .setResolutionSelector(uhd4kSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)

        // Attach Camera2Interop session callback + initial <= 1/500s shutter clamp
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
                    latestTelemetry = latestTelemetry.copy(
                        exposureTimeNs = expNs,
                        sensorIso = iso
                    )
                }
            }
        )

        val preview = previewBuilder.build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }

        val imageAnalysis = analysisBuilder.build().also { analysis ->
            analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                val curTelemetry = latestTelemetry.copy(
                    streamWidth = imageProxy.width,
                    streamHeight = imageProxy.height
                )
                latestTelemetry = curTelemetry

                // Update closed-loop floating ISO from frame luma when AE_MODE_OFF shutter clamp is active
                if (shutterClampEnabled) {
                    maybeServoFloatingIsoFromCenterLuma(imageProxy)
                }

                onFrameAvailable(imageProxy, curTelemetry)
            }
        }

        val camera = provider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            imageAnalysis
        )
        boundCamera = camera

        // Query hardware ISO & exposure limits via Camera2CameraInfo
        runCatching {
            val chars = Camera2CameraInfo.from(camera.cameraInfo)
            chars.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)?.let {
                sensorIsoRange = it
            }
            chars.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)?.let {
                sensorExposureRangeNs = it
            }
        }

        applyZoomAndExposureControls()
    }

    fun setShutterClamp(enabled: Boolean) {
        shutterClampEnabled = enabled
        applyZoomAndExposureControls()
    }

    fun setExposureBiasEv(ev: Float) {
        exposureBiasEv = ev.coerceIn(-2.0f, 2.0f)
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
        latestTelemetry = latestTelemetry.copy(activeZoomRatio = clampedZoom)

        val c2Control = Camera2CameraControl.from(camera.cameraControl)
        if (shutterClampEnabled) {
            val clampedExpNs = MAX_SHUTTER_EXPOSURE_NS.coerceIn(
                sensorExposureRangeNs.lower,
                sensorExposureRangeNs.upper
            )
            // Apply user EV bias multiplier onto floating ISO while keeping shutter <= 1/500s
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
            // Release manual AE override and apply standard CameraX EV bias
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
     * Lightweight auto-ISO servo sampling 16 points in the Y-plane strip when `CONTROL_AE_MODE_OFF`
     * clamps shutter to `<= 1/500 s`, allowing ISO to float smoothly from bright gravel bars (ISO 100)
     * into Western Cascade tree canopy (ISO 3200+) without motion blur.
     */
    private var lastIsoServoNs = 0L
    private fun maybeServoFloatingIsoFromCenterLuma(imageProxy: ImageProxy) {
        val nowNs = imageProxy.imageInfo.timestamp
        if (nowNs - lastIsoServoNs < 350_000_000L) return // Update ~3 Hz to prevent flicker
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
        // Target mid-gray luma ~118 prior to EV bias offset
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
        const val TARGET_WIDTH_4K = 3840
        const val TARGET_HEIGHT_4K = 2160
        const val MAX_SHUTTER_EXPOSURE_NS = 2_000_000L // 1/500 s = 2,000,000 ns
    }
}
