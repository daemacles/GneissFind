/*
 * CameraX + Camera2Interop controller for the landscape push-broom camera pipeline.
 *
 * Binds Preview and YUV_420_888 ImageAnalysis to the activity lifecycle, enforces the shutter-speed
 * clamp with floating ISO servoing and lossless sensor zoom crop, and reports per-frame exposure and
 * calibrated focal-length telemetry to ScannerViewModel.
 *
 * Main types:
 * - CameraExposureTelemetry: immutable per-frame exposure, ISO, zoom, and calibrated vertical focal-length snapshot.
 * - PushBroomCameraController: lifecycle-aware CameraX + Camera2Interop session manager and closed-loop luma ISO servo.
 */
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
 * Live exposure, zoom, and calibrated vertical focal-length telemetry reported from [TotalCaptureResult].
 *
 * @property exposureTimeNs Sensor integration time (`SENSOR_EXPOSURE_TIME`) in nanoseconds.
 * @property sensorIso Sensor sensitivity (`SENSOR_SENSITIVITY`) in ISO units.
 * @property activeZoomRatio Active CameraX digital crop zoom ratio (`2.0f` nominal when `2X CROP` is enabled, `1.0f` otherwise).
 * @property verticalFocalLengthPx Calibrated vertical pinhole focal length $f_y$ in output stream pixels (`~2985 px` at `2.0x` on `1920x1080`).
 * @property streamWidth Width of the analyzed `YUV_420_888` [ImageProxy] stream in pixels ([PushBroomCameraController.TARGET_WIDTH_1080P]).
 * @property streamHeight Height of the analyzed `YUV_420_888` [ImageProxy] stream in pixels ([PushBroomCameraController.TARGET_HEIGHT_1080P]).
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
 *    ([MAX_SHUTTER_EXPOSURE_NS]) shutter clamp to **ON** (`true`) at startup.
 *  - Synchronizes `Preview.targetRotation` and `ImageAnalysis.targetRotation` across
 *    [Surface.ROTATION_90] and [Surface.ROTATION_270] reversible landscape flips.
 *  - Reads `SENSOR_EXPOSURE_TIME`, `SENSOR_SENSITIVITY`, `LENS_INTRINSIC_CALIBRATION`, and
 *    `SCALER_CROP_REGION` via `Camera2Interop.Extender.setSessionCaptureCallback` to feed accurate
 *    telemetry (`1/500s`, `ISO`, `fy`, and live `GSD mm/px`) to the LCARS HUD.
 *
 * Lifecycle & Threading:
 * - Call [bindToLifecycle] on the main thread once the Compose `PreviewView` is attached.
 * - [onFrameAvailable] is invoked sequentially on the dedicated single-threaded `GneissFind-Analyzer`
 *   executor; the callee (`ScannerViewModel`) is responsible for calling `imageProxy.close()`.
 * - Call [close] when the owning ViewModel is cleared to unbind use cases and shut down [analysisExecutor].
 *
 * @param appContext Application [Context] used to obtain [ProcessCameraProvider] and main executor.
 * @param onFrameAvailable Callback invoked on [analysisExecutor] for each `YUV_420_888` frame alongside its [CameraExposureTelemetry].
 */
@OptIn(ExperimentalCamera2Interop::class)
class PushBroomCameraController(
    private val appContext: Context,
    private val onFrameAvailable: (ImageProxy, CameraExposureTelemetry) -> Unit
) : AutoCloseable {

    /** Single-threaded executor (`GneissFind-Analyzer`) running `ImageAnalysis` and LiteRT GPU inference. */
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "GneissFind-Analyzer").apply { priority = Thread.NORM_PRIORITY + 1 }
    }

    private var cameraProvider: ProcessCameraProvider? = null
    private var boundCamera: Camera? = null
    private var previewUseCase: Preview? = null
    private var imageAnalysisUseCase: ImageAnalysis? = null

    /** Hardware-supported ISO sensitivity range queried from `SENSOR_INFO_SENSITIVITY_RANGE`. */
    private var sensorIsoRange: Range<Int> = Range(100, 6400)

    /** Hardware-supported exposure time range (`ns`) queried from `SENSOR_INFO_EXPOSURE_TIME_RANGE`. */
    private var sensorExposureRangeNs: Range<Long> = Range(100_000L, 100_000_000L)

    /** Full active pixel array rectangle queried from `SENSOR_INFO_ACTIVE_ARRAY_SIZE`. */
    private var activeArraySize: Rect? = null

    /** Maximum hardware digital zoom ratio queried from `SCALER_AVAILABLE_MAX_DIGITAL_ZOOM`. */
    private var hardwareMaxZoomRatio: Float = 8.0f

    /** Last zoom ratio pushed to `cameraControl.setZoomRatio(...)` to avoid redundant IPC calls. */
    private var lastAppliedZoomRatio: Float = -1.0f

    /** Active landscape display rotation ([Surface.ROTATION_90] or [Surface.ROTATION_270]). */
    @Volatile
    var targetSurfaceRotation: Int = Surface.ROTATION_90
        private set

    /** Whether manual `CONTROL_AE_MODE_OFF` shutter clamping (`<=` [MAX_SHUTTER_EXPOSURE_NS]) is active. */
    @Volatile
    var shutterClampEnabled: Boolean = true
        private set

    /** User-selected canopy exposure compensation bias in EV stops within `[MIN_CANOPY_EV_BIAS]..[MAX_CANOPY_EV_BIAS]`. */
    @Volatile
    var exposureBiasEv: Float = 0.0f
        private set

    /** Whether `2.0x` lossless center sensor crop (`0.59 mm/px` GSD) is enabled (`1.0x` when `false`). */
    @Volatile
    var twoXCropEnabled: Boolean = true
        private set

    /** Latest per-frame exposure and focal-length telemetry snapshot. */
    @Volatile
    var latestTelemetry: CameraExposureTelemetry = CameraExposureTelemetry()
        private set

    /** Adaptive floating ISO estimate servoed by [maybeServoFloatingIsoFromCenterLuma] when `CONTROL_AE_MODE_OFF` clamps shutter at `<=` [MAX_SHUTTER_EXPOSURE_NS]. */
    @Volatile
    private var clampedFloatingIso: Int = 400

    /**
     * Asynchronously resolves [ProcessCameraProvider] and binds `Preview` + `1920x1080` `YUV_420_888`
     * `ImageAnalysis` to [lifecycleOwner] on the main executor.
     *
     * @param lifecycleOwner Activity/Fragment lifecycle governing the camera session.
     * @param previewView Compose-hosted [PreviewView] receiving the camera preview surface.
     * @param initialRotation Initial display rotation constant ([Surface.ROTATION_90] or [Surface.ROTATION_270]).
     */
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
     * Dynamically updates CameraX [Preview] and [ImageAnalysis] `targetRotation` whenever the user
     * flips the phone 180 degrees between [Surface.ROTATION_90] and [Surface.ROTATION_270].
     *
     * @param surfaceRotation New Android [Surface] rotation constant (normalized via [normalizeLandscapeRotation]).
     */
    fun updateTargetRotation(surfaceRotation: Int) {
        val normalized = normalizeLandscapeRotation(surfaceRotation)
        targetSurfaceRotation = normalized
        previewUseCase?.targetRotation = normalized
        imageAnalysisUseCase?.targetRotation = normalized
    }

    /**
     * Configures and binds the `16:9` [Preview] and `1920x1080` `STRATEGY_KEEP_ONLY_LATEST`
     * [ImageAnalysis] use cases, attaches the Camera2 `CaptureCallback` for metadata extraction,
     * queries static sensor characteristics, and applies the initial `2.0x` zoom + shutter clamp.
     */
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

        // Attach Camera2Interop session callback for per-frame exposure/ISO/intrinsics telemetry.
        // Dynamic AE clamp options are driven exclusively via Camera2CameraControl so Extender's
        // ALWAYS_OVERRIDE priority never shadows runtime EXP / EV toggle updates.
        val analysisExtender = Camera2Interop.Extender(analysisBuilder)

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
        lastAppliedZoomRatio = -1.0f

        // Query hardware ISO, exposure, max digital zoom & sensor array dimensions synchronously
        // via Camera2CameraInfo so initial 2.0x zoom never races CameraX's uninitialized zoomState.
        runCatching {
            val chars = Camera2CameraInfo.from(camera.cameraInfo)
            chars.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)?.let {
                sensorIsoRange = it
            }
            chars.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)?.let {
                sensorExposureRangeNs = it
            }
            chars.getCameraCharacteristic(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM)?.let {
                if (it > 1.0f) hardwareMaxZoomRatio = it
            }
            activeArraySize = chars.getCameraCharacteristic(
                CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE
            )
        }

        // Immediately apply default 2.0x lossless sensor crop + <= 1/500s shutter clamp
        applyZoomAndExposureControls(forceZoomUpdate = true)
    }

    /**
     * Enables or disables the `<= 1/500 s` ([MAX_SHUTTER_EXPOSURE_NS]) manual shutter clamp (`EXP CLAMP` / `EXP AUTO`).
     *
     * @param enabled `true` to lock `CONTROL_AE_MODE_OFF` at [MAX_SHUTTER_EXPOSURE_NS] with floating ISO servoing;
     *   `false` to restore native `CONTROL_AE_MODE_ON`.
     */
    fun setShutterClamp(enabled: Boolean) {
        shutterClampEnabled = enabled
        applyZoomAndExposureControls(forceZoomUpdate = false)
    }

    /**
     * Sets the user canopy exposure compensation bias in EV stops (`CANOPY EV`), clamped to `[MIN_CANOPY_EV_BIAS]..[MAX_CANOPY_EV_BIAS]`.
     *
     * @param ev Desired exposure bias in EV stops (e.g. `-1.0f` to protect wet silica highlights).
     */
    fun setExposureBiasEv(ev: Float) {
        exposureBiasEv = ev.coerceIn(MIN_CANOPY_EV_BIAS, MAX_CANOPY_EV_BIAS)
        applyZoomAndExposureControls(forceZoomUpdate = false)
    }

    /**
     * Toggles between `2.0x` lossless center sensor crop (`0.59 mm/px` GSD) and `1.0x` full-width field of view.
     *
     * @param enabled `true` for `2.0x` zoom ratio; `false` for `1.0x` zoom ratio.
     */
    fun setTwoXSensorCrop(enabled: Boolean) {
        twoXCropEnabled = enabled
        applyZoomAndExposureControls(forceZoomUpdate = true)
    }

    /**
     * Pushes the current [twoXCropEnabled], [shutterClampEnabled], [exposureBiasEv], and [clampedFloatingIso]
     * state to the bound CameraX `CameraControl` and `Camera2CameraControl`.
     */
    private fun applyZoomAndExposureControls(forceZoomUpdate: Boolean = false) {
        val camera = boundCamera ?: return
        val targetZoom = if (twoXCropEnabled) 2.0f else 1.0f
        val zoomState = camera.cameraInfo.zoomState.value
        val minZoom = zoomState?.minZoomRatio?.takeIf { it > 0f } ?: 1.0f
        // CameraX initializes zoomState.maxZoomRatio to 1.0f before session open; fall back to
        // CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM until zoomState populates.
        val maxZoom = zoomState?.maxZoomRatio?.takeIf { it > 1.01f } ?: hardwareMaxZoomRatio
        val clampedZoom = targetZoom.coerceIn(minZoom, maxZoom.coerceAtLeast(minZoom))
        if (forceZoomUpdate || kotlin.math.abs(clampedZoom - lastAppliedZoomRatio) > 0.01f) {
            lastAppliedZoomRatio = clampedZoom
            camera.cameraControl.setZoomRatio(clampedZoom)
        }
        latestTelemetry = latestTelemetry.copy(
            activeZoomRatio = clampedZoom,
            verticalFocalLengthPx = fallbackVerticalFocalLengthPx(clampedZoom)
        )

        val c2Control = Camera2CameraControl.from(camera.cameraControl)
        val evState = camera.cameraInfo.exposureState
        if (shutterClampEnabled) {
            if (evState.isExposureCompensationSupported && evState.exposureCompensationIndex != 0) {
                camera.cameraControl.setExposureCompensationIndex(0)
            }
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
            // Clear manual AE/shutter/ISO overrides so CameraX restores native CONTROL_AE_MODE_ON.
            c2Control.clearCaptureRequestOptions()

            if (evState.isExposureCompensationSupported) {
                val step = evState.exposureCompensationStep.toFloat().coerceAtLeast(0.1f)
                val index = (exposureBiasEv / step).roundToInt()
                    .coerceIn(evState.exposureCompensationRange.lower, evState.exposureCompensationRange.upper)
                camera.cameraControl.setExposureCompensationIndex(index)
            }
        }
    }

    /** Timestamp (`ns`) of the last closed-loop ISO servo adjustment (`350 ms` minimum interval). */
    private var lastIsoServoNs = 0L

    /**
     * Closed-loop auto-ISO servo sampling 16 points in the middle half of the Y-plane buffer when
     * `CONTROL_AE_MODE_OFF` clamps shutter to `<=` [MAX_SHUTTER_EXPOSURE_NS], allowing ISO to float
     * smoothly from bright gravel bars (`ISO 100`) into Western Cascade tree canopy (`ISO 3200+`)
     * without walking motion blur.
     */
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
        // Bias target Y-luma in gamma (~0.5) space so auto-ISO preserves user Canopy EV stops.
        // RATIONALE-UNKNOWN: why was 118f chosen as the 0-EV reference Y-luma target and 0.12f as the deadband threshold?
        val targetLuma = (118f * 2.0.pow((exposureBiasEv * 0.5f).toDouble()).toFloat())
            .coerceIn(42f, 210f)
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

        /** Target width (`1920 px`) of the `16:9` landscape `YUV_420_888` analysis stream (`AGENTS.md` §3). */
        const val TARGET_WIDTH_1080P = 1920

        /** Target height (`1080 px`) of the `16:9` landscape `YUV_420_888` analysis stream (`AGENTS.md` §3). */
        const val TARGET_HEIGHT_1080P = 1080

        /**
         * Maximum manual exposure time (`2,000,000 ns` = `1/500 s`) when `EXP CLAMP` is enabled.
         * Relaxed from the outdoor `1/2000 s` (`500,000 ns`) spec ceiling during Milestone 1B hardware
         * bring-up (`docs/ENGINEERING_LEDGER.md`, commit `8707282`) so indoor/shaded scenes maintain image brightness.
         */
        const val MAX_SHUTTER_EXPOSURE_NS = 2_000_000L // 1/500 s = 2,000,000 ns

        /** Minimum selectable canopy EV bias (`-2.0 EV`). */
        const val MIN_CANOPY_EV_BIAS = -2.0f

        /** Maximum selectable canopy EV bias (`+1.0 EV`). */
        const val MAX_CANOPY_EV_BIAS = 1.0f

        /**
         * Optical vertical focal length at `2.0x` crop on `1920x1080` (`~20.5 deg` VFOV -> `540 / tan(10.25 deg) = 2985 px`).
         */
        val DEFAULT_FY_2X_1080P_PX: Float =
            ((TARGET_HEIGHT_1080P / 2.0) / tan(Math.toRadians(10.25))).toFloat()

        /**
         * Computes the nominal vertical pinhole focal length $f_y$ in `1080p` stream pixels for [zoomRatio]
         * when hardware `LENS_INTRINSIC_CALIBRATION` is unavailable.
         *
         * @param zoomRatio Active digital zoom ratio (clamped to `1.0..8.0`).
         * @return Estimated vertical focal length in pixels (`~2985 px` at `2.0x`, `~1492 px` at `1.0x`).
         */
        fun fallbackVerticalFocalLengthPx(zoomRatio: Float): Float {
            val safeZoom = zoomRatio.coerceIn(1.0f, 8.0f)
            return DEFAULT_FY_2X_1080P_PX * (safeZoom / 2.0f)
        }

        /**
         * Coerces an arbitrary Android [Surface] rotation constant to either [Surface.ROTATION_270]
         * or [Surface.ROTATION_90] for reversible-landscape operation.
         */
        fun normalizeLandscapeRotation(rotation: Int): Int {
            return if (rotation == Surface.ROTATION_270) {
                Surface.ROTATION_270
            } else {
                Surface.ROTATION_90
            }
        }

        /**
         * Scales the sensor-native vertical focal length (`LENS_INTRINSIC_CALIBRATION[1]`) into
         * `1920x1080` (`16:9`) output stream pixels, accounting for both `SCALER_CROP_REGION` digital
         * zoom and the vertical letterbox crop from a `4:3` active sensor array to `16:9`.
         * Falls back to [fallbackVerticalFocalLengthPx] if calibration arrays are missing or out of bounds.
         *
         * @param intrinsics 5-element `[fx, fy, cx, cy, s]` array from `TotalCaptureResult.LENS_INTRINSIC_CALIBRATION`, or `null`.
         * @param cropRegion Active `SCALER_CROP_REGION` in sensor pixel coordinates, or `null`.
         * @param activeArray Full `SENSOR_INFO_ACTIVE_ARRAY_SIZE` rectangle, or `null`.
         * @param activeZoomRatio Current digital zoom ratio (`2.0f` or `1.0f`).
         * @param outputStreamHeightPx Height of the analyzed stream in pixels (defaults to [TARGET_HEIGHT_1080P]).
         * @return Vertical focal length $f_y$ in output stream pixels.
         */
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
