/*
 * AndroidViewModel coordinating the live push-broom camera stream, 120 Hz IMU pitch stabilization,
 * tile cropping, stone detection, 3D temporal voting, audio sonification, and field strip logging.
 *
 * Exposes immutable ScannerUiState to Compose while decoupling display-rate EIS translation and
 * 120 Hz world-ray reticle reprojection from analyzer-rate inference sweeps.
 *
 * Main types:
 * - DetectorSourceOption: selectable live GPU and synthetic simulation sources in the LCARS SOURCE drawer.
 * - ScannerUiState: immutable UI state snapshot observed by ScannerScreen.
 * - ScannerViewModel: lifecycle-aware coordinator owning camera, detector, raycaster, tracker, sonifier, and logger.
 */
package com.gneissfind.app.ui

import android.app.Application
import android.view.Surface
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import com.gneissfind.app.audio.MetalDetectorSonifier
import com.gneissfind.app.camera.CameraExposureTelemetry
import com.gneissfind.app.camera.PushBroomCameraController
import com.gneissfind.app.inference.RfDetrTwoGraphGpuDetector
import com.gneissfind.app.inference.SyntheticStoneDetector
import com.gneissfind.app.inference.SyntheticTargetProfile
import com.gneissfind.app.logging.FieldDataLogger
import com.gneissfind.app.logging.FieldStripMetadata
import com.gneissfind.app.model.ConfirmedTarget
import com.gneissfind.app.model.DetectorBackend
import com.gneissfind.app.model.OperatingMode
import com.gneissfind.app.model.ScanSpeedState
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripDetection
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.SwathTileMode
import com.gneissfind.app.tracking.GroundPlaneRaycaster
import com.gneissfind.app.tracking.ProjectedCandidate
import com.gneissfind.app.tracking.TemporalTargetTracker
import com.gneissfind.app.vision.ImageProxyYuv420Frame
import com.gneissfind.app.vision.PushBroomCropper
import com.gneissfind.app.vision.PushBroomSweepBuffers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlin.math.atan2
import kotlin.math.roundToInt

/**
 * Selectable detection input sources presented inside the scrollable LCARS `SOURCE` drawer.
 *
 * @property code Short LCARS index prefix displayed on the drawer row (`S-01`..`S-05`).
 * @property title Primary uppercase label shown on the drawer row and collapsed `SOURCE` button.
 * @property subtitle Secondary description displayed inside the expanded selection drawer.
 * @property backend Underlying [DetectorBackend] activated when this source is selected.
 * @property syntheticProfile Optional [SyntheticTargetProfile] when [backend] is synthetic.
 */
enum class DetectorSourceOption(
    val code: String,
    val title: String,
    val subtitle: String,
    val backend: DetectorBackend,
    val syntheticProfile: SyntheticTargetProfile?
) {
    LIVE_CAMERA_GPU(
        code = "S-01",
        title = "RF-DETR NANO GPU",
        subtitle = "COCO PROXY • REALTIME CAMERA",
        backend = DetectorBackend.RF_DETR_NANO_GPU,
        syntheticProfile = null
    ),
    SIM_MIXED_GRAVEL(
        code = "S-02",
        title = "MIXED GRAVEL SIM",
        subtitle = "MULTI-CLASS SYNTHETIC WALK STREAM",
        backend = DetectorBackend.SYNTHETIC_DEMO,
        syntheticProfile = SyntheticTargetProfile.MIXED_GRAVEL_WALK
    ),
    SIM_AGATE_20MM(
        code = "S-03",
        title = "20MM AGATE (780HZ)",
        subtitle = "SMALL PEBBLE • HIGH-PITCH SONIFIER TEST",
        backend = DetectorBackend.SYNTHETIC_DEMO,
        syntheticProfile = SyntheticTargetProfile.PEBBLE_20MM_HIGH_PITCH
    ),
    SIM_JASPER_55MM(
        code = "S-04",
        title = "55MM JASPER (320HZ)",
        subtitle = "LARGE COBBLE • LOW-PITCH SONIFIER TEST",
        backend = DetectorBackend.SYNTHETIC_DEMO,
        syntheticProfile = SyntheticTargetProfile.COBBLE_55MM_LOW_PITCH
    ),
    SIM_CONFUSER_MUTE(
        code = "S-05",
        title = "CONFUSER MUTE SIM",
        subtitle = "BASALT / QUARTZITE • SILENT REJECTION",
        backend = DetectorBackend.SYNTHETIC_DEMO,
        syntheticProfile = SyntheticTargetProfile.QUIET_GRAVEL_CONFUSERS
    )
}

/**
 * Immutable snapshot of the scanner's complete state consumed by `ScannerScreen`.
 *
 * @property operatingMode Active operating mode ([OperatingMode.SCAN] or [OperatingMode.LOG_DATA]).
 * @property detectorBackend Active inference backend ([DetectorBackend.RF_DETR_NANO_GPU] default).
 * @property syntheticProfile Active simulation scenario when [detectorBackend] is [DetectorBackend.SYNTHETIC_DEMO].
 * @property detectorStatus Human-readable backend status string rendered in the LCARS `SOURCE` drawer.
 * @property enabledClasses Subset of user-alertable [StoneClass]es (`01..04`) currently enabled in the left rail.
 * @property swathTileMode Active horizontal tile swath ([SwathTileMode.FIVE_TILE] `1792x384` or [SwathTileMode.THREE_TILE] `1088x384`).
 * @property shutterClampEnabled Whether manual `<= 1/500 s` shutter clamping (`EXP CLAMP`) is enabled.
 * @property twoXCropEnabled Whether `2.0x` lossless center sensor crop (`2X CROP`) is enabled.
 * @property eisPitchLockEnabled Whether IMU software EIS pitch stabilization (`EIS LOCK`) is enabled.
 * @property exposureBiasEv Active canopy exposure compensation bias in EV stops (`-2.0..+1.0 EV`).
 * @property isAudioMuted Whether `MetalDetectorSonifier` audio output is muted.
 * @property simulatedSpeedOverrideMps Non-null when the synthetic overspeed test (`OVERSPEED SIM`) is active.
 * @property displaySurfaceRotation Active reversible-landscape [Surface] rotation ([Surface.ROTATION_90] or [Surface.ROTATION_270]).
 * @property stripGeometry Active push-broom strip ROI geometry inside the `1920x1080` camera frame.
 * @property rawStripDetections Unconfirmed single-sweep detections in unified strip pixel coordinates.
 * @property confirmedTargets Confirmed targets (with 120 Hz ego-motion-reprojected strip coordinates and visibility flags).
 * @property speedState Current walking speed state ([ScanSpeedState.Normal] or [ScanSpeedState.SpeedWarning]).
 * @property imuPitchDeg Smoothed downward optical pitch in degrees (`0` = horizon, `45` = nominal, `90` = nadir).
 * @property rollDeg Smoothed landscape screen roll angle in degrees.
 * @property referencePitchDeg Active EIS pitch setpoint in degrees (`45.0 deg` default).
 * @property cropY Quantized integer top Y offset (`0..696 px`) of the stabilized `384 px` strip on the `1080p` frame.
 * @property displayCropYFloat Continuous sub-pixel top Y offset (`0f..696f px`) driving 120 Hz GPU `PreviewView.translationY`.
 * @property isPitchClamped `true` when [displayCropYFloat] hits `0` or [StripRoiGeometry.MAX_CROP_Y_PX] (`PITCH LIMIT`).
 * @property verticalFocalLengthPx Calibrated vertical focal length $f_y$ in `1080p` stream pixels.
 * @property slantRangeMeters Estimated optical slant range $R$ in meters along the strip center.
 * @property gsdMmPerPx Estimated ground sample distance in millimeters per pixel (`0.59 mm/px` nominal).
 * @property sweepLatencyMs Wall-clock latency of the most recent inference sweep in milliseconds.
 * @property rollingLatencyMs Exponential moving average of sweep latency in milliseconds.
 * @property sweepHz Smoothed camera analyzer sweep rate in Hz.
 * @property part1LatencyMs Latency of Part 1 (backbone + encoder) in milliseconds.
 * @property bridgeLatencyMs Latency of the Host CPU Top-K query bridge in milliseconds.
 * @property part2LatencyMs Latency of Part 2 (decoder + head) in milliseconds.
 * @property exposureTimeNs Current sensor exposure time in nanoseconds.
 * @property sensorIso Current sensor sensitivity in ISO units.
 * @property activeZoomRatio Current CameraX zoom ratio (`2.0f` or `1.0f`).
 * @property audioPitchHz Current `MetalDetectorSonifier` tone frequency in Hz (`0f` when silent).
 * @property audioCadenceHz Current `MetalDetectorSonifier` ping repetition rate in Hz (`0f` when silent).
 * @property loggedFrameCount Number of lossless PNG strip frames saved in the current `LOG DATA` session.
 * @property fieldLogDirectory Absolute filesystem path of the active (or base) `LOG DATA` session directory.
 */
data class ScannerUiState(
    val operatingMode: OperatingMode = OperatingMode.SCAN,
    val detectorBackend: DetectorBackend = DetectorBackend.RF_DETR_NANO_GPU,
    val syntheticProfile: SyntheticTargetProfile = SyntheticTargetProfile.MIXED_GRAVEL_WALK,
    val detectorStatus: String = "RF-DETR NANO GPU [COCO PROXY]",
    val enabledClasses: Set<StoneClass> = StoneClass.alertableClasses.toSet(),
    val swathTileMode: SwathTileMode = SwathTileMode.FIVE_TILE,
    val shutterClampEnabled: Boolean = true,
    val twoXCropEnabled: Boolean = true,
    val eisPitchLockEnabled: Boolean = true,
    val exposureBiasEv: Float = 0.0f,
    val isAudioMuted: Boolean = false,
    val simulatedSpeedOverrideMps: Float? = null,
    val displaySurfaceRotation: Int = Surface.ROTATION_90,
    val stripGeometry: StripRoiGeometry = StripRoiGeometry(),
    val rawStripDetections: List<StripDetection> = emptyList(),
    val confirmedTargets: List<ConfirmedTarget> = emptyList(),
    val speedState: ScanSpeedState = ScanSpeedState.Normal(0f),
    val imuPitchDeg: Float = GroundPlaneRaycaster.DEFAULT_PITCH_DEGREES,
    val rollDeg: Float = 0f,
    val referencePitchDeg: Float = GroundPlaneRaycaster.DEFAULT_PITCH_DEGREES,
    val cropY: Int = StripRoiGeometry.NOMINAL_CROP_Y_PX,
    val displayCropYFloat: Float = StripRoiGeometry.NOMINAL_CROP_Y_PX.toFloat(),
    val isPitchClamped: Boolean = false,
    val verticalFocalLengthPx: Float = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX,
    val slantRangeMeters: Float = PushBroomCropper.DEFAULT_SLANT_RANGE_METERS,
    val gsdMmPerPx: Float = PushBroomCropper.DEFAULT_GSD_MM_PER_PX,
    val sweepLatencyMs: Float = 0f,
    val rollingLatencyMs: Float = 0f,
    val sweepHz: Float = 0f,
    val part1LatencyMs: Float = 0f,
    val bridgeLatencyMs: Float = 0f,
    val part2LatencyMs: Float = 0f,
    val exposureTimeNs: Long = PushBroomCameraController.MAX_SHUTTER_EXPOSURE_NS,
    val sensorIso: Int = 400,
    val activeZoomRatio: Float = 2.0f,
    val audioPitchHz: Float = 0f,
    val audioCadenceHz: Float = 0f,
    val loggedFrameCount: Int = 0,
    val fieldLogDirectory: String = ""
) {
    /** Resolves the currently active [DetectorSourceOption] from [detectorBackend] and [syntheticProfile]. */
    val activeSourceOption: DetectorSourceOption
        get() = if (detectorBackend == DetectorBackend.RF_DETR_NANO_GPU) {
            DetectorSourceOption.LIVE_CAMERA_GPU
        } else {
            DetectorSourceOption.entries.firstOrNull {
                it.backend == DetectorBackend.SYNTHETIC_DEMO && it.syntheticProfile == syntheticProfile
            } ?: DetectorSourceOption.SIM_MIXED_GRAVEL
        }
}

/**
 * Primary [AndroidViewModel] orchestrating the GneissFind perception, tracking, sonification, and logging pipeline.
 *
 * Threading & Decoupled Cadence:
 * - **120 Hz IMU Display Loop (Main Thread):** [GroundPlaneRaycaster.onRealtimeImuPitchListener] updates
 *   [ScannerUiState.displayCropYFloat] and reprojects [ScannerUiState.confirmedTargets] through the live `3x3`
 *   rotation matrix so the GPU `PreviewView` and target reticles remain ego-motion-locked at display refresh rate.
 * - **Analyzer Sweep Loop (`GneissFind-Analyzer` Thread):** [processCameraFrameOnAnalyzerThread] extracts the
 *   stabilized YUV strip into [sweepBuffers], runs [RfDetrTwoGraphGpuDetector] or [SyntheticStoneDetector],
 *   merges cross-tile boxes via [PushBroomCropper], back-projects 3D world rays at the frame's capture timestamp,
 *   updates [TemporalTargetTracker], and drives [MetalDetectorSonifier].
 */
class ScannerViewModel(application: Application) : AndroidViewModel(application) {

    private val appContext = application.applicationContext

    private val cropper = PushBroomCropper(applyImageNetNormalization = false)
    private val sweepBuffers = PushBroomSweepBuffers()

    private val syntheticDetector = SyntheticStoneDetector()
    private val rfDetrGpuDetector = RfDetrTwoGraphGpuDetector(appContext)

    private val raycaster = GroundPlaneRaycaster(appContext)
    private val tracker = TemporalTargetTracker()
    private val sonifier = MetalDetectorSonifier()
    private val fieldLogger = FieldDataLogger(appContext)

    private val cameraController = PushBroomCameraController(
        appContext = appContext,
        onFrameAvailable = ::processCameraFrameOnAnalyzerThread
    )

    private var lastFrameTimestampNs: Long = 0L
    private var smoothedSweepHz: Float = 0f
    private val displayRotationMatrixScratch = FloatArray(9)

    private val _uiState = MutableStateFlow(
        ScannerUiState(
            fieldLogDirectory = FieldDataLogger.resolveBaseSessionsDir(appContext).absolutePath
        )
    )

    /** Hot state stream of [ScannerUiState] collected by `ScannerScreen`. */
    val uiState: StateFlow<ScannerUiState> = _uiState.asStateFlow()

    init {
        raycaster.onRealtimeImuPitchListener = { smoothedPitchDeg, rollDeg, refPitchDeg ->
            var updatedVisibleTargetsForAudio: List<ConfirmedTarget>? = null
            var currentSpeedForAudio: ScanSpeedState = ScanSpeedState.Normal(0f)
            _uiState.update { prev ->
                val fy = prev.verticalFocalLengthPx
                val floatStab = GroundPlaneRaycaster.computeStabilizedCropYFloat(
                    currentPitchDeg = smoothedPitchDeg,
                    referencePitchDeg = refPitchDeg,
                    verticalFocalLengthPx = fy,
                    eisEnabled = prev.eisPitchLockEnabled
                )
                val stripCenterOffsetPx = floatStab.clampedCropYFloat - StripRoiGeometry.NOMINAL_CROP_Y_PX
                val lookPitchDeg = smoothedPitchDeg + Math.toDegrees(
                    atan2(stripCenterOffsetPx.toDouble(), fy.toDouble())
                ).toFloat()
                val slantR = GroundPlaneRaycaster.computeSlantRangeForPitch(
                    forwardPitchDeg = lookPitchDeg
                )
                val gsd = PushBroomCropper.computeGsdMmPerPx(slantR, prev.activeZoomRatio)
                val reprojectedTargets = if (prev.confirmedTargets.isNotEmpty()) {
                    val liveMat = raycaster.copyLatestRotationMatrix(displayRotationMatrixScratch)
                    raycaster.reprojectConfirmedTargetsToCurrentPose(
                        targets = prev.confirmedTargets,
                        stripGeometry = prev.stripGeometry,
                        currentCropYFloat = floatStab.clampedCropYFloat,
                        currentFocalLengthPx = fy,
                        currentRotationMatrix = liveMat
                    )
                } else {
                    prev.confirmedTargets
                }
                val prevVisibleCount = prev.confirmedTargets.count { it.isVisibleInStrip }
                val newVisibleCount = reprojectedTargets.count { it.isVisibleInStrip }
                if (prevVisibleCount != newVisibleCount) {
                    updatedVisibleTargetsForAudio = reprojectedTargets.filter { it.isVisibleInStrip }
                    currentSpeedForAudio = prev.speedState
                }
                prev.copy(
                    imuPitchDeg = smoothedPitchDeg,
                    rollDeg = rollDeg,
                    referencePitchDeg = refPitchDeg,
                    cropY = floatStab.clampedCropYFloat.roundToInt()
                        .coerceIn(0, StripRoiGeometry.MAX_CROP_Y_PX),
                    displayCropYFloat = floatStab.clampedCropYFloat,
                    isPitchClamped = floatStab.isPitchClamped,
                    slantRangeMeters = slantR,
                    gsdMmPerPx = gsd,
                    confirmedTargets = reprojectedTargets
                )
            }
            updatedVisibleTargetsForAudio?.let { visibleTargets ->
                sonifier.updateTargets(visibleTargets, currentSpeedForAudio)
            }
        }
    }

    /** Binds the CameraX preview and `YUV_420_888` analyzer stream to [lifecycleOwner] and [previewView]. */
    fun bindCamera(
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView,
        initialSurfaceRotation: Int = _uiState.value.displaySurfaceRotation
    ) {
        cameraController.bindToLifecycle(lifecycleOwner, previewView, initialSurfaceRotation)
    }

    /**
     * Synchronizes CameraX `targetRotation`, ARCore `setDisplayGeometry`, and IMU `remapCoordinateSystem`
     * whenever reversible `sensorLandscape` flips between [Surface.ROTATION_90] and [Surface.ROTATION_270].
     */
    fun onDisplayGeometryChanged(
        surfaceRotation: Int,
        viewportWidthPx: Int = 0,
        viewportHeightPx: Int = 0
    ) {
        val normalized = PushBroomCameraController.normalizeLandscapeRotation(surfaceRotation)
        cameraController.updateTargetRotation(normalized)
        raycaster.updateDisplayGeometry(
            surfaceRotation = normalized,
            viewportWidthPx = viewportWidthPx,
            viewportHeightPx = viewportHeightPx
        )
        _uiState.update { it.copy(displaySurfaceRotation = normalized) }
    }

    /** Starts the 125 Hz IMU listener and the background `AudioTrack` sonification thread on Activity `ON_RESUME`. */
    fun onResumeSensorsAndAudio() {
        raycaster.startImuListening()
        sonifier.start()
    }

    /** Silences and stops the `AudioTrack` sonification thread and unregisters IMU listeners on Activity `ON_PAUSE`. */
    fun onPauseSensorsAndAudio() {
        sonifier.updateTargets(emptyList(), ScanSpeedState.Normal(0f))
        sonifier.stop()
        raycaster.stopImuListening()
    }

    /**
     * Switches between [OperatingMode.SCAN] and [OperatingMode.LOG_DATA].
     * Entering [OperatingMode.LOG_DATA] automatically enables the shutter clamp and opens a new [FieldDataLogger] session;
     * returning to [OperatingMode.SCAN] closes the active logging session.
     */
    fun setOperatingMode(mode: OperatingMode) {
        if (mode == OperatingMode.LOG_DATA) {
            cameraController.setShutterClamp(true)
            val dir = fieldLogger.startSession()
            _uiState.update {
                it.copy(
                    operatingMode = mode,
                    shutterClampEnabled = true,
                    loggedFrameCount = 0,
                    fieldLogDirectory = dir.absolutePath
                )
            }
        } else {
            fieldLogger.stopSession()
            _uiState.update {
                it.copy(
                    operatingMode = mode,
                    loggedFrameCount = fieldLogger.savedFrameCount
                )
            }
        }
    }

    /** Switches the active [DetectorBackend] and resets [TemporalTargetTracker] tracks. */
    fun setDetectorBackend(backend: DetectorBackend) {
        tracker.reset()
        _uiState.update { it.copy(detectorBackend = backend) }
    }

    /** Selects a [SyntheticTargetProfile], switches to [DetectorBackend.SYNTHETIC_DEMO], and resets tracks. */
    fun setSyntheticProfile(profile: SyntheticTargetProfile) {
        syntheticDetector.profile = profile
        tracker.reset()
        _uiState.update {
            it.copy(
                detectorBackend = DetectorBackend.SYNTHETIC_DEMO,
                syntheticProfile = profile
            )
        }
    }

    /**
     * Activates the chosen [DetectorSourceOption] from the scrollable LCARS `SOURCE` drawer and
     * clears any prior spatial target tracks.
     */
    fun selectSourceOption(option: DetectorSourceOption) {
        tracker.reset()
        option.syntheticProfile?.let { profile ->
            syntheticDetector.profile = profile
        }
        _uiState.update { prev ->
            prev.copy(
                detectorBackend = option.backend,
                syntheticProfile = option.syntheticProfile ?: prev.syntheticProfile
            )
        }
    }

    /** Toggles between 5-tile (`1792x384`) and 3-tile (`1088x384`) swath geometry and resets active tracks. */
    fun toggleSwathTileMode() {
        val next = _uiState.value.swathTileMode.toggled()
        tracker.reset()
        _uiState.update { state ->
            state.copy(
                swathTileMode = next,
                stripGeometry = state.stripGeometry.copy(
                    stripLeftPx = next.canonicalStripLeftPx,
                    stripWidthPx = next.stripWidthPx,
                    tileMode = next
                )
            )
        }
    }

    /** Enables or disables IMU software EIS pitch stabilization (`EIS LOCK` / `EIS FREE`). */
    fun setEisPitchLockEnabled(enabled: Boolean) {
        raycaster.eisPitchLockEnabled = enabled
        _uiState.update { prev ->
            val floatStab = GroundPlaneRaycaster.computeStabilizedCropYFloat(
                currentPitchDeg = prev.imuPitchDeg,
                referencePitchDeg = prev.referencePitchDeg,
                verticalFocalLengthPx = prev.verticalFocalLengthPx,
                eisEnabled = enabled
            )
            prev.copy(
                eisPitchLockEnabled = enabled,
                cropY = floatStab.clampedCropYFloat.roundToInt()
                    .coerceIn(0, StripRoiGeometry.MAX_CROP_Y_PX),
                displayCropYFloat = floatStab.clampedCropYFloat,
                isPitchClamped = floatStab.isPitchClamped
            )
        }
    }

    /** Locks the current IMU inclination as the new EIS reference pitch (`SYNC PITCH`) and recenters `cropY` to `348`. */
    fun syncReferencePitch() {
        val synced = raycaster.syncReferencePitch()
        _uiState.update {
            it.copy(
                imuPitchDeg = synced,
                referencePitchDeg = synced,
                cropY = StripRoiGeometry.NOMINAL_CROP_Y_PX,
                displayCropYFloat = StripRoiGeometry.NOMINAL_CROP_Y_PX.toFloat(),
                isPitchClamped = false
            )
        }
    }

    /** Toggles whether a user-alertable [StoneClass] (`01..04`) is enabled in the left mineral rail. */
    fun toggleStoneClass(stoneClass: StoneClass) {
        if (!stoneClass.isUserAlertable) return
        _uiState.update { state ->
            val updated = state.enabledClasses.toMutableSet()
            if (!updated.add(stoneClass)) {
                updated.remove(stoneClass)
            }
            state.copy(enabledClasses = updated)
        }
    }

    /** Enables or disables the Camera2 `<= 1/500 s` shutter clamp (`EXP CLAMP` / `EXP AUTO`). */
    fun setShutterClampEnabled(enabled: Boolean) {
        cameraController.setShutterClamp(enabled)
        _uiState.update { it.copy(shutterClampEnabled = enabled) }
    }

    /** Enables or disables the `2.0x` lossless center sensor crop (`2X CROP` / `1X WIDE`). */
    fun setTwoXCropEnabled(enabled: Boolean) {
        cameraController.setTwoXSensorCrop(enabled)
        val zoom = if (enabled) 2.0f else 1.0f
        _uiState.update {
            it.copy(
                twoXCropEnabled = enabled,
                activeZoomRatio = zoom,
                gsdMmPerPx = PushBroomCropper.computeGsdMmPerPx(it.slantRangeMeters, zoom)
            )
        }
    }

    /** Updates the canopy exposure compensation bias (`CANOPY EV`) in EV stops. */
    fun setExposureBiasEv(ev: Float) {
        val clamped = ev.coerceIn(
            PushBroomCameraController.MIN_CANOPY_EV_BIAS,
            PushBroomCameraController.MAX_CANOPY_EV_BIAS
        )
        cameraController.setExposureBiasEv(clamped)
        _uiState.update { it.copy(exposureBiasEv = clamped) }
    }

    /** Mutes or unmutes `MetalDetectorSonifier` audio output (`AUDIO ON` / `MUTED`). */
    fun setAudioMuted(muted: Boolean) {
        sonifier.isMuted = muted
        _uiState.update { it.copy(isAudioMuted = muted) }
    }

    /** Sets or clears a simulated walking speed override (`m/s`) to test the `> 1.10 m/s` overspeed gate. */
    fun setSimulatedWalkSpeedOverride(speedMps: Float?) {
        raycaster.simulatedWalkSpeedOverrideMps = speedMps
        val immediateSpeedState = when {
            speedMps == null -> ScanSpeedState.Normal(0f)
            speedMps > TemporalTargetTracker.MAX_SWEEP_SPEED_MPS -> ScanSpeedState.SpeedWarning(speedMps)
            else -> ScanSpeedState.Normal(speedMps)
        }
        sonifier.updateTargets(
            confirmedTargets = _uiState.value.confirmedTargets,
            speedState = immediateSpeedState
        )
        val audioTel = sonifier.latestTelemetry
        _uiState.update {
            it.copy(
                simulatedSpeedOverrideMps = speedMps,
                speedState = immediateSpeedState,
                audioPitchHz = audioTel.currentPitchHz,
                audioCadenceHz = audioTel.currentPingCadenceHz
            )
        }
    }

    /** Toggles the `1.38 m/s` simulated overspeed test (`OVERSPEED SIM`) in the LCARS `SOURCE` drawer. */
    fun toggleOverspeedTest() {
        val next = if (_uiState.value.simulatedSpeedOverrideMps != null) null else 1.38f
        setSimulatedWalkSpeedOverride(next)
    }

    /**
     * Executes one full perception-and-tracking sweep on the single-threaded `GneissFind-Analyzer` worker
     * for each incoming `1920x1080` `YUV_420_888` [imageProxy] and always closes [imageProxy] in `finally`.
     */
    private fun processCameraFrameOnAnalyzerThread(
        imageProxy: ImageProxy,
        exposureTelemetry: CameraExposureTelemetry
    ) {
        try {
            val stateSnapshot = _uiState.value
            val isLoggingMode = stateSnapshot.operatingMode == OperatingMode.LOG_DATA
            val tileMode = stateSnapshot.swathTileMode
            val frameTs = imageProxy.imageInfo.timestamp

            if (lastFrameTimestampNs != 0L && frameTs > lastFrameTimestampNs) {
                val dtSec = (frameTs - lastFrameTimestampNs) * 1e-9f
                if (dtSec in 0.005f..1.0f) {
                    val instHz = 1.0f / dtSec
                    smoothedSweepHz = if (smoothedSweepHz <= 0.1f) {
                        instHz
                    } else {
                        smoothedSweepHz * 0.80f + instHz * 0.20f
                    }
                }
            }
            lastFrameTimestampNs = frameTs

            // 1. Capture timestamp-matched IMU pose, stabilized cropY in [0..696], slant range R, & speed
            val poseSnapshot = raycaster.capturePoseSnapshot(
                arFrame = null,
                fallbackTimestampNs = frameTs,
                verticalFocalLengthPx = exposureTelemetry.verticalFocalLengthPx
            )
            val activeZoom = exposureTelemetry.activeZoomRatio
            val centerGsdMmPerPx = PushBroomCropper.computeGsdMmPerPx(
                slantRangeMeters = poseSnapshot.centerSlantRangeMeters,
                zoomRatio = activeZoom
            )

            // 2. Zero-copy YUV_420_888 stabilized strip extraction (1792x384 5-tile or 1088x384 3-tile)
            val yuvFrame = ImageProxyYuv420Frame(imageProxy)
            val geometry = cropper.extractTiles(
                frame = yuvFrame,
                outBuffers = sweepBuffers,
                cropY = poseSnapshot.cropY,
                tileMode = tileMode,
                isPitchClamped = poseSnapshot.isPitchClamped,
                populateArgbStrip = isLoggingMode
            )

            // 3. Optional LOG DATA lossless strip + metadata.jsonl persistence (4 Hz)
            if (isLoggingMode && sweepBuffers.hasArgbPixels) {
                fieldLogger.maybeRecordStripFrame(
                    stripArgbPixels = sweepBuffers.stripArgbPixels,
                    metadata = FieldStripMetadata(
                        timestampNs = yuvFrame.timestampNs,
                        wallTimeMs = System.currentTimeMillis(),
                        stripWidthPx = tileMode.stripWidthPx,
                        stripHeightPx = StripRoiGeometry.STRIP_HEIGHT_PX,
                        tileCount = tileMode.tileCount,
                        zoomRatio = activeZoom,
                        exposureTimeNs = exposureTelemetry.exposureTimeNs,
                        sensorIso = exposureTelemetry.sensorIso,
                        shutterClamped = stateSnapshot.shutterClampEnabled,
                        exposureBiasEv = stateSnapshot.exposureBiasEv,
                        imuPitchDeg = poseSnapshot.forwardPitchDeg,
                        referencePitchDeg = poseSnapshot.referencePitchDeg,
                        cropY = poseSnapshot.cropY,
                        isPitchClamped = poseSnapshot.isPitchClamped,
                        slantRangeMeters = poseSnapshot.centerSlantRangeMeters,
                        gsdMmPerPx = centerGsdMmPerPx,
                        horizontalSpeedMps = poseSnapshot.horizontalSpeedMps
                    )
                )
            }

            // 4. Run active StoneDetector across the 5 (or 3) tiles
            val activeDetector = when (stateSnapshot.detectorBackend) {
                DetectorBackend.SYNTHETIC_DEMO -> syntheticDetector
                DetectorBackend.RF_DETR_NANO_GPU -> rfDetrGpuDetector
            }
            val inferenceResult = activeDetector.detectSweep(sweepBuffers)

            // 5. Map tile boxes to unified strip, run Cross-Tile NMS (IoU >= 0.45),
            //    mute Class 4 (hard_confuser), and enforce >= 20 mm physical diameter
            val filteredDetections = cropper.mergeAndFilterTileDetections(
                tileDetections = inferenceResult.tileDetections,
                tileMode = tileMode,
                gsdMmPerPx = centerGsdMmPerPx,
                enabledClasses = stateSnapshot.enabledClasses
            )

            // 6. Raycast detections onto ground plane & update 3-of-6 spatial voting + 1.10 m/s speed gate
            val anchorWorldRays = stateSnapshot.detectorBackend == DetectorBackend.RF_DETR_NANO_GPU
            val projectedCandidates = filteredDetections.map { det ->
                val raycast = raycaster.raycastStripDetection(
                    detection = det,
                    geometry = geometry,
                    poseSnapshot = poseSnapshot,
                    zoomRatio = activeZoom
                )
                ProjectedCandidate(
                    stripDetection = det,
                    worldPosition = raycast.worldPoint,
                    slantRangeMeters = raycast.slantRangeMeters,
                    refinedDiameterMm = det.maxSpanPx * raycast.gsdMmPerPx,
                    worldRay = if (anchorWorldRays) raycast.worldRay else null,
                    angularWidthRad = if (anchorWorldRays) raycast.angularWidthRad else 0f,
                    angularHeightRad = if (anchorWorldRays) raycast.angularHeightRad else 0f
                )
            }

            val trackerOutput = tracker.updateSweep(
                candidates = projectedCandidates,
                timestampNs = yuvFrame.timestampNs,
                cameraWorldPos = poseSnapshot.cameraWorldPosition,
                currentSpeedMps = poseSnapshot.horizontalSpeedMps
            )

            // Reproject confirmed targets from frame capture time (t_capture) to current live
            // display orientation (t_now) so ~390 ms GPU inference latency does not offset reticles.
            val useLiveImuDisplay = raycaster.hasLiveImuStream
            val currentDisplayCropY = if (useLiveImuDisplay) {
                _uiState.value.displayCropYFloat
            } else {
                poseSnapshot.cropY.toFloat()
            }
            val reprojectedConfirmed = if (anchorWorldRays && trackerOutput.confirmedTargets.isNotEmpty()) {
                raycaster.reprojectConfirmedTargetsToCurrentPose(
                    targets = trackerOutput.confirmedTargets,
                    stripGeometry = geometry,
                    currentCropYFloat = currentDisplayCropY,
                    currentFocalLengthPx = exposureTelemetry.verticalFocalLengthPx
                )
            } else {
                trackerOutput.confirmedTargets
            }

            // 7. Drive AudioTrack sonifier (strictly reserved for stone pings & overspeed warning)
            sonifier.updateTargets(
                confirmedTargets = reprojectedConfirmed.filter { it.isVisibleInStrip },
                speedState = trackerOutput.speedState
            )
            val audioTel = sonifier.latestTelemetry

            _uiState.update { prev ->
                prev.copy(
                    detectorStatus = activeDetector.statusSummary,
                    stripGeometry = geometry,
                    rawStripDetections = filteredDetections,
                    confirmedTargets = reprojectedConfirmed,
                    speedState = trackerOutput.speedState,
                    imuPitchDeg = if (useLiveImuDisplay) prev.imuPitchDeg else poseSnapshot.forwardPitchDeg,
                    rollDeg = if (useLiveImuDisplay) prev.rollDeg else poseSnapshot.rollDeg,
                    referencePitchDeg = if (useLiveImuDisplay) prev.referencePitchDeg else poseSnapshot.referencePitchDeg,
                    cropY = if (useLiveImuDisplay) prev.cropY else poseSnapshot.cropY,
                    displayCropYFloat = if (useLiveImuDisplay) {
                        prev.displayCropYFloat
                    } else {
                        poseSnapshot.cropY.toFloat()
                    },
                    isPitchClamped = if (useLiveImuDisplay) prev.isPitchClamped else poseSnapshot.isPitchClamped,
                    verticalFocalLengthPx = exposureTelemetry.verticalFocalLengthPx,
                    slantRangeMeters = if (useLiveImuDisplay) prev.slantRangeMeters else poseSnapshot.centerSlantRangeMeters,
                    gsdMmPerPx = if (useLiveImuDisplay) {
                        PushBroomCropper.computeGsdMmPerPx(prev.slantRangeMeters, activeZoom)
                    } else {
                        centerGsdMmPerPx
                    },
                    sweepLatencyMs = inferenceResult.totalSweepLatencyMs,
                    rollingLatencyMs = inferenceResult.rollingAvgLatencyMs,
                    sweepHz = smoothedSweepHz,
                    part1LatencyMs = inferenceResult.part1LatencyMs,
                    bridgeLatencyMs = inferenceResult.hostTopKBridgeLatencyMs,
                    part2LatencyMs = inferenceResult.part2LatencyMs,
                    exposureTimeNs = exposureTelemetry.exposureTimeNs,
                    sensorIso = exposureTelemetry.sensorIso,
                    activeZoomRatio = activeZoom,
                    audioPitchHz = audioTel.currentPitchHz,
                    audioCadenceHz = audioTel.currentPingCadenceHz,
                    loggedFrameCount = fieldLogger.savedFrameCount,
                    fieldLogDirectory = fieldLogger.currentSessionPath
                )
            }
        } finally {
            imageProxy.close()
        }
    }

    override fun onCleared() {
        super.onCleared()
        sonifier.close()
        raycaster.close()
        fieldLogger.close()
        cameraController.close()
        syntheticDetector.close()
        rfDetrGpuDetector.close()
    }
}
