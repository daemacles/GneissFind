/*
 * AndroidViewModel coordinating the live push-broom camera stream, 120 Hz IMU pitch stabilization,
 * double-buffered async tile cropping, stone detection, 3D temporal voting, audio sonification,
 * and field strip logging.
 *
 * Exposes immutable ScannerUiState to Compose while decoupling display-rate EIS translation,
 * camera-rate YUV strip extraction (< 4 ms ImageProxy lifetime), and inference-rate GPU/NPU sweeps.
 *
 * Main types:
 * - DetectorSourceOption: selectable live GPU and synthetic simulation sources in the LCARS SOURCE drawer.
 * - ScannerUiState: immutable UI state snapshot observed by ScannerScreen.
 * - ScannerViewModel: lifecycle-aware coordinator owning camera, detector, raycaster, tracker, sonifier, and logger.
 */
package com.gneissfind.app.ui

import android.app.Application
import android.os.SystemClock
import android.view.Surface
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import com.gneissfind.app.audio.MetalDetectorSonifier
import com.gneissfind.app.camera.CameraExposureTelemetry
import com.gneissfind.app.camera.PushBroomCameraController
import com.gneissfind.app.inference.LiveModelProfile
import com.gneissfind.app.inference.RfDetrTwoGraphGpuDetector
import com.gneissfind.app.inference.SyntheticStoneDetector
import com.gneissfind.app.inference.SyntheticTargetProfile
import com.gneissfind.app.logging.FieldDataLogger
import com.gneissfind.app.logging.FieldStripMetadata
import com.gneissfind.app.model.AcquisitionStage
import com.gneissfind.app.model.ConfirmedTarget
import com.gneissfind.app.model.DetectorBackend
import com.gneissfind.app.model.LockOnHudState
import com.gneissfind.app.model.OperatingMode
import com.gneissfind.app.model.PatrolPowerState
import com.gneissfind.app.model.ScanSpeedState
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripDetection
import com.gneissfind.app.inference.SweepInferenceResult
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.SwathTileMode
import com.gneissfind.app.model.TileDetection
import com.gneissfind.app.model.WorldPoint3D
import com.gneissfind.app.tracking.GroundPlaneRaycaster
import com.gneissfind.app.tracking.LockOnPinpointController
import com.gneissfind.app.tracking.MotionPoseSnapshot
import com.gneissfind.app.tracking.PatrolDutyCycleController
import com.gneissfind.app.tracking.ProjectedCandidate
import com.gneissfind.app.tracking.TemporalTargetTracker
import com.gneissfind.app.vision.ImageProxyYuv420Frame
import com.gneissfind.app.vision.LockOnFrameExtraction
import com.gneissfind.app.vision.PushBroomCropper
import com.gneissfind.app.vision.PushBroomSweepBuffers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.atan2
import kotlin.math.roundToInt

/**
 * Selectable detection input sources presented inside the scrollable LCARS `SOURCE` drawer.
 *
 * @property code Short LCARS index prefix displayed on the drawer row (`S-01`..`S-09`).
 * @property title Primary uppercase label shown on the drawer row and collapsed `SOURCE` button.
 * @property subtitle Secondary description displayed inside the expanded selection drawer.
 * @property backend Underlying [DetectorBackend] activated when this source is selected.
 * @property liveProfile Optional [LiveModelProfile] when [backend] is [DetectorBackend.RF_DETR_NANO_GPU].
 * @property syntheticProfile Optional [SyntheticTargetProfile] when [backend] is synthetic.
 */
enum class DetectorSourceOption(
    val code: String,
    val title: String,
    val subtitle: String,
    val backend: DetectorBackend,
    val liveProfile: LiveModelProfile? = null,
    val syntheticProfile: SyntheticTargetProfile? = null
) {
    LIVE_CAMERA_GPU(
        code = "S-01",
        title = "RF-DETR 256 B=3 W8-DYN",
        subtitle = "APACHE-2.0 • 9.2 FPS COLD / 6.6 FPS SOAK",
        backend = DetectorBackend.RF_DETR_NANO_GPU,
        liveProfile = LiveModelProfile.RFDETR_NANO_256_B3_INTERLEAVED
    ),
    LIVE_RFDETR_320_INTERLEAVED(
        code = "S-02",
        title = "RF-DETR 320 B=3 W8-DYN",
        subtitle = "APACHE-2.0 • 100% SMALL RECALL • 7.1 FPS",
        backend = DetectorBackend.RF_DETR_NANO_GPU,
        liveProfile = LiveModelProfile.RFDETR_NANO_320_B3_INTERLEAVED
    ),
    LIVE_RFDETR_256_FULL_B5(
        code = "S-03",
        title = "RF-DETR 256 B=5 W8-DYN",
        subtitle = "APACHE-2.0 • ALL 5 TILES EVERY SWEEP • 6.3 FPS",
        backend = DetectorBackend.RF_DETR_NANO_GPU,
        liveProfile = LiveModelProfile.RFDETR_NANO_256_B5_K100
    ),
    LIVE_YOLO26S_384_GPU(
        code = "S-04",
        title = "YOLO26-S 384 B=5 GPU",
        subtitle = "AGPL-3.0 BENCHMARK • 13.1 FPS COLD / 7.8 FPS SOAK",
        backend = DetectorBackend.RF_DETR_NANO_GPU,
        liveProfile = LiveModelProfile.YOLO26S_384_B5_GPU
    ),
    LIVE_RFDETR_384_BASELINE(
        code = "S-05",
        title = "RF-DETR 384 B=5 BASE",
        subtitle = "APACHE-2.0 • M2 BASELINE (K=300 • 2.8 FPS)",
        backend = DetectorBackend.RF_DETR_NANO_GPU,
        liveProfile = LiveModelProfile.RFDETR_NANO_384_BASELINE
    ),
    SIM_MIXED_GRAVEL(
        code = "S-06",
        title = "MIXED GRAVEL SIM",
        subtitle = "MULTI-CLASS SYNTHETIC WALK STREAM",
        backend = DetectorBackend.SYNTHETIC_DEMO,
        syntheticProfile = SyntheticTargetProfile.MIXED_GRAVEL_WALK
    ),
    SIM_AGATE_20MM(
        code = "S-07",
        title = "20MM AGATE (780HZ)",
        subtitle = "SMALL PEBBLE • HIGH-PITCH SONIFIER TEST",
        backend = DetectorBackend.SYNTHETIC_DEMO,
        syntheticProfile = SyntheticTargetProfile.PEBBLE_20MM_HIGH_PITCH
    ),
    SIM_JASPER_55MM(
        code = "S-08",
        title = "55MM JASPER (320HZ)",
        subtitle = "LARGE COBBLE • LOW-PITCH SONIFIER TEST",
        backend = DetectorBackend.SYNTHETIC_DEMO,
        syntheticProfile = SyntheticTargetProfile.COBBLE_55MM_LOW_PITCH
    ),
    SIM_CONFUSER_MUTE(
        code = "S-09",
        title = "CONFUSER MUTE SIM",
        subtitle = "BASALT / QUARTZITE • SILENT REJECTION",
        backend = DetectorBackend.SYNTHETIC_DEMO,
        syntheticProfile = SyntheticTargetProfile.QUIET_GRAVEL_CONFUSERS
    )
}

/**
 * Immutable snapshot of the scanner's complete state consumed by `ScannerScreen`.
 */
data class ScannerUiState(
    val operatingMode: OperatingMode = OperatingMode.SCAN,
    val detectorBackend: DetectorBackend = DetectorBackend.RF_DETR_NANO_GPU,
    val liveModelProfile: LiveModelProfile = LiveModelProfile.RFDETR_NANO_256_B3_INTERLEAVED,
    val syntheticProfile: SyntheticTargetProfile = SyntheticTargetProfile.MIXED_GRAVEL_WALK,
    val detectorStatus: String = LiveModelProfile.RFDETR_NANO_256_B3_INTERLEAVED.statusBadge,
    val isCompilingModel: Boolean = false,
    val enabledClasses: Set<StoneClass> = StoneClass.alertableClasses.toSet(),
    val swathTileMode: SwathTileMode = SwathTileMode.FIVE_TILE,
    val shutterClampEnabled: Boolean = true,
    val twoXCropEnabled: Boolean = true,
    val eisPitchLockEnabled: Boolean = true,
    val eyesUpPatrolEnabled: Boolean = true,
    val patrolPowerState: PatrolPowerState = PatrolPowerState.PATROL_ACTIVE,
    val acquisitionStage: AcquisitionStage = AcquisitionStage.PATROL,
    val lockOnHudState: LockOnHudState = LockOnHudState(),
    val cameraStreamFps: Float = 30.0f,
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
    /** Resolves the currently active [DetectorSourceOption] from [detectorBackend], [liveModelProfile], and [syntheticProfile]. */
    val activeSourceOption: DetectorSourceOption
        get() = if (detectorBackend == DetectorBackend.RF_DETR_NANO_GPU) {
            DetectorSourceOption.entries.firstOrNull {
                it.backend == DetectorBackend.RF_DETR_NANO_GPU && it.liveProfile == liveModelProfile
            } ?: DetectorSourceOption.LIVE_CAMERA_GPU
        } else {
            DetectorSourceOption.entries.firstOrNull {
                it.backend == DetectorBackend.SYNTHETIC_DEMO && it.syntheticProfile == syntheticProfile
            } ?: DetectorSourceOption.SIM_MIXED_GRAVEL
        }
}

/**
 * Pre-allocated staging slot used by the lock-free producer-consumer handoff between
 * `GneissFind-Analyzer` (YUV strip extraction) and `GneissFind-Inference` (GPU/NPU execution).
 */
private class PreparedSweepSlot {
    val sweepBuffers = PushBroomSweepBuffers()
    val lockOnCrop320RgbFloats = FloatArray(
        PushBroomCropper.LOCK_ON_CROP_SIZE_PX * PushBroomCropper.LOCK_ON_CROP_SIZE_PX * 3
    )
    val inUseByConsumer = AtomicBoolean(false)
    var isLockOnSlot: Boolean = false
    var lockOnCropLeftPx: Int = 0
    var lockOnCropTopPx: Int = 0
    var poseSnapshot: MotionPoseSnapshot? = null
    var geometry: StripRoiGeometry = StripRoiGeometry()
    var exposureTelemetry: CameraExposureTelemetry = CameraExposureTelemetry()
    var stateSnapshot: ScannerUiState = ScannerUiState()
    var centerGsdMmPerPx: Float = PushBroomCropper.DEFAULT_GSD_MM_PER_PX
    var frameTimestampNs: Long = 0L
}

/**
 * Primary [AndroidViewModel] orchestrating the GneissFind perception, tracking, sonification, and logging pipeline.
 *
 * Threading & Decoupled Cadence:
 * - **120 Hz IMU Display Loop (Main Thread):** [GroundPlaneRaycaster.onRealtimeImuPitchListener] updates
 *   [ScannerUiState.displayCropYFloat] and reprojects [ScannerUiState.confirmedTargets] through the live `3x3`
 *   rotation matrix so the GPU `PreviewView` and target reticles remain ego-motion-locked at display refresh rate.
 * - **Camera YUV Producer (`GneissFind-Analyzer` Thread):** [processCameraFrameOnAnalyzerThread] extracts the
 *   stabilized YUV strip into a free [PreparedSweepSlot] in `~3 ms` and **closes [ImageProxy] immediately** so
 *   Camera2's Gralloc queue is never blocked during GPU inference.
 * - **GPU/NPU Inference Consumer (`GneissFind-Inference` Thread):** [drainLatestSweepOnInferenceThread] consumes
 *   the newest prepared slot with zero wait time, runs [RfDetrTwoGraphGpuDetector] or [SyntheticStoneDetector],
 *   merges cross-tile boxes, back-projects 3D world rays, updates [TemporalTargetTracker], and drives [MetalDetectorSonifier].
 */
class ScannerViewModel(application: Application) : AndroidViewModel(application) {

    private val appContext = application.applicationContext

    private val cropper = PushBroomCropper(applyImageNetNormalization = false)

    // 3-buffer ring so the camera analyzer thread and GPU inference thread never contend or allocate
    private val slotRing = Array(3) { PreparedSweepSlot() }
    private val latestReadySlot = AtomicReference<PreparedSweepSlot?>(null)
    private val inferenceRunning = AtomicBoolean(false)
    private val isShuttingDown = AtomicBoolean(false)
    private var lastProducerExtractWallMs: Long = 0L

    private val inferenceExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "GneissFind-Inference").apply {
            priority = Thread.MAX_PRIORITY - 1
        }
    }

    private val syntheticDetector = SyntheticStoneDetector()
    private val rfDetrGpuDetector = RfDetrTwoGraphGpuDetector(
        appContext = appContext,
        initialProfile = LiveModelProfile.RFDETR_NANO_256_B3_INTERLEAVED
    )

    private val raycaster = GroundPlaneRaycaster(appContext)
    private val tracker = TemporalTargetTracker()
    private val patrolDutyController = PatrolDutyCycleController()
    private val lockOnController = LockOnPinpointController()
    private val sonifier = MetalDetectorSonifier()
    private val fieldLogger = FieldDataLogger(appContext)

    // Pre-allocated analyzer scratch buffers for 30 FPS Lock-On / Pinpoint extraction
    private val analyzerLumaScratch = ByteArray(
        StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX * StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX
    )
    private val analyzerCrop320Scratch = FloatArray(
        PushBroomCropper.LOCK_ON_CROP_SIZE_PX * PushBroomCropper.LOCK_ON_CROP_SIZE_PX * 3
    )

    private val cameraController = PushBroomCameraController(
        appContext = appContext,
        onFrameAvailable = ::processCameraFrameOnAnalyzerThread
    )

    private var lastCompletedSweepWallNs: Long = 0L
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
        rfDetrGpuDetector.onCompilationStateChanged = { isCompiling, statusBadge ->
            _uiState.update { prev ->
                prev.copy(
                    isCompilingModel = isCompiling,
                    detectorStatus = statusBadge.ifEmpty { prev.detectorStatus }
                )
            }
        }
        raycaster.onRealtimeImuPitchListener = { smoothedPitchDeg, rollDeg, refPitchDeg ->
            var updatedVisibleTargetsForAudio: List<ConfirmedTarget>? = null
            var currentSpeedForAudio: ScanSpeedState = ScanSpeedState.Normal(0f)
            val nowNs = System.nanoTime()
            _uiState.update { prev ->
                val fy = prev.verticalFocalLengthPx
                val liveMat = raycaster.copyLatestRotationMatrix(displayRotationMatrixScratch)
                val floatStab = GroundPlaneRaycaster.computeStabilizedCropYFloat(
                    currentPitchDeg = smoothedPitchDeg,
                    referencePitchDeg = refPitchDeg,
                    verticalFocalLengthPx = fy,
                    eisEnabled = prev.eisPitchLockEnabled && prev.acquisitionStage == AcquisitionStage.PATROL
                )
                val stripCenterOffsetPx = floatStab.clampedCropYFloat - StripRoiGeometry.NOMINAL_CROP_Y_PX
                val lookPitchDeg = smoothedPitchDeg + Math.toDegrees(
                    atan2(stripCenterOffsetPx.toDouble(), fy.toDouble())
                ).toFloat()
                val slantR = GroundPlaneRaycaster.computeSlantRangeForPitch(
                    forwardPitchDeg = lookPitchDeg
                )
                val gsd = PushBroomCropper.computeGsdMmPerPx(slantR, prev.activeZoomRatio)

                if (prev.acquisitionStage != AcquisitionStage.PATROL) {
                    val reprojectedLock = lockOnController.reprojectForDisplay120Hz(
                        currentRotationMatrix = liveMat,
                        currentFocalLengthPx = fy,
                        nowTimestampNs = nowNs
                    )
                    if (reprojectedLock.stage == AcquisitionStage.PATROL) {
                        updatedVisibleTargetsForAudio = emptyList()
                        currentSpeedForAudio = ScanSpeedState.Normal(0f)
                        prev.copy(
                            acquisitionStage = AcquisitionStage.PATROL,
                            lockOnHudState = reprojectedLock,
                            imuPitchDeg = smoothedPitchDeg,
                            rollDeg = rollDeg,
                            referencePitchDeg = refPitchDeg
                        )
                    } else {
                        updatedVisibleTargetsForAudio = buildLockOnAudioTargets(reprojectedLock)
                        currentSpeedForAudio = ScanSpeedState.Normal(0f)
                        prev.copy(
                            acquisitionStage = reprojectedLock.stage,
                            lockOnHudState = reprojectedLock,
                            imuPitchDeg = smoothedPitchDeg,
                            rollDeg = rollDeg,
                            referencePitchDeg = refPitchDeg,
                            slantRangeMeters = reprojectedLock.slantRangeMeters,
                            gsdMmPerPx = PushBroomCropper.computeGsdMmPerPx(
                                reprojectedLock.slantRangeMeters,
                                prev.activeZoomRatio
                            )
                        )
                    }
                } else {
                    val reprojectedTargets = if (prev.confirmedTargets.isNotEmpty()) {
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
            }
            updatedVisibleTargetsForAudio?.let { visibleTargets ->
                sonifier.updateTargets(visibleTargets, currentSpeedForAudio)
            }
        }
        scheduleInferenceDrainIfNeeded()
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
     * Records a user screen/UI touch interaction and wakes `PATROL_IDLE` to `PATROL_ACTIVE`
     * (`30 FPS` Camera2 + `120 Hz` display refresh) in `< 50 ms`.
     */
    fun onUserTouchInteraction() {
        val nextPower = patrolDutyController.onUserInteraction(System.nanoTime())
        cameraController.setTargetCameraFps(nextPower.targetCameraFps)
        if (_uiState.value.patrolPowerState != nextPower) {
            _uiState.update { it.copy(patrolPowerState = nextPower) }
        }
    }

    /** Enables or disables the `"EYES-UP"` (`EYE`) `90%` OLED-saving veil during `PATROL_IDLE`. */
    fun setEyesUpPatrolEnabled(enabled: Boolean) {
        val nextPower = patrolDutyController.onUserInteraction(System.nanoTime())
        cameraController.setTargetCameraFps(nextPower.targetCameraFps)
        _uiState.update {
            it.copy(
                eyesUpPatrolEnabled = enabled,
                patrolPowerState = nextPower
            )
        }
    }

    /** Toggles the `"EYES-UP"` (`EYE`) OLED-saving veil preference. */
    fun toggleEyesUpPatrol() {
        setEyesUpPatrolEnabled(!_uiState.value.eyesUpPatrolEnabled)
    }

    /**
     * Enters Stage 2 (`LOCK_ON_NEURAL`) on the visible [ConfirmedTarget] matching [trackId].
     */
    fun lockTargetById(trackId: Long) {
        val current = _uiState.value
        if (current.operatingMode != OperatingMode.SCAN) return
        val target = current.confirmedTargets.firstOrNull { it.trackId == trackId && it.isVisibleInStrip }
            ?: return
        activateLockOnTarget(target)
    }

    /**
     * Enters Stage 2 (`LOCK_ON_NEURAL`) on the visible [ConfirmedTarget] with the smallest
     * [ConfirmedTarget.slantRangeMeters] (tie-breaking by horizontal proximity to viewport center).
     */
    fun lockNearestTarget() {
        val current = _uiState.value
        if (current.operatingMode != OperatingMode.SCAN) return
        val nearest = LockOnPinpointController.selectNearestVisibleTarget(
            targets = current.confirmedTargets,
            stripGeometry = current.stripGeometry
        ) ?: return
        activateLockOnTarget(nearest)
    }

    /**
     * Hit-tests a tap inside the `PATROL` push-broom viewport (`normalizedStripX`, `normalizedStripY`
     * in `[0.0, 1.0]`) and enters `LOCK_ON_NEURAL` if a visible reticle was tapped.
     */
    fun onPatrolViewportTapped(normalizedStripX: Float, normalizedStripY: Float) {
        onUserTouchInteraction()
        val current = _uiState.value
        if (current.operatingMode != OperatingMode.SCAN ||
            current.acquisitionStage != AcquisitionStage.PATROL
        ) {
            return
        }
        val tapX = normalizedStripX.coerceIn(0f, 1f) * current.stripGeometry.stripWidthPx
        val tapY = normalizedStripY.coerceIn(0f, 1f) * StripRoiGeometry.STRIP_HEIGHT_PX
        val hit = LockOnPinpointController.findTappedTargetInStrip(
            targets = current.confirmedTargets,
            tapStripXPx = tapX,
            tapStripYPx = tapY
        ) ?: return
        activateLockOnTarget(hit)
    }

    /**
     * Handles the central stateful **`LOCK NEAREST` / `BREAK LOCK`** LCARS button:
     * breaks lock immediately (`0 ms`) when locked on, or locks onto the nearest visible target in `PATROL`.
     */
    fun toggleCentralLockButton() {
        onUserTouchInteraction()
        if (_uiState.value.acquisitionStage != AcquisitionStage.PATROL) {
            breakLock()
        } else {
            lockNearestTarget()
        }
    }

    /**
     * Immediately exits `LOCK_ON_NEURAL` / `PINPOINT_CONSTELLATION` (`0 ms`) and restores `PATROL`
     * push-broom scanning and EIS.
     */
    fun breakLock() {
        val unlockedHud = lockOnController.breakLock()
        latestReadySlot.set(null)
        lastCompletedSweepWallNs = 0L
        smoothedSweepHz = 0f
        val nextPower = patrolDutyController.onContextChanged(
            operatingMode = _uiState.value.operatingMode,
            acquisitionStage = AcquisitionStage.PATROL,
            nowNs = System.nanoTime()
        )
        cameraController.setTargetCameraFps(nextPower.targetCameraFps)
        sonifier.updateTargets(
            confirmedTargets = _uiState.value.confirmedTargets.filter { it.isVisibleInStrip },
            speedState = _uiState.value.speedState
        )
        val audioTel = sonifier.latestTelemetry
        _uiState.update { prev ->
            prev.copy(
                acquisitionStage = AcquisitionStage.PATROL,
                lockOnHudState = unlockedHud,
                patrolPowerState = nextPower,
                audioPitchHz = audioTel.currentPitchHz,
                audioCadenceHz = audioTel.currentPingCadenceHz
            )
        }
    }

    private fun activateLockOnTarget(target: ConfirmedTarget) {
        val nowNs = System.nanoTime()
        val prev = _uiState.value
        val rotMat = raycaster.copyLatestRotationMatrix()
        val lockHud = lockOnController.lockOntoTarget(
            target = target,
            stripGeometry = prev.stripGeometry,
            currentCropYFloat = prev.displayCropYFloat,
            currentFocalLengthPx = prev.verticalFocalLengthPx,
            currentRotationMatrix = rotMat,
            timestampNs = nowNs
        )
        latestReadySlot.set(null)
        lastCompletedSweepWallNs = 0L
        smoothedSweepHz = 0f
        val nextPower = patrolDutyController.onContextChanged(
            operatingMode = prev.operatingMode,
            acquisitionStage = lockHud.stage,
            nowNs = nowNs
        )
        cameraController.setTargetCameraFps(nextPower.targetCameraFps)
        sonifier.updateTargets(
            confirmedTargets = buildLockOnAudioTargets(lockHud),
            speedState = ScanSpeedState.Normal(0f)
        )
        val audioTel = sonifier.latestTelemetry
        _uiState.update {
            it.copy(
                acquisitionStage = lockHud.stage,
                lockOnHudState = lockHud,
                patrolPowerState = nextPower,
                audioPitchHz = audioTel.currentPitchHz,
                audioCadenceHz = audioTel.currentPingCadenceHz
            )
        }
    }

    private fun buildLockOnAudioTargets(hud: LockOnHudState): List<ConfirmedTarget> {
        if (!hud.isLocked || !hud.isTargetInFrame || !hud.stoneClass.isUserAlertable) {
            return emptyList()
        }
        // Boost cadence smoothly as range closes from 1.50m -> 0.25m for tactile homing feedback
        val proximityBoost = ((1.50f - hud.slantRangeMeters) / 1.25f).coerceIn(0f, 0.35f)
        val effectiveConf = (hud.smoothedConfidence + proximityBoost).coerceIn(0.45f, 0.99f)
        return listOf(
            ConfirmedTarget(
                trackId = hud.lockedTrackId,
                stoneClass = hud.stoneClass,
                worldPosition = WorldPoint3D(0f, 0f, hud.slantRangeMeters),
                stripLeftPx = hud.frameLeftPx,
                stripTopPx = hud.frameTopPx,
                stripRightPx = hud.frameRightPx,
                stripBottomPx = hud.frameBottomPx,
                smoothedConfidence = effectiveConf,
                estimatedDiameterMm = hud.estimatedDiameterMm,
                slantRangeMeters = hud.slantRangeMeters,
                hitCountInWindow = 6,
                windowSize = 6,
                lastUpdatedTimestampNs = 0L,
                isVisibleInStrip = true
            )
        )
    }

    /**
     * Switches between [OperatingMode.SCAN] and [OperatingMode.LOG_DATA].
     */
    fun setOperatingMode(mode: OperatingMode) {
        val nextPower = patrolDutyController.onContextChanged(
            operatingMode = mode,
            acquisitionStage = _uiState.value.acquisitionStage,
            nowNs = System.nanoTime()
        )
        cameraController.setTargetCameraFps(nextPower.targetCameraFps)
        if (mode == OperatingMode.LOG_DATA) {
            cameraController.setShutterClamp(true)
            val dir = fieldLogger.startSession()
            _uiState.update {
                it.copy(
                    operatingMode = mode,
                    patrolPowerState = nextPower,
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
                    patrolPowerState = nextPower,
                    loggedFrameCount = fieldLogger.savedFrameCount
                )
            }
        }
    }

    /** Switches the active [DetectorBackend] and resets [TemporalTargetTracker] tracks. */
    fun setDetectorBackend(backend: DetectorBackend) {
        tracker.reset()
        lastCompletedSweepWallNs = 0L
        smoothedSweepHz = 0f
        _uiState.update { it.copy(detectorBackend = backend) }
    }

    /** Selects a [SyntheticTargetProfile], switches to [DetectorBackend.SYNTHETIC_DEMO], and resets tracks. */
    fun setSyntheticProfile(profile: SyntheticTargetProfile) {
        syntheticDetector.profile = profile
        tracker.reset()
        lastCompletedSweepWallNs = 0L
        smoothedSweepHz = 0f
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
        lastCompletedSweepWallNs = 0L
        smoothedSweepHz = 0f
        option.liveProfile?.let { profile ->
            rfDetrGpuDetector.setLiveModelProfile(profile)
        }
        option.syntheticProfile?.let { profile ->
            syntheticDetector.profile = profile
        }
        val needsAsyncCompile = option.backend == DetectorBackend.RF_DETR_NANO_GPU &&
            !rfDetrGpuDetector.isDesiredProfileCompiled
        _uiState.update { prev ->
            prev.copy(
                detectorBackend = option.backend,
                liveModelProfile = option.liveProfile ?: prev.liveModelProfile,
                syntheticProfile = option.syntheticProfile ?: prev.syntheticProfile,
                detectorStatus = option.liveProfile?.statusBadge ?: prev.detectorStatus,
                isCompilingModel = needsAsyncCompile
            )
        }
        if (option.backend == DetectorBackend.RF_DETR_NANO_GPU) {
            scheduleInferenceDrainIfNeeded()
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
     * Producer stage on `GneissFind-Analyzer`: extracts the pitch-stabilized YUV strip into a free
     * [PreparedSweepSlot] in `~3 ms`, **closes [imageProxy] immediately**, and triggers the
     * decoupled `GneissFind-Inference` worker.
     */
    private fun processCameraFrameOnAnalyzerThread(
        imageProxy: ImageProxy,
        exposureTelemetry: CameraExposureTelemetry
    ) {
        if (isShuttingDown.get()) {
            imageProxy.close()
            return
        }

        if (lockOnController.currentStage != AcquisitionStage.PATROL) {
            processLockOnFrameOnAnalyzerThread(imageProxy, exposureTelemetry)
            return
        }

        val slot: PreparedSweepSlot
        val stateSnapshot: ScannerUiState
        val isLoggingMode: Boolean
        val tileMode: SwathTileMode
        val poseSnapshot: MotionPoseSnapshot
        val centerGsdMmPerPx: Float
        val geometry: StripRoiGeometry
        val frameTs: Long

        try {
            val nowWallMs = SystemClock.elapsedRealtime()
            // Avoid redundant 30 Hz CPU YUV decoding if an unread frame is already staged and < 22 ms old
            if (inferenceRunning.get() &&
                latestReadySlot.get() != null &&
                (nowWallMs - lastProducerExtractWallMs) < 22L
            ) {
                return
            }

            val freeSlot = slotRing.firstOrNull { candidate ->
                !candidate.inUseByConsumer.get() && candidate !== latestReadySlot.get()
            } ?: return

            slot = freeSlot
            slot.isLockOnSlot = false
            lastProducerExtractWallMs = nowWallMs
            stateSnapshot = _uiState.value
            isLoggingMode = stateSnapshot.operatingMode == OperatingMode.LOG_DATA
            tileMode = stateSnapshot.swathTileMode
            frameTs = imageProxy.imageInfo.timestamp

            poseSnapshot = raycaster.capturePoseSnapshot(
                arFrame = null,
                fallbackTimestampNs = frameTs,
                verticalFocalLengthPx = exposureTelemetry.verticalFocalLengthPx
            )
            val activeZoom = exposureTelemetry.activeZoomRatio
            centerGsdMmPerPx = PushBroomCropper.computeGsdMmPerPx(
                slantRangeMeters = poseSnapshot.centerSlantRangeMeters,
                zoomRatio = activeZoom
            )

            val yuvFrame = ImageProxyYuv420Frame(imageProxy)
            geometry = cropper.extractTiles(
                frame = yuvFrame,
                outBuffers = slot.sweepBuffers,
                cropY = poseSnapshot.cropY,
                tileMode = tileMode,
                isPitchClamped = poseSnapshot.isPitchClamped,
                populateArgbStrip = isLoggingMode
            )
        } finally {
            // Release Camera2 ImageReader Gralloc buffer immediately after ~3 ms CPU YUV extraction!
            imageProxy.close()
        }

        // Optional LOG DATA lossless strip + metadata.jsonl persistence (4 Hz)
        if (isLoggingMode && slot.sweepBuffers.hasArgbPixels) {
            fieldLogger.maybeRecordStripFrame(
                stripArgbPixels = slot.sweepBuffers.stripArgbPixels,
                metadata = FieldStripMetadata(
                    timestampNs = frameTs,
                    wallTimeMs = System.currentTimeMillis(),
                    stripWidthPx = tileMode.stripWidthPx,
                    stripHeightPx = StripRoiGeometry.STRIP_HEIGHT_PX,
                    tileCount = tileMode.tileCount,
                    zoomRatio = exposureTelemetry.activeZoomRatio,
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

        slot.poseSnapshot = poseSnapshot
        slot.geometry = geometry
        slot.exposureTelemetry = exposureTelemetry
        slot.stateSnapshot = stateSnapshot
        slot.centerGsdMmPerPx = centerGsdMmPerPx
        slot.frameTimestampNs = frameTs

        latestReadySlot.set(slot)
        scheduleInferenceDrainIfNeeded()
    }

    /**
     * Fast Lock-On / Pinpoint frame producer (`< 1.5 ms` [ImageProxy] hold time):
     * copies upright `1920x1080` `Y` luma + clamped `320x320` RGB crop, closes [imageProxy]
     * immediately, runs `30 FPS` OpenCV VO / `AKAZE` homography, and schedules `B=1` GPU inference
     * whenever requested (`~15-22 FPS` in Stage 2, `1.0 Hz` in Stage 3).
     */
    private fun processLockOnFrameOnAnalyzerThread(
        imageProxy: ImageProxy,
        exposureTelemetry: CameraExposureTelemetry
    ) {
        val frameTs: Long
        val poseSnapshot: MotionPoseSnapshot
        val extraction: LockOnFrameExtraction
        val stateSnapshot = _uiState.value

        try {
            frameTs = imageProxy.imageInfo.timestamp
            poseSnapshot = raycaster.capturePoseSnapshot(
                arFrame = null,
                fallbackTimestampNs = frameTs,
                verticalFocalLengthPx = exposureTelemetry.verticalFocalLengthPx
            )
            val (predCx, predCy) = lockOnController.predictCurrentCropCenterPx(
                rotationMatrix = poseSnapshot.rotationMatrix,
                focalLengthPx = exposureTelemetry.verticalFocalLengthPx
            )
            val yuvFrame = ImageProxyYuv420Frame(imageProxy)
            extraction = cropper.extractUprightLumaAndLockOnCrop320(
                frame = yuvFrame,
                outLumaBytes = analyzerLumaScratch,
                outCrop320RgbFloats = analyzerCrop320Scratch,
                cropCenterXPx = predCx,
                cropCenterYPx = predCy,
                extractRgbCrop = true
            )
        } finally {
            imageProxy.close()
        }

        val directive = lockOnController.onCameraLumaFrame(
            lumaBytes = analyzerLumaScratch,
            widthPx = extraction.uprightWidthPx,
            heightPx = extraction.uprightHeightPx,
            poseSnapshot = poseSnapshot,
            timestampNs = frameTs
        )

        if (directive.autoExitedToPatrol || directive.hudState.stage == AcquisitionStage.PATROL) {
            val nextPower = patrolDutyController.onContextChanged(
                operatingMode = _uiState.value.operatingMode,
                acquisitionStage = AcquisitionStage.PATROL,
                nowNs = System.nanoTime()
            )
            cameraController.setTargetCameraFps(nextPower.targetCameraFps)
            sonifier.updateTargets(emptyList(), ScanSpeedState.Normal(0f))
            val audioTel = sonifier.latestTelemetry
            _uiState.update { prev ->
                prev.copy(
                    acquisitionStage = AcquisitionStage.PATROL,
                    lockOnHudState = directive.hudState,
                    patrolPowerState = nextPower,
                    cameraStreamFps = exposureTelemetry.measuredCameraFps,
                    audioPitchHz = audioTel.currentPitchHz,
                    audioCadenceHz = audioTel.currentPingCadenceHz
                )
            }
            return
        }

        // Publish 30 FPS AKAZE / Lock-On HUD + homing audio telemetry immediately
        sonifier.updateTargets(
            confirmedTargets = buildLockOnAudioTargets(directive.hudState),
            speedState = ScanSpeedState.Normal(0f)
        )
        val audioTel = sonifier.latestTelemetry
        val useAkazeTelemetry = directive.hudState.stage == AcquisitionStage.PINPOINT_CONSTELLATION &&
            directive.openCvLatencyMs > 0f
        _uiState.update { prev ->
            prev.copy(
                acquisitionStage = directive.hudState.stage,
                lockOnHudState = directive.hudState,
                cameraStreamFps = exposureTelemetry.measuredCameraFps,
                exposureTimeNs = exposureTelemetry.exposureTimeNs,
                sensorIso = exposureTelemetry.sensorIso,
                activeZoomRatio = exposureTelemetry.activeZoomRatio,
                verticalFocalLengthPx = exposureTelemetry.verticalFocalLengthPx,
                slantRangeMeters = directive.hudState.slantRangeMeters,
                gsdMmPerPx = PushBroomCropper.computeGsdMmPerPx(
                    directive.hudState.slantRangeMeters,
                    exposureTelemetry.activeZoomRatio
                ),
                sweepLatencyMs = if (useAkazeTelemetry) directive.openCvLatencyMs else prev.sweepLatencyMs,
                sweepHz = if (useAkazeTelemetry) exposureTelemetry.measuredCameraFps else prev.sweepHz,
                audioPitchHz = audioTel.currentPitchHz,
                audioCadenceHz = audioTel.currentPingCadenceHz
            )
        }

        if (directive.shouldRunNeuralB1) {
            val freeSlot = slotRing.firstOrNull { candidate ->
                !candidate.inUseByConsumer.get() && candidate !== latestReadySlot.get()
            } ?: return
            System.arraycopy(
                analyzerCrop320Scratch,
                0,
                freeSlot.lockOnCrop320RgbFloats,
                0,
                analyzerCrop320Scratch.size
            )
            freeSlot.isLockOnSlot = true
            freeSlot.lockOnCropLeftPx = extraction.cropLeftPx
            freeSlot.lockOnCropTopPx = extraction.cropTopPx
            freeSlot.poseSnapshot = poseSnapshot
            freeSlot.exposureTelemetry = exposureTelemetry
            freeSlot.stateSnapshot = stateSnapshot
            freeSlot.frameTimestampNs = frameTs
            latestReadySlot.set(freeSlot)
            scheduleInferenceDrainIfNeeded()
        }
    }

    /** Schedules [drainLatestSweepOnInferenceThread] on [inferenceExecutor] when idle. */
    private fun scheduleInferenceDrainIfNeeded() {
        if (isShuttingDown.get()) return
        if (inferenceRunning.compareAndSet(false, true)) {
            try {
                inferenceExecutor.execute(::drainLatestSweepOnInferenceThread)
            } catch (_: Throwable) {
                inferenceRunning.set(false)
            }
        }
    }

    /**
     * Consumer stage on `GneissFind-Inference`: continuously drains the newest [PreparedSweepSlot]
     * from [latestReadySlot], running GPU/NPU inference, cross-tile NMS, 3D raycasting, temporal
     * tracking, and audio sonification without waiting for the next camera frame arrival.
     */
    private fun drainLatestSweepOnInferenceThread() {
        try {
            if (!isShuttingDown.get() &&
                _uiState.value.detectorBackend == DetectorBackend.RF_DETR_NANO_GPU &&
                (!rfDetrGpuDetector.isDesiredProfileCompiled || !rfDetrGpuDetector.isLockOnHardwareReady)
            ) {
                rfDetrGpuDetector.ensureCompiledAndWarmOnInferenceThread()
            }
            while (!isShuttingDown.get()) {
                if (_uiState.value.detectorBackend == DetectorBackend.RF_DETR_NANO_GPU &&
                    !rfDetrGpuDetector.isDesiredProfileCompiled
                ) {
                    rfDetrGpuDetector.ensureCompiledAndWarmOnInferenceThread()
                }
                val slot = latestReadySlot.getAndSet(null) ?: break
                slot.inUseByConsumer.set(true)
                try {
                    runInferenceSweepForSlot(slot)
                } finally {
                    slot.inUseByConsumer.set(false)
                }
            }
        } finally {
            inferenceRunning.set(false)
            val needsFollowUpCompile = _uiState.value.detectorBackend == DetectorBackend.RF_DETR_NANO_GPU &&
                !rfDetrGpuDetector.isDesiredProfileCompiled
            if (!isShuttingDown.get() && (latestReadySlot.get() != null || needsFollowUpCompile)) {
                scheduleInferenceDrainIfNeeded()
            }
        }
    }

    /** Executes inference, cross-tile stitch, 3D raycast, tracking, and UI update for [slot]. */
    private fun runInferenceSweepForSlot(slot: PreparedSweepSlot) {
        if (slot.isLockOnSlot) {
            runLockOnNeuralInferenceForSlot(slot)
            return
        }
        // Discard stale PATROL sweep if user entered Lock-On while it was queued
        if (lockOnController.currentStage != AcquisitionStage.PATROL) {
            return
        }

        val stateSnapshot = slot.stateSnapshot
        val poseSnapshot = slot.poseSnapshot ?: return
        val geometry = slot.geometry
        val exposureTelemetry = slot.exposureTelemetry
        val tileMode = stateSnapshot.swathTileMode
        val activeZoom = exposureTelemetry.activeZoomRatio
        val centerGsdMmPerPx = slot.centerGsdMmPerPx

        val activeDetector = when (stateSnapshot.detectorBackend) {
            DetectorBackend.SYNTHETIC_DEMO -> syntheticDetector
            DetectorBackend.RF_DETR_NANO_GPU -> rfDetrGpuDetector
        }
        val inferenceResult = activeDetector.detectSweep(slot.sweepBuffers)

        // Compute true completed inference sweep rate (Hz) from wall-clock completion delta
        val nowWallNs = System.nanoTime()
        if (lastCompletedSweepWallNs != 0L && nowWallNs > lastCompletedSweepWallNs) {
            val dtSec = (nowWallNs - lastCompletedSweepWallNs) * 1e-9f
            if (dtSec in 0.005f..2.0f) {
                val instHz = 1.0f / dtSec
                smoothedSweepHz = if (smoothedSweepHz <= 0.1f) {
                    instHz
                } else {
                    smoothedSweepHz * 0.80f + instHz * 0.20f
                }
            }
        }
        lastCompletedSweepWallNs = nowWallNs

        val filteredDetections = cropper.mergeAndFilterTileDetections(
            tileDetections = inferenceResult.tileDetections,
            tileMode = tileMode,
            gsdMmPerPx = centerGsdMmPerPx,
            enabledClasses = _uiState.value.enabledClasses
        )

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
            timestampNs = slot.frameTimestampNs,
            cameraWorldPos = poseSnapshot.cameraWorldPosition,
            currentSpeedMps = poseSnapshot.horizontalSpeedMps
        )

        val enabledSet = _uiState.value.enabledClasses
        val maxEnabledDetScore = filteredDetections.maxOfOrNull { it.confidence } ?: 0f
        val effectiveAlertableScore = if (enabledSet.size == StoneClass.alertableClasses.size) {
            maxOf(inferenceResult.maxAlertableCandidateScore, maxEnabledDetScore)
        } else {
            maxEnabledDetScore
        }
        val currentUi = _uiState.value
        val nextPowerState = patrolDutyController.onSweepCompleted(
            maxAlertableScore = effectiveAlertableScore,
            operatingMode = currentUi.operatingMode,
            acquisitionStage = currentUi.acquisitionStage,
            nowNs = nowWallNs
        )
        cameraController.setTargetCameraFps(nextPowerState.targetCameraFps)

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

        if (lockOnController.currentStage != AcquisitionStage.PATROL) return

        sonifier.updateTargets(
            confirmedTargets = reprojectedConfirmed.filter { it.isVisibleInStrip },
            speedState = trackerOutput.speedState
        )
        val audioTel = sonifier.latestTelemetry

        _uiState.update { prev ->
            prev.copy(
                detectorStatus = activeDetector.statusSummary,
                patrolPowerState = nextPowerState,
                cameraStreamFps = exposureTelemetry.measuredCameraFps,
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
    }

    /**
     * Evaluates `rf_detr_nano_320_b1_w8dyn` (`B=1`, `~15-22 FPS`) on [slot]'s clamped `320x320`
     * crop and updates [LockOnPinpointController] triangulation / confidence state.
     */
    private fun runLockOnNeuralInferenceForSlot(slot: PreparedSweepSlot) {
        if (lockOnController.currentStage == AcquisitionStage.PATROL) return
        val poseSnapshot = slot.poseSnapshot ?: return
        val exposureTelemetry = slot.exposureTelemetry
        val currentUi = _uiState.value

        val lockResult: SweepInferenceResult = if (currentUi.detectorBackend == DetectorBackend.RF_DETR_NANO_GPU) {
            rfDetrGpuDetector.detectLockOnCrop320(slot.lockOnCrop320RgbFloats)
        } else {
            val lockedCls = currentUi.lockOnHudState.stoneClass
            SweepInferenceResult(
                tileDetections = listOf(
                    TileDetection(
                        tileIndex = 0,
                        classId = lockedCls.id,
                        confidence = 0.91f,
                        leftPx = 132f,
                        topPx = 132f,
                        rightPx = 188f,
                        bottomPx = 188f
                    )
                ),
                totalSweepLatencyMs = 8.0f,
                rollingAvgLatencyMs = 8.0f,
                part1LatencyMs = 4.0f,
                hostTopKBridgeLatencyMs = 0.5f,
                part2LatencyMs = 3.5f,
                isHardwareGraphActive = false,
                maxAlertableCandidateScore = 0.91f
            )
        }

        val nowWallNs = System.nanoTime()
        if (lastCompletedSweepWallNs != 0L && nowWallNs > lastCompletedSweepWallNs) {
            val dtSec = (nowWallNs - lastCompletedSweepWallNs) * 1e-9f
            if (dtSec in 0.005f..2.0f) {
                val instHz = 1.0f / dtSec
                smoothedSweepHz = if (smoothedSweepHz <= 0.1f) {
                    instHz
                } else {
                    smoothedSweepHz * 0.75f + instHz * 0.25f
                }
            }
        }
        lastCompletedSweepWallNs = nowWallNs

        val updatedHud = lockOnController.onNeuralB1Detections(
            cropDetections = lockResult.tileDetections,
            cropLeftPx = slot.lockOnCropLeftPx,
            cropTopPx = slot.lockOnCropTopPx,
            poseSnapshot = poseSnapshot,
            zoomRatio = exposureTelemetry.activeZoomRatio,
            timestampNs = slot.frameTimestampNs
        )

        if (updatedHud.stage == AcquisitionStage.PATROL) {
            val nextPower = patrolDutyController.onContextChanged(
                operatingMode = _uiState.value.operatingMode,
                acquisitionStage = AcquisitionStage.PATROL,
                nowNs = nowWallNs
            )
            cameraController.setTargetCameraFps(nextPower.targetCameraFps)
            sonifier.updateTargets(emptyList(), ScanSpeedState.Normal(0f))
            val audioTel = sonifier.latestTelemetry
            _uiState.update { prev ->
                prev.copy(
                    acquisitionStage = AcquisitionStage.PATROL,
                    lockOnHudState = updatedHud,
                    patrolPowerState = nextPower,
                    audioPitchHz = audioTel.currentPitchHz,
                    audioCadenceHz = audioTel.currentPingCadenceHz
                )
            }
            return
        }

        sonifier.updateTargets(
            confirmedTargets = buildLockOnAudioTargets(updatedHud),
            speedState = ScanSpeedState.Normal(0f)
        )
        val audioTel = sonifier.latestTelemetry
        val showNeuralTiming = updatedHud.stage == AcquisitionStage.LOCK_ON_NEURAL ||
            updatedHud.akazeInliers < 12

        _uiState.update { prev ->
            prev.copy(
                acquisitionStage = updatedHud.stage,
                lockOnHudState = updatedHud,
                detectorStatus = if (showNeuralTiming) "RF-DETR 320 B=1 W8-DYN" else "PINPOINT AKAZE + 1HZ B=1",
                sweepLatencyMs = if (showNeuralTiming) lockResult.totalSweepLatencyMs else prev.sweepLatencyMs,
                rollingLatencyMs = lockResult.rollingAvgLatencyMs,
                sweepHz = if (showNeuralTiming) smoothedSweepHz else prev.sweepHz,
                part1LatencyMs = lockResult.part1LatencyMs,
                bridgeLatencyMs = lockResult.hostTopKBridgeLatencyMs,
                part2LatencyMs = lockResult.part2LatencyMs,
                slantRangeMeters = updatedHud.slantRangeMeters,
                gsdMmPerPx = PushBroomCropper.computeGsdMmPerPx(
                    updatedHud.slantRangeMeters,
                    exposureTelemetry.activeZoomRatio
                ),
                audioPitchHz = audioTel.currentPitchHz,
                audioCadenceHz = audioTel.currentPingCadenceHz
            )
        }
    }

    override fun onCleared() {
        super.onCleared()
        isShuttingDown.set(true)
        latestReadySlot.set(null)
        lockOnController.close()
        sonifier.close()
        raycaster.close()
        fieldLogger.close()
        cameraController.close()
        runCatching {
            inferenceExecutor.submit {
                syntheticDetector.close()
                rfDetrGpuDetector.close()
            }.get(2, TimeUnit.SECONDS)
        }
        inferenceExecutor.shutdownNow()
    }
}
