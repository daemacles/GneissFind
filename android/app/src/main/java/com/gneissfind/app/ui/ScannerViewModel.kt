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
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewModelScope
import com.gneissfind.app.audio.MetalDetectorSonifier
import com.gneissfind.app.camera.CameraExposureTelemetry
import com.gneissfind.app.camera.PushBroomCameraController
import com.gneissfind.app.camera.ReplayCameraSource
import com.gneissfind.app.camera.ReplayRunSummary
import com.gneissfind.app.camera.ReplaySessionItem
import com.gneissfind.app.inference.LiveModelProfile
import com.gneissfind.app.inference.RfDetrTwoGraphGpuDetector
import com.gneissfind.app.inference.SweepInferenceResult
import com.gneissfind.app.inference.SyntheticStoneDetector
import com.gneissfind.app.inference.SyntheticTargetProfile
import com.gneissfind.app.logging.FieldDataLogger
import com.gneissfind.app.logging.FieldFrameTelemetry
import com.gneissfind.app.model.AcquisitionStage
import com.gneissfind.app.model.ConfirmedTarget
import com.gneissfind.app.model.DetectorBackend
import com.gneissfind.app.model.LockOnHudState
import com.gneissfind.app.model.OperatingMode
import com.gneissfind.app.model.PatrolPowerState
import com.gneissfind.app.model.ScanSpeedState
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripDetection
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
import com.gneissfind.app.vision.Yuv420Frame
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.atan2
import kotlin.math.hypot
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
    val loggedDurationSec: Int = 0,
    val fieldLogDirectory: String = "",
    val isReplayActive: Boolean = false,
    val activeReplaySessionName: String? = null,
    val availableReplaySessions: List<ReplaySessionItem> = emptyList(),
    val replayPreviewBitmap: Bitmap? = null,
    val replayFrameSeq: Long = 0L
) {
    /** Resolves the currently active [DetectorSourceOption] from [detectorBackend], [liveModelProfile], and [syntheticProfile]. */
    val activeSourceOption: DetectorSourceOption
        get() = if (detectorBackend == DetectorBackend.RF_DETR_NANO_GPU) {
            DetectorSourceOption.entries.firstOrNull {
                it.backend == DetectorBackend.RF_DETR_NANO_GPU && it.liveProfile == liveModelProfile
            } ?: DetectorSourceOption.LIVE_CAMERA_GPU
        } else {
            DetectorSourceOption.SIM_MIXED_GRAVEL
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
    var lockOnCropSizePx: Int = PushBroomCropper.LOCK_ON_CROP_SIZE_PX
    val lockOnCamPosEnu = FloatArray(3)
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
    /**
     * Lock-free guard set atomically at the top of `setOperatingMode(OperatingMode.LOG_DATA)` so
     * in-flight GPU sweeps and `120 Hz` IMU reprojection callbacks can never repopulate
     * `confirmedTargets` or fire `MetalDetectorSonifier` audio pings while recording.
     */
    private val isLoggingDataActive = AtomicBoolean(false)
    private val loggingDiscardedSweepsCount = AtomicInteger(0)
    private val loggingAudioPingsCount = AtomicInteger(0)
    private var lastProducerExtractWallMs: Long = 0L
    private var lastLogTelemetryUiWallMs: Long = 0L

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

    private val replayLocksCount = AtomicInteger(0)
    private val replayHandoversCount = AtomicInteger(0)
    private val replayLostLocksCount = AtomicInteger(0)
    private val replayDistractorJumpsCount = AtomicInteger(0)
    @Volatile
    private var replayInitialLockedClass: StoneClass? = null

    private val replayCameraSource = ReplayCameraSource(
        appContext = appContext,
        raycaster = raycaster,
        onSessionHeaderLoaded = { header ->
            _uiState.update { prev ->
                prev.copy(
                    activeZoomRatio = header.zoomRatio,
                    twoXCropEnabled = header.zoomRatio >= 1.5f,
                    verticalFocalLengthPx = header.focalLengthPx,
                    referencePitchDeg = header.refPitchDeg,
                    displaySurfaceRotation = header.displayRotation
                )
            }
        },
        onReplayFrameAvailable = { yuvFrame, telemetry, previewBmp, frameSeq, releaseCallback ->
            _uiState.update { prev ->
                prev.copy(
                    replayPreviewBitmap = previewBmp,
                    replayFrameSeq = frameSeq
                )
            }
            processYuvFrameFromSource(
                yuvFrame = yuvFrame,
                exposureTelemetry = telemetry,
                isFromLiveCamera = false,
                releaseSourceBuffer = releaseCallback
            )
        },
        onTryAutoLockAtFrame = { _ ->
            tryAutoLockNearestTargetForReplay()
        },
        onBuildRunSummary = { sessionName, totalFrames, caughtExceptions ->
            ReplayRunSummary(
                sessionName = sessionName,
                frames = totalFrames,
                locks = replayLocksCount.get(),
                handovers = replayHandoversCount.get(),
                lostLocks = replayLostLocksCount.get(),
                distractorJumps = replayDistractorJumpsCount.get(),
                distractorsRejected = lockOnController.rejectedDistractorCount,
                exceptions = caughtExceptions
            )
        },
        onLoopRestarted = {
            val unlockedHud = lockOnController.breakLock()
            tracker.reset()
            latestReadySlot.set(null)
            lastCompletedSweepWallNs = 0L
            smoothedSweepHz = 0f
            sonifier.updateTargets(emptyList(), ScanSpeedState.Normal(0f))
            val audioTel = sonifier.latestTelemetry
            _uiState.update { prev ->
                prev.copy(
                    acquisitionStage = AcquisitionStage.PATROL,
                    lockOnHudState = unlockedHud,
                    rawStripDetections = emptyList(),
                    confirmedTargets = emptyList(),
                    audioPitchHz = audioTel.currentPitchHz,
                    audioCadenceHz = audioTel.currentPingCadenceHz
                )
            }
        }
    )

    private var lastCompletedSweepWallNs: Long = 0L
    private var smoothedSweepHz: Float = 0f
    private val displayRotationMatrixScratch = FloatArray(9)

    private val _uiState = MutableStateFlow(
        ScannerUiState(
            fieldLogDirectory = FieldDataLogger.resolveBaseSessionsDir(appContext).absolutePath,
            availableReplaySessions = ReplayCameraSource.discoverReplaySessionItems(appContext)
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
        raycaster.onImuTraceSampleListener = { tMonoNs, rotMat9, rawPitchDeg ->
            fieldLogger.recordImuSample(
                timestampMonoNs = tMonoNs,
                rotationMatrix9 = rotMat9,
                pitchDeg = rawPitchDeg
            )
        }
        raycaster.onRealtimeImuPitchListener = { smoothedPitchDeg, rollDeg, refPitchDeg ->
            var updatedVisibleTargetsForAudio: List<ConfirmedTarget>? = null
            var currentSpeedForAudio: ScanSpeedState = ScanSpeedState.Normal(0f)
            val nowNs = System.nanoTime()
            val loggingNow = isLoggingDataActive.get()
            _uiState.update { prev ->
                val fy = prev.verticalFocalLengthPx
                val liveMat = raycaster.copyLatestRotationMatrix(displayRotationMatrixScratch)
                val isLoggingMode = loggingNow || prev.operatingMode == OperatingMode.LOG_DATA
                val floatStab = GroundPlaneRaycaster.computeStabilizedCropYFloat(
                    currentPitchDeg = smoothedPitchDeg,
                    referencePitchDeg = refPitchDeg,
                    verticalFocalLengthPx = fy,
                    eisEnabled = prev.eisPitchLockEnabled &&
                        prev.acquisitionStage == AcquisitionStage.PATROL &&
                        !isLoggingMode
                )
                val stripCenterOffsetPx = floatStab.clampedCropYFloat - StripRoiGeometry.NOMINAL_CROP_Y_PX
                val lookPitchDeg = smoothedPitchDeg + Math.toDegrees(
                    atan2(stripCenterOffsetPx.toDouble(), fy.toDouble())
                ).toFloat()
                val slantR = GroundPlaneRaycaster.computeSlantRangeForPitch(
                    forwardPitchDeg = lookPitchDeg
                )
                val gsd = PushBroomCropper.computeGsdMmPerPx(slantR, prev.activeZoomRatio)

                if (isLoggingMode) {
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
                        confirmedTargets = emptyList(),
                        rawStripDetections = emptyList(),
                        loggedFrameCount = fieldLogger.savedFrameCount,
                        loggedDurationSec = fieldLogger.recordedDurationSec
                    )
                } else if (prev.acquisitionStage != AcquisitionStage.PATROL) {
                    val reprojectedLock = lockOnController.reprojectForDisplay120Hz(
                        currentRotationMatrix = liveMat,
                        currentFocalLengthPx = fy,
                        nowTimestampNs = nowNs
                    )
                    if (reprojectedLock.stage == AcquisitionStage.PATROL) {
                        if (prev.isReplayActive) {
                            replayLostLocksCount.incrementAndGet()
                        }
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
            if (!isLoggingDataActive.get()) {
                updatedVisibleTargetsForAudio?.let { visibleTargets ->
                    sonifier.updateTargets(visibleTargets, currentSpeedForAudio)
                }
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
     * Enters Stage 2 (`LOCK_ON_NEURAL`) on the visible enabled [ConfirmedTarget] with the smallest
     * [ConfirmedTarget.slantRangeMeters] (tie-breaking by horizontal proximity to viewport center).
     */
    fun lockNearestTarget() {
        val current = _uiState.value
        if (current.operatingMode != OperatingMode.SCAN) return
        val enabledTargets = current.confirmedTargets.filter {
            current.enabledClasses.contains(it.stoneClass)
        }
        val nearest = LockOnPinpointController.selectNearestVisibleTarget(
            targets = enabledTargets,
            stripGeometry = current.stripGeometry
        ) ?: return
        activateLockOnTarget(nearest)
    }

    /**
     * Hit-tests a tap inside the `PATROL` push-broom viewport (`normalizedStripX`, `normalizedStripY`
     * relative to the active strip ROI) and enters `LOCK_ON_NEURAL` if a visible enabled reticle
     * was tapped within [minTouchSlopStripPx] (`>= 48.dp` screen-space equivalent).
     */
    fun onPatrolViewportTapped(
        normalizedStripX: Float,
        normalizedStripY: Float,
        minTouchSlopStripPx: Float = 56f
    ) {
        onUserTouchInteraction()
        val current = _uiState.value
        if (current.operatingMode != OperatingMode.SCAN ||
            current.acquisitionStage != AcquisitionStage.PATROL
        ) {
            return
        }
        val tapX = normalizedStripX.coerceIn(-0.15f, 1.15f) * current.stripGeometry.stripWidthPx
        val tapY = normalizedStripY.coerceIn(-0.25f, 1.25f) * StripRoiGeometry.STRIP_HEIGHT_PX
        val enabledTargets = current.confirmedTargets.filter {
            current.enabledClasses.contains(it.stoneClass)
        }
        val hit = LockOnPinpointController.findTappedTargetInStrip(
            targets = enabledTargets,
            tapStripXPx = tapX,
            tapStripYPx = tapY,
            touchSlopPx = minTouchSlopStripPx.coerceAtLeast(56f)
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
        if (prev.isReplayActive && prev.acquisitionStage == AcquisitionStage.PATROL) {
            replayLocksCount.incrementAndGet()
            replayInitialLockedClass = target.stoneClass
        }
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

    private fun tryAutoLockNearestTargetForReplay(): Boolean {
        val current = _uiState.value
        if (current.acquisitionStage != AcquisitionStage.PATROL) return true
        val enabledTargets = current.confirmedTargets.filter {
            current.enabledClasses.contains(it.stoneClass)
        }
        val nearest = LockOnPinpointController.selectNearestVisibleTarget(
            targets = enabledTargets,
            stripGeometry = current.stripGeometry
        ) ?: return false
        activateLockOnTarget(nearest)
        return true
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
        if (mode == OperatingMode.LOG_DATA && _uiState.value.isReplayActive) {
            stopReplayAndRestoreLiveCamera()
        }
        val nextPower = patrolDutyController.onContextChanged(
            operatingMode = mode,
            acquisitionStage = if (mode == OperatingMode.LOG_DATA) {
                AcquisitionStage.PATROL
            } else {
                _uiState.value.acquisitionStage
            },
            nowNs = System.nanoTime()
        )
        if (mode == OperatingMode.LOG_DATA) {
            // Set atomic logging guard FIRST so any in-flight GPU sweep or 120 Hz IMU callback
            // cannot repopulate confirmedTargets or fire MetalDetectorSonifier audio pings.
            isLoggingDataActive.set(true)
            loggingDiscardedSweepsCount.set(0)
            loggingAudioPingsCount.set(0)
            val unlockedHud = lockOnController.breakLock()
            tracker.reset()
            latestReadySlot.set(null)
            lastCompletedSweepWallNs = 0L
            smoothedSweepHz = 0f
            sonifier.updateTargets(emptyList(), ScanSpeedState.Normal(0f))
            val audioTel = sonifier.latestTelemetry
            cameraController.setTargetCameraFps(PushBroomCameraController.ACTIVE_PATROL_FPS)
            val snap = _uiState.value
            val dir = fieldLogger.startSession(
                initialZoomRatio = snap.activeZoomRatio,
                initialFocalLengthPx = snap.verticalFocalLengthPx,
                initialRefPitchDeg = snap.referencePitchDeg,
                initialDisplayRotation = snap.displaySurfaceRotation
            )
            _uiState.update {
                it.copy(
                    operatingMode = mode,
                    acquisitionStage = AcquisitionStage.PATROL,
                    lockOnHudState = unlockedHud,
                    patrolPowerState = nextPower,
                    detectorStatus = "LOG 1080P30 (DETECTOR PAUSED)",
                    rawStripDetections = emptyList(),
                    confirmedTargets = emptyList(),
                    sweepLatencyMs = 0f,
                    rollingLatencyMs = 0f,
                    sweepHz = 0f,
                    audioPitchHz = audioTel.currentPitchHz,
                    audioCadenceHz = audioTel.currentPingCadenceHz,
                    loggedFrameCount = 0,
                    loggedDurationSec = 0,
                    fieldLogDirectory = dir.absolutePath
                )
            }
        } else {
            fieldLogger.stopSession()
            isLoggingDataActive.set(false)
            cameraController.setTargetCameraFps(nextPower.targetCameraFps)
            val discovered = ReplayCameraSource.discoverReplaySessionItems(appContext)
            _uiState.update { prev ->
                prev.copy(
                    operatingMode = mode,
                    patrolPowerState = nextPower,
                    detectorStatus = prev.liveModelProfile.statusBadge,
                    loggedFrameCount = fieldLogger.savedFrameCount,
                    loggedDurationSec = fieldLogger.recordedDurationSec,
                    availableReplaySessions = discovered
                )
            }
        }
    }

    /** Refreshes the list of recorded `<session>` clips shown in the LCARS `SRC` drawer. */
    fun refreshAvailableReplaySessions() {
        val discovered = ReplayCameraSource.discoverReplaySessionItems(appContext)
        _uiState.update { it.copy(availableReplaySessions = discovered) }
    }

    /** Switches the active [DetectorBackend] and resets [TemporalTargetTracker] tracks. */
    fun setDetectorBackend(backend: DetectorBackend) {
        if (_uiState.value.isReplayActive) {
            stopReplayAndRestoreLiveCamera()
        }
        tracker.reset()
        lastCompletedSweepWallNs = 0L
        smoothedSweepHz = 0f
        _uiState.update { it.copy(detectorBackend = backend) }
    }

    /** Selects a [SyntheticTargetProfile], switches to [DetectorBackend.SYNTHETIC_DEMO], and resets tracks. */
    fun setSyntheticProfile(profile: SyntheticTargetProfile) {
        if (_uiState.value.isReplayActive) {
            stopReplayAndRestoreLiveCamera()
        }
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
     * clears any prior spatial target tracks (stopping any active replay session first).
     */
    fun selectSourceOption(option: DetectorSourceOption) {
        if (_uiState.value.isReplayActive || replayCameraSource.isRunning) {
            stopReplayAndRestoreLiveCamera()
        }
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
                isCompilingModel = needsAsyncCompile,
                isReplayActive = false,
                activeReplaySessionName = null,
                replayPreviewBitmap = null
            )
        }
        if (option.backend == DetectorBackend.RF_DETR_NANO_GPU) {
            scheduleInferenceDrainIfNeeded()
        }
    }

    /**
     * Activates deterministic MP4 + `120 Hz` IMU replay from a discovered [ReplaySessionItem] in
     * the LCARS `SRC` drawer, routing frames through the live GPU detector and tracker pipeline.
     */
    fun selectReplaySession(
        item: ReplaySessionItem,
        loopReplay: Boolean = true,
        autoLockFrameIdx: Int = -1
    ) {
        startReplayDirectory(
            sessionDir = File(item.sessionDirPath),
            loopReplay = loopReplay,
            autoLockFrameIdx = autoLockFrameIdx
        )
    }

    /**
     * Headless `adb shell am start` entrypoint (`--es debug_replay_session`, `--ei debug_auto_lock_frame`,
     * `--ez debug_loop_replay`).
     */
    fun startDebugReplaySessionFromIntent(
        sessionNameOrPath: String,
        autoLockFrameIdx: Int = -1,
        loopReplay: Boolean = true
    ): Boolean {
        val dir = ReplayCameraSource.resolveSessionDirectory(appContext, sessionNameOrPath)
        if (dir == null) {
            Log.e(
                ReplayCameraSource.SUMMARY_LOG_TAG,
                "Could not resolve debug_replay_session='$sessionNameOrPath'"
            )
            return false
        }
        return startReplayDirectory(
            sessionDir = dir,
            loopReplay = loopReplay,
            autoLockFrameIdx = autoLockFrameIdx
        )
    }

    /**
     * Headless `adb shell am start --ei debug_record_seconds <N>` entrypoint that waits for live
     * CameraX frames to arrive, records a detector-free `1080p30` `MOD: LOG` session for
     * [durationSec] seconds, returns to `SCAN`, and logs `GNEISSFIND_RECORD_SUMMARY`.
     */
    fun startDebugRecordSessionFromIntent(durationSec: Int) {
        val clampedSec = durationSec.coerceIn(1, 300)
        viewModelScope.launch {
            if (_uiState.value.isReplayActive) {
                stopReplayAndRestoreLiveCamera()
            }
            // Wait up to 4s for live CameraX ImageAnalysis frames to begin flowing
            val deadlineMs = SystemClock.elapsedRealtime() + 4_000L
            while (_uiState.value.cameraStreamFps <= 0f && SystemClock.elapsedRealtime() < deadlineMs) {
                delay(50L)
            }
            delay(250L)
            val startWallMs = SystemClock.elapsedRealtime()
            setOperatingMode(OperatingMode.LOG_DATA)
            val sessionPath = _uiState.value.fieldLogDirectory
            delay(clampedSec * 1_000L)
            val elapsedSec = ((SystemClock.elapsedRealtime() - startWallMs) / 1000.0).coerceAtLeast(0.001)
            val discardedSweeps = loggingDiscardedSweepsCount.get()
            val audioPingsDuringLog = loggingAudioPingsCount.get()
            setOperatingMode(OperatingMode.SCAN)
            val frames = fieldLogger.savedFrameCount
            val avgFps = frames / elapsedSec
            Log.i(
                ReplayCameraSource.SUMMARY_LOG_TAG,
                "GNEISSFIND_RECORD_SUMMARY session=$sessionPath frames=$frames elapsed_s=${"%.2f".format(elapsedSec)} avg_fps=${"%.2f".format(avgFps)} discarded_sweeps=$discardedSweeps audio_pings=$audioPingsDuringLog"
            )
        }
    }

    private fun startReplayDirectory(
        sessionDir: File,
        loopReplay: Boolean,
        autoLockFrameIdx: Int
    ): Boolean {
        if (_uiState.value.operatingMode == OperatingMode.LOG_DATA) {
            fieldLogger.stopSession()
        }
        val unlockedHud = lockOnController.breakLock()
        tracker.reset()
        latestReadySlot.set(null)
        lastCompletedSweepWallNs = 0L
        smoothedSweepHz = 0f
        replayLocksCount.set(0)
        replayHandoversCount.set(0)
        replayLostLocksCount.set(0)
        replayDistractorJumpsCount.set(0)
        replayInitialLockedClass = null

        cameraController.suspendCaptureForReplay()
        val needsAsyncCompile = !rfDetrGpuDetector.isDesiredProfileCompiled
        val discovered = ReplayCameraSource.discoverReplaySessionItems(appContext)
        _uiState.update { prev ->
            prev.copy(
                operatingMode = OperatingMode.SCAN,
                acquisitionStage = AcquisitionStage.PATROL,
                lockOnHudState = unlockedHud,
                detectorBackend = DetectorBackend.RF_DETR_NANO_GPU,
                isCompilingModel = needsAsyncCompile,
                rawStripDetections = emptyList(),
                confirmedTargets = emptyList(),
                isReplayActive = true,
                activeReplaySessionName = sessionDir.name,
                availableReplaySessions = discovered
            )
        }
        scheduleInferenceDrainIfNeeded()

        val started = replayCameraSource.startSession(
            sessionDir = sessionDir,
            loopReplay = loopReplay,
            autoLockFrameIdx = autoLockFrameIdx
        )
        if (!started) {
            stopReplayAndRestoreLiveCamera()
        }
        return started
    }

    private fun stopReplayAndRestoreLiveCamera() {
        replayCameraSource.stopSession()
        raycaster.resetRingBuffer()
        cameraController.resumeCaptureFromReplay()
        _uiState.update {
            it.copy(
                isReplayActive = false,
                activeReplaySessionName = null,
                replayPreviewBitmap = null
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
     * Producer stage on `GneissFind-Analyzer`: extracts the pitch-stabilized YUV strip into a free
     * [PreparedSweepSlot] in `~3 ms`, **closes [imageProxy] immediately**, and triggers the
     * decoupled `GneissFind-Inference` worker.
     */
    private fun processCameraFrameOnAnalyzerThread(
        imageProxy: ImageProxy,
        exposureTelemetry: CameraExposureTelemetry
    ) {
        if (isShuttingDown.get() || _uiState.value.isReplayActive || raycaster.isReplayModeActive) {
            imageProxy.close()
            return
        }
        processYuvFrameFromSource(
            yuvFrame = ImageProxyYuv420Frame(imageProxy),
            exposureTelemetry = exposureTelemetry,
            isFromLiveCamera = true,
            releaseSourceBuffer = { imageProxy.close() }
        )
    }

    /**
     * Unified producer stage shared by live CameraX (`GneissFind-Analyzer`) and deterministic
     * [ReplayCameraSource] (`GneissFind-Replay`): extracts stabilized `PATROL` tiles or `LOCK-ON`
     * `Y` + `320x320` crop in `< 3 ms`, releases the source buffer immediately, and schedules GPU inference.
     */
    private fun processYuvFrameFromSource(
        yuvFrame: Yuv420Frame,
        exposureTelemetry: CameraExposureTelemetry,
        isFromLiveCamera: Boolean,
        releaseSourceBuffer: () -> Unit
    ) {
        if (isShuttingDown.get()) {
            releaseSourceBuffer()
            return
        }

        if (!isLoggingDataActive.get() && lockOnController.currentStage != AcquisitionStage.PATROL) {
            processLockOnYuvFrameFromSource(
                yuvFrame = yuvFrame,
                exposureTelemetry = exposureTelemetry,
                isFromLiveCamera = isFromLiveCamera,
                releaseSourceBuffer = releaseSourceBuffer
            )
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
            stateSnapshot = _uiState.value
            isLoggingMode = isLoggingDataActive.get() || stateSnapshot.operatingMode == OperatingMode.LOG_DATA
            frameTs = yuvFrame.timestampNs

            // Feed 30 FPS 1080p hardware MediaCodec + trace.jsonl on every frame while MOD == LOG_DATA,
            // returning immediately without running YUV float tile extraction or GPU neural inference!
            if (isLoggingMode && isFromLiveCamera && !raycaster.isReplayModeActive) {
                fieldLogger.recordCameraFrame(
                    frame = yuvFrame,
                    cropper = cropper,
                    telemetry = FieldFrameTelemetry(
                        timestampMonoNs = GroundPlaneRaycaster.alignHardwareTimestampToMonotonicNs(frameTs),
                        exposureTimeNs = exposureTelemetry.exposureTimeNs,
                        sensorIso = exposureTelemetry.sensorIso,
                        zoomRatio = exposureTelemetry.activeZoomRatio,
                        focalLengthPx = exposureTelemetry.verticalFocalLengthPx,
                        referencePitchDeg = stateSnapshot.referencePitchDeg,
                        displayRotation = stateSnapshot.displaySurfaceRotation,
                        shutterClamped = stateSnapshot.shutterClampEnabled,
                        exposureBiasEv = stateSnapshot.exposureBiasEv
                    )
                )
                val nowLogWallMs = SystemClock.elapsedRealtime()
                if (nowLogWallMs - lastLogTelemetryUiWallMs >= 100L) {
                    lastLogTelemetryUiWallMs = nowLogWallMs
                    _uiState.update { prev ->
                        prev.copy(
                            cameraStreamFps = exposureTelemetry.measuredCameraFps,
                            exposureTimeNs = exposureTelemetry.exposureTimeNs,
                            sensorIso = exposureTelemetry.sensorIso,
                            activeZoomRatio = exposureTelemetry.activeZoomRatio,
                            verticalFocalLengthPx = exposureTelemetry.verticalFocalLengthPx,
                            loggedFrameCount = fieldLogger.savedFrameCount,
                            loggedDurationSec = fieldLogger.recordedDurationSec,
                            fieldLogDirectory = fieldLogger.currentSessionPath
                        )
                    }
                }
                return
            }

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
            tileMode = stateSnapshot.swathTileMode

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

            geometry = cropper.extractTiles(
                frame = yuvFrame,
                outBuffers = slot.sweepBuffers,
                cropY = poseSnapshot.cropY,
                tileMode = tileMode,
                isPitchClamped = poseSnapshot.isPitchClamped,
                populateArgbStrip = false
            )
        } finally {
            // Release Camera2 ImageReader Gralloc or MediaCodec buffer immediately after ~3 ms CPU YUV extraction!
            releaseSourceBuffer()
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
     * Fast Lock-On / Pinpoint frame producer (`< 1.5 ms` buffer hold time) shared by live CameraX
     * and [ReplayCameraSource]: copies upright `1920x1080` `Y` luma + clamped `320x320` RGB crop,
     * releases the source buffer immediately, runs `30 FPS` OpenCV VO / `AKAZE` homography, and
     * schedules `B=1` GPU inference whenever requested.
     */
    private fun processLockOnYuvFrameFromSource(
        yuvFrame: Yuv420Frame,
        exposureTelemetry: CameraExposureTelemetry,
        isFromLiveCamera: Boolean,
        releaseSourceBuffer: () -> Unit
    ) {
        val frameTs: Long
        val poseSnapshot: MotionPoseSnapshot
        val extraction: LockOnFrameExtraction
        val stateSnapshot = _uiState.value
        val prevStage = lockOnController.currentStage

        try {
            frameTs = yuvFrame.timestampNs
            poseSnapshot = raycaster.capturePoseSnapshot(
                arFrame = null,
                fallbackTimestampNs = frameTs,
                verticalFocalLengthPx = exposureTelemetry.verticalFocalLengthPx
            )
            val (predCx, predCy, predCropSize) = lockOnController.predictCurrentCropWindow(
                rotationMatrix = poseSnapshot.rotationMatrix,
                focalLengthPx = exposureTelemetry.verticalFocalLengthPx
            )
            if (stateSnapshot.operatingMode == OperatingMode.LOG_DATA &&
                isFromLiveCamera &&
                !raycaster.isReplayModeActive
            ) {
                fieldLogger.recordCameraFrame(
                    frame = yuvFrame,
                    cropper = cropper,
                    telemetry = FieldFrameTelemetry(
                        timestampMonoNs = GroundPlaneRaycaster.alignHardwareTimestampToMonotonicNs(frameTs),
                        exposureTimeNs = exposureTelemetry.exposureTimeNs,
                        sensorIso = exposureTelemetry.sensorIso,
                        zoomRatio = exposureTelemetry.activeZoomRatio,
                        focalLengthPx = exposureTelemetry.verticalFocalLengthPx,
                        referencePitchDeg = stateSnapshot.referencePitchDeg,
                        displayRotation = stateSnapshot.displaySurfaceRotation,
                        shutterClamped = stateSnapshot.shutterClampEnabled,
                        exposureBiasEv = stateSnapshot.exposureBiasEv
                    )
                )
            }
            extraction = cropper.extractUprightLumaAndLockOnCrop320(
                frame = yuvFrame,
                outLumaBytes = analyzerLumaScratch,
                outCrop320RgbFloats = analyzerCrop320Scratch,
                cropCenterXPx = predCx,
                cropCenterYPx = predCy,
                sourceCropSizePx = predCropSize,
                extractRgbCrop = true
            )
        } finally {
            releaseSourceBuffer()
        }

        val nowMonoNs = System.nanoTime()
        val directive = lockOnController.onCameraLumaFrame(
            lumaBytes = analyzerLumaScratch,
            widthPx = extraction.uprightWidthPx,
            heightPx = extraction.uprightHeightPx,
            poseSnapshot = poseSnapshot,
            timestampNs = nowMonoNs
        )

        if (directive.autoExitedToPatrol || directive.hudState.stage == AcquisitionStage.PATROL) {
            if (stateSnapshot.isReplayActive && prevStage != AcquisitionStage.PATROL) {
                replayLostLocksCount.incrementAndGet()
            }
            val nextPower = patrolDutyController.onContextChanged(
                operatingMode = _uiState.value.operatingMode,
                acquisitionStage = AcquisitionStage.PATROL,
                nowNs = nowMonoNs
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

        if (stateSnapshot.isReplayActive &&
            prevStage == AcquisitionStage.LOCK_ON_NEURAL &&
            directive.hudState.stage == AcquisitionStage.PINPOINT_CONSTELLATION
        ) {
            replayHandoversCount.incrementAndGet()
        }

        // Publish 30 FPS AKAZE / Lock-On HUD + homing audio telemetry immediately
        val voOrImuSpeedMps = kotlin.math.max(
            poseSnapshot.horizontalSpeedMps,
            lockOnController.estimatedHorizontalSpeedMps
        )
        val lockSpeedState = ScanSpeedState.Normal(voOrImuSpeedMps)
        sonifier.updateTargets(
            confirmedTargets = buildLockOnAudioTargets(directive.hudState),
            speedState = lockSpeedState
        )
        val audioTel = sonifier.latestTelemetry
        val useVisualTrackerTelemetry = directive.hudState.isVisuallyLocked &&
            directive.openCvLatencyMs > 0f
        _uiState.update { prev ->
            if (isLoggingDataActive.get() || prev.operatingMode == OperatingMode.LOG_DATA) {
                return@update prev
            }
            val updatedRollingMs = if (useVisualTrackerTelemetry) {
                if (prev.rollingLatencyMs <= 0f || prev.rollingLatencyMs > 25f) {
                    directive.openCvLatencyMs
                } else {
                    prev.rollingLatencyMs * 0.85f + directive.openCvLatencyMs * 0.15f
                }
            } else {
                prev.rollingLatencyMs
            }
            prev.copy(
                acquisitionStage = directive.hudState.stage,
                lockOnHudState = directive.hudState,
                speedState = lockSpeedState,
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
                sweepLatencyMs = if (useVisualTrackerTelemetry) directive.openCvLatencyMs else prev.sweepLatencyMs,
                rollingLatencyMs = updatedRollingMs,
                sweepHz = if (useVisualTrackerTelemetry) exposureTelemetry.measuredCameraFps else prev.sweepHz,
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
            freeSlot.lockOnCropSizePx = extraction.cropSizePx
            lockOnController.copyCurrentCamPosEnu(freeSlot.lockOnCamPosEnu)
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
                !isLoggingDataActive.get() &&
                _uiState.value.detectorBackend == DetectorBackend.RF_DETR_NANO_GPU &&
                (!rfDetrGpuDetector.isDesiredProfileCompiled || !rfDetrGpuDetector.isLockOnHardwareReady)
            ) {
                rfDetrGpuDetector.ensureCompiledAndWarmOnInferenceThread()
            }
            while (!isShuttingDown.get()) {
                if (isLoggingDataActive.get()) {
                    latestReadySlot.set(null)
                    break
                }
                if (_uiState.value.detectorBackend == DetectorBackend.RF_DETR_NANO_GPU &&
                    !rfDetrGpuDetector.isDesiredProfileCompiled
                ) {
                    rfDetrGpuDetector.ensureCompiledAndWarmOnInferenceThread()
                }
                val slot = latestReadySlot.getAndSet(null) ?: break
                if (isLoggingDataActive.get()) {
                    loggingDiscardedSweepsCount.incrementAndGet()
                    break
                }
                slot.inUseByConsumer.set(true)
                try {
                    runInferenceSweepForSlot(slot)
                } finally {
                    slot.inUseByConsumer.set(false)
                }
            }
        } finally {
            inferenceRunning.set(false)
            val needsFollowUpCompile = !isLoggingDataActive.get() &&
                _uiState.value.detectorBackend == DetectorBackend.RF_DETR_NANO_GPU &&
                !rfDetrGpuDetector.isDesiredProfileCompiled
            if (!isShuttingDown.get() && !isLoggingDataActive.get() && (latestReadySlot.get() != null || needsFollowUpCompile)) {
                scheduleInferenceDrainIfNeeded()
            }
        }
    }

    /** Executes inference, cross-tile stitch, 3D raycast, tracking, and UI update for [slot]. */
    private fun runInferenceSweepForSlot(slot: PreparedSweepSlot) {
        if (isLoggingDataActive.get() || _uiState.value.operatingMode == OperatingMode.LOG_DATA) {
            loggingDiscardedSweepsCount.incrementAndGet()
            return
        }
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

        if (isLoggingDataActive.get() || _uiState.value.operatingMode == OperatingMode.LOG_DATA) {
            loggingDiscardedSweepsCount.incrementAndGet()
            return
        }
        if (lockOnController.currentStage != AcquisitionStage.PATROL) return

        val visibleTargets = reprojectedConfirmed.filter { it.isVisibleInStrip }
        if (isLoggingDataActive.get()) {
            loggingDiscardedSweepsCount.incrementAndGet()
            return
        }
        sonifier.updateTargets(
            confirmedTargets = visibleTargets,
            speedState = trackerOutput.speedState
        )
        val audioTel = sonifier.latestTelemetry

        _uiState.update { prev ->
            if (isLoggingDataActive.get() || prev.operatingMode == OperatingMode.LOG_DATA) {
                return@update prev
            }
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
                loggedDurationSec = fieldLogger.recordedDurationSec,
                fieldLogDirectory = fieldLogger.currentSessionPath
            )
        }
    }

    /**
     * Evaluates `rf_detr_nano_320_b1_w8dyn` (`B=1`, `~15-22 FPS`) on [slot]'s clamped `320x320`
     * crop and updates [LockOnPinpointController] triangulation / confidence state.
     */
    private fun runLockOnNeuralInferenceForSlot(slot: PreparedSweepSlot) {
        if (isLoggingDataActive.get() || _uiState.value.operatingMode == OperatingMode.LOG_DATA) {
            loggingDiscardedSweepsCount.incrementAndGet()
            return
        }
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
        val liveRotMat = raycaster.copyLatestRotationMatrix()

        val prevStage = lockOnController.currentStage
        val prevHudBeforeUpdate = _uiState.value.lockOnHudState
        val updatedHud = lockOnController.onNeuralB1Detections(
            cropDetections = lockResult.tileDetections,
            cropLeftPx = slot.lockOnCropLeftPx,
            cropTopPx = slot.lockOnCropTopPx,
            poseSnapshot = poseSnapshot,
            zoomRatio = exposureTelemetry.activeZoomRatio,
            timestampNs = nowWallNs,
            captureCamPosEnu = slot.lockOnCamPosEnu,
            displayRotationMatrix = liveRotMat,
            cropSizePx = slot.lockOnCropSizePx
        )

        if (updatedHud.stage == AcquisitionStage.PATROL) {
            if (currentUi.isReplayActive && prevStage != AcquisitionStage.PATROL) {
                replayLostLocksCount.incrementAndGet()
            }
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

        if (currentUi.isReplayActive) {
            if (prevStage == AcquisitionStage.LOCK_ON_NEURAL &&
                updatedHud.stage == AcquisitionStage.PINPOINT_CONSTELLATION
            ) {
                replayHandoversCount.incrementAndGet()
            }
            val expectedCls = replayInitialLockedClass
            if (expectedCls != null && updatedHud.stoneClass != expectedCls) {
                replayDistractorJumpsCount.incrementAndGet()
            } else if (prevHudBeforeUpdate.isTargetInFrame && updatedHud.isTargetInFrame) {
                val prevCx = (prevHudBeforeUpdate.frameLeftPx + prevHudBeforeUpdate.frameRightPx) * 0.5f
                val prevCy = (prevHudBeforeUpdate.frameTopPx + prevHudBeforeUpdate.frameBottomPx) * 0.5f
                val nextCx = (updatedHud.frameLeftPx + updatedHud.frameRightPx) * 0.5f
                val nextCy = (updatedHud.frameTopPx + updatedHud.frameBottomPx) * 0.5f
                val jumpPx = hypot(nextCx - prevCx, nextCy - prevCy)
                if (jumpPx > 460f) {
                    replayDistractorJumpsCount.incrementAndGet()
                }
            }
        }

        if (isLoggingDataActive.get() || _uiState.value.operatingMode == OperatingMode.LOG_DATA) {
            loggingDiscardedSweepsCount.incrementAndGet()
            return
        }
        sonifier.updateTargets(
            confirmedTargets = buildLockOnAudioTargets(updatedHud),
            speedState = ScanSpeedState.Normal(0f)
        )
        val audioTel = sonifier.latestTelemetry
        val showNeuralTiming = !updatedHud.isVisuallyLocked &&
            (updatedHud.stage == AcquisitionStage.LOCK_ON_NEURAL || updatedHud.akazeInliers < 12)

        _uiState.update { prev ->
            if (isLoggingDataActive.get() || prev.operatingMode == OperatingMode.LOG_DATA) {
                return@update prev
            }
            prev.copy(
                acquisitionStage = updatedHud.stage,
                lockOnHudState = updatedHud,
                detectorStatus = if (updatedHud.isVisuallyLocked) {
                    updatedHud.visualAnchorLabel
                } else {
                    "RF-DETR 320 B=1 W8-DYN"
                },
                sweepLatencyMs = if (showNeuralTiming) lockResult.totalSweepLatencyMs else prev.sweepLatencyMs,
                rollingLatencyMs = if (showNeuralTiming) lockResult.rollingAvgLatencyMs else prev.rollingLatencyMs,
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
        replayCameraSource.close()
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
