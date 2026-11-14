package com.gneissfind.app.ui

import android.app.Application
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

data class ScannerUiState(
    val operatingMode: OperatingMode = OperatingMode.SCAN,
    val detectorBackend: DetectorBackend = DetectorBackend.SYNTHETIC_DEMO,
    val syntheticProfile: SyntheticTargetProfile = SyntheticTargetProfile.MIXED_GRAVEL_WALK,
    val detectorStatus: String = "Synthetic Demo (Mixed Gravel Walk)",
    val enabledClasses: Set<StoneClass> = StoneClass.alertableClasses.toSet(),
    val shutterClampEnabled: Boolean = true,
    val twoXCropEnabled: Boolean = true,
    val exposureBiasEv: Float = -0.5f,
    val isAudioMuted: Boolean = false,
    val simulatedSpeedOverrideMps: Float? = null,
    val stripGeometry: StripRoiGeometry = StripRoiGeometry(
        uprightFrameWidth = 2160,
        uprightFrameHeight = 3840,
        stripLeftPx = (2160 - StripRoiGeometry.STRIP_WIDTH_PX) / 2,
        stripTopPx = 1190
    ),
    val rawStripDetections: List<StripDetection> = emptyList(),
    val confirmedTargets: List<ConfirmedTarget> = emptyList(),
    val speedState: ScanSpeedState = ScanSpeedState.Normal(0f),
    val imuPitchDeg: Float = GroundPlaneRaycaster.DEFAULT_PITCH_DEGREES,
    val slantRangeMeters: Float = PushBroomCropper.DEFAULT_SLANT_RANGE_METERS,
    val gsdMmPerPx: Float = PushBroomCropper.DEFAULT_GSD_MM_PER_PX,
    val sweepLatencyMs: Float = 0f,
    val rollingLatencyMs: Float = 0f,
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
)

class ScannerViewModel(application: Application) : AndroidViewModel(application) {

    private val appContext = application.applicationContext

    private val cropper = PushBroomCropper(applyImageNetNormalization = true)
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

    private val _uiState = MutableStateFlow(
        ScannerUiState(
            fieldLogDirectory = FieldDataLogger.resolveBaseFieldLogsDir(appContext).absolutePath
        )
    )
    val uiState: StateFlow<ScannerUiState> = _uiState.asStateFlow()

    fun bindCamera(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        cameraController.bindToLifecycle(lifecycleOwner, previewView)
    }

    fun onResumeSensorsAndAudio() {
        raycaster.startImuListening()
        sonifier.start()
    }

    fun onPauseSensorsAndAudio() {
        sonifier.updateTargets(emptyList(), ScanSpeedState.Normal(0f))
        sonifier.stop()
        raycaster.stopImuListening()
    }

    fun setOperatingMode(mode: OperatingMode) {
        if (mode == OperatingMode.LOG_DATA) {
            // Mandatory <= 1/500s shutter clamp when Field Logger mode activates
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

    fun setDetectorBackend(backend: DetectorBackend) {
        tracker.reset()
        _uiState.update { it.copy(detectorBackend = backend) }
    }

    fun setSyntheticProfile(profile: SyntheticTargetProfile) {
        syntheticDetector.profile = profile
        tracker.reset()
        _uiState.update { it.copy(syntheticProfile = profile) }
    }

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

    fun setShutterClampEnabled(enabled: Boolean) {
        cameraController.setShutterClamp(enabled)
        _uiState.update { it.copy(shutterClampEnabled = enabled) }
    }

    fun setTwoXCropEnabled(enabled: Boolean) {
        cameraController.setTwoXSensorCrop(enabled)
        _uiState.update { it.copy(twoXCropEnabled = enabled) }
    }

    fun setExposureBiasEv(ev: Float) {
        val clamped = ev.coerceIn(-2.0f, 2.0f)
        cameraController.setExposureBiasEv(clamped)
        _uiState.update { it.copy(exposureBiasEv = clamped) }
    }

    fun setAudioMuted(muted: Boolean) {
        sonifier.isMuted = muted
        _uiState.update { it.copy(isAudioMuted = muted) }
    }

    fun setSimulatedWalkSpeedOverride(speedMps: Float?) {
        raycaster.simulatedWalkSpeedOverrideMps = speedMps
        _uiState.update { it.copy(simulatedSpeedOverrideMps = speedMps) }
    }

    private fun processCameraFrameOnAnalyzerThread(
        imageProxy: ImageProxy,
        exposureTelemetry: CameraExposureTelemetry
    ) {
        try {
            val stateSnapshot = _uiState.value
            val isLoggingMode = stateSnapshot.operatingMode == OperatingMode.LOG_DATA

            // 1. Zero-copy YUV_420_888 strip extraction (1088x384 -> 3 overlapping 384x384 tiles)
            val yuvFrame = ImageProxyYuv420Frame(imageProxy)
            val geometry = cropper.extractTiles(
                frame = yuvFrame,
                outBuffers = sweepBuffers,
                populateArgbStrip = isLoggingMode
            )

            // 2. Capture IMU / ARCore pose, slant range R, and ground velocity v_xy
            val poseSnapshot = raycaster.capturePoseSnapshot(
                arFrame = null,
                fallbackTimestampNs = yuvFrame.timestampNs
            )
            val centerGsdMmPerPx = PushBroomCropper.computeGsdMmPerPx(poseSnapshot.centerSlantRangeMeters)

            // 3. Optional LOG DATA lossless 1088x384 strip + metadata.jsonl persistence (5–10 Hz)
            if (isLoggingMode && sweepBuffers.hasArgbPixels) {
                fieldLogger.maybeRecordStripFrame(
                    stripArgbPixels = sweepBuffers.stripArgbPixels,
                    metadata = FieldStripMetadata(
                        timestampNs = yuvFrame.timestampNs,
                        wallTimeMs = System.currentTimeMillis(),
                        imuPitchDeg = poseSnapshot.forwardPitchDeg,
                        slantRangeMeters = poseSnapshot.centerSlantRangeMeters,
                        gsdMmPerPx = centerGsdMmPerPx,
                        horizontalSpeedMps = poseSnapshot.horizontalSpeedMps,
                        exposureTimeNs = exposureTelemetry.exposureTimeNs,
                        sensorIso = exposureTelemetry.sensorIso,
                        shutterClamped = stateSnapshot.shutterClampEnabled,
                        exposureBiasEv = stateSnapshot.exposureBiasEv
                    )
                )
            }

            // 4. Run active StoneDetector across the 3 tiles
            val activeDetector = when (stateSnapshot.detectorBackend) {
                DetectorBackend.SYNTHETIC_DEMO -> syntheticDetector
                DetectorBackend.RF_DETR_NANO_GPU -> rfDetrGpuDetector
            }
            val inferenceResult = activeDetector.detectSweep(sweepBuffers)

            // 5. Map tile boxes to 1088x384 strip, run Cross-Tile NMS (IoU = 0.50),
            //    mute Class 4 (hard_confuser), and enforce >= 20 mm physical diameter
            val filteredDetections = cropper.mergeAndFilterTileDetections(
                tileDetections = inferenceResult.tileDetections,
                gsdMmPerPx = centerGsdMmPerPx,
                enabledClasses = stateSnapshot.enabledClasses
            )

            // 6. Raycast detections onto ground plane and update 3-of-6 spatial voting + 1.1 m/s speed gate
            val projectedCandidates = filteredDetections.map { det ->
                val raycast = raycaster.raycastStripDetection(
                    detection = det,
                    geometry = geometry,
                    poseSnapshot = poseSnapshot
                )
                ProjectedCandidate(
                    stripDetection = det,
                    worldPosition = raycast.worldPoint,
                    slantRangeMeters = raycast.slantRangeMeters,
                    refinedDiameterMm = det.maxSpanPx * raycast.gsdMmPerPx
                )
            }

            val trackerOutput = tracker.updateSweep(
                candidates = projectedCandidates,
                timestampNs = yuvFrame.timestampNs,
                cameraWorldPos = poseSnapshot.cameraWorldPosition,
                currentSpeedMps = poseSnapshot.horizontalSpeedMps
            )

            // 7. Drive AudioTrack sonifier (780..320 Hz pitch, 2..14 Hz cadence, or SLOW SWEEP buzz)
            sonifier.updateTargets(
                confirmedTargets = trackerOutput.confirmedTargets,
                speedState = trackerOutput.speedState
            )
            val audioTel = sonifier.latestTelemetry

            _uiState.update { prev ->
                prev.copy(
                    detectorStatus = activeDetector.statusSummary,
                    stripGeometry = geometry,
                    rawStripDetections = filteredDetections,
                    confirmedTargets = trackerOutput.confirmedTargets,
                    speedState = trackerOutput.speedState,
                    imuPitchDeg = poseSnapshot.forwardPitchDeg,
                    slantRangeMeters = poseSnapshot.centerSlantRangeMeters,
                    gsdMmPerPx = centerGsdMmPerPx,
                    sweepLatencyMs = inferenceResult.totalSweepLatencyMs,
                    rollingLatencyMs = inferenceResult.rollingAvgLatencyMs,
                    part1LatencyMs = inferenceResult.part1LatencyMs,
                    bridgeLatencyMs = inferenceResult.hostTopKBridgeLatencyMs,
                    part2LatencyMs = inferenceResult.part2LatencyMs,
                    exposureTimeNs = exposureTelemetry.exposureTimeNs,
                    sensorIso = exposureTelemetry.sensorIso,
                    activeZoomRatio = exposureTelemetry.activeZoomRatio,
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
