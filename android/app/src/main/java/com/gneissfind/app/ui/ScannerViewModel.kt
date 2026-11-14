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

data class ScannerUiState(
    val operatingMode: OperatingMode = OperatingMode.SCAN,
    val detectorBackend: DetectorBackend = DetectorBackend.SYNTHETIC_DEMO,
    val syntheticProfile: SyntheticTargetProfile = SyntheticTargetProfile.MIXED_GRAVEL_WALK,
    val detectorStatus: String = "SYNTHETIC (MIXED GRAVEL SIM)",
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

    private var lastFrameTimestampNs: Long = 0L
    private var smoothedSweepHz: Float = 0f

    private val _uiState = MutableStateFlow(
        ScannerUiState(
            fieldLogDirectory = FieldDataLogger.resolveBaseSessionsDir(appContext).absolutePath
        )
    )
    val uiState: StateFlow<ScannerUiState> = _uiState.asStateFlow()

    fun bindCamera(
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView,
        initialSurfaceRotation: Int = _uiState.value.displaySurfaceRotation
    ) {
        cameraController.bindToLifecycle(lifecycleOwner, previewView, initialSurfaceRotation)
    }

    /**
     * Synchronizes CameraX `targetRotation`, ARCore `setDisplayGeometry`, and IMU `remapCoordinateSystem`
     * whenever reversible `sensorLandscape` flips between `Surface.ROTATION_90` and `Surface.ROTATION_270`.
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
        _uiState.update {
            it.copy(
                detectorBackend = DetectorBackend.SYNTHETIC_DEMO,
                syntheticProfile = profile
            )
        }
    }

    /**
     * Inline LCARS scenario cycler (`LIVE CAM (GPU)` -> `MIXED GRAVEL SIM` -> `21MM AGATE` ->
     * `56MM JASPER` -> `CONFUSERS MUTED`) so testing never requires a modal sheet.
     */
    fun cycleScenarioPreset() {
        val cur = _uiState.value
        tracker.reset()
        if (cur.detectorBackend == DetectorBackend.RF_DETR_NANO_GPU) {
            syntheticDetector.profile = SyntheticTargetProfile.MIXED_GRAVEL_WALK
            _uiState.update {
                it.copy(
                    detectorBackend = DetectorBackend.SYNTHETIC_DEMO,
                    syntheticProfile = SyntheticTargetProfile.MIXED_GRAVEL_WALK
                )
            }
            return
        }

        val profiles = SyntheticTargetProfile.entries
        val nextIdx = profiles.indexOf(cur.syntheticProfile) + 1
        if (nextIdx < profiles.size) {
            val nextProfile = profiles[nextIdx]
            syntheticDetector.profile = nextProfile
            _uiState.update { it.copy(syntheticProfile = nextProfile) }
        } else {
            _uiState.update { it.copy(detectorBackend = DetectorBackend.RF_DETR_NANO_GPU) }
        }
    }

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

    fun setEisPitchLockEnabled(enabled: Boolean) {
        raycaster.eisPitchLockEnabled = enabled
        _uiState.update {
            it.copy(
                eisPitchLockEnabled = enabled,
                cropY = if (enabled) it.cropY else StripRoiGeometry.NOMINAL_CROP_Y_PX,
                isPitchClamped = if (enabled) it.isPitchClamped else false
            )
        }
    }

    fun syncReferencePitch() {
        val synced = raycaster.syncReferencePitch()
        _uiState.update {
            it.copy(
                referencePitchDeg = synced,
                cropY = StripRoiGeometry.NOMINAL_CROP_Y_PX,
                isPitchClamped = false
            )
        }
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
        val zoom = if (enabled) 2.0f else 1.0f
        _uiState.update {
            it.copy(
                twoXCropEnabled = enabled,
                activeZoomRatio = zoom,
                gsdMmPerPx = PushBroomCropper.computeGsdMmPerPx(it.slantRangeMeters, zoom)
            )
        }
    }

    fun setExposureBiasEv(ev: Float) {
        val clamped = ev.coerceIn(
            PushBroomCameraController.MIN_CANOPY_EV_BIAS,
            PushBroomCameraController.MAX_CANOPY_EV_BIAS
        )
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

    fun toggleOverspeedTest() {
        val next = if (_uiState.value.simulatedSpeedOverrideMps != null) null else 1.38f
        setSimulatedWalkSpeedOverride(next)
    }

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
                    refinedDiameterMm = det.maxSpanPx * raycast.gsdMmPerPx
                )
            }

            val trackerOutput = tracker.updateSweep(
                candidates = projectedCandidates,
                timestampNs = yuvFrame.timestampNs,
                cameraWorldPos = poseSnapshot.cameraWorldPosition,
                currentSpeedMps = poseSnapshot.horizontalSpeedMps
            )

            // 7. Drive AudioTrack sonifier (strictly reserved for stone pings & overspeed warning)
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
                    rollDeg = poseSnapshot.rollDeg,
                    referencePitchDeg = poseSnapshot.referencePitchDeg,
                    cropY = poseSnapshot.cropY,
                    isPitchClamped = poseSnapshot.isPitchClamped,
                    verticalFocalLengthPx = exposureTelemetry.verticalFocalLengthPx,
                    slantRangeMeters = poseSnapshot.centerSlantRangeMeters,
                    gsdMmPerPx = centerGsdMmPerPx,
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
