/*
 * AndroidViewModel coordinating the live push-broom camera stream, 120 Hz IMU pitch stabilization,
 * tile cropping, stone detection, 3D temporal voting, audio sonification, and field strip logging.
 * Exposes immutable ScannerUiState to Compose while decoupling display-rate EIS translation from
 * analyzer-rate inference sweeps.
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
    val activeSourceOption: DetectorSourceOption
        get() = if (detectorBackend == DetectorBackend.RF_DETR_NANO_GPU) {
            DetectorSourceOption.LIVE_CAMERA_GPU
        } else {
            DetectorSourceOption.entries.firstOrNull {
                it.backend == DetectorBackend.SYNTHETIC_DEMO && it.syntheticProfile == syntheticProfile
            } ?: DetectorSourceOption.SIM_MIXED_GRAVEL
        }
}

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
