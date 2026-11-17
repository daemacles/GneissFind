/*
 * Stateful 3-stage acquisition funnel controller (PATROL -> LOCK_ON_NEURAL -> PINPOINT_CONSTELLATION).
 *
 * Coordinates target selection (tap-to-lock and LOCK NEAREST), 16:9 full-frame 3D pinhole
 * reprojection, rotation-gated OpenCV ground-patch visual odometry, multi-view ray-bundle +
 * Calib3d triangulation (sigma3dMm sphere-of-confusion), automatic handover to 30 FPS AKAZE
 * gravel-constellation similarity tracking with 1 Hz B=1 neural confidence refresh, 3D-anchored
 * neural detector box tracking, off-screen directional edge chevrons, and the 15.0-second
 * lost-track grace window.
 */
package com.gneissfind.app.tracking

import com.gneissfind.app.model.AcquisitionStage
import com.gneissfind.app.model.ConfirmedTarget
import com.gneissfind.app.model.LockOnHudState
import com.gneissfind.app.model.OffScreenChevron
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.TileDetection
import com.gneissfind.app.model.WorldRay3D
import com.gneissfind.app.vision.PushBroomCropper
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Per-frame directive emitted by [LockOnPinpointController.onCameraLumaFrame] telling
 * `ScannerViewModel` whether to schedule `B=1` `rf_detr_nano_320_b1_w8dyn` GPU inference on the
 * current camera frame and where to center the clamped square RGB crop.
 *
 * @property hudState Updated [LockOnHudState] snapshot (`stage == PATROL` if the `15.0 s` grace window expired).
 * @property shouldRunNeuralB1 True when `GneissFind-Inference` should evaluate `detectLockOnCrop320` on this frame.
 * @property desiredCropCenterXPx Target horizontal center (`[0..1920]`) for the neural crop.
 * @property desiredCropCenterYPx Target vertical center (`[0..1080]`) for the neural crop.
 * @property desiredCropSizePx Source square ROI width/height in `1920x1080` pixels (`[320..640]`).
 * @property autoExitedToPatrol True when the `15.0 s` continuous lost-track timeout expired on this frame.
 * @property openCvLatencyMs CPU execution latency (ms) of OpenCV KLT / `AKAZE` on this frame.
 */
data class LockOnFrameDirective(
    val hudState: LockOnHudState,
    val shouldRunNeuralB1: Boolean,
    val desiredCropCenterXPx: Float,
    val desiredCropCenterYPx: Float,
    val desiredCropSizePx: Int = PushBroomCropper.LOCK_ON_CROP_SIZE_PX,
    val autoExitedToPatrol: Boolean = false,
    val openCvLatencyMs: Float = 0f
)

/**
 * Single calibrated camera viewpoint and unit bearing ray stored in the multi-view parallax buffer
 * for Stage 2 3D target triangulation.
 *
 * @property camPosEnu 3-element `(eastM, northM, upM)` camera optical center in meters.
 * @property rCvFromWorld 9-element row-major `3x3` rotation mapping ENU world vectors into OpenCV camera space.
 * @property rayEnu Unit 3D bearing vector from [camPosEnu] toward the detected target center in ENU coordinates.
 * @property pixelU Detected target horizontal center in `1920x1080` pixels.
 * @property pixelV Detected target vertical center in `1920x1080` pixels.
 * @property boxSpanPx Detected bounding-box major-axis span in pixels.
 */
class TriangulationViewObservation(
    val camPosEnu: FloatArray,
    val rCvFromWorld: FloatArray,
    val rayEnu: WorldRay3D,
    val pixelU: Float,
    val pixelV: Float,
    val boxSpanPx: Float
)

/**
 * Result of triangulating `>= 2` multi-view target observations in 3D ENU coordinates.
 *
 * @property targetEnuM 3-element `(eastM, northM, upM)` estimated 3D stone center in meters.
 * @property sigma3dMm 3D sphere-of-confusion radius in millimeters (combining ray residual and baseline geometry).
 * @property maxParallaxDeg Maximum pairwise ray parallax angle in degrees across baseline-separated viewpoints.
 * @property viewCount Number of distinct viewpoints contributing to the estimate.
 */
class MultiViewTriangulationResult(
    val targetEnuM: FloatArray,
    val sigma3dMm: Float,
    val maxParallaxDeg: Float,
    val viewCount: Int
)

/**
 * Orchestrates Stage 2 (`LOCK_ON_NEURAL`) and Stage 3 (`PINPOINT_CONSTELLATION`) target homing.
 *
 * Threading: All state transitions and frame updates are synchronized on [stateLock] so UI actions
 * (Main thread), 120 Hz IMU display reprojections (Main thread), 30 Hz OpenCV updates
 * (`GneissFind-Analyzer`), and `B=1` neural updates (`GneissFind-Inference`) never race.
 * All timestamps passed to this controller must use the same monotonic clock (`System.nanoTime()`).
 *
 * @param openCvTracker Native OpenCV KLT / triangulation / `AKAZE` constellation engine.
 * @param waistHeightMeters Nominal initial camera height above the gravel plane in meters (`1.06 m`).
 */
class LockOnPinpointController(
    private val openCvTracker: OpenCvConstellationTracker = OpenCvConstellationTracker(),
    private val waistHeightMeters: Float = GroundPlaneRaycaster.DEFAULT_WAIST_HEIGHT_METERS
) : AutoCloseable {

    private val stateLock = Any()

    private var stage: AcquisitionStage = AcquisitionStage.PATROL
    private var lockedTrackId: Long = -1L
    private var stoneClass: StoneClass = StoneClass.CHALCEDONY_AGATE
    private var smoothedConfidence: Float = 0.80f
    private var estimatedDiameterMm: Float = 30f

    // Current metric ENU coordinates: [0] = East (m), [1] = North (m), [2] = Up (m, ground = 0)
    private val camPosEnu = floatArrayOf(0f, 0f, waistHeightMeters)
    private val prevFrameCamPosEnu = floatArrayOf(0f, 0f, waistHeightMeters)
    private val prevFrameRotationScr = FloatArray(9)
    private var hasPrevAnalyzerPose = false

    private val targetPosEnu = floatArrayOf(0f, 1.0f, 0f)
    private var angularWidthRad: Float = 0.03f
    private var angularHeightRad: Float = 0.03f

    // Separate 3D-anchored neural detector bounding box so 1 Hz neural box stays world-locked on screen
    private var hasNeuralDetBox: Boolean = false
    private val neuralBoxPosEnu = floatArrayOf(0f, 1.0f, 0f)
    private var neuralAngularWidthRad: Float = 0.03f
    private var neuralAngularHeightRad: Float = 0.03f
    private var missedNeuralCropCount: Int = 0

    private var sigma3dMm: Float = INITIAL_SIGMA_3D_MM
    private var maxParallaxDeg: Float = 0f
    private val viewObservations = ArrayList<TriangulationViewObservation>(MAX_TRIANGULATION_VIEWS)

    private var pinpointAnchorRangeM: Float = 1.0f
    private var smoothedPinpointRangeM: Float = 1.0f
    private var akazeInliers: Int = 0
    private var consecutiveAkazeSpikeFrames: Int = 0
    private var lastPinpointNeuralRefreshNs: Long = 0L

    private var latestFrameLeftPx: Float = 900f
    private var latestFrameTopPx: Float = 480f
    private var latestFrameRightPx: Float = 1020f
    private var latestFrameBottomPx: Float = 600f
    private var isTargetInFrame: Boolean = true
    private var wasTargetOffScreen: Boolean = false

    private var lostTrackStartNs: Long = 0L
    private var latestLumaSnapshot: ByteArray? = null
    private var latestLumaWidth: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX
    private var latestLumaHeight: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX

    /** Current acquisition stage (`PATROL`, `LOCK_ON_NEURAL`, or `PINPOINT_CONSTELLATION`). */
    val currentStage: AcquisitionStage
        get() = synchronized(stateLock) { stage }

    /**
     * Copies the current estimated metric ENU camera center `(eastM, northM, upM)` into [outEnu]
     * so `GneissFind-Analyzer` can attach capture-time camera translation to a queued neural slot.
     */
    fun copyCurrentCamPosEnu(outEnu: FloatArray = FloatArray(3)): FloatArray = synchronized(stateLock) {
        if (outEnu.size >= 3) {
            outEnu[0] = camPosEnu[0]
            outEnu[1] = camPosEnu[1]
            outEnu[2] = camPosEnu[2]
        }
        outEnu
    }

    /**
     * Enters Stage 2 (`LOCK_ON_NEURAL`) locked onto [target], anchoring its 3D ENU position from
     * the current camera orientation and slant range.
     *
     * @param target Selected [ConfirmedTarget] from `PATROL`.
     * @param stripGeometry Active push-broom strip geometry when the target was selected.
     * @param currentCropYFloat Active continuous EIS `cropY` offset when the target was selected.
     * @param currentFocalLengthPx Calibrated vertical pinhole focal length in `1080p` pixels.
     * @param currentRotationMatrix Current `3x3` `R_world_from_scr` rotation matrix.
     * @param timestampNs Monotonic wall timestamp in nanoseconds (`System.nanoTime()`).
     * @return Initial [LockOnHudState] for Stage 2 (`LOCK_ON_NEURAL`).
     */
    fun lockOntoTarget(
        target: ConfirmedTarget,
        stripGeometry: StripRoiGeometry,
        currentCropYFloat: Float,
        currentFocalLengthPx: Float,
        currentRotationMatrix: FloatArray,
        timestampNs: Long
    ): LockOnHudState = synchronized(stateLock) {
        openCvTracker.resetSession()
        viewObservations.clear()

        stage = AcquisitionStage.LOCK_ON_NEURAL
        lockedTrackId = target.trackId
        stoneClass = target.stoneClass
        smoothedConfidence = target.smoothedConfidence.coerceIn(0.35f, 0.99f)
        estimatedDiameterMm = target.estimatedDiameterMm.coerceIn(18f, 250f)

        val f = currentFocalLengthPx.coerceAtLeast(100f)
        val initStripCenterX = (target.stripLeftPx + target.stripRightPx) * 0.5f
        val initStripCenterY = (target.stripTopPx + target.stripBottomPx) * 0.5f
        val initFrameX = stripGeometry.stripLeftPx + initStripCenterX
        val initFrameY = currentCropYFloat + initStripCenterY

        val ray = target.worldRay?.normalized() ?: GroundPlaneRaycaster.pixelToWorldRay(
            frameXPx = initFrameX,
            frameYPx = initFrameY,
            focalLengthPx = f,
            rotationMatrix = currentRotationMatrix,
            frameWidthPx = stripGeometry.uprightFrameWidth,
            frameHeightPx = stripGeometry.uprightFrameHeight
        )

        val r0 = target.slantRangeMeters.coerceIn(0.35f, 4.0f)
        val downSin = (-ray.upZ).coerceIn(0.15f, 0.999f)
        val initHeightM = (r0 * downSin).coerceIn(0.25f, 1.80f)

        camPosEnu[0] = 0f
        camPosEnu[1] = 0f
        camPosEnu[2] = initHeightM
        hasPrevAnalyzerPose = false

        targetPosEnu[0] = camPosEnu[0] + r0 * ray.eastX
        targetPosEnu[1] = camPosEnu[1] + r0 * ray.northY
        targetPosEnu[2] = (camPosEnu[2] + r0 * ray.upZ).coerceIn(-0.05f, 0.05f)

        val initW = (target.stripRightPx - target.stripLeftPx).coerceAtLeast(24f)
        val initH = (target.stripBottomPx - target.stripTopPx).coerceAtLeast(24f)
        angularWidthRad = if (target.angularWidthRad > 0.002f) target.angularWidthRad else initW / f
        angularHeightRad = if (target.angularHeightRad > 0.002f) target.angularHeightRad else initH / f

        // Initialize 3D-anchored neural detector box from the PATROL lock-on detection
        hasNeuralDetBox = true
        System.arraycopy(targetPosEnu, 0, neuralBoxPosEnu, 0, 3)
        neuralAngularWidthRad = angularWidthRad
        neuralAngularHeightRad = angularHeightRad
        missedNeuralCropCount = 0

        sigma3dMm = max(85f, r0 * 110f)
        maxParallaxDeg = 0f
        pinpointAnchorRangeM = r0
        smoothedPinpointRangeM = r0
        akazeInliers = 0
        consecutiveAkazeSpikeFrames = 0
        lastPinpointNeuralRefreshNs = 0L
        wasTargetOffScreen = false
        lostTrackStartNs = 0L

        // Record initial Lock-On viewpoint t=0 as the first baseline ray
        val rCv0 = scrRotationToCvFromWorld(currentRotationMatrix)
        viewObservations.add(
            TriangulationViewObservation(
                camPosEnu = camPosEnu.clone(),
                rCvFromWorld = rCv0,
                rayEnu = ray,
                pixelU = initFrameX,
                pixelV = initFrameY,
                boxSpanPx = max(initW, initH)
            )
        )

        buildProjectedHudStateLocked(
            rWorldFromScr = currentRotationMatrix,
            focalLengthPx = f,
            nowNs = timestampNs,
            sourceBadge = "RF-DETR B=1"
        )
    }

    /**
     * Immediately exits Stage 2 / Stage 3 (`0 ms`) and resets back to Stage 1 (`PATROL`).
     *
     * @return Default unlocked [LockOnHudState] (`stage == AcquisitionStage.PATROL`).
     */
    fun breakLock(): LockOnHudState = synchronized(stateLock) {
        resetUnlockedLocked()
        LockOnHudState(stage = AcquisitionStage.PATROL)
    }

    /**
     * Reprojects the locked 3D ENU target anchor into `1920x1080` screen space at 120 Hz using
     * the live `TYPE_GAME_ROTATION_VECTOR` orientation, updating off-screen directional chevrons
     * and checking the `15.0 s` lost-track grace countdown.
     *
     * @param currentRotationMatrix Live `3x3` `R_world_from_scr` rotation matrix.
     * @param currentFocalLengthPx Calibrated vertical pinhole focal length in `1080p` pixels.
     * @param nowTimestampNs Monotonic wall timestamp in nanoseconds (`System.nanoTime()`).
     * @return Updated [LockOnHudState] (`stage == PATROL` if the `15.0 s` countdown expired).
     */
    fun reprojectForDisplay120Hz(
        currentRotationMatrix: FloatArray,
        currentFocalLengthPx: Float,
        nowTimestampNs: Long
    ): LockOnHudState = synchronized(stateLock) {
        if (stage == AcquisitionStage.PATROL) {
            return LockOnHudState(stage = AcquisitionStage.PATROL)
        }
        if (checkAndExpireGraceWindowLocked(nowTimestampNs)) {
            return LockOnHudState(stage = AcquisitionStage.PATROL)
        }
        val badge = resolveSourceBadgeLocked()
        buildProjectedHudStateLocked(
            rWorldFromScr = currentRotationMatrix,
            focalLengthPx = currentFocalLengthPx,
            nowNs = nowTimestampNs,
            sourceBadge = badge
        )
    }

    /**
     * Returns the current predicted `(centerXPx, centerYPx)` in `1920x1080` coordinates around
     * which `GneissFind-Analyzer` should extract the clamped neural crop.
     */
    fun predictCurrentCropCenterPx(
        rotationMatrix: FloatArray,
        focalLengthPx: Float
    ): Pair<Float, Float> {
        val (cx, cy, _) = predictCurrentCropWindow(rotationMatrix, focalLengthPx)
        return Pair(cx, cy)
    }

    /**
     * Returns the current predicted `(centerXPx, centerYPx, sourceCropSizePx)` in `1920x1080`
     * coordinates for extracting the adaptive `[320..640]` -> `320x320` neural crop.
     */
    fun predictCurrentCropWindow(
        rotationMatrix: FloatArray,
        focalLengthPx: Float
    ): Triple<Float, Float, Int> = synchronized(stateLock) {
        if (stage == AcquisitionStage.PATROL) {
            return Triple(
                StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX * 0.5f,
                StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX * 0.5f,
                PushBroomCropper.LOCK_ON_CROP_SIZE_PX
            )
        }
        val proj = projectEnuToFrame(
            targetEnu = targetPosEnu,
            camPosEnu = camPosEnu,
            rWorldFromScr = rotationMatrix,
            focalLengthPx = focalLengthPx
        )
        Triple(
            proj.uPx.coerceIn(0f, StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX.toFloat()),
            proj.vPx.coerceIn(0f, StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX.toFloat()),
            desiredCropSizePxLocked(focalLengthPx)
        )
    }

    /**
     * Runs on each `30 FPS` camera frame (after `imageProxy.close()`) to update metric camera
     * translation from background gravel optical flow (gated against fast pure-rotation pans) and—
     * when in Stage 3 (`PINPOINT_CONSTELLATION`)—execute `< 5 ms` OpenCV `AKAZE` constellation
     * similarity tracking.
     *
     * @param lumaBytes Upright `1920x1080` single-channel luminance buffer.
     * @param widthPx Upright frame width in pixels (`1920`).
     * @param heightPx Upright frame height in pixels (`1080`).
     * @param poseSnapshot Synchronized IMU/camera pose for the frame.
     * @param timestampNs Monotonic wall timestamp in nanoseconds (`System.nanoTime()`).
     * @return [LockOnFrameDirective] specifying updated HUD state and whether `B=1` GPU inference is needed.
     */
    fun onCameraLumaFrame(
        lumaBytes: ByteArray,
        widthPx: Int,
        heightPx: Int,
        poseSnapshot: MotionPoseSnapshot,
        timestampNs: Long
    ): LockOnFrameDirective = synchronized(stateLock) {
        if (stage == AcquisitionStage.PATROL) {
            return LockOnFrameDirective(
                hudState = LockOnHudState(stage = AcquisitionStage.PATROL),
                shouldRunNeuralB1 = false,
                desiredCropCenterXPx = widthPx * 0.5f,
                desiredCropCenterYPx = heightPx * 0.5f
            )
        }

        // Retain reference to latest luma snapshot so Stage 2 -> Stage 3 handover can immediately seed AKAZE
        val snapshotBuf = latestLumaSnapshot?.takeIf { it.size == widthPx * heightPx }
            ?: ByteArray(widthPx * heightPx).also { latestLumaSnapshot = it }
        System.arraycopy(lumaBytes, 0, snapshotBuf, 0, min(lumaBytes.size, snapshotBuf.size))
        latestLumaWidth = widthPx
        latestLumaHeight = heightPx

        val focalPx = poseSnapshot.verticalFocalLengthPx.coerceAtLeast(100f)
        val rotScr = poseSnapshot.rotationMatrix

        // 1. Update metric 3D camera center C_enu via OpenCV background gravel KLT + 3D ray-ground solver,
        //    gated against fast pure-rotation tilts/pans AND frozen whenever the target is off-screen or
        //    unanchored so panning across non-ground clutter never drifts camPosEnu.
        val allowGroundVoIntegration = isTargetInFrame &&
            lostTrackStartNs == 0L &&
            missedNeuralCropCount < 2 &&
            (stage == AcquisitionStage.LOCK_ON_NEURAL ||
                akazeInliers >= OpenCvConstellationTracker.MIN_HOMOGRAPHY_INLIERS)
        val gravelFlow = openCvTracker.trackGravelFlowCorrespondences(
            lumaBytes = snapshotBuf,
            widthPx = widthPx,
            heightPx = heightPx,
            excludeLeftPx = latestFrameLeftPx,
            excludeTopPx = latestFrameTopPx,
            excludeRightPx = latestFrameRightPx,
            excludeBottomPx = latestFrameBottomPx
        )
        if (allowGroundVoIntegration && hasPrevAnalyzerPose && gravelFlow != null && gravelFlow.inlierCount >= 12) {
            val rotStepDeg = rotationAngleBetweenMatricesDeg(prevFrameRotationScr, rotScr)
            if (rotStepDeg <= MAX_VO_ROTATION_STEP_DEG) {
                val solvedCam = solveCameraPositionFromGravelFlow(
                    prevCamPosEnu = prevFrameCamPosEnu,
                    prevRotationScr = prevFrameRotationScr,
                    currRotationScr = rotScr,
                    prevPointsXY = gravelFlow.prevFramePointsXY,
                    currPointsXY = gravelFlow.currFramePointsXY,
                    count = gravelFlow.inlierCount,
                    focalLengthPx = focalPx,
                    frameWidthPx = widthPx,
                    frameHeightPx = heightPx
                )
                if (solvedCam != null) {
                    val rotDamp = (1.0f - (rotStepDeg / 1.6f)).coerceIn(0.25f, 1.0f)
                    val stepEast = ((solvedCam[0] - camPosEnu[0]) * rotDamp)
                        .coerceIn(-MAX_VO_HORIZONTAL_STEP_M, MAX_VO_HORIZONTAL_STEP_M)
                    val stepNorth = ((solvedCam[1] - camPosEnu[1]) * rotDamp)
                        .coerceIn(-MAX_VO_HORIZONTAL_STEP_M, MAX_VO_HORIZONTAL_STEP_M)
                    val stepUp = ((solvedCam[2] - camPosEnu[2]) * rotDamp)
                        .coerceIn(-MAX_VO_VERTICAL_STEP_M, MAX_VO_VERTICAL_STEP_M)
                    camPosEnu[0] += stepEast
                    camPosEnu[1] += stepNorth
                    camPosEnu[2] = (camPosEnu[2] + stepUp).coerceIn(0.18f, 1.90f)
                }
            }
        }
        System.arraycopy(camPosEnu, 0, prevFrameCamPosEnu, 0, 3)
        System.arraycopy(rotScr, 0, prevFrameRotationScr, 0, 9)
        hasPrevAnalyzerPose = true

        // 2. Stage 3: 30 FPS OpenCV AKAZE gravel-constellation similarity guidance (< 5 ms CPU)
        var openCvMs = 0f
        if (stage == AcquisitionStage.PINPOINT_CONSTELLATION) {
            val predProjection = projectEnuToFrame(
                targetEnu = targetPosEnu,
                camPosEnu = camPosEnu,
                rWorldFromScr = rotScr,
                focalLengthPx = focalPx,
                frameWidthPx = widthPx,
                frameHeightPx = heightPx
            )
            // Only search AKAZE when the predicted target lies within or near the visible viewport
            val nearViewport = predProjection.inFront &&
                predProjection.uPx in -160f..(widthPx + 160f) &&
                predProjection.vPx in -160f..(heightPx + 160f)

            if (nearViewport) {
                val predX = predProjection.uPx.coerceIn(0f, widthPx.toFloat())
                val predY = predProjection.vPx.coerceIn(0f, heightPx.toFloat())
                val match = openCvTracker.matchConstellationFrame(
                    lumaBytes = null, // Already uploaded into fullGrayMat above
                    widthPx = widthPx,
                    heightPx = heightPx,
                    predictedCenterXPx = predX,
                    predictedCenterYPx = predY
                )
                openCvMs = match.latencyMs

                val matchCx = (match.frameLeftPx + match.frameRightPx) * 0.5f
                val matchCy = (match.frameTopPx + match.frameBottomPx) * 0.5f
                val matchInFrame = matchCx in 0f..widthPx.toFloat() && matchCy in 0f..heightPx.toFloat()

                if (match.matched &&
                    match.inliers >= OpenCvConstellationTracker.MIN_HOMOGRAPHY_INLIERS &&
                    matchInFrame
                ) {
                    akazeInliers = match.inliers
                    wasTargetOffScreen = false
                    lostTrackStartNs = 0L
                    missedNeuralCropCount = 0

                    val currRay = currentTargetRayLocked()
                    val rawRay = GroundPlaneRaycaster.pixelToWorldRay(
                        frameXPx = matchCx,
                        frameYPx = matchCy,
                        focalLengthPx = focalPx,
                        rotationMatrix = rotScr,
                        frameWidthPx = widthPx,
                        frameHeightPx = heightPx
                    )
                    val deltaDeg = currRay.angularDistanceDegTo(rawRay)
                    val rayAlpha = if (deltaDeg > 4.5f && consecutiveAkazeSpikeFrames == 0 && match.inliers < 26) {
                        consecutiveAkazeSpikeFrames = 1
                        0.12f
                    } else {
                        consecutiveAkazeSpikeFrames = 0
                        when {
                            deltaDeg < 0.25f -> 0.16f
                            deltaDeg < 1.20f -> 0.48f
                            else -> 0.82f
                        }
                    }
                    val smoothedRay = blendWorldRays(currRay, rawRay, rayAlpha)

                    val rawRangeM = (pinpointAnchorRangeM / match.cumulativeScale).coerceIn(0.18f, 3.50f)
                    smoothedPinpointRangeM = (smoothedPinpointRangeM * 0.78f + rawRangeM * 0.22f)
                        .coerceIn(0.18f, 3.50f)

                    targetPosEnu[0] = camPosEnu[0] + smoothedPinpointRangeM * smoothedRay.eastX
                    targetPosEnu[1] = camPosEnu[1] + smoothedPinpointRangeM * smoothedRay.northY
                    targetPosEnu[2] = camPosEnu[2] + smoothedPinpointRangeM * smoothedRay.upZ

                    val rawAngW = ((match.frameRightPx - match.frameLeftPx).coerceAtLeast(12f)) / focalPx
                    val rawAngH = ((match.frameBottomPx - match.frameTopPx).coerceAtLeast(12f)) / focalPx
                    angularWidthRad = angularWidthRad * 0.78f + rawAngW * 0.22f
                    angularHeightRad = angularHeightRad * 0.78f + rawAngH * 0.22f

                    val needs1HzRefresh = lastPinpointNeuralRefreshNs == 0L ||
                        (timestampNs - lastPinpointNeuralRefreshNs) >= PINPOINT_NEURAL_REFRESH_INTERVAL_NS
                    if (needs1HzRefresh) {
                        lastPinpointNeuralRefreshNs = timestampNs
                    }

                    val hud = buildProjectedHudStateLocked(
                        rWorldFromScr = rotScr,
                        focalLengthPx = focalPx,
                        nowNs = timestampNs,
                        sourceBadge = "AKAZE"
                    )
                    val cropCx = ((hud.frameLeftPx + hud.frameRightPx) * 0.5f)
                        .finiteOr(widthPx * 0.5f)
                        .coerceIn(0f, widthPx.coerceAtLeast(1).toFloat())
                    val cropCy = ((hud.frameTopPx + hud.frameBottomPx) * 0.5f)
                        .finiteOr(heightPx * 0.5f)
                        .coerceIn(0f, heightPx.coerceAtLeast(1).toFloat())
                    return LockOnFrameDirective(
                        hudState = hud,
                        shouldRunNeuralB1 = needs1HzRefresh,
                        desiredCropCenterXPx = cropCx,
                        desiredCropCenterYPx = cropCy,
                        desiredCropSizePx = hud.cropSizePx,
                        openCvLatencyMs = openCvMs
                    )
                } else {
                    akazeInliers = if (matchInFrame) match.inliers else 0
                }
            } else {
                akazeInliers = 0
            }
        }

        if (checkAndExpireGraceWindowLocked(timestampNs)) {
            return LockOnFrameDirective(
                hudState = LockOnHudState(stage = AcquisitionStage.PATROL),
                shouldRunNeuralB1 = false,
                desiredCropCenterXPx = widthPx * 0.5f,
                desiredCropCenterYPx = heightPx * 0.5f,
                autoExitedToPatrol = true,
                openCvLatencyMs = openCvMs
            )
        }

        val hud = buildProjectedHudStateLocked(
            rWorldFromScr = rotScr,
            focalLengthPx = focalPx,
            nowNs = timestampNs,
            sourceBadge = resolveSourceBadgeLocked()
        )
        val rawCenterX = (hud.frameLeftPx + hud.frameRightPx) * 0.5f
        val rawCenterY = (hud.frameTopPx + hud.frameBottomPx) * 0.5f
        val reachMarginPx = hud.cropSizePx * 0.55f
        // Only run GPU B=1 crop inference when the projected target is inside or reachable by the edge crop
        val canReachInEdgeCrop = hud.isTargetInFrame || (
            rawCenterX in -reachMarginPx..(widthPx + reachMarginPx) &&
                rawCenterY in -reachMarginPx..(heightPx + reachMarginPx) &&
                (hud.offScreenChevron?.angularOffsetDeg ?: 0f) <= 45f
            )
        val cropCx = rawCenterX.finiteOr(widthPx * 0.5f).coerceIn(0f, widthPx.coerceAtLeast(1).toFloat())
        val cropCy = rawCenterY.finiteOr(heightPx * 0.5f).coerceIn(0f, heightPx.coerceAtLeast(1).toFloat())
        return LockOnFrameDirective(
            hudState = hud,
            shouldRunNeuralB1 = canReachInEdgeCrop,
            desiredCropCenterXPx = cropCx,
            desiredCropCenterYPx = cropCy,
            desiredCropSizePx = hud.cropSizePx,
            openCvLatencyMs = openCvMs
        )
    }

    /**
     * Updates Lock-On / Pinpoint state with detections from `rf_detr_nano_320_b1_w8dyn` evaluated
     * on the clamped `[cropSizePx x cropSizePx]` ROI (sampled into `320x320`) at `(cropLeftPx, cropTopPx)`.
     *
     * Uses [poseSnapshot] (and [captureCamPosEnu] when provided) from capture time to back-project
     * detections into 3D ENU coordinates—eliminating GPU inference latency lag—and projects the
     * returned [LockOnHudState] with [displayRotationMatrix] at [timestampNs].
     *
     * @param cropDetections Detections in `[0..320]` tensor-local pixel coordinates.
     * @param cropLeftPx Left X origin of the source square ROI in `1920x1080` pixels.
     * @param cropTopPx Top Y origin of the source square ROI in `1920x1080` pixels.
     * @param poseSnapshot Synchronized IMU/camera pose at crop capture time.
     * @param zoomRatio Active zoom ratio (`2.0f` default).
     * @param timestampNs Monotonic wall timestamp in nanoseconds (`System.nanoTime()`).
     * @param syntheticCameraStepEnu Optional explicit camera position override for pure-JVM unit tests.
     * @param captureCamPosEnu Optional camera ENU position captured when the crop was extracted.
     * @param displayRotationMatrix Optional live `3x3` `R_world_from_scr` rotation matrix at completion time.
     * @param cropSizePx Source square ROI width/height in `1920x1080` pixels (`[320..640]`).
     * @return Updated [LockOnHudState] (`stage == PATROL` if the `15.0 s` grace window expired).
     */
    fun onNeuralB1Detections(
        cropDetections: List<TileDetection>,
        cropLeftPx: Int,
        cropTopPx: Int,
        poseSnapshot: MotionPoseSnapshot,
        zoomRatio: Float = PushBroomCropper.DEFAULT_ZOOM_RATIO,
        timestampNs: Long,
        syntheticCameraStepEnu: FloatArray? = null,
        captureCamPosEnu: FloatArray? = null,
        displayRotationMatrix: FloatArray = poseSnapshot.rotationMatrix,
        cropSizePx: Int = PushBroomCropper.LOCK_ON_CROP_SIZE_PX
    ): LockOnHudState = synchronized(stateLock) {
        if (stage == AcquisitionStage.PATROL) {
            return LockOnHudState(stage = AcquisitionStage.PATROL)
        }
        if (syntheticCameraStepEnu != null && syntheticCameraStepEnu.size >= 3) {
            camPosEnu[0] = syntheticCameraStepEnu[0]
            camPosEnu[1] = syntheticCameraStepEnu[1]
            camPosEnu[2] = syntheticCameraStepEnu[2]
        }

        val obsCamPos = if (syntheticCameraStepEnu != null && syntheticCameraStepEnu.size >= 3) {
            camPosEnu
        } else if (captureCamPosEnu != null && captureCamPosEnu.size >= 3) {
            captureCamPosEnu
        } else {
            camPosEnu
        }

        val effectiveCropSizePx = cropSizePx.coerceIn(
            PushBroomCropper.LOCK_ON_CROP_SIZE_PX,
            PushBroomCropper.LOCK_ON_WIDE_SEARCH_CROP_SIZE_PX
        )
        val cropScale = effectiveCropSizePx.toFloat() / PushBroomCropper.LOCK_ON_CROP_SIZE_PX.toFloat()
        val gateDistPx = max(MAX_GATE_DISTANCE_PX, effectiveCropSizePx * 0.52f)

        val focalPx = poseSnapshot.verticalFocalLengthPx.coerceAtLeast(100f)
        val captureRotScr = poseSnapshot.rotationMatrix
        val expectedProj = projectEnuToFrame(
            targetEnu = targetPosEnu,
            camPosEnu = obsCamPos,
            rWorldFromScr = captureRotScr,
            focalLengthPx = focalPx
        )

        val matchedDet = selectBestLockOnCropDetection(
            cropDetections = cropDetections,
            cropLeftPx = cropLeftPx,
            cropTopPx = cropTopPx,
            cropScale = cropScale,
            maxGateDistancePx = gateDistPx,
            lockedClass = stoneClass,
            expectedFrameX = expectedProj.uPx,
            expectedFrameY = expectedProj.vPx
        )

        val akazeActivelyLocked = stage == AcquisitionStage.PINPOINT_CONSTELLATION &&
            akazeInliers >= OpenCvConstellationTracker.MIN_HOMOGRAPHY_INLIERS

        if (matchedDet != null) {
            val detClass = StoneClass.fromId(matchedDet.classId)
            if (detClass != null && detClass.isUserAlertable) {
                if (detClass == stoneClass || matchedDet.confidence >= smoothedConfidence + 0.12f) {
                    stoneClass = detClass
                }
            }
            smoothedConfidence = smoothedConfidence * 0.65f + matchedDet.confidence * 0.35f
            missedNeuralCropCount = 0
            wasTargetOffScreen = false
            lostTrackStartNs = 0L

            val clampedDetBox = clampBoxToFrame(
                rawLeft = cropLeftPx + matchedDet.leftPx * cropScale,
                rawTop = cropTopPx + matchedDet.topPx * cropScale,
                rawRight = cropLeftPx + matchedDet.rightPx * cropScale,
                rawBottom = cropTopPx + matchedDet.bottomPx * cropScale,
                minSpanPx = 4f,
                frameWidthPx = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
                frameHeightPx = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX
            )
            val fLeft = clampedDetBox[0]
            val fTop = clampedDetBox[1]
            val fRight = clampedDetBox[2]
            val fBottom = clampedDetBox[3]

            val obsU = (fLeft + fRight) * 0.5f
            val obsV = (fTop + fBottom) * 0.5f
            val boxW = fRight - fLeft
            val boxH = fBottom - fTop
            val boxSpan = max(boxW, boxH)
            val isBoxUnclipped = matchedDet.leftPx > 8f &&
                matchedDet.topPx > 8f &&
                matchedDet.rightPx < 312f &&
                matchedDet.bottomPx < 312f

            val obsRay = GroundPlaneRaycaster.pixelToWorldRay(
                frameXPx = obsU,
                frameYPx = obsV,
                focalLengthPx = focalPx,
                rotationMatrix = captureRotScr
            )

            if (akazeActivelyLocked) {
                // Stage 3 1 Hz neural refresh: update 3D-anchored neural detector box + gentle drift correction
                // WITHOUT yanking the 30 FPS AKAZE Pinpoint reticle to 145ms-stale capture coordinates!
                val anchorRange = smoothedPinpointRangeM.coerceIn(0.18f, 3.50f)
                hasNeuralDetBox = true
                neuralBoxPosEnu[0] = obsCamPos[0] + anchorRange * obsRay.eastX
                neuralBoxPosEnu[1] = obsCamPos[1] + anchorRange * obsRay.northY
                neuralBoxPosEnu[2] = obsCamPos[2] + anchorRange * obsRay.upZ
                neuralAngularWidthRad = boxW / focalPx
                neuralAngularHeightRad = boxH / focalPx

                val currAkazeRay = currentTargetRayLocked()
                if (currAkazeRay.angularDistanceDegTo(obsRay) <= 2.5f) {
                    val nudgedEast = obsCamPos[0] + anchorRange * obsRay.eastX
                    val nudgedNorth = obsCamPos[1] + anchorRange * obsRay.northY
                    val nudgedUp = obsCamPos[2] + anchorRange * obsRay.upZ
                    targetPosEnu[0] = targetPosEnu[0] * 0.92f + nudgedEast * 0.08f
                    targetPosEnu[1] = targetPosEnu[1] * 0.92f + nudgedNorth * 0.08f
                    targetPosEnu[2] = targetPosEnu[2] * 0.92f + nudgedUp * 0.08f
                }

                if (isBoxUnclipped) {
                    val gsd = PushBroomCropper.computeGsdMmPerPx(anchorRange, zoomRatio)
                    estimatedDiameterMm = (estimatedDiameterMm * 0.80f + (boxSpan * gsd) * 0.20f)
                        .coerceIn(18f, 250f)
                }

                // Proactive keyframe refresh when AKAZE inliers dip near floor (12..15) so texture lock stays crisp
                if (akazeInliers <= 15 && matchedDet.confidence >= 0.55f) {
                    val seedProj = projectEnuToFrame(
                        targetEnu = targetPosEnu,
                        camPosEnu = camPosEnu,
                        rWorldFromScr = displayRotationMatrix,
                        focalLengthPx = focalPx,
                        frameWidthPx = latestLumaWidth,
                        frameHeightPx = latestLumaHeight
                    )
                    val seedHalfW = max(12f, angularWidthRad * focalPx * 0.5f)
                    val seedHalfH = max(12f, angularHeightRad * focalPx * 0.5f)
                    val clampedSeed = clampBoxToFrame(
                        rawLeft = seedProj.uPx - seedHalfW,
                        rawTop = seedProj.vPx - seedHalfH,
                        rawRight = seedProj.uPx + seedHalfW,
                        rawBottom = seedProj.vPx + seedHalfH,
                        minSpanPx = 8f,
                        frameWidthPx = latestLumaWidth,
                        frameHeightPx = latestLumaHeight
                    )
                    val sLeft = clampedSeed[0]
                    val sTop = clampedSeed[1]
                    val sRight = clampedSeed[2]
                    val sBottom = clampedSeed[3]
                    if (seedProj.inFront &&
                        seedProj.uPx in 64f..(latestLumaWidth - 64f) &&
                        seedProj.vPx in 64f..(latestLumaHeight - 64f)
                    ) {
                        openCvTracker.seedConstellationCollar(
                            lumaBytes = latestLumaSnapshot,
                            widthPx = latestLumaWidth,
                            heightPx = latestLumaHeight,
                            stoneLeftPx = sLeft,
                            stoneTopPx = sTop,
                            stoneRightPx = sRight,
                            stoneBottomPx = sBottom,
                            baseScale = 1.0f
                        )
                        pinpointAnchorRangeM = anchorRange
                    }
                }
            } else {
                // Stage 2 (LOCK_ON_NEURAL) or Stage 3 AKAZE recovery
                if (isBoxUnclipped) {
                    angularWidthRad = angularWidthRad * 0.55f + (boxW / focalPx) * 0.45f
                    angularHeightRad = angularHeightRad * 0.55f + (boxH / focalPx) * 0.45f
                }

                val obsRCv = scrRotationToCvFromWorld(captureRotScr)
                appendObservationIfNewBaselineLocked(
                    TriangulationViewObservation(
                        camPosEnu = obsCamPos.clone(),
                        rCvFromWorld = obsRCv,
                        rayEnu = obsRay,
                        pixelU = obsU,
                        pixelV = obsV,
                        boxSpanPx = boxSpan
                    )
                )

                var updatedFromStereo = false
                if (viewObservations.size >= 2) {
                    val tri = triangulateMultiViewObservations(
                        observations = viewObservations,
                        priorTargetEnu = targetPosEnu,
                        focalLengthPx = focalPx,
                        openCvTracker = openCvTracker
                    )
                    if (tri != null && tri.maxParallaxDeg >= MIN_PARALLAX_FOR_3D_UPDATE_DEG) {
                        val triDistM = computeDistanceEnu(camPosEnu, tri.targetEnuM)
                        val triPlausible = triDistM in 0.18f..3.80f && abs(tri.targetEnuM[2]) <= 0.35f
                        if (triPlausible) {
                            val blend = if (tri.sigma3dMm < 35f) 0.85f else 0.55f
                            targetPosEnu[0] = targetPosEnu[0] * (1f - blend) + tri.targetEnuM[0] * blend
                            targetPosEnu[1] = targetPosEnu[1] * (1f - blend) + tri.targetEnuM[1] * blend
                            targetPosEnu[2] = targetPosEnu[2] * (1f - blend) + tri.targetEnuM[2] * blend
                            sigma3dMm = tri.sigma3dMm
                            maxParallaxDeg = max(maxParallaxDeg, tri.maxParallaxDeg)
                            updatedFromStereo = true
                        }
                    } else if (tri != null) {
                        maxParallaxDeg = max(maxParallaxDeg, tri.maxParallaxDeg)
                    }
                }

                if (!updatedFromStereo) {
                    // Single-view / sub-baseline bearing update with deadband smoothing
                    val currRay = currentTargetRayLocked(fromCamEnu = obsCamPos)
                    val rayDeltaDeg = currRay.angularDistanceDegTo(obsRay)
                    val rayAlpha = when {
                        rayDeltaDeg < 0.30f -> 0.25f
                        rayDeltaDeg < 1.20f -> 0.55f
                        else -> 0.85f
                    }
                    val smoothedRay = blendWorldRays(currRay, obsRay, rayAlpha)
                    val curRange = computeDistanceEnu(obsCamPos, targetPosEnu).coerceIn(0.25f, 3.80f)
                    targetPosEnu[0] = obsCamPos[0] + curRange * smoothedRay.eastX
                    targetPosEnu[1] = obsCamPos[1] + curRange * smoothedRay.northY
                    targetPosEnu[2] = obsCamPos[2] + curRange * smoothedRay.upZ
                }

                val currentRangeM = computeDistanceEnu(obsCamPos, targetPosEnu).coerceIn(0.18f, 3.80f)
                hasNeuralDetBox = true
                neuralBoxPosEnu[0] = obsCamPos[0] + currentRangeM * obsRay.eastX
                neuralBoxPosEnu[1] = obsCamPos[1] + currentRangeM * obsRay.northY
                neuralBoxPosEnu[2] = obsCamPos[2] + currentRangeM * obsRay.upZ
                neuralAngularWidthRad = boxW / focalPx
                neuralAngularHeightRad = boxH / focalPx

                if (isBoxUnclipped) {
                    val gsd = PushBroomCropper.computeGsdMmPerPx(currentRangeM, zoomRatio)
                    estimatedDiameterMm = (estimatedDiameterMm * 0.70f + (boxSpan * gsd) * 0.30f)
                        .coerceIn(18f, 250f)
                }

                // Reproject current target to latestLumaSnapshot pose for seeding AKAZE collar accurately
                val seedProj = projectEnuToFrame(
                    targetEnu = targetPosEnu,
                    camPosEnu = camPosEnu,
                    rWorldFromScr = displayRotationMatrix,
                    focalLengthPx = focalPx,
                    frameWidthPx = latestLumaWidth,
                    frameHeightPx = latestLumaHeight
                )
                val seedHalfW = max(12f, angularWidthRad * focalPx * 0.5f)
                val seedHalfH = max(12f, angularHeightRad * focalPx * 0.5f)
                val clampedSeed = clampBoxToFrame(
                    rawLeft = seedProj.uPx - seedHalfW,
                    rawTop = seedProj.vPx - seedHalfH,
                    rawRight = seedProj.uPx + seedHalfW,
                    rawBottom = seedProj.vPx + seedHalfH,
                    minSpanPx = 8f,
                    frameWidthPx = latestLumaWidth,
                    frameHeightPx = latestLumaHeight
                )
                val sLeft = clampedSeed[0]
                val sTop = clampedSeed[1]
                val sRight = clampedSeed[2]
                val sBottom = clampedSeed[3]

                // Check Stage 2 -> Stage 3 handover criterion: sigma_3D < 0.5 * D, >= 4 views, >= 3.0 deg parallax
                if (stage == AcquisitionStage.LOCK_ON_NEURAL &&
                    viewObservations.size >= MIN_VIEWS_FOR_PINPOINT_HANDOVER &&
                    maxParallaxDeg >= MIN_PARALLAX_DEG_FOR_PINPOINT_HANDOVER &&
                    sigma3dMm < HANDOVER_SIGMA_TO_DIAMETER_RATIO * estimatedDiameterMm
                ) {
                    val seeded = openCvTracker.seedConstellationCollar(
                        lumaBytes = latestLumaSnapshot,
                        widthPx = latestLumaWidth,
                        heightPx = latestLumaHeight,
                        stoneLeftPx = sLeft,
                        stoneTopPx = sTop,
                        stoneRightPx = sRight,
                        stoneBottomPx = sBottom,
                        baseScale = 1.0f
                    )
                    stage = AcquisitionStage.PINPOINT_CONSTELLATION
                    pinpointAnchorRangeM = currentRangeM
                    smoothedPinpointRangeM = currentRangeM
                    lastPinpointNeuralRefreshNs = timestampNs
                    if (seeded) {
                        akazeInliers = OpenCvConstellationTracker.MIN_HOMOGRAPHY_INLIERS
                    }
                } else if (stage == AcquisitionStage.PINPOINT_CONSTELLATION &&
                    akazeInliers < OpenCvConstellationTracker.MIN_HOMOGRAPHY_INLIERS
                ) {
                    // Neural B=1 re-acquired stone after temporary occlusion / off-screen pan -> re-seed AKAZE!
                    val reseeded = openCvTracker.seedConstellationCollar(
                        lumaBytes = latestLumaSnapshot,
                        widthPx = latestLumaWidth,
                        heightPx = latestLumaHeight,
                        stoneLeftPx = sLeft,
                        stoneTopPx = sTop,
                        stoneRightPx = sRight,
                        stoneBottomPx = sBottom,
                        baseScale = 1.0f
                    )
                    pinpointAnchorRangeM = currentRangeM
                    smoothedPinpointRangeM = currentRangeM
                    lastPinpointNeuralRefreshNs = timestampNs
                    if (reseeded) {
                        akazeInliers = OpenCvConstellationTracker.MIN_HOMOGRAPHY_INLIERS
                    }
                }
            }

            return buildProjectedHudStateLocked(
                rWorldFromScr = displayRotationMatrix,
                focalLengthPx = focalPx,
                nowNs = timestampNs,
                sourceBadge = resolveSourceBadgeLocked()
            )
        }

        // Missed neural detection on this crop
        missedNeuralCropCount++
        if (missedNeuralCropCount >= 2 && !akazeActivelyLocked) {
            hasNeuralDetBox = false
        }
        if (!akazeActivelyLocked) {
            if (missedNeuralCropCount >= MIN_CONSECUTIVE_NEURAL_MISSES_FOR_GRACE && lostTrackStartNs == 0L) {
                lostTrackStartNs = timestampNs
            }
            if (checkAndExpireGraceWindowLocked(timestampNs)) {
                return LockOnHudState(stage = AcquisitionStage.PATROL)
            }
        }

        return buildProjectedHudStateLocked(
            rWorldFromScr = displayRotationMatrix,
            focalLengthPx = focalPx,
            nowNs = timestampNs,
            sourceBadge = resolveSourceBadgeLocked()
        )
    }

    override fun close() {
        synchronized(stateLock) {
            resetUnlockedLocked()
            openCvTracker.close()
        }
    }

    private fun resetUnlockedLocked() {
        stage = AcquisitionStage.PATROL
        lockedTrackId = -1L
        viewObservations.clear()
        hasPrevAnalyzerPose = false
        hasNeuralDetBox = false
        missedNeuralCropCount = 0
        wasTargetOffScreen = false
        lostTrackStartNs = 0L
        akazeInliers = 0
        consecutiveAkazeSpikeFrames = 0
        maxParallaxDeg = 0f
        sigma3dMm = INITIAL_SIGMA_3D_MM
        openCvTracker.resetSession()
    }

    private fun desiredCropSizePxLocked(focalLengthPx: Float): Int {
        val f = focalLengthPx.coerceAtLeast(100f)
        val needsWideSearch = lostTrackStartNs != 0L ||
            missedNeuralCropCount >= 1 ||
            (stage == AcquisitionStage.PINPOINT_CONSTELLATION &&
                akazeInliers < OpenCvConstellationTracker.MIN_HOMOGRAPHY_INLIERS)
        if (needsWideSearch) {
            return PushBroomCropper.LOCK_ON_WIDE_SEARCH_CROP_SIZE_PX
        }
        val apparentSpanPx = (max(angularWidthRad, angularHeightRad) * f)
            .finiteOr(PushBroomCropper.LOCK_ON_CROP_SIZE_PX.toFloat())
        return (apparentSpanPx * 2.4f)
            .finiteOr(PushBroomCropper.LOCK_ON_CROP_SIZE_PX.toFloat())
            .roundToInt()
            .coerceIn(
                PushBroomCropper.LOCK_ON_CROP_SIZE_PX,
                PushBroomCropper.LOCK_ON_WIDE_SEARCH_CROP_SIZE_PX
            )
    }

    private fun resolveSourceBadgeLocked(): String = when {
        lostTrackStartNs != 0L -> "STANDBY"
        stage == AcquisitionStage.PINPOINT_CONSTELLATION &&
            akazeInliers >= OpenCvConstellationTracker.MIN_HOMOGRAPHY_INLIERS -> "AKAZE"
        else -> "RF-DETR B=1"
    }

    private fun currentTargetRayLocked(fromCamEnu: FloatArray = camPosEnu): WorldRay3D {
        return WorldRay3D(
            eastX = targetPosEnu[0] - fromCamEnu[0],
            northY = targetPosEnu[1] - fromCamEnu[1],
            upZ = targetPosEnu[2] - fromCamEnu[2]
        ).normalized()
    }

    private fun checkAndExpireGraceWindowLocked(nowNs: Long): Boolean {
        if (lostTrackStartNs == 0L) return false
        if (nowNs >= lostTrackStartNs && (nowNs - lostTrackStartNs) >= MAX_LOST_TRACK_GRACE_NS) {
            resetUnlockedLocked()
            return true
        }
        return false
    }

    private fun appendObservationIfNewBaselineLocked(obs: TriangulationViewObservation) {
        if (viewObservations.isEmpty()) {
            viewObservations.add(obs)
            return
        }
        val last = viewObservations.last()
        val baselineDist = computeDistanceEnu(last.camPosEnu, obs.camPosEnu)
        val anchorDist = computeDistanceEnu(viewObservations.first().camPosEnu, obs.camPosEnu)
        // Require real >= 3.5 cm camera translation before adding a new triangulation view so pure
        // wrist rotation never accumulates zero-baseline noisy rays.
        if (baselineDist >= MIN_STEREO_BASELINE_M && anchorDist >= MIN_STEREO_BASELINE_M) {
            if (viewObservations.size >= MAX_TRIANGULATION_VIEWS) {
                // Preserve observation 0 (initial widest anchor) and drop oldest intermediate sample
                viewObservations.removeAt(1)
            }
            viewObservations.add(obs)
        } else if (viewObservations.size == 1 && anchorDist < 0.015f) {
            // Refresh initial stationary anchor ray before camera begins translating
            viewObservations[0] = obs
        } else if (viewObservations.size > 1 && baselineDist < 0.015f) {
            viewObservations[viewObservations.lastIndex] = obs
        }
    }

    private fun buildProjectedHudStateLocked(
        rWorldFromScr: FloatArray,
        focalLengthPx: Float,
        nowNs: Long,
        sourceBadge: String
    ): LockOnHudState {
        val frameW = latestLumaWidth
        val frameH = latestLumaHeight
        val proj = projectEnuToFrame(
            targetEnu = targetPosEnu,
            camPosEnu = camPosEnu,
            rWorldFromScr = rWorldFromScr,
            focalLengthPx = focalLengthPx,
            frameWidthPx = frameW,
            frameHeightPx = frameH
        )
        val halfW = max(12f, angularWidthRad * focalLengthPx * 0.5f)
        val halfH = max(12f, angularHeightRad * focalLengthPx * 0.5f)
        val bLeft = proj.uPx - halfW
        val bTop = proj.vPx - halfH
        val bRight = proj.uPx + halfW
        val bBottom = proj.vPx + halfH

        val inFrame = proj.inFront &&
            proj.uPx in 0f..frameW.toFloat() &&
            proj.vPx in 0f..frameH.toFloat()
        isTargetInFrame = inFrame
        if (inFrame) {
            latestFrameLeftPx = bLeft
            latestFrameTopPx = bTop
            latestFrameRightPx = bRight
            latestFrameBottomPx = bBottom
            if (wasTargetOffScreen) {
                // Target re-entered the visible viewport after an off-screen pan -> immediately reset 15s grace timer!
                wasTargetOffScreen = false
                lostTrackStartNs = 0L
                missedNeuralCropCount = 0
            }
        } else {
            wasTargetOffScreen = true
            if (lostTrackStartNs == 0L) {
                lostTrackStartNs = nowNs
            }
        }

        // Reproject 3D-anchored neural detector bounding box so it stays world-locked on screen
        val neuralProj = projectEnuToFrame(
            targetEnu = neuralBoxPosEnu,
            camPosEnu = camPosEnu,
            rWorldFromScr = rWorldFromScr,
            focalLengthPx = focalLengthPx,
            frameWidthPx = frameW,
            frameHeightPx = frameH
        )
        val neuralVisible = hasNeuralDetBox &&
            neuralProj.inFront &&
            neuralProj.uPx in 0f..frameW.toFloat() &&
            neuralProj.vPx in 0f..frameH.toFloat()
        val nHalfW = max(10f, neuralAngularWidthRad * focalLengthPx * 0.5f)
        val nHalfH = max(10f, neuralAngularHeightRad * focalLengthPx * 0.5f)

        val activeCropSizePx = desiredCropSizePxLocked(focalLengthPx)
        val (cropL, cropT) = PushBroomCropper.computeClampedSquareCropOrigin(
            centerXPx = proj.uPx,
            centerYPx = proj.vPx,
            cropSizePx = activeCropSizePx,
            frameWidthPx = frameW,
            frameHeightPx = frameH
        )
        val sigmaPx = ((sigma3dMm * 0.001f) * focalLengthPx / proj.slantRangeM.coerceAtLeast(0.18f))
            .coerceIn(8f, 220f)
        val graceActive = lostTrackStartNs != 0L
        val remSec = if (graceActive && nowNs >= lostTrackStartNs) {
            (MAX_LOST_TRACK_GRACE_SEC - (nowNs - lostTrackStartNs) * 1e-9f).coerceIn(0f, MAX_LOST_TRACK_GRACE_SEC)
        } else {
            MAX_LOST_TRACK_GRACE_SEC
        }
        val chevron = if (!inFrame) {
            computeOffScreenChevron(proj, frameW, frameH)
        } else {
            null
        }
        val effectiveBadge = when {
            !inFrame || graceActive -> "STANDBY"
            stage == AcquisitionStage.PINPOINT_CONSTELLATION &&
                akazeInliers >= OpenCvConstellationTracker.MIN_HOMOGRAPHY_INLIERS -> "AKAZE"
            else -> sourceBadge
        }

        return LockOnHudState(
            stage = stage,
            lockedTrackId = lockedTrackId,
            stoneClass = stoneClass,
            smoothedConfidence = smoothedConfidence,
            estimatedDiameterMm = estimatedDiameterMm,
            slantRangeMeters = proj.slantRangeM,
            sigma3dMm = sigma3dMm,
            sigmaRadiusPx = sigmaPx,
            parallaxAngleDeg = maxParallaxDeg,
            triangulatedViews = viewObservations.size,
            akazeInliers = if (inFrame) akazeInliers else 0,
            frameLeftPx = bLeft,
            frameTopPx = bTop,
            frameRightPx = bRight,
            frameBottomPx = bBottom,
            cropLeftPx = cropL,
            cropTopPx = cropT,
            cropSizePx = activeCropSizePx,
            hasNeuralDetectorBox = neuralVisible,
            neuralBoxLeftPx = neuralProj.uPx - nHalfW,
            neuralBoxTopPx = neuralProj.vPx - nHalfH,
            neuralBoxRightPx = neuralProj.uPx + nHalfW,
            neuralBoxBottomPx = neuralProj.vPx + nHalfH,
            isTargetInFrame = inFrame,
            isLostTrackGraceActive = graceActive,
            lostTrackRemainingSec = remSec,
            offScreenChevron = chevron,
            trackingSourceBadge = effectiveBadge
        )
    }

    private data class PinholeProjection(
        val uPx: Float,
        val vPx: Float,
        val xCv: Float,
        val yCv: Float,
        val zCv: Float,
        val slantRangeM: Float,
        val inFront: Boolean
    )

    companion object {
        const val INITIAL_SIGMA_3D_MM = 140.0f
        const val HANDOVER_SIGMA_TO_DIAMETER_RATIO = 0.50f
        const val MIN_VIEWS_FOR_PINPOINT_HANDOVER = 4
        const val MIN_PARALLAX_DEG_FOR_PINPOINT_HANDOVER = 3.0f
        const val MIN_PARALLAX_FOR_3D_UPDATE_DEG = 0.8f
        const val MIN_STEREO_BASELINE_M = 0.035f
        const val MAX_TRIANGULATION_VIEWS = 16

        /** Skip KLT translation updates when inter-frame rotation exceeds `1.25 deg/frame` (`~37.5 deg/s`). */
        const val MAX_VO_ROTATION_STEP_DEG = 1.25f
        private const val MAX_VO_HORIZONTAL_STEP_M = 0.032f
        private const val MAX_VO_VERTICAL_STEP_M = 0.022f

        /** `1.0 Hz` neural confidence refresh period during Stage 3 `PINPOINT_CONSTELLATION`. */
        const val PINPOINT_NEURAL_REFRESH_INTERVAL_NS = 1_000_000_000L

        /** `15.0 s` continuous lost-track grace standby duration before auto-returning to `PATROL`. */
        const val MAX_LOST_TRACK_GRACE_SEC = 15.0f
        const val MAX_LOST_TRACK_GRACE_NS = 15_000_000_000L
        private const val MIN_CONSECUTIVE_NEURAL_MISSES_FOR_GRACE = 3

        private const val LOCK_ON_SCORE_FLOOR = 0.28f
        private const val MAX_GATE_DISTANCE_PX = 210.0f

        /**
         * Selects the visible alertable [ConfirmedTarget] for **`LOCK NEAREST`** using smallest
         * [ConfirmedTarget.slantRangeMeters] and tie-breaking by smallest horizontal pixel distance
         * from the `1920x1080` viewport center (`960 px`).
         *
         * @param targets Active confirmed targets in `PATROL`.
         * @param stripGeometry Active push-broom strip geometry.
         * @return Nearest visible [ConfirmedTarget], or `null` when zero targets are visible.
         */
        fun selectNearestVisibleTarget(
            targets: List<ConfirmedTarget>,
            stripGeometry: StripRoiGeometry = StripRoiGeometry()
        ): ConfirmedTarget? {
            val frameCenterX = stripGeometry.uprightFrameWidth * 0.5f
            var best: ConfirmedTarget? = null
            var bestCenterDist = Float.POSITIVE_INFINITY

            for (candidate in targets) {
                if (!candidate.isVisibleInStrip || !candidate.stoneClass.isUserAlertable) continue
                val candFrameCenterX =
                    stripGeometry.stripLeftPx + (candidate.stripLeftPx + candidate.stripRightPx) * 0.5f
                val centerDist = abs(candFrameCenterX - frameCenterX)

                val currentBest = best
                if (currentBest == null) {
                    best = candidate
                    bestCenterDist = centerDist
                } else {
                    val rangeDiff = candidate.slantRangeMeters - currentBest.slantRangeMeters
                    if (rangeDiff < -1e-3f || (abs(rangeDiff) <= 1e-3f && centerDist < bestCenterDist)) {
                        best = candidate
                        bestCenterDist = centerDist
                    }
                }
            }
            return best
        }

        /**
         * Hit-tests a user tap at `(tapStripXPx, tapStripYPx)` in unified strip pixels against
         * visible [ConfirmedTarget] reticles, enforcing a generous `touchSlopPx` (`>= 48.dp` minimum
         * hit box) around each reticle.
         *
         * @param targets Active confirmed targets in `PATROL`.
         * @param tapStripXPx Tap X coordinate in unified strip pixels (`[0..stripWidthPx]`).
         * @param tapStripYPx Tap Y coordinate in unified strip pixels (`[0..STRIP_HEIGHT_PX]`).
         * @param touchSlopPx Extra hit-padding radius in strip pixels (`>= 40f`).
         * @return Closest tapped [ConfirmedTarget], or `null` if no reticle lies within touch slop.
         */
        fun findTappedTargetInStrip(
            targets: List<ConfirmedTarget>,
            tapStripXPx: Float,
            tapStripYPx: Float,
            touchSlopPx: Float = 56f
        ): ConfirmedTarget? {
            var best: ConfirmedTarget? = null
            var bestDistSq = Float.POSITIVE_INFINITY

            for (target in targets) {
                if (!target.isVisibleInStrip || !target.stoneClass.isUserAlertable) continue
                val cx = (target.stripLeftPx + target.stripRightPx) * 0.5f
                val cy = (target.stripTopPx + target.stripBottomPx) * 0.5f
                val halfW = max((target.stripRightPx - target.stripLeftPx) * 0.5f + touchSlopPx * 0.5f, touchSlopPx)
                val halfH = max((target.stripBottomPx - target.stripTopPx) * 0.5f + touchSlopPx * 0.5f, touchSlopPx)

                if (abs(tapStripXPx - cx) <= halfW && abs(tapStripYPx - cy) <= halfH) {
                    val dx = tapStripXPx - cx
                    val dy = tapStripYPx - cy
                    val dSq = dx * dx + dy * dy
                    if (dSq < bestDistSq) {
                        bestDistSq = dSq
                        best = target
                    }
                }
            }
            return best
        }

        /**
         * Returns the 3D rotation angle in degrees (`[0, 180]`) between two `3x3` orthogonal
         * rotation matrices `R_a` and `R_b` via `acos((trace(R_a^T R_b) - 1) / 2)`.
         */
        fun rotationAngleBetweenMatricesDeg(rA: FloatArray, rB: FloatArray): Float {
            if (rA.size < 9 || rB.size < 9) return 0f
            var trace = 0.0
            for (i in 0 until 9) {
                trace += rA[i].toDouble() * rB[i].toDouble()
            }
            val cosTheta = ((trace - 1.0) * 0.5).coerceIn(-1.0, 1.0)
            return Math.toDegrees(acos(cosTheta)).toFloat()
        }

        /**
         * Solves the metric 3D ENU camera optical center `C_curr = (eastM, northM, upM)` (including
         * vertical crouch height!) from RANSAC-inlier static gravel correspondences between two frames
         * via exact `3x3` orthogonal ray-projection linear least squares:
         * `sum_i (I - d_curr_i d_curr_i^T) C_curr = sum_i (I - d_curr_i d_curr_i^T) G_i`.
         */
        fun solveCameraPositionFromGravelFlow(
            prevCamPosEnu: FloatArray,
            prevRotationScr: FloatArray,
            currRotationScr: FloatArray,
            prevPointsXY: FloatArray,
            currPointsXY: FloatArray,
            count: Int,
            focalLengthPx: Float,
            frameWidthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
            frameHeightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX
        ): FloatArray? {
            val prevHeight = prevCamPosEnu[2]
            if (prevHeight <= 0.08f || count < 6) return null

            var a00 = 0.0; var a01 = 0.0; var a02 = 0.0
            var a11 = 0.0; var a12 = 0.0; var a22 = 0.0
            var b0 = 0.0; var b1 = 0.0; var b2 = 0.0
            var validRays = 0

            for (i in 0 until count) {
                val uPrev = prevPointsXY[i * 2]
                val vPrev = prevPointsXY[i * 2 + 1]
                val uCurr = currPointsXY[i * 2]
                val vCurr = currPointsXY[i * 2 + 1]

                val dPrev = GroundPlaneRaycaster.pixelToWorldRay(
                    frameXPx = uPrev,
                    frameYPx = vPrev,
                    focalLengthPx = focalLengthPx,
                    rotationMatrix = prevRotationScr,
                    frameWidthPx = frameWidthPx,
                    frameHeightPx = frameHeightPx
                )
                if (-dPrev.upZ < 0.12f) continue
                val rangePrev = prevHeight / (-dPrev.upZ)
                if (rangePrev !in 0.12f..6.50f) continue

                val gEast = (prevCamPosEnu[0] + rangePrev * dPrev.eastX).toDouble()
                val gNorth = (prevCamPosEnu[1] + rangePrev * dPrev.northY).toDouble()
                val gUp = 0.0

                val dCurr = GroundPlaneRaycaster.pixelToWorldRay(
                    frameXPx = uCurr,
                    frameYPx = vCurr,
                    focalLengthPx = focalLengthPx,
                    rotationMatrix = currRotationScr,
                    frameWidthPx = frameWidthPx,
                    frameHeightPx = frameHeightPx
                )
                val dx = dCurr.eastX.toDouble()
                val dy = dCurr.northY.toDouble()
                val dz = dCurr.upZ.toDouble()

                val p00 = 1.0 - dx * dx
                val p01 = -dx * dy
                val p02 = -dx * dz
                val p11 = 1.0 - dy * dy
                val p12 = -dy * dz
                val p22 = 1.0 - dz * dz

                a00 += p00; a01 += p01; a02 += p02
                a11 += p11; a12 += p12; a22 += p22

                b0 += p00 * gEast + p01 * gNorth + p02 * gUp
                b1 += p01 * gEast + p11 * gNorth + p12 * gUp
                b2 += p02 * gEast + p12 * gNorth + p22 * gUp
                validRays++
            }

            if (validRays < 6) return null
            // Gentle Tikhonov regularization anchored to prevCamPosEnu prevents ill-conditioning on narrow FOV patches
            val reg = 2e-3
            a00 += reg; a11 += reg; a22 += reg
            b0 += reg * prevCamPosEnu[0]
            b1 += reg * prevCamPosEnu[1]
            b2 += reg * prevCamPosEnu[2]

            return solveSymmetric3x3(a00, a01, a02, a11, a12, a22, b0, b1, b2)
        }

        /**
         * Intersects `>= 2` calibrated multi-view target observations `(C_k, d_k)` to estimate the
         * 3D stone coordinate `P_enu` and its sphere-of-confusion radius `sigma3dMm`.
         *
         * Only viewpoint pairs separated by `>= MIN_STEREO_BASELINE_M` (`3.5 cm`) contribute to
         * stereo parallax so pure camera rotation never hallucinates depth convergence.
         */
        fun triangulateMultiViewObservations(
            observations: List<TriangulationViewObservation>,
            priorTargetEnu: FloatArray,
            focalLengthPx: Float,
            openCvTracker: OpenCvConstellationTracker? = null
        ): MultiViewTriangulationResult? {
            if (observations.size < 2) return null

            // 1. Find maximum pairwise ray parallax angle across genuinely baseline-separated viewpoints
            var maxAngleDeg = 0f
            var wideA = 0
            var wideB = observations.lastIndex
            for (i in observations.indices) {
                for (j in i + 1 until observations.size) {
                    val pairBaselineM = computeDistanceEnu(
                        observations[i].camPosEnu,
                        observations[j].camPosEnu
                    )
                    if (pairBaselineM < MIN_STEREO_BASELINE_M) continue
                    val ang = observations[i].rayEnu.angularDistanceDegTo(observations[j].rayEnu)
                    if (ang > maxAngleDeg) {
                        maxAngleDeg = ang
                        wideA = i
                        wideB = j
                    }
                }
            }
            if (maxAngleDeg <= 0f) {
                return MultiViewTriangulationResult(
                    targetEnuM = priorTargetEnu.clone(),
                    sigma3dMm = INITIAL_SIGMA_3D_MM,
                    maxParallaxDeg = 0f,
                    viewCount = observations.size
                )
            }

            // 2. Solve multi-view ray-bundle midpoint least squares: sum_k (I - d_k d_k^T) P = sum_k (I - d_k d_k^T) C_k
            var a00 = 0.0; var a01 = 0.0; var a02 = 0.0
            var a11 = 0.0; var a12 = 0.0; var a22 = 0.0
            var b0 = 0.0; var b1 = 0.0; var b2 = 0.0

            for (obs in observations) {
                val dx = obs.rayEnu.eastX.toDouble()
                val dy = obs.rayEnu.northY.toDouble()
                val dz = obs.rayEnu.upZ.toDouble()
                val cx = obs.camPosEnu[0].toDouble()
                val cy = obs.camPosEnu[1].toDouble()
                val cz = obs.camPosEnu[2].toDouble()

                val p00 = 1.0 - dx * dx
                val p01 = -dx * dy
                val p02 = -dx * dz
                val p11 = 1.0 - dy * dy
                val p12 = -dy * dz
                val p22 = 1.0 - dz * dz

                a00 += p00; a01 += p01; a02 += p02
                a11 += p11; a12 += p12; a22 += p22
                b0 += p00 * cx + p01 * cy + p02 * cz
                b1 += p01 * cx + p11 * cy + p12 * cz
                b2 += p02 * cx + p12 * cy + p22 * cz
            }

            val priorWeight = if (maxAngleDeg < 2.0f) 0.008 else 0.0005
            a00 += priorWeight; a11 += priorWeight; a22 += priorWeight
            b0 += priorWeight * priorTargetEnu[0]
            b1 += priorWeight * priorTargetEnu[1]
            b2 += priorWeight * priorTargetEnu[2]

            var solved = solveSymmetric3x3(a00, a01, a02, a11, a12, a22, b0, b1, b2)

            // Cross-fuse with OpenCV C++ Calib3d.triangulatePoints on the widest-baseline pair when native OpenCV is loaded
            if (openCvTracker != null && maxAngleDeg >= 1.5f) {
                val obsA = observations[wideA]
                val obsB = observations[wideB]
                val cvTri = openCvTracker.triangulateTwoViewsOpenCv(
                    camPosAEnu = obsA.camPosEnu,
                    rCvFromWorldA = obsA.rCvFromWorld,
                    pixelUA = obsA.pixelU,
                    pixelVA = obsA.pixelV,
                    camPosBEnu = obsB.camPosEnu,
                    rCvFromWorldB = obsB.rCvFromWorld,
                    pixelUB = obsB.pixelU,
                    pixelVB = obsB.pixelV,
                    focalLengthPx = focalLengthPx
                )
                if (cvTri != null) {
                    solved = if (solved != null) {
                        // Blend N-view ray-bundle (75%) with OpenCV 2-view projective triangulation (25%)
                        floatArrayOf(
                            solved[0] * 0.75f + cvTri[0] * 0.25f,
                            solved[1] * 0.75f + cvTri[1] * 0.25f,
                            solved[2] * 0.75f + cvTri[2] * 0.25f
                        )
                    } else {
                        cvTri
                    }
                }
            }

            val pEnu = solved ?: return null

            // 3. Compute 3D sphere-of-confusion radius sigma3dMm from orthogonal ray residuals + baseline geometry
            var sumSqResidualM = 0.0
            var meanSpanPx = 0f
            for (obs in observations) {
                val vx = pEnu[0] - obs.camPosEnu[0]
                val vy = pEnu[1] - obs.camPosEnu[1]
                val vz = pEnu[2] - obs.camPosEnu[2]
                val along = vx * obs.rayEnu.eastX + vy * obs.rayEnu.northY + vz * obs.rayEnu.upZ
                val ox = vx - along * obs.rayEnu.eastX
                val oy = vy - along * obs.rayEnu.northY
                val oz = vz - along * obs.rayEnu.upZ
                sumSqResidualM += (ox * ox + oy * oy + oz * oz).toDouble()
                meanSpanPx += obs.boxSpanPx
            }
            val rmsResidualM = sqrt(sumSqResidualM / observations.size).toFloat()
            meanSpanPx /= observations.size

            val latestCam = observations.last().camPosEnu
            val currentRangeM = computeDistanceEnu(latestCam, pEnu).coerceIn(0.18f, 4.50f)
            val pixelJitterPx = max(1.6f, 0.045f * meanSpanPx)
            val sigmaAngRad = pixelJitterPx / focalLengthPx.coerceAtLeast(100f)
            val halfParallaxSin = sin(Math.toRadians((maxAngleDeg * 0.5).coerceIn(0.35, 60.0))).toFloat()
            val multiViewFactor = 1.0f / sqrt((observations.size - 1).coerceAtLeast(1).toFloat())
            val geometricDepthSigmaM = (currentRangeM * sigmaAngRad / halfParallaxSin) * multiViewFactor

            val sigmaMm = (1000.0f * hypot(rmsResidualM, geometricDepthSigmaM)).coerceIn(2.5f, 250.0f)
            return MultiViewTriangulationResult(
                targetEnuM = pEnu,
                sigma3dMm = sigmaMm,
                maxParallaxDeg = maxAngleDeg,
                viewCount = observations.size
            )
        }

        private fun blendWorldRays(a: WorldRay3D, b: WorldRay3D, alpha: Float): WorldRay3D {
            val t = alpha.coerceIn(0f, 1f)
            val inv = 1f - t
            return WorldRay3D(
                eastX = a.eastX * inv + b.eastX * t,
                northY = a.northY * inv + b.northY * t,
                upZ = a.upZ * inv + b.upZ * t
            ).normalized()
        }

        private fun selectBestLockOnCropDetection(
            cropDetections: List<TileDetection>,
            cropLeftPx: Int,
            cropTopPx: Int,
            cropScale: Float,
            maxGateDistancePx: Float,
            lockedClass: StoneClass,
            expectedFrameX: Float,
            expectedFrameY: Float
        ): TileDetection? {
            if (cropDetections.isEmpty()) return null
            var best: TileDetection? = null
            var bestUtility = Float.NEGATIVE_INFINITY

            for (det in cropDetections) {
                if (det.confidence < LOCK_ON_SCORE_FLOOR) continue
                val cls = StoneClass.fromId(det.classId) ?: continue
                if (!cls.isUserAlertable) continue

                val fx = cropLeftPx + (det.leftPx + det.rightPx) * 0.5f * cropScale
                val fy = cropTopPx + (det.topPx + det.bottomPx) * 0.5f * cropScale
                val distPx = hypot(fx - expectedFrameX, fy - expectedFrameY)
                if (distPx > maxGateDistancePx) continue

                val classBonus = if (cls == lockedClass) 0.20f else 0.0f
                val proximityBonus = (1.0f - (distPx / maxGateDistancePx)).coerceIn(0f, 1f) * 0.25f
                val utility = det.confidence + classBonus + proximityBonus
                if (utility > bestUtility) {
                    bestUtility = utility
                    best = det
                }
            }
            return best
        }

        private fun scrRotationToCvFromWorld(rWorldFromScr: FloatArray): FloatArray {
            // R_world_from_cv = R_world_from_scr * diag(1, -1, -1); R_cv_from_world = R_world_from_cv^T
            return floatArrayOf(
                rWorldFromScr[0], rWorldFromScr[3], rWorldFromScr[6],
                -rWorldFromScr[1], -rWorldFromScr[4], -rWorldFromScr[7],
                -rWorldFromScr[2], -rWorldFromScr[5], -rWorldFromScr[8]
            )
        }

        private fun projectEnuToFrame(
            targetEnu: FloatArray,
            camPosEnu: FloatArray,
            rWorldFromScr: FloatArray,
            focalLengthPx: Float,
            frameWidthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
            frameHeightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX
        ): PinholeProjection {
            val safeW = frameWidthPx.coerceAtLeast(1)
            val safeH = frameHeightPx.coerceAtLeast(1)
            val dEast = (targetEnu[0] - camPosEnu[0]).finiteOr(0f)
            val dNorth = (targetEnu[1] - camPosEnu[1]).finiteOr(1f)
            val dUp = (targetEnu[2] - camPosEnu[2]).finiteOr(-1f)
            val rangeM = sqrt(dEast * dEast + dNorth * dNorth + dUp * dUp)
                .finiteOr(1.5f)
                .coerceAtLeast(1e-4f)

            val xScr = (rWorldFromScr[0] * dEast + rWorldFromScr[3] * dNorth + rWorldFromScr[6] * dUp).finiteOr(0f)
            val yScr = (rWorldFromScr[1] * dEast + rWorldFromScr[4] * dNorth + rWorldFromScr[7] * dUp).finiteOr(0f)
            val zScr = (rWorldFromScr[2] * dEast + rWorldFromScr[5] * dNorth + rWorldFromScr[8] * dUp).finiteOr(-1f)

            val xCv = xScr
            val yCv = -yScr
            val zCv = -zScr
            val f = focalLengthPx.finiteOr(2000f).coerceAtLeast(100f)
            val cx = safeW * 0.5f
            val cy = safeH * 0.5f

            if (zCv <= 0.05f) {
                val norm2d = hypot(xCv, yCv).finiteOr(1f).coerceAtLeast(1e-4f)
                return PinholeProjection(
                    uPx = (cx + (xCv / norm2d) * safeW * 2.0f).finiteOr(cx),
                    vPx = (cy + (yCv / norm2d) * safeH * 2.0f).finiteOr(cy),
                    xCv = xCv,
                    yCv = yCv,
                    zCv = zCv,
                    slantRangeM = rangeM,
                    inFront = false
                )
            }

            return PinholeProjection(
                uPx = (cx + f * (xCv / zCv)).finiteOr(cx),
                vPx = (cy + f * (yCv / zCv)).finiteOr(cy),
                xCv = xCv,
                yCv = yCv,
                zCv = zCv,
                slantRangeM = rangeM,
                inFront = true
            )
        }

        private fun computeOffScreenChevron(
            proj: PinholeProjection,
            frameWidthPx: Int,
            frameHeightPx: Int
        ): OffScreenChevron {
            val safeW = frameWidthPx.coerceAtLeast(1).toFloat()
            val safeH = frameHeightPx.coerceAtLeast(1).toFloat()
            val cx = safeW * 0.5f
            val cy = safeH * 0.5f
            val rawDx = if (proj.inFront) proj.uPx - cx else proj.xCv
            val rawDy = if (proj.inFront) proj.vPx - cy else proj.yCv
            val dx = rawDx.finiteOr(0f)
            var dy = rawDy.finiteOr(1f)
            if (abs(dx) < 1e-4f && abs(dy) < 1e-4f) {
                dy = 1f
            }

            val dirRad = atan2(dy, dx).finiteOr(0f)
            val ux = cos(dirRad)
            val uy = sin(dirRad)
            val halfSpanX = safeW * 0.45f
            val halfSpanY = safeH * 0.43f
            val scaleX = if (abs(ux) > 1e-5f) halfSpanX / abs(ux) else Float.POSITIVE_INFINITY
            val scaleY = if (abs(uy) > 1e-5f) halfSpanY / abs(uy) else Float.POSITIVE_INFINITY
            val distToEdge = min(scaleX, scaleY).finiteOr(min(halfSpanX, halfSpanY))

            val edgeXNorm = ((cx + ux * distToEdge) / safeW).finiteOr(0.5f).coerceIn(0.05f, 0.95f)
            val edgeYNorm = ((cy + uy * distToEdge) / safeH).finiteOr(0.5f).coerceIn(0.07f, 0.93f)

            val cosAngle = (proj.zCv / proj.slantRangeM.coerceAtLeast(1e-4f))
                .finiteOr(0f)
                .coerceIn(-1.0f, 1.0f)
            val angularOffsetDeg = Math.toDegrees(acos(cosAngle.toDouble())).toFloat().finiteOr(45f)

            return OffScreenChevron(
                edgeXNorm = edgeXNorm,
                edgeYNorm = edgeYNorm,
                directionRad = dirRad,
                angularOffsetDeg = angularOffsetDeg
            )
        }

        /**
         * Clamps `[rawLeft, rawTop, rawRight, rawBottom]` strictly inside `[0..frameWidthPx] x [0..frameHeightPx]`
         * with guaranteed non-empty `[min..max]` ranges even when `rawLeft >= frameWidthPx` or `NaN`/`Infinity`
         * occurs at the right/bottom edges (`x >= 1920`, `y >= 1080`).
         */
        internal fun clampBoxToFrame(
            rawLeft: Float,
            rawTop: Float,
            rawRight: Float,
            rawBottom: Float,
            minSpanPx: Float = 8f,
            frameWidthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
            frameHeightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX
        ): FloatArray {
            val frameW = frameWidthPx.coerceAtLeast(1).toFloat()
            val frameH = frameHeightPx.coerceAtLeast(1).toFloat()
            val spanX = minSpanPx.finiteOr(4f).coerceIn(1f, frameW)
            val spanY = minSpanPx.finiteOr(4f).coerceIn(1f, frameH)
            val maxLeft = (frameW - spanX).coerceAtLeast(0f)
            val maxTop = (frameH - spanY).coerceAtLeast(0f)
            val safeLeft = rawLeft.finiteOr(0f).coerceIn(0f, maxLeft)
            val safeTop = rawTop.finiteOr(0f).coerceIn(0f, maxTop)
            val minRight = (safeLeft + spanX).coerceAtMost(frameW)
            val minBottom = (safeTop + spanY).coerceAtMost(frameH)
            val safeRight = rawRight.finiteOr(minRight).coerceIn(minRight, frameW)
            val safeBottom = rawBottom.finiteOr(minBottom).coerceIn(minBottom, frameH)
            return floatArrayOf(safeLeft, safeTop, safeRight, safeBottom)
        }

        private fun Float.finiteOr(fallback: Float): Float =
            if (isFinite()) coerceIn(-100_000f, 100_000f) else fallback

        private fun computeDistanceEnu(a: FloatArray, b: FloatArray): Float {
            val dx = a[0] - b[0]
            val dy = a[1] - b[1]
            val dz = a[2] - b[2]
            return sqrt(dx * dx + dy * dy + dz * dz)
        }

        /**
         * Solves a symmetric `3x3` linear system `A x = b` via analytical Cramer / cofactor inversion.
         */
        private fun solveSymmetric3x3(
            a00: Double,
            a01: Double,
            a02: Double,
            a11: Double,
            a12: Double,
            a22: Double,
            b0: Double,
            b1: Double,
            b2: Double
        ): FloatArray? {
            val c00 = a11 * a22 - a12 * a12
            val c01 = -(a01 * a22 - a12 * a02)
            val c02 = a01 * a12 - a11 * a02
            val c11 = a00 * a22 - a02 * a02
            val c12 = -(a00 * a12 - a01 * a02)
            val c22 = a00 * a11 - a01 * a01

            val det = a00 * c00 + a01 * c01 + a02 * c02
            if (!det.isFinite() || abs(det) < 1e-9) return null

            val invDet = 1.0 / det
            val x = (c00 * b0 + c01 * b1 + c02 * b2) * invDet
            val y = (c01 * b0 + c11 * b1 + c12 * b2) * invDet
            val z = (c02 * b0 + c12 * b1 + c22 * b2) * invDet
            if (!x.isFinite() || !y.isFinite() || !z.isFinite()) return null

            return floatArrayOf(x.toFloat(), y.toFloat(), z.toFloat())
        }
    }
}
