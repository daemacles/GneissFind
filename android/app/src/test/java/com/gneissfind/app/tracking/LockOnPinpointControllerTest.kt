/*
 * Unit tests for LockOnPinpointController and PushBroomCropper Lock-On helpers.
 *
 * Verifies clamped 320x320 crop origin geometry, LOCK NEAREST target selection and center-distance
 * tie-breaking, tap-to-lock reticle hit-testing, 3D camera visual odometry from gravel ground flow
 * (including vertical crouch recovery), multi-view ray-bundle triangulation and automatic Stage 2
 * -> Stage 3 handover (LOCK_ON_NEURAL -> PINPOINT_CONSTELLATION), off-screen directional chevrons,
 * the 15.0-second lost-track grace timeout, and instant 0 ms breakLock().
 */
package com.gneissfind.app.tracking

import com.gneissfind.app.camera.PushBroomCameraController
import com.gneissfind.app.model.AcquisitionStage
import com.gneissfind.app.model.ConfirmedTarget
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.TileDetection
import com.gneissfind.app.model.WorldPoint3D
import com.gneissfind.app.vision.PushBroomCropper
import kotlin.math.sqrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LockOnPinpointControllerTest {

    @Test
    fun `computeClampedSquareCropOrigin clamps cleanly at frame corners and preserves 320x320 span`() {
        // Top-left corner clamp
        val (left0, top0) = PushBroomCropper.computeClampedSquareCropOrigin(
            centerXPx = 25f,
            centerYPx = 12f,
            cropSizePx = 320,
            frameWidthPx = 1920,
            frameHeightPx = 1080
        )
        assertEquals(0, left0)
        assertEquals(0, top0)

        // Bottom-right corner clamp (1920 - 320 = 1600, 1080 - 320 = 760)
        val (leftMax, topMax) = PushBroomCropper.computeClampedSquareCropOrigin(
            centerXPx = 1905f,
            centerYPx = 1068f,
            cropSizePx = 320,
            frameWidthPx = 1920,
            frameHeightPx = 1080
        )
        assertEquals(1600, leftMax)
        assertEquals(760, topMax)

        // Exact center (960 - 160 = 800, 540 - 160 = 380)
        val (leftMid, topMid) = PushBroomCropper.computeClampedSquareCropOrigin(
            centerXPx = 960f,
            centerYPx = 540f,
            cropSizePx = 320,
            frameWidthPx = 1920,
            frameHeightPx = 1080
        )
        assertEquals(800, leftMid)
        assertEquals(380, topMid)
    }

    @Test
    fun `selectNearestVisibleTarget chooses smallest slant range and breaks ties by viewport center`() {
        val geom = StripRoiGeometry() // stripLeftPx = 64, frameCenter = 960 -> stripCenter = 896

        val farCenter = buildConfirmedTarget(
            trackId = 1L,
            stripCenterX = 896f,
            slantRangeMeters = 1.65f,
            isVisible = true
        )
        val nearOuter = buildConfirmedTarget(
            trackId = 2L,
            stripCenterX = 300f,
            slantRangeMeters = 1.20f,
            isVisible = true
        )
        val nearInner = buildConfirmedTarget(
            trackId = 3L,
            stripCenterX = 820f,
            slantRangeMeters = 1.2004f, // within 1e-3f tie-break window of nearOuter
            isVisible = true
        )
        val closerButInvisible = buildConfirmedTarget(
            trackId = 4L,
            stripCenterX = 896f,
            slantRangeMeters = 0.85f,
            isVisible = false
        )

        val picked = LockOnPinpointController.selectNearestVisibleTarget(
            targets = listOf(farCenter, nearOuter, nearInner, closerButInvisible),
            stripGeometry = geom
        )
        assertNotNull(picked)
        assertEquals(3L, picked!!.trackId)
    }

    @Test
    fun `findTappedTargetInStrip enforces generous touch slop around visible reticles`() {
        val t1 = buildConfirmedTarget(trackId = 10L, stripCenterX = 400f, stripCenterY = 192f, boxSizePx = 32f)
        val t2 = buildConfirmedTarget(trackId = 11L, stripCenterX = 950f, stripCenterY = 200f, boxSizePx = 36f)

        // Tap 42 px offset from small 32 px target center -> still hits within 56 px touch slop
        val hit = LockOnPinpointController.findTappedTargetInStrip(
            targets = listOf(t1, t2),
            tapStripXPx = 442f,
            tapStripYPx = 198f,
            touchSlopPx = 56f
        )
        assertNotNull(hit)
        assertEquals(10L, hit!!.trackId)

        // Tap far away in empty gravel -> returns null
        val miss = LockOnPinpointController.findTappedTargetInStrip(
            targets = listOf(t1, t2),
            tapStripXPx = 1500f,
            tapStripYPx = 80f,
            touchSlopPx = 56f
        )
        assertNull(miss)
    }

    @Test
    fun `solveCameraPositionFromGravelFlow recovers 3D lateral step and vertical crouch`() {
        val focalPx = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX
        val rotPrev = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 45f, rollDeg = 0f, yawDeg = 0f)
        val rotCurr = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 48f, rollDeg = 1.5f, yawDeg = 2.0f)

        val prevCamEnu = floatArrayOf(0f, 0f, 1.06f)
        val trueCurrCamEnu = floatArrayOf(0.08f, 0.11f, 0.98f) // 8 cm East, 11 cm North, 8 cm crouch down

        // Generate 24 static ground points (z_enu = 0) visible in both frames
        val prevPts = FloatArray(48)
        val currPts = FloatArray(48)
        var idx = 0
        for (gx in listOf(-0.35f, -0.18f, -0.05f, 0.08f, 0.22f, 0.38f)) {
            for (gy in listOf(0.80f, 0.98f, 1.18f, 1.40f)) {
                val gEnu = floatArrayOf(gx, gy, 0f)
                val (u0, v0) = projectGroundPoint(gEnu, prevCamEnu, rotPrev, focalPx)
                val (u1, v1) = projectGroundPoint(gEnu, trueCurrCamEnu, rotCurr, focalPx)
                prevPts[idx * 2] = u0
                prevPts[idx * 2 + 1] = v0
                currPts[idx * 2] = u1
                currPts[idx * 2 + 1] = v1
                idx++
            }
        }

        val solved = LockOnPinpointController.solveCameraPositionFromGravelFlow(
            prevCamPosEnu = prevCamEnu,
            prevRotationScr = rotPrev,
            currRotationScr = rotCurr,
            prevPointsXY = prevPts,
            currPointsXY = currPts,
            count = 24,
            focalLengthPx = focalPx
        )
        assertNotNull(solved)
        assertEquals(trueCurrCamEnu[0], solved!![0], 0.015f)
        assertEquals(trueCurrCamEnu[1], solved[1], 0.015f)
        assertEquals(trueCurrCamEnu[2], solved[2], 0.015f)
    }

    @Test
    fun `multi-view triangulation converges sigma3dMm and triggers Stage 3 PINPOINT handover`() {
        val controller = LockOnPinpointController()
        val geom = StripRoiGeometry()
        val focalPx = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX
        val rot45 = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 45f, rollDeg = 0f, yawDeg = 0f)

        // True physical 46 mm stone at (East = 0.05 m, North = 1.06 m, Up = 0.00 m)
        val trueStoneEnu = floatArrayOf(0.05f, 1.06f, 0.0f)
        val initCamEnu = floatArrayOf(0f, 0f, 1.06f)
        val (initU, initV) = projectGroundPoint(trueStoneEnu, initCamEnu, rot45, focalPx)
        val initRay = GroundPlaneRaycaster.pixelToWorldRay(initU, initV, focalPx, rot45)
        val initRange = distance3d(initCamEnu, trueStoneEnu)

        val stripCenterX = initU - geom.stripLeftPx
        val stripCenterY = initV - StripRoiGeometry.NOMINAL_CROP_Y_PX
        val target = ConfirmedTarget(
            trackId = 77L,
            stoneClass = StoneClass.CHALCEDONY_AGATE,
            worldPosition = WorldPoint3D(trueStoneEnu[0], 0f, trueStoneEnu[1]),
            stripLeftPx = stripCenterX - 38f,
            stripTopPx = stripCenterY - 38f,
            stripRightPx = stripCenterX + 38f,
            stripBottomPx = stripCenterY + 38f,
            smoothedConfidence = 0.88f,
            estimatedDiameterMm = 46f,
            slantRangeMeters = initRange,
            hitCountInWindow = 3,
            windowSize = 4,
            lastUpdatedTimestampNs = 1_000_000_000L,
            worldRay = initRay,
            angularWidthRad = 76f / focalPx,
            angularHeightRad = 76f / focalPx,
            isVisibleInStrip = true
        )

        val initialHud = controller.lockOntoTarget(
            target = target,
            stripGeometry = geom,
            currentCropYFloat = StripRoiGeometry.NOMINAL_CROP_Y_PX.toFloat(),
            currentFocalLengthPx = focalPx,
            currentRotationMatrix = rot45,
            timestampNs = 1_000_000_000L
        )
        assertEquals(AcquisitionStage.LOCK_ON_NEURAL, initialHud.stage)
        assertTrue("Initial single-view sigma should be broad", initialHud.sigma3dMm >= 85f)

        // Simulate user stepping sideways & crouching down across 5 calibrated viewpoints (> 10 deg parallax)
        val cameraTrajectory = listOf(
            floatArrayOf(0.08f, 0.08f, 0.96f),
            floatArrayOf(0.16f, 0.18f, 0.85f),
            floatArrayOf(0.24f, 0.28f, 0.74f),
            floatArrayOf(0.30f, 0.36f, 0.65f)
        )

        var latestHud = initialHud
        cameraTrajectory.forEachIndexed { stepIdx, camPos ->
            val tsNs = 1_200_000_000L + stepIdx * 100_000_000L
            val (uTrue, vTrue) = projectGroundPoint(trueStoneEnu, camPos, rot45, focalPx)
            val (cropLeft, cropTop) = PushBroomCropper.computeClampedSquareCropOrigin(uTrue, vTrue)
            val detCx = uTrue - cropLeft
            val detCy = vTrue - cropTop
            val rangeNow = distance3d(camPos, trueStoneEnu)
            val gsdNow = PushBroomCropper.computeGsdMmPerPx(rangeNow, PushBroomCropper.DEFAULT_ZOOM_RATIO)
            val halfSpanPx = (46f / gsdNow) * 0.5f

            val pose = MotionPoseSnapshot(
                cameraWorldPosition = WorldPoint3D(camPos[0], camPos[2], camPos[1]),
                forwardPitchDeg = 45f,
                rawPitchDeg = 45f,
                rollDeg = 0f,
                referencePitchDeg = 45f,
                cropY = StripRoiGeometry.NOMINAL_CROP_Y_PX,
                isPitchClamped = false,
                centerSlantRangeMeters = rangeNow,
                horizontalSpeedMps = 0.25f,
                arPlaneLocked = false,
                timestampNs = tsNs,
                verticalFocalLengthPx = focalPx,
                rotationMatrix = rot45
            )

            latestHud = controller.onNeuralB1Detections(
                cropDetections = listOf(
                    TileDetection(
                        tileIndex = 0,
                        classId = StoneClass.CHALCEDONY_AGATE.id,
                        confidence = 0.94f,
                        leftPx = detCx - halfSpanPx,
                        topPx = detCy - halfSpanPx,
                        rightPx = detCx + halfSpanPx,
                        bottomPx = detCy + halfSpanPx
                    )
                ),
                cropLeftPx = cropLeft,
                cropTopPx = cropTop,
                poseSnapshot = pose,
                timestampNs = tsNs,
                syntheticCameraStepEnu = camPos
            )
        }

        assertTrue("Parallax should exceed 3.0 deg (actual=${latestHud.parallaxAngleDeg})", latestHud.parallaxAngleDeg >= 3.0f)
        assertTrue(
            "Sigma3D should converge below 0.5 * D (sigma=${latestHud.sigma3dMm}mm, D=${latestHud.estimatedDiameterMm}mm)",
            latestHud.sigma3dMm < 0.5f * latestHud.estimatedDiameterMm
        )
        assertEquals(AcquisitionStage.PINPOINT_CONSTELLATION, latestHud.stage)
        controller.close()
    }

    @Test
    fun `off-screen rotation emits directional chevron, 15s grace expires to PATROL, and breakLock exits immediately`() {
        val controller = LockOnPinpointController()
        val geom = StripRoiGeometry()
        val focalPx = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX
        val rotForward = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 45f, rollDeg = 0f, yawDeg = 0f)

        val target = buildConfirmedTarget(trackId = 99L, stripCenterX = 896f, slantRangeMeters = 1.45f)
        val t0Ns = 5_000_000_000L
        val locked = controller.lockOntoTarget(
            target = target,
            stripGeometry = geom,
            currentCropYFloat = StripRoiGeometry.NOMINAL_CROP_Y_PX.toFloat(),
            currentFocalLengthPx = focalPx,
            currentRotationMatrix = rotForward,
            timestampNs = t0Ns
        )
        assertTrue(locked.isTargetInFrame)
        assertNull(locked.offScreenChevron)

        // Pan camera 55 degrees clockwise to the right (yawDeg = +55f) -> North stone moves off the left edge
        val rotPannedRight = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 45f, rollDeg = 0f, yawDeg = 55f)
        val t4sNs = t0Ns + 4_000_000_000L
        val pannedHud = controller.reprojectForDisplay120Hz(
            currentRotationMatrix = rotPannedRight,
            currentFocalLengthPx = focalPx,
            nowTimestampNs = t4sNs
        )
        assertEquals(AcquisitionStage.LOCK_ON_NEURAL, pannedHud.stage)
        assertFalse(pannedHud.isTargetInFrame)
        assertTrue(pannedHud.isLostTrackGraceActive)
        val chevron = pannedHud.offScreenChevron
        assertNotNull(chevron)
        assertTrue(
            "Chevron should clamp near left viewport edge when camera pans right (edgeX=${chevron!!.edgeXNorm})",
            chevron.edgeXNorm <= 0.12f
        )

        // Advance 10 seconds into grace period -> still holding lock with ~5s remaining
        val midGraceHud = controller.reprojectForDisplay120Hz(
            currentRotationMatrix = rotPannedRight,
            currentFocalLengthPx = focalPx,
            nowTimestampNs = t4sNs + 10_000_000_000L
        )
        assertEquals(AcquisitionStage.LOCK_ON_NEURAL, midGraceHud.stage)
        assertEquals(5.0f, midGraceHud.lostTrackRemainingSec, 0.2f)

        // Advance past 15.0 seconds -> automatically exits back to PATROL
        val expiredHud = controller.reprojectForDisplay120Hz(
            currentRotationMatrix = rotPannedRight,
            currentFocalLengthPx = focalPx,
            nowTimestampNs = t4sNs + 15_100_000_000L
        )
        assertEquals(AcquisitionStage.PATROL, expiredHud.stage)

        // Re-lock and verify instant 0 ms breakLock()
        controller.lockOntoTarget(
            target = target,
            stripGeometry = geom,
            currentCropYFloat = StripRoiGeometry.NOMINAL_CROP_Y_PX.toFloat(),
            currentFocalLengthPx = focalPx,
            currentRotationMatrix = rotForward,
            timestampNs = t0Ns + 20_000_000_000L
        )
        assertEquals(AcquisitionStage.LOCK_ON_NEURAL, controller.currentStage)
        val broken = controller.breakLock()
        assertEquals(AcquisitionStage.PATROL, broken.stage)
        assertEquals(AcquisitionStage.PATROL, controller.currentStage)
        controller.close()
    }

    @Test
    fun `pure camera pitch rotation without translation preserves slant range and does not hallucinate baseline`() {
        val controller = LockOnPinpointController()
        val geom = StripRoiGeometry()
        val focalPx = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX
        val stationaryCam = floatArrayOf(0f, 0f, 1.06f)
        val trueStoneEnu = floatArrayOf(0.02f, 1.48f, 0.0f)
        val trueRange = distance3d(stationaryCam, trueStoneEnu)

        val rot36 = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 36f, rollDeg = 0f, yawDeg = 0f)
        val (u0, v0) = projectGroundPoint(trueStoneEnu, stationaryCam, rot36, focalPx)
        val initRay = GroundPlaneRaycaster.pixelToWorldRay(u0, v0, focalPx, rot36)

        val target = ConfirmedTarget(
            trackId = 81L,
            stoneClass = StoneClass.PETRIFIED_WOOD,
            worldPosition = WorldPoint3D(trueStoneEnu[0], 0f, trueStoneEnu[1]),
            stripLeftPx = u0 - geom.stripLeftPx - 36f,
            stripTopPx = v0 - StripRoiGeometry.NOMINAL_CROP_Y_PX - 36f,
            stripRightPx = u0 - geom.stripLeftPx + 36f,
            stripBottomPx = v0 - StripRoiGeometry.NOMINAL_CROP_Y_PX + 36f,
            smoothedConfidence = 0.78f,
            estimatedDiameterMm = 68f,
            slantRangeMeters = trueRange,
            hitCountInWindow = 3,
            windowSize = 4,
            lastUpdatedTimestampNs = 2_000_000_000L,
            worldRay = initRay,
            isVisibleInStrip = true
        )

        var hud = controller.lockOntoTarget(
            target = target,
            stripGeometry = geom,
            currentCropYFloat = StripRoiGeometry.NOMINAL_CROP_Y_PX.toFloat(),
            currentFocalLengthPx = focalPx,
            currentRotationMatrix = rot36,
            timestampNs = 2_000_000_000L
        )
        assertEquals(trueRange, hud.slantRangeMeters, 0.05f)

        // Rotate pitch purely in place across 30 -> 42 -> 34 deg with slight +/-3px bbox noise
        val pitches = listOf(32f, 38f, 42f, 35f)
        pitches.forEachIndexed { idx, pitch ->
            val rot = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = pitch, rollDeg = 0f, yawDeg = 0f)
            val (uTrue, vTrue) = projectGroundPoint(trueStoneEnu, stationaryCam, rot, focalPx)
            val jitter = if (idx % 2 == 0) 3.0f else -3.0f
            val (cropLeft, cropTop) = PushBroomCropper.computeClampedSquareCropOrigin(uTrue, vTrue)
            val cx = (uTrue + jitter) - cropLeft
            val cy = (vTrue - jitter) - cropTop
            val ts = 2_100_000_000L + idx * 80_000_000L

            val pose = MotionPoseSnapshot(
                cameraWorldPosition = WorldPoint3D(0f, stationaryCam[2], 0f),
                forwardPitchDeg = pitch,
                rawPitchDeg = pitch,
                rollDeg = 0f,
                referencePitchDeg = 36f,
                cropY = StripRoiGeometry.NOMINAL_CROP_Y_PX,
                isPitchClamped = false,
                centerSlantRangeMeters = trueRange,
                horizontalSpeedMps = 0f,
                arPlaneLocked = false,
                timestampNs = ts,
                verticalFocalLengthPx = focalPx,
                rotationMatrix = rot
            )
            hud = controller.onNeuralB1Detections(
                cropDetections = listOf(
                    TileDetection(
                        tileIndex = 0,
                        classId = StoneClass.PETRIFIED_WOOD.id,
                        confidence = 0.80f,
                        leftPx = cx - 34f,
                        topPx = cy - 34f,
                        rightPx = cx + 34f,
                        bottomPx = cy + 34f
                    )
                ),
                cropLeftPx = cropLeft,
                cropTopPx = cropTop,
                poseSnapshot = pose,
                timestampNs = ts
            )
        }

        assertEquals("Zero translation must report 0 deg stereo parallax", 0f, hud.parallaxAngleDeg, 0.01f)
        assertEquals("Pure rotation must not inflate slant range", trueRange, hud.slantRangeMeters, 0.06f)
        assertEquals(1, hud.triangulatedViews)
        controller.close()
    }

    @Test
    fun `pinpoint off-screen pan holds lock across analyzer frames for 15s and delayed B1 stays world-locked`() {
        val controller = LockOnPinpointController()
        val geom = StripRoiGeometry()
        val focalPx = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX
        val rot45 = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 45f, rollDeg = 0f, yawDeg = 0f)
        val trueStoneEnu = floatArrayOf(0.22f, 1.06f, 0.0f)
        val initCamEnu = floatArrayOf(0f, 0f, 1.06f)
        val (initU, initV) = projectGroundPoint(trueStoneEnu, initCamEnu, rot45, focalPx)
        val initRay = GroundPlaneRaycaster.pixelToWorldRay(initU, initV, focalPx, rot45)
        val initRange = distance3d(initCamEnu, trueStoneEnu)

        val target = ConfirmedTarget(
            trackId = 88L,
            stoneClass = StoneClass.PETRIFIED_WOOD,
            worldPosition = WorldPoint3D(trueStoneEnu[0], 0f, trueStoneEnu[1]),
            stripLeftPx = initU - geom.stripLeftPx - 38f,
            stripTopPx = initV - StripRoiGeometry.NOMINAL_CROP_Y_PX - 38f,
            stripRightPx = initU - geom.stripLeftPx + 38f,
            stripBottomPx = initV - StripRoiGeometry.NOMINAL_CROP_Y_PX + 38f,
            smoothedConfidence = 0.86f,
            estimatedDiameterMm = 48f,
            slantRangeMeters = initRange,
            hitCountInWindow = 4,
            windowSize = 4,
            lastUpdatedTimestampNs = 3_000_000_000L,
            worldRay = initRay,
            isVisibleInStrip = true
        )

        var tNs = 3_000_000_000L
        controller.lockOntoTarget(
            target = target,
            stripGeometry = geom,
            currentCropYFloat = StripRoiGeometry.NOMINAL_CROP_Y_PX.toFloat(),
            currentFocalLengthPx = focalPx,
            currentRotationMatrix = rot45,
            timestampNs = tNs
        )

        // Converge into PINPOINT_CONSTELLATION via 4 multi-view steps keeping stone well inside 1920x1080
        val steps = listOf(
            floatArrayOf(0.06f, 0.08f, 0.96f),
            floatArrayOf(0.12f, 0.16f, 0.86f),
            floatArrayOf(0.18f, 0.24f, 0.76f),
            floatArrayOf(0.24f, 0.30f, 0.68f)
        )
        var lastCam = initCamEnu
        for (cam in steps) {
            tNs += 100_000_000L
            lastCam = cam
            val (u, v) = projectGroundPoint(trueStoneEnu, cam, rot45, focalPx)
            val (cL, cT) = PushBroomCropper.computeClampedSquareCropOrigin(u, v)
            val pose = MotionPoseSnapshot(
                cameraWorldPosition = WorldPoint3D(cam[0], cam[2], cam[1]),
                forwardPitchDeg = 45f,
                rawPitchDeg = 45f,
                rollDeg = 0f,
                referencePitchDeg = 45f,
                cropY = StripRoiGeometry.NOMINAL_CROP_Y_PX,
                isPitchClamped = false,
                centerSlantRangeMeters = distance3d(cam, trueStoneEnu),
                horizontalSpeedMps = 0.2f,
                arPlaneLocked = false,
                timestampNs = tNs,
                verticalFocalLengthPx = focalPx,
                rotationMatrix = rot45
            )
            controller.onNeuralB1Detections(
                cropDetections = listOf(
                    TileDetection(
                        tileIndex = 0,
                        classId = StoneClass.PETRIFIED_WOOD.id,
                        confidence = 0.92f,
                        leftPx = (u - cL) - 36f,
                        topPx = (v - cT) - 36f,
                        rightPx = (u - cL) + 36f,
                        bottomPx = (v - cT) + 36f
                    )
                ),
                cropLeftPx = cL,
                cropTopPx = cT,
                poseSnapshot = pose,
                timestampNs = tNs,
                syntheticCameraStepEnu = cam
            )
        }
        assertEquals(AcquisitionStage.PINPOINT_CONSTELLATION, controller.currentStage)

        // Verify 145ms-delayed neural B=1 detection captured at rot45 while live display is tilted to rot43
        // (-2 deg = ~95 px shift at 2.0x zoom) projects both reticle and neuralBox to the live rot43 pose!
        val rot43 = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 43f, rollDeg = 0f, yawDeg = 0f)
        val (expectedLiveU, expectedLiveV) = projectGroundPoint(trueStoneEnu, lastCam, rot43, focalPx)
        val (captureU, captureV) = projectGroundPoint(trueStoneEnu, lastCam, rot45, focalPx)
        val (cL45, cT45) = PushBroomCropper.computeClampedSquareCropOrigin(captureU, captureV)
        val staleCapturePose = MotionPoseSnapshot(
            cameraWorldPosition = WorldPoint3D(lastCam[0], lastCam[2], lastCam[1]),
            forwardPitchDeg = 45f,
            rawPitchDeg = 45f,
            rollDeg = 0f,
            referencePitchDeg = 45f,
            cropY = StripRoiGeometry.NOMINAL_CROP_Y_PX,
            isPitchClamped = false,
            centerSlantRangeMeters = distance3d(lastCam, trueStoneEnu),
            horizontalSpeedMps = 0f,
            arPlaneLocked = false,
            timestampNs = tNs,
            verticalFocalLengthPx = focalPx,
            rotationMatrix = rot45
        )
        tNs += 145_000_000L
        val delayedHud = controller.onNeuralB1Detections(
            cropDetections = listOf(
                TileDetection(
                    tileIndex = 0,
                    classId = StoneClass.PETRIFIED_WOOD.id,
                    confidence = 0.91f,
                    leftPx = (captureU - cL45) - 34f,
                    topPx = (captureV - cT45) - 34f,
                    rightPx = (captureU - cL45) + 34f,
                    bottomPx = (captureV - cT45) + 34f
                )
            ),
            cropLeftPx = cL45,
            cropTopPx = cT45,
            poseSnapshot = staleCapturePose,
            timestampNs = tNs,
            captureCamPosEnu = lastCam,
            displayRotationMatrix = rot43
        )
        assertTrue(delayedHud.hasNeuralDetectorBox)
        val liveReticleCenterY = (delayedHud.frameTopPx + delayedHud.frameBottomPx) * 0.5f
        val liveNeuralCenterY = (delayedHud.neuralBoxTopPx + delayedHud.neuralBoxBottomPx) * 0.5f
        assertEquals(expectedLiveU, (delayedHud.frameLeftPx + delayedHud.frameRightPx) * 0.5f, 8.0f)
        assertEquals(expectedLiveV, liveReticleCenterY, 8.0f)
        assertEquals(expectedLiveV, liveNeuralCenterY, 8.0f)

        // Now pan camera 60 degrees away in PINPOINT_CONSTELLATION and verify both 120Hz display and
        // 30Hz onCameraLumaFrame hold lock for the full 15s grace window with offScreenChevron!
        val rotPannedAway = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 45f, rollDeg = 0f, yawDeg = 60f)
        val dummyLuma = ByteArray(1920 * 1080)
        val panStartNs = tNs + 50_000_000L
        val displayOff = controller.reprojectForDisplay120Hz(
            currentRotationMatrix = rotPannedAway,
            currentFocalLengthPx = focalPx,
            nowTimestampNs = panStartNs
        )
        assertEquals(AcquisitionStage.PINPOINT_CONSTELLATION, displayOff.stage)
        assertFalse(displayOff.isTargetInFrame)
        assertNotNull(displayOff.offScreenChevron)

        val pannedPose = staleCapturePose.copy(rotationMatrix = rotPannedAway, timestampNs = panStartNs + 8_000_000L)
        val analyzerDirective = controller.onCameraLumaFrame(
            lumaBytes = dummyLuma,
            widthPx = 1920,
            heightPx = 1080,
            poseSnapshot = pannedPose,
            timestampNs = panStartNs + 8_000_000L
        )
        assertFalse("Must NOT prematurely break lock on off-screen frame", analyzerDirective.autoExitedToPatrol)
        assertEquals(AcquisitionStage.PINPOINT_CONSTELLATION, analyzerDirective.hudState.stage)
        assertNotNull(analyzerDirective.hudState.offScreenChevron)
        assertEquals(15.0f, analyzerDirective.hudState.lostTrackRemainingSec, 0.25f)

        // Pan back to target at +6.0s -> immediately re-enters frame and clears grace timer
        val backInView = controller.reprojectForDisplay120Hz(
            currentRotationMatrix = rot45,
            currentFocalLengthPx = focalPx,
            nowTimestampNs = panStartNs + 6_000_000_000L
        )
        assertTrue(backInView.isTargetInFrame)
        assertFalse("Grace countdown must clear immediately on viewport re-entry", backInView.isLostTrackGraceActive)
        assertEquals(15.0f, backInView.lostTrackRemainingSec, 0.01f)
        assertNull(backInView.offScreenChevron)
        controller.close()
    }

    @Test
    fun `off-screen countdown resets immediately when target reappears and starts fresh on subsequent pan`() {
        val controller = LockOnPinpointController()
        val geom = StripRoiGeometry()
        val focalPx = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX
        val rotForward = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 45f, rollDeg = 0f, yawDeg = 0f)
        val rotPannedAway = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 45f, rollDeg = 0f, yawDeg = 58f)
        val dummyLuma = ByteArray(1920 * 1080)

        val target = buildConfirmedTarget(trackId = 104L, stripCenterX = 896f, slantRangeMeters = 1.35f)
        val t0Ns = 10_000_000_000L
        controller.lockOntoTarget(
            target = target,
            stripGeometry = geom,
            currentCropYFloat = StripRoiGeometry.NOMINAL_CROP_Y_PX.toFloat(),
            currentFocalLengthPx = focalPx,
            currentRotationMatrix = rotForward,
            timestampNs = t0Ns
        )

        // 1. Pan away for 8.5 seconds -> countdown drops to ~6.5 seconds
        val tPan1StartNs = t0Ns + 100_000_000L
        controller.reprojectForDisplay120Hz(rotPannedAway, focalPx, tPan1StartNs)
        val midOffScreen = controller.reprojectForDisplay120Hz(
            currentRotationMatrix = rotPannedAway,
            currentFocalLengthPx = focalPx,
            nowTimestampNs = tPan1StartNs + 8_500_000_000L
        )
        assertFalse(midOffScreen.isTargetInFrame)
        assertTrue(midOffScreen.isLostTrackGraceActive)
        assertEquals(6.5f, midOffScreen.lostTrackRemainingSec, 0.2f)

        // 2. Pan back so target reappears in-frame (even before neural B=1 finishes!) -> timer MUST reset to 15.0s
        val tReenterNs = tPan1StartNs + 9_000_000_000L
        val reenteredDisplay = controller.reprojectForDisplay120Hz(
            currentRotationMatrix = rotForward,
            currentFocalLengthPx = focalPx,
            nowTimestampNs = tReenterNs
        )
        assertTrue(reenteredDisplay.isTargetInFrame)
        assertFalse(reenteredDisplay.isLostTrackGraceActive)
        assertEquals(15.0f, reenteredDisplay.lostTrackRemainingSec, 0.01f)

        // Analyzer frame arriving right after re-entry must also keep grace reset AND request wide 640px search if needed
        val forwardPose = MotionPoseSnapshot(
            cameraWorldPosition = WorldPoint3D(0f, 1.06f, 0f),
            forwardPitchDeg = 45f,
            rawPitchDeg = 45f,
            rollDeg = 0f,
            referencePitchDeg = 45f,
            cropY = StripRoiGeometry.NOMINAL_CROP_Y_PX,
            isPitchClamped = false,
            centerSlantRangeMeters = 1.35f,
            horizontalSpeedMps = 0f,
            arPlaneLocked = false,
            timestampNs = tReenterNs + 16_000_000L,
            verticalFocalLengthPx = focalPx,
            rotationMatrix = rotForward
        )
        val reenteredAnalyzer = controller.onCameraLumaFrame(
            lumaBytes = dummyLuma,
            widthPx = 1920,
            heightPx = 1080,
            poseSnapshot = forwardPose,
            timestampNs = tReenterNs + 16_000_000L
        )
        assertFalse(reenteredAnalyzer.autoExitedToPatrol)
        assertFalse(reenteredAnalyzer.hudState.isLostTrackGraceActive)
        assertEquals(15.0f, reenteredAnalyzer.hudState.lostTrackRemainingSec, 0.01f)

        // 3. Pan away a second time for 8.0 seconds (total 17.0s since first pan!) -> MUST start fresh from 15.0s (~7.0s left, NOT expire!)
        val tPan2StartNs = tReenterNs + 500_000_000L
        controller.reprojectForDisplay120Hz(rotPannedAway, focalPx, tPan2StartNs)
        val secondPanAt8s = controller.reprojectForDisplay120Hz(
            currentRotationMatrix = rotPannedAway,
            currentFocalLengthPx = focalPx,
            nowTimestampNs = tPan2StartNs + 8_000_000_000L
        )
        assertEquals(AcquisitionStage.LOCK_ON_NEURAL, secondPanAt8s.stage)
        assertTrue(secondPanAt8s.isLostTrackGraceActive)
        assertEquals(7.0f, secondPanAt8s.lostTrackRemainingSec, 0.2f)
        controller.close()
    }

    @Test
    fun `wide 640x640 re-acquisition crop recovers 220px offset target and ignores border-clipped diameter shrink`() {
        val controller = LockOnPinpointController()
        val geom = StripRoiGeometry()
        val focalPx = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX
        val rot45 = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 45f, rollDeg = 0f, yawDeg = 0f)

        val target = buildConfirmedTarget(
            trackId = 105L,
            stripCenterX = 896f,
            slantRangeMeters = 1.15f,
            boxSizePx = 180f
        ).copy(estimatedDiameterMm = 165f)
        val t0Ns = 20_000_000_000L
        controller.lockOntoTarget(
            target = target,
            stripGeometry = geom,
            currentCropYFloat = StripRoiGeometry.NOMINAL_CROP_Y_PX.toFloat(),
            currentFocalLengthPx = focalPx,
            currentRotationMatrix = rot45,
            timestampNs = t0Ns
        )

        val pose = MotionPoseSnapshot(
            cameraWorldPosition = WorldPoint3D(0f, 1.06f, 0f),
            forwardPitchDeg = 45f,
            rawPitchDeg = 45f,
            rollDeg = 0f,
            referencePitchDeg = 45f,
            cropY = StripRoiGeometry.NOMINAL_CROP_Y_PX,
            isPitchClamped = false,
            centerSlantRangeMeters = 1.15f,
            horizontalSpeedMps = 0f,
            arPlaneLocked = false,
            timestampNs = t0Ns + 60_000_000L,
            verticalFocalLengthPx = focalPx,
            rotationMatrix = rot45
        )

        // One missed pass triggers wide 640x640 search window
        controller.onNeuralB1Detections(
            cropDetections = emptyList(),
            cropLeftPx = 800,
            cropTopPx = 380,
            poseSnapshot = pose,
            timestampNs = t0Ns + 60_000_000L
        )
        val (predCx, predCy, predSize) = controller.predictCurrentCropWindow(rot45, focalPx)
        assertEquals(PushBroomCropper.LOCK_ON_WIDE_SEARCH_CROP_SIZE_PX, predSize)

        // Target is 220 px to the right of predicted center (would miss 320x320 crop completely, fits inside 640x640!)
        val (wideLeft, wideTop) = PushBroomCropper.computeClampedSquareCropOrigin(
            centerXPx = predCx,
            centerYPx = predCy,
            cropSizePx = 640
        )
        val trueCenterX = predCx + 220f
        val trueCenterY = predCy - 40f
        // Map 1920x1080 box into [0..320] tensor coordinates (cropScale = 2.0)
        val tensorCx = (trueCenterX - wideLeft) / 2.0f
        val tensorCy = (trueCenterY - wideTop) / 2.0f
        val reacquiredHud = controller.onNeuralB1Detections(
            cropDetections = listOf(
                TileDetection(
                    tileIndex = 0,
                    classId = StoneClass.CHALCEDONY_AGATE.id,
                    confidence = 0.89f,
                    leftPx = tensorCx - 45f,
                    topPx = tensorCy - 45f,
                    rightPx = tensorCx + 45f,
                    bottomPx = tensorCy + 45f
                )
            ),
            cropLeftPx = wideLeft,
            cropTopPx = wideTop,
            poseSnapshot = pose,
            timestampNs = t0Ns + 130_000_000L,
            cropSizePx = 640
        )
        assertTrue(reacquiredHud.hasNeuralDetectorBox)
        val neuralCenterX = (reacquiredHud.neuralBoxLeftPx + reacquiredHud.neuralBoxRightPx) * 0.5f
        assertEquals("Wide 640px crop must map [0..320] box back to exact 1920x1080 X", trueCenterX, neuralCenterX, 4.0f)

        // Border-clipped detection (rightPx = 319.5 >= 312) must NOT shrink estimatedDiameterMm
        val sizeBeforeClip = reacquiredHud.estimatedDiameterMm
        val clippedHud = controller.onNeuralB1Detections(
            cropDetections = listOf(
                TileDetection(
                    tileIndex = 0,
                    classId = StoneClass.CHALCEDONY_AGATE.id,
                    confidence = 0.84f,
                    leftPx = 295f,
                    topPx = 140f,
                    rightPx = 319.5f,
                    bottomPx = 165f
                )
            ),
            cropLeftPx = wideLeft,
            cropTopPx = wideTop,
            poseSnapshot = pose,
            timestampNs = t0Ns + 210_000_000L,
            cropSizePx = 640
        )
        assertEquals(
            "Clipped edge detection must not shrink estimatedDiameterMm",
            sizeBeforeClip,
            clippedHud.estimatedDiameterMm,
            0.01f
        )
        controller.close()
    }

    private fun buildConfirmedTarget(
        trackId: Long,
        stripCenterX: Float,
        stripCenterY: Float = 192f,
        slantRangeMeters: Float = 1.40f,
        boxSizePx: Float = 40f,
        isVisible: Boolean = true
    ): ConfirmedTarget {
        val half = boxSizePx * 0.5f
        return ConfirmedTarget(
            trackId = trackId,
            stoneClass = StoneClass.CHALCEDONY_AGATE,
            worldPosition = WorldPoint3D(0f, 0f, 1.0f),
            stripLeftPx = stripCenterX - half,
            stripTopPx = stripCenterY - half,
            stripRightPx = stripCenterX + half,
            stripBottomPx = stripCenterY + half,
            smoothedConfidence = 0.85f,
            estimatedDiameterMm = 32f,
            slantRangeMeters = slantRangeMeters,
            hitCountInWindow = 3,
            windowSize = 4,
            lastUpdatedTimestampNs = 1_000_000_000L,
            isVisibleInStrip = isVisible
        )
    }

    private fun projectGroundPoint(
        pointEnu: FloatArray,
        camPosEnu: FloatArray,
        rWorldFromScr: FloatArray,
        focalLengthPx: Float
    ): Pair<Float, Float> {
        val dEast = pointEnu[0] - camPosEnu[0]
        val dNorth = pointEnu[1] - camPosEnu[1]
        val dUp = pointEnu[2] - camPosEnu[2]

        val xScr = rWorldFromScr[0] * dEast + rWorldFromScr[3] * dNorth + rWorldFromScr[6] * dUp
        val yScr = rWorldFromScr[1] * dEast + rWorldFromScr[4] * dNorth + rWorldFromScr[7] * dUp
        val zScr = rWorldFromScr[2] * dEast + rWorldFromScr[5] * dNorth + rWorldFromScr[8] * dUp

        val xCv = xScr
        val yCv = -yScr
        val zCv = (-zScr).coerceAtLeast(1e-4f)
        val u = 960f + focalLengthPx * (xCv / zCv)
        val v = 540f + focalLengthPx * (yCv / zCv)
        return Pair(u, v)
    }

    private fun distance3d(a: FloatArray, b: FloatArray): Float {
        val dx = a[0] - b[0]
        val dy = a[1] - b[1]
        val dz = a[2] - b[2]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
}
