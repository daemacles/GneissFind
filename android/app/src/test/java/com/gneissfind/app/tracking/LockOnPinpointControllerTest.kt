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
