/*
 * Unit test suite for [TemporalTargetTracker] and [GroundPlaneRaycaster] 3D world-ray AR tracking.
 *
 * Covers the spatial-temporal tracking and ego-motion reprojection invariants
 * (`.agents/rules/android-camera-ar.md`, `docs/ENGINEERING_LEDGER.md`):
 * - `3-of-6` sliding-window spatial voting within a `40 mm` ground-plane radius.
 * - Rapid 2-sweep consecutive high-confidence (`>= 0.72`) acquisition and confirmation hysteresis
 *   (retaining confirmed lock down to `>= 2` hits in the last 6 sweeps).
 * - Rotation-invariant 3D IMU world-ray (`<= 3.8 deg` cone) track association across handheld
 *   camera pitch, roll, and yaw rotation.
 * - 120 Hz ego-motion reticle reprojection compensating for GPU inference latency and suppressing
 *   off-strip ghost reticles (`isVisibleInStrip = false`).
 * - Walking speed gate alerting above `1.10 m/s` ([ScanSpeedState.SpeedWarning]).
 *
 * Main types:
 * - [TemporalTargetTrackerTest]: JUnit4 test suite.
 */
package com.gneissfind.app.tracking

import com.gneissfind.app.model.ScanSpeedState
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripDetection
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.SwathTileMode
import com.gneissfind.app.model.WorldPoint3D
import com.gneissfind.app.model.WorldRay3D
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

/**
 * Verifies [TemporalTargetTracker] 3-of-6 spatial voting, 2-sweep high-confidence acquisition,
 * 3D IMU world-ray association, 120 Hz ego-motion reprojection, and walking-speed gate alerts.
 */
@RunWith(JUnit4::class)
class TemporalTargetTrackerTest {

    /**
     * Builds a synthetic [ProjectedCandidate] anchored at ground-plane coordinates
     * `([xMeters], 0, [zMeters])` for flat ground-plane voting tests.
     */
    private fun candidateAt(
        xMeters: Float,
        zMeters: Float,
        stoneClass: StoneClass = StoneClass.JASPER_RED_YELLOW,
        confidence: Float = 0.85f,
        diameterMm: Float = 38f
    ): ProjectedCandidate {
        return ProjectedCandidate(
            stripDetection = StripDetection(
                stoneClass = stoneClass,
                confidence = confidence,
                leftPx = 500f,
                topPx = 180f,
                rightPx = 545f,
                bottomPx = 225f,
                estimatedDiameterMm = diameterMm,
                sourceTileIndex = 1
            ),
            worldPosition = WorldPoint3D(xMeters = xMeters, yMeters = 0f, zMeters = zMeters),
            slantRangeMeters = 1.50f,
            refinedDiameterMm = diameterMm
        )
    }

    @Test
    fun threeOfSixSpatialVoting_promotesWithin40mmRadius_andExpiresAfterWindow() {
        val tracker = TemporalTargetTracker()
        val baseTs = 1_000_000_000L
        val stepNs = 80_000_000L

        // Sweep 1: 1/6 votes -> not yet confirmed
        val out1 = tracker.updateSweep(
            candidates = listOf(candidateAt(xMeters = 0.100f, zMeters = 1.060f)),
            timestampNs = baseTs
        )
        assertThat(out1.confirmedTargets).isEmpty()

        // Sweep 2: missed detection (1/6 votes) -> still unconfirmed
        val out2 = tracker.updateSweep(
            candidates = emptyList(),
            timestampNs = baseTs + stepNs
        )
        assertThat(out2.confirmedTargets).isEmpty()

        // Sweep 3: 2nd hit within 22 mm (< 40 mm radius) -> 2/6 votes -> still unconfirmed
        val out3 = tracker.updateSweep(
            candidates = listOf(candidateAt(xMeters = 0.115f, zMeters = 1.075f)),
            timestampNs = baseTs + 2 * stepNs
        )
        assertThat(out3.confirmedTargets).isEmpty()

        // Sweep 4: 3rd hit within 40 mm radius -> 3/6 votes -> PROMOTED to ConfirmedTarget!
        val out4 = tracker.updateSweep(
            candidates = listOf(candidateAt(xMeters = 0.108f, zMeters = 1.068f)),
            timestampNs = baseTs + 3 * stepNs
        )
        assertThat(out4.confirmedTargets).hasSize(1)
        assertThat(out4.confirmedTargets.single().hitCountInWindow).isEqualTo(3)

        // Advance 5 empty sweeps so hits drop below 3-of-6 and eventually expire completely
        var latestOut = out4
        for (i in 4..8) {
            latestOut = tracker.updateSweep(
                candidates = emptyList(),
                timestampNs = baseTs + i * stepNs
            )
        }
        assertThat(latestOut.confirmedTargets).isEmpty()
    }

    @Test
    fun speedGate_triggersSlowSweepWarningAbove1Point1Mps() {
        val tracker = TemporalTargetTracker()

        // Move 0.08 m in 0.10 s = 0.80 m/s -> Normal
        tracker.updateSweep(
            candidates = emptyList(),
            timestampNs = 1_000_000_000L,
            cameraWorldPos = WorldPoint3D(0f, 1.06f, 0f)
        )
        val normalSweep = tracker.updateSweep(
            candidates = emptyList(),
            timestampNs = 1_100_000_000L,
            cameraWorldPos = WorldPoint3D(0f, 1.06f, 0.08f)
        )
        assertThat(normalSweep.speedState).isInstanceOf(ScanSpeedState.Normal::class.java)

        // Move 0.14 m in 0.10 s = 1.40 m/s (> 1.10 m/s) -> SpeedWarning ("SLOW SWEEP")!
        val fastSweep = tracker.updateSweep(
            candidates = emptyList(),
            timestampNs = 1_200_000_000L,
            cameraWorldPos = WorldPoint3D(0f, 1.06f, 0.22f)
        )
        assertThat(fastSweep.speedState).isInstanceOf(ScanSpeedState.SpeedWarning::class.java)
        val warning = fastSweep.speedState as ScanSpeedState.SpeedWarning
        assertThat(warning.currentSpeedMps).isWithin(0.02f).of(1.40f)
    }

    @Test
    fun worldRayAssociation_locksInTwoConsecutiveSweepsAndMaintainsTrackAcrossPitchAndYawMotion() {
        val raycaster = GroundPlaneRaycaster(context = null)
        raycaster.eisPitchLockEnabled = false
        val tracker = TemporalTargetTracker()
        val geometry = StripRoiGeometry(
            stripLeftPx = SwathTileMode.FIVE_TILE.canonicalStripLeftPx,
            stripTopPx = StripRoiGeometry.NOMINAL_CROP_Y_PX,
            stripWidthPx = SwathTileMode.FIVE_TILE.stripWidthPx,
            stripHeightPx = StripRoiGeometry.STRIP_HEIGHT_PX,
            tileMode = SwathTileMode.FIVE_TILE
        )
        val focalPx = 1475.0f // 1.0x wide focal length matching screen grab 3

        // Stationary physical target in the world: initially at strip center (896, 192) when
        // camera is at pitch = 22.0 deg, roll = 0.0 deg, yaw = 0.0 deg.
        val t1Ns = 1_000_000_000L
        raycaster.recordSyntheticOrientationSample(timestampNs = t1Ns, pitchDeg = 22.0f, rollDeg = 0f, yawDeg = 0f)
        val pose1 = raycaster.capturePoseSnapshot(fallbackTimestampNs = t1Ns, verticalFocalLengthPx = focalPx)
        val det1 = StripDetection(
            stoneClass = StoneClass.CHALCEDONY_AGATE,
            confidence = 0.84f,
            leftPx = 860f,
            topPx = 156f,
            rightPx = 932f,
            bottomPx = 228f,
            estimatedDiameterMm = 180f,
            sourceTileIndex = 2
        )
        val ray1 = raycaster.raycastStripDetection(det1, geometry, pose1, zoomRatio = 1.0f)
        val trueWorldRay = ray1.worldRay

        val out1 = tracker.updateSweep(
            candidates = listOf(
                ProjectedCandidate(
                    stripDetection = det1,
                    worldPosition = ray1.worldPoint,
                    slantRangeMeters = ray1.slantRangeMeters,
                    refinedDiameterMm = 180f,
                    worldRay = ray1.worldRay,
                    angularWidthRad = ray1.angularWidthRad,
                    angularHeightRad = ray1.angularHeightRad
                )
            ),
            timestampNs = t1Ns
        )
        // Single sweep is never promoted (rejects 1-frame false positives)
        assertThat(out1.confirmedTargets).isEmpty()

        // Sweep 2 (400 ms later): user tilts camera UP to 19.0 deg and pans RIGHT by +3.5 deg.
        // Compute where the stationary physical object appears in the new camera image:
        val t2Ns = t1Ns + 400_000_000L
        raycaster.recordSyntheticOrientationSample(timestampNs = t2Ns, pitchDeg = 19.0f, rollDeg = 1.2f, yawDeg = 3.5f)
        val pose2 = raycaster.capturePoseSnapshot(fallbackTimestampNs = t2Ns, verticalFocalLengthPx = focalPx)
        val expectedBox2 = GroundPlaneRaycaster.reprojectWorldRayToStripBox(
            worldRay = trueWorldRay,
            angularWidthRad = ray1.angularWidthRad,
            angularHeightRad = ray1.angularHeightRad,
            currentRotationMatrix = pose2.rotationMatrix,
            currentFocalLengthPx = focalPx,
            currentCropYFloat = pose2.cropY.toFloat(),
            stripGeometry = geometry
        )
        // Verify the image of the mug shifted significantly (> 70 px) in strip coordinates due to camera motion
        assertThat(kotlin.math.abs(expectedBox2.stripLeftPx - det1.leftPx)).isGreaterThan(70f)
        assertThat(kotlin.math.abs(expectedBox2.stripTopPx - det1.topPx)).isGreaterThan(60f)

        val det2 = StripDetection(
            stoneClass = StoneClass.CHALCEDONY_AGATE,
            confidence = 0.88f,
            leftPx = expectedBox2.stripLeftPx,
            topPx = expectedBox2.stripTopPx,
            rightPx = expectedBox2.stripRightPx,
            bottomPx = expectedBox2.stripBottomPx,
            estimatedDiameterMm = 180f,
            sourceTileIndex = 2
        )
        val ray2 = raycaster.raycastStripDetection(det2, geometry, pose2, zoomRatio = 1.0f)
        // Back-projected world ray is invariant to camera rotation (< 0.05 deg error)
        assertThat(ray2.worldRay.angularDistanceDegTo(trueWorldRay)).isLessThan(0.05f)

        val out2 = tracker.updateSweep(
            candidates = listOf(
                ProjectedCandidate(
                    stripDetection = det2,
                    worldPosition = ray2.worldPoint,
                    slantRangeMeters = ray2.slantRangeMeters,
                    refinedDiameterMm = 180f,
                    worldRay = ray2.worldRay,
                    angularWidthRad = ray2.angularWidthRad,
                    angularHeightRad = ray2.angularHeightRad
                )
            ),
            timestampNs = t2Ns
        )
        // Promoted on 2nd consecutive high-confidence world-ray hit (~400 ms acquisition!)
        assertThat(out2.confirmedTargets).hasSize(1)
        val lockedTrackId = out2.confirmedTargets.single().trackId
        assertThat(out2.confirmedTargets.single().hitCountInWindow).isEqualTo(2)

        // Sweep 3 (800 ms): momentary missed detection during motion -> confirmation hysteresis
        // retains the confirmed lock at 2/6 instead of dropping it!
        val t3Ns = t2Ns + 400_000_000L
        val out3 = tracker.updateSweep(
            candidates = emptyList(),
            timestampNs = t3Ns
        )
        assertThat(out3.confirmedTargets).hasSize(1)
        assertThat(out3.confirmedTargets.single().trackId).isEqualTo(lockedTrackId)
    }

    @Test
    fun egoMotionReprojection_compensatesInferenceLatencyAndSuppressesOffStripGhostsAt120Hz() {
        val raycaster = GroundPlaneRaycaster(context = null)
        raycaster.eisPitchLockEnabled = false
        val tracker = TemporalTargetTracker()
        val geometry = StripRoiGeometry()
        val focalPx = 1475.0f

        // Capture frame at t = 1.0 s (pitch = 22 deg, yaw = 0 deg)
        val tCapture1 = 1_000_000_000L
        raycaster.recordSyntheticOrientationSample(tCapture1, pitchDeg = 22.0f, rollDeg = 0f, yawDeg = 0f)
        val pose1 = raycaster.capturePoseSnapshot(fallbackTimestampNs = tCapture1, verticalFocalLengthPx = focalPx)
        val det = StripDetection(
            stoneClass = StoneClass.PETRIFIED_WOOD,
            confidence = 0.90f,
            leftPx = 850f,
            topPx = 150f,
            rightPx = 942f,
            bottomPx = 234f,
            estimatedDiameterMm = 220f,
            sourceTileIndex = 2
        )
        val ray1 = raycaster.raycastStripDetection(det, geometry, pose1, zoomRatio = 1.0f)
        val cand = ProjectedCandidate(
            stripDetection = det,
            worldPosition = ray1.worldPoint,
            slantRangeMeters = ray1.slantRangeMeters,
            refinedDiameterMm = 220f,
            worldRay = ray1.worldRay,
            angularWidthRad = ray1.angularWidthRad,
            angularHeightRad = ray1.angularHeightRad
        )

        tracker.updateSweep(listOf(cand), timestampNs = tCapture1)
        val out2 = tracker.updateSweep(listOf(cand), timestampNs = tCapture1 + 400_000_000L)
        assertThat(out2.confirmedTargets).hasSize(1)

        // Simulate 395 ms of GPU inference latency during which the user pans left (-4 deg) and tilts up (20 deg)
        val tDisplayNow = tCapture1 + 795_000_000L
        raycaster.recordSyntheticOrientationSample(tDisplayNow, pitchDeg = 20.0f, rollDeg = 0f, yawDeg = -4.0f)
        val reprojectedOnScreen = raycaster.reprojectConfirmedTargetsToCurrentPose(
            targets = out2.confirmedTargets,
            stripGeometry = geometry,
            currentCropYFloat = StripRoiGeometry.NOMINAL_CROP_Y_PX.toFloat(),
            currentFocalLengthPx = focalPx
        ).single()

        assertThat(reprojectedOnScreen.isVisibleInStrip).isTrue()
        // Panning camera left (-4 deg yaw) moves the stationary target to the RIGHT (+X) in strip coordinates;
        // tilting camera up (22 -> 20 deg) moves the target DOWN (+Y) in strip coordinates.
        assertThat(reprojectedOnScreen.stripLeftPx).isGreaterThan(det.leftPx + 80f)
        assertThat(reprojectedOnScreen.stripTopPx).isGreaterThan(det.topPx + 40f)

        // Now user tilts phone sharply up toward the horizon (pitch = 4 deg) so the desk target is below the strip
        raycaster.recordSyntheticOrientationSample(tDisplayNow + 100_000_000L, pitchDeg = 4.0f, rollDeg = 0f, yawDeg = -4.0f)
        val reprojectedOffScreen = raycaster.reprojectConfirmedTargetsToCurrentPose(
            targets = out2.confirmedTargets,
            stripGeometry = geometry,
            currentCropYFloat = StripRoiGeometry.NOMINAL_CROP_Y_PX.toFloat(),
            currentFocalLengthPx = focalPx
        ).single()

        // Target is marked not visible in strip -> suppresses floating ghost reticles!
        assertThat(reprojectedOffScreen.isVisibleInStrip).isFalse()
    }

    @Test
    fun trackLevelSpatialNms_mergesDuplicateSameClassTracksWithin3Point2DegOr75Mm_andKeepsDistinctStones() {
        val tracker = TemporalTargetTracker()
        val rayCenter = WorldRay3D(eastX = 0.0f, northY = 0.70710677f, upZ = -0.70710677f).normalized()
        // ~2.3 deg East offset (< 3.2 deg NMS gate) and 60 mm ground separation (< 75 mm NMS gate)
        val rayDuplicate = WorldRay3D(eastX = 0.040f, northY = 0.70710677f, upZ = -0.70710677f).normalized()
        // ~18 deg East offset (> 3.2 deg) and 0.48 m ground separation -> genuinely separate stone
        val raySeparate = WorldRay3D(eastX = 0.320f, northY = 0.70710677f, upZ = -0.70710677f).normalized()

        val primaryCand = ProjectedCandidate(
            stripDetection = StripDetection(
                stoneClass = StoneClass.CHALCEDONY_AGATE,
                confidence = 0.91f,
                leftPx = 700f,
                topPx = 160f,
                rightPx = 760f,
                bottomPx = 220f,
                estimatedDiameterMm = 35f,
                sourceTileIndex = 1
            ),
            worldPosition = WorldPoint3D(xMeters = 0.00f, yMeters = 0f, zMeters = 1.06f),
            slantRangeMeters = 1.50f,
            refinedDiameterMm = 35f,
            worldRay = rayCenter,
            angularWidthRad = 0.020f,
            angularHeightRad = 0.020f
        )
        val seamDuplicateCand = ProjectedCandidate(
            stripDetection = StripDetection(
                stoneClass = StoneClass.CHALCEDONY_AGATE,
                confidence = 0.78f,
                leftPx = 810f,
                topPx = 162f,
                rightPx = 865f,
                bottomPx = 218f,
                estimatedDiameterMm = 34f,
                sourceTileIndex = 2
            ),
            worldPosition = WorldPoint3D(xMeters = 0.060f, yMeters = 0f, zMeters = 1.06f),
            slantRangeMeters = 1.50f,
            refinedDiameterMm = 34f,
            worldRay = rayDuplicate,
            angularWidthRad = 0.019f,
            angularHeightRad = 0.019f
        )
        val separateAgateCand = ProjectedCandidate(
            stripDetection = StripDetection(
                stoneClass = StoneClass.CHALCEDONY_AGATE,
                confidence = 0.86f,
                leftPx = 1320f,
                topPx = 160f,
                rightPx = 1380f,
                bottomPx = 220f,
                estimatedDiameterMm = 38f,
                sourceTileIndex = 3
            ),
            worldPosition = WorldPoint3D(xMeters = 0.48f, yMeters = 0f, zMeters = 1.06f),
            slantRangeMeters = 1.55f,
            refinedDiameterMm = 38f,
            worldRay = raySeparate,
            angularWidthRad = 0.020f,
            angularHeightRad = 0.020f
        )
        val adjacentJasperCand = primaryCand.copy(
            stripDetection = primaryCand.stripDetection.copy(stoneClass = StoneClass.JASPER_RED_YELLOW)
        )

        val sweepCandidates = listOf(primaryCand, seamDuplicateCand, separateAgateCand, adjacentJasperCand)
        tracker.updateSweep(sweepCandidates, timestampNs = 1_000_000_000L)
        val out2 = tracker.updateSweep(sweepCandidates, timestampNs = 1_150_000_000L)

        // Expect exactly 2 Agates (duplicate merged!) and 1 Jasper = 3 confirmed targets total
        val confirmedAgates = out2.confirmedTargets.filter { it.stoneClass == StoneClass.CHALCEDONY_AGATE }
        val confirmedJaspers = out2.confirmedTargets.filter { it.stoneClass == StoneClass.JASPER_RED_YELLOW }
        assertThat(confirmedAgates).hasSize(2)
        assertThat(confirmedJaspers).hasSize(1)
        assertThat(out2.totalCandidateTracks).isEqualTo(3)
    }

    @Test
    fun trackLevelSpatialNms_mergesOverlappingStripBoxesByIomEvenBeyond75Mm() {
        val tracker = TemporalTargetTracker()
        val outerBox = ProjectedCandidate(
            stripDetection = StripDetection(
                stoneClass = StoneClass.GREEN_GEM_ROCK,
                confidence = 0.89f,
                leftPx = 600f,
                topPx = 140f,
                rightPx = 700f,
                bottomPx = 240f,
                estimatedDiameterMm = 58f,
                sourceTileIndex = 1
            ),
            worldPosition = WorldPoint3D(xMeters = 0.00f, yMeters = 0f, zMeters = 1.00f),
            slantRangeMeters = 1.45f,
            refinedDiameterMm = 58f
        )
        // Nested 40x40 box inside the 100x100 box: IoU = 1600/10000 = 0.16 (< 0.35),
        // but IoM = 1600/1600 = 1.00 (>= 0.60), and 3D distance = 110 mm (> 75 mm).
        val nestedFragmentBox = ProjectedCandidate(
            stripDetection = StripDetection(
                stoneClass = StoneClass.GREEN_GEM_ROCK,
                confidence = 0.76f,
                leftPx = 630f,
                topPx = 170f,
                rightPx = 670f,
                bottomPx = 210f,
                estimatedDiameterMm = 24f,
                sourceTileIndex = 2
            ),
            worldPosition = WorldPoint3D(xMeters = 0.11f, yMeters = 0f, zMeters = 1.00f),
            slantRangeMeters = 1.45f,
            refinedDiameterMm = 24f
        )

        val both = listOf(outerBox, nestedFragmentBox)
        tracker.updateSweep(both, timestampNs = 1_000_000_000L)
        tracker.updateSweep(both, timestampNs = 1_100_000_000L)
        val out3 = tracker.updateSweep(both, timestampNs = 1_200_000_000L)

        assertThat(out3.confirmedTargets).hasSize(1)
        assertThat(out3.confirmedTargets.single().smoothedConfidence).isGreaterThan(0.85f)
    }
}
