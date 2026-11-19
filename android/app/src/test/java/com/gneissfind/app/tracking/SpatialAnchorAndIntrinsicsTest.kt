package com.gneissfind.app.tracking

import com.gneissfind.app.camera.PushBroomCameraController
import com.gneissfind.app.model.AcquisitionStage
import com.gneissfind.app.model.ConfirmedTarget
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.WorldPoint3D
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

/**
 * Deterministic JVM unit tests covering the M2D-2 Unified 3-Tier Spatial Anchor,
 * API 30+ Camera2 `CONTROL_ZOOM_RATIO` focal-length calibration, interpolated `SO(3)` IMU
 * ring-buffer lookup with lag compensation, and 4-Gate geometric + NCC verification primitives.
 */
class SpatialAnchorAndIntrinsicsTest {

    @Test
    fun computeStreamVerticalFocalLength_scalesByControlZoomRatioWhenCropMatchesActiveArray() {
        // Pixel 10 Pro XL: 4080x3072 active array, 16:9 effective height = 2295 px,
        // fy_sensor = 2952 px -> 1.0x stream fy = 2952 * (1080/2295) = 1389.2 px.
        // At 2.0x zoom via API 30+ CONTROL_ZOOM_RATIO, SCALER_CROP_REGION stays 4080x3072.
        val fy1x = PushBroomCameraController.computeStreamVerticalFocalLengthPxFromDims(
            fySensorPx = 2952f,
            cropWidthPx = 4080,
            cropHeightPx = 3072,
            activeWidthPx = 4080,
            activeHeightPx = 3072,
            activeZoomRatio = 1.0f,
            captureZoomRatio = 1.0f
        )
        val fy2x = PushBroomCameraController.computeStreamVerticalFocalLengthPxFromDims(
            fySensorPx = 2952f,
            cropWidthPx = 4080,
            cropHeightPx = 3072,
            activeWidthPx = 4080,
            activeHeightPx = 3072,
            activeZoomRatio = 2.0f,
            captureZoomRatio = 2.0f
        )

        assertThat(fy1x).isWithin(5f).of(1389f)
        assertThat(fy2x).isWithin(10f).of(2778f)
    }

    @Test
    fun computeStreamVerticalFocalLength_doesNotDoubleScaleWhenCropRegionAlreadyCropped() {
        // Legacy HAL that already halves SCALER_CROP_REGION to 2040x1536 at 2.0x zoom must NOT double-scale.
        val fyLegacyCropped = PushBroomCameraController.computeStreamVerticalFocalLengthPxFromDims(
            fySensorPx = 2952f,
            cropWidthPx = 2040,
            cropHeightPx = 1536,
            activeWidthPx = 4080,
            activeHeightPx = 3072,
            activeZoomRatio = 2.0f,
            captureZoomRatio = 2.0f
        )

        assertThat(fyLegacyCropped).isWithin(10f).of(2778f)
    }

    @Test
    fun replayFocalLengthRepair_fixesLegacyUnscaled2xWithoutDoubleScalingCalibratedClips() {
        // Legacy 20261119_011006 trace logged 1389 px at 2.0x zoom -> repairs to 2778 px
        val repaired = com.gneissfind.app.camera.ReplayCameraSource.resolveReplayVerticalFocalLengthPx(
            recordedFyPx = 1389f,
            zoomRatio = 2.0f
        )
        assertThat(repaired).isWithin(1f).of(2778f)

        // New M2D-2 trace already logged 2736.9 px at 2.0x zoom -> untouched (no double scaling)
        val calibrated = com.gneissfind.app.camera.ReplayCameraSource.resolveReplayVerticalFocalLengthPx(
            recordedFyPx = 2736.9f,
            zoomRatio = 2.0f
        )
        assertThat(calibrated).isWithin(0.1f).of(2736.9f)

        // 1.0x wide clip logged 1389 px -> untouched
        val wide1x = com.gneissfind.app.camera.ReplayCameraSource.resolveReplayVerticalFocalLengthPx(
            recordedFyPx = 1389f,
            zoomRatio = 1.0f
        )
        assertThat(wide1x).isWithin(0.1f).of(1389f)
    }

    @Test
    fun groundPlaneRaycaster_interpolatesOrthonormalRotationAndAppliesLagCompensation() {
        val raycaster = GroundPlaneRaycaster(context = null, waistHeightMeters = 1.15f)
        val r0 = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 40f, rollDeg = 0f, yawDeg = 0f)
        val r1 = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 60f, rollDeg = 0f, yawDeg = 20f)

        raycaster.injectReplayOrientationMatrix(
            timestampNs = 1_000_000_000L,
            rotationMatrix9 = r0,
            rawPitchOverrideDeg = 40f
        )
        raycaster.injectReplayOrientationMatrix(
            timestampNs = 1_020_000_000L,
            rotationMatrix9 = r1,
            rawPitchOverrideDeg = 60f
        )

        // Midpoint query (10 ms) -> orthonormal interpolated rotation matrix with det = +1 and pitch = 50 deg
        val midR = raycaster.lookupRotationMatrixAtTimestampNs(1_010_000_000L)
        assertThat(midR).isNotNull()
        assertThat(det3x3(midR!!)).isWithin(1e-4f).of(1.0f)
        val midPitch = raycaster.lookupPitchAtTimestampNs(1_010_000_000L)
        assertThat(midPitch).isWithin(0.1f).of(50f)

        // Apply +10 ms camera-to-IMU lag compensation (camera lags IMU by 10 ms) -> querying 1_020ms returns 1_010ms pose
        raycaster.cameraImuLagCompensationNs = 10_000_000L
        val shiftedR = raycaster.lookupRotationMatrixAtTimestampNs(1_020_000_000L)
        assertThat(shiftedR).isNotNull()
        for (i in 0 until 9) {
            assertThat(shiftedR!![i]).isWithin(1e-3f).of(midR[i])
        }
        val shiftedPitch = raycaster.lookupPitchAtTimestampNs(1_020_000_000L)
        assertThat(shiftedPitch).isWithin(0.1f).of(50f)
        raycaster.close()
    }

    @Test
    fun verifyHomography2x2Conditioning_acceptsValidSimilarityAndRejectsDegenerateWarps() {
        val rad = Math.toRadians(15.0)
        val s = 1.15
        assertThat(
            OpenCvConstellationTracker.verifyHomography2x2Conditioning(
                j00 = s * cos(rad),
                j01 = -s * sin(rad),
                j10 = s * sin(rad),
                j11 = s * cos(rad),
                h20 = 1e-5,
                h21 = -1e-5,
                widthPx = 1920
            )
        ).isTrue()

        // Squashed pancake (singular value ratio = 2.0 / 0.5 = 4.0 > 2.4) -> REJECT
        assertThat(
            OpenCvConstellationTracker.verifyHomography2x2Conditioning(
                j00 = 2.0,
                j01 = 0.0,
                j10 = 0.0,
                j11 = 0.5,
                h20 = 0.0,
                h21 = 0.0,
                widthPx = 1920
            )
        ).isFalse()

        // Negative determinant reflection -> REJECT
        assertThat(
            OpenCvConstellationTracker.verifyHomography2x2Conditioning(
                j00 = -1.0,
                j01 = 0.0,
                j10 = 0.0,
                j11 = 1.0,
                h20 = 0.0,
                h21 = 0.0,
                widthPx = 1920
            )
        ).isFalse()

        // Excessive perspective warp -> REJECT
        assertThat(
            OpenCvConstellationTracker.verifyHomography2x2Conditioning(
                j00 = 1.0,
                j01 = 0.0,
                j10 = 0.0,
                j11 = 1.0,
                h20 = 0.0005,
                h21 = 0.0,
                widthPx = 1920
            )
        ).isFalse()
    }

    @Test
    fun normalizeZeroMeanUnitNcc_rejectsBlankBeigeWallAndDistinguishesTexturedPatches() {
        // Uniform beige wall (std ~ 0.8 < 3.5) -> MUST return null (0% false lock on blank wall)
        val blankWall = ByteArray(32 * 32) { idx -> (180 + (idx % 3) - 1).toByte() }
        assertThat(OpenCvConstellationTracker.normalizeZeroMeanUnitNcc(blankWall)).isNull()

        // Textured mug patch vs brightness-shifted mug patch -> NCC ~ 1.0
        val mugPatchA = ByteArray(32 * 32) { idx ->
            val x = idx % 32
            val y = idx / 32
            (if ((x / 4 + y / 4) % 2 == 0) 60 else 190).toByte()
        }
        val mugPatchBrighter = ByteArray(32 * 32) { idx ->
            ((mugPatchA[idx].toInt() and 0xFF) + 25).coerceIn(0, 255).toByte()
        }
        val invertedPatch = ByteArray(32 * 32) { idx ->
            (255 - (mugPatchA[idx].toInt() and 0xFF)).toByte()
        }

        val normA = OpenCvConstellationTracker.normalizeZeroMeanUnitNcc(mugPatchA)
        val normB = OpenCvConstellationTracker.normalizeZeroMeanUnitNcc(mugPatchBrighter)
        val normInv = OpenCvConstellationTracker.normalizeZeroMeanUnitNcc(invertedPatch)

        assertThat(normA).isNotNull()
        assertThat(normB).isNotNull()
        assertThat(normInv).isNotNull()
        assertThat(OpenCvConstellationTracker.dotNcc(normA!!, normB!!)).isGreaterThan(0.99f)
        assertThat(OpenCvConstellationTracker.dotNcc(normA, normInv!!)).isLessThan(-0.99f)
    }

    @Test
    fun lockOnPinpointController_runsTier1KltAndTier2MiniVioFromStage2Frame1AndReportsHonestCoasting() {
        var tier1ReturnsMatch = true
        var tier1Calls = 0
        var tier2Calls = 0

        val fakeTracker = object : OpenCvConstellationTracker() {
            override fun beginTrackingFrame(lumaBytes: ByteArray, widthPx: Int, heightPx: Int): Boolean = true

            override fun stepRotationKnownMiniVio30Hz(
                prevCamPosEnu: FloatArray,
                prevRotationScr: FloatArray,
                currRotationScr: FloatArray,
                focalLengthPx: Float,
                nominalSceneDepthM: Float,
                fullWidthPx: Int,
                fullHeightPx: Int,
                excludeLeftPx: Float,
                excludeTopPx: Float,
                excludeRightPx: Float,
                excludeBottomPx: Float,
                excludeTargetRegion: Boolean
            ): MiniVioStepResult {
                tier2Calls++
                return MiniVioStepResult(
                    updated = true,
                    camPosEnu = floatArrayOf(prevCamPosEnu[0] + 0.01f, prevCamPosEnu[1], prevCamPosEnu[2]),
                    activeLandmarks = 82,
                    triangulatedLandmarks = 18,
                    inlierLandmarks = 42,
                    latencyMs = 1.4f
                )
            }

            override fun trackLocalTargetSimilarity30Hz(
                prevBoxLeftPx: Float,
                prevBoxTopPx: Float,
                prevBoxRightPx: Float,
                prevBoxBottomPx: Float,
                prevRotationScr: FloatArray,
                currRotationScr: FloatArray,
                focalLengthPx: Float,
                fullWidthPx: Int,
                fullHeightPx: Int
            ): LocalSimilarityTrackResult {
                tier1Calls++
                return if (tier1ReturnsMatch) {
                    LocalSimilarityTrackResult(
                        matched = true,
                        inliers = 28,
                        frameLeftPx = prevBoxLeftPx + 2f,
                        frameTopPx = prevBoxTopPx - 1f,
                        frameRightPx = prevBoxRightPx + 2f,
                        frameBottomPx = prevBoxBottomPx - 1f,
                        scaleRatio = 1.0f,
                        nccScore = 0.84f,
                        latencyMs = 0.7f
                    )
                } else {
                    LocalSimilarityTrackResult(matched = false, inliers = 0, latencyMs = 0.4f)
                }
            }
        }

        val controller = LockOnPinpointController(openCvTracker = fakeTracker)
        val r0 = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 45f, rollDeg = 0f, yawDeg = 0f)
        val pose0 = MotionPoseSnapshot(
            timestampNs = 1_000_000_000L,
            forwardPitchDeg = 45f,
            rawPitchDeg = 45f,
            referencePitchDeg = 45f,
            rollDeg = 0f,
            cropY = 348,
            isPitchClamped = false,
            centerSlantRangeMeters = 1.35f,
            cameraWorldPosition = WorldPoint3D(0f, 0f, 1.15f),
            horizontalSpeedMps = 0f,
            arPlaneLocked = false,
            rotationMatrix = r0,
            verticalFocalLengthPx = 2200f
        )
        val confirmed = ConfirmedTarget(
            trackId = 7L,
            stoneClass = StoneClass.CHALCEDONY_AGATE,
            worldPosition = WorldPoint3D(0f, 0f, 1.15f),
            stripLeftPx = 836f,
            stripTopPx = 150f,
            stripRightPx = 956f,
            stripBottomPx = 270f,
            smoothedConfidence = 0.90f,
            estimatedDiameterMm = 42f,
            slantRangeMeters = 1.35f,
            hitCountInWindow = 3,
            windowSize = 4,
            lastUpdatedTimestampNs = 1_000_000_000L,
            isVisibleInStrip = true
        )

        val enterHud = controller.lockOntoTarget(
            target = confirmed,
            stripGeometry = StripRoiGeometry(),
            currentCropYFloat = 348f,
            currentFocalLengthPx = 2200f,
            currentRotationMatrix = r0,
            timestampNs = 1_000_000_000L
        )
        assertThat(enterHud.stage).isEqualTo(AcquisitionStage.LOCK_ON_NEURAL)
        assertThat(enterHud.hasMetric3dDepth).isFalse()

        val luma = ByteArray(1920 * 1080)
        // Frame 0 establishes previous pose; Frame 1 runs Tier 1 KLT + Tier 2 Mini-VIO immediately!
        controller.onCameraLumaFrame(
            lumaBytes = luma,
            widthPx = 1920,
            heightPx = 1080,
            poseSnapshot = pose0.copy(timestampNs = 1_000_000_000L),
            timestampNs = 1_000_000_000L
        )
        val dir1 = controller.onCameraLumaFrame(
            lumaBytes = luma,
            widthPx = 1920,
            heightPx = 1080,
            poseSnapshot = pose0.copy(timestampNs = 1_033_333_333L),
            timestampNs = 1_033_333_333L
        )
        assertThat(tier1Calls).isGreaterThan(0)
        assertThat(tier2Calls).isGreaterThan(0)
        assertThat(dir1.hudState.isVisuallyLocked).isTrue()
        assertThat(dir1.hudState.visualAnchorLabel).contains("KLT")
        assertThat(dir1.hudState.hasMetric3dDepth).isFalse()

        // Frame 2: Tier 1 loses visual texture (e.g., blank wall) -> MUST honestly report coasting!
        tier1ReturnsMatch = false
        val dir2 = controller.onCameraLumaFrame(
            lumaBytes = luma,
            widthPx = 1920,
            heightPx = 1080,
            poseSnapshot = pose0.copy(timestampNs = 1_066_666_666L),
            timestampNs = 1_066_666_666L
        )
        assertThat(dir2.hudState.isVisuallyLocked).isFalse()
        assertThat(dir2.hudState.visualAnchorLabel).isEqualTo("COAST 3D/IMU")
        controller.close()
    }

    private fun det3x3(m: FloatArray): Float {
        return m[0] * (m[4] * m[8] - m[5] * m[7]) -
            m[1] * (m[3] * m[8] - m[5] * m[6]) +
            m[2] * (m[3] * m[7] - m[4] * m[6])
    }
}
