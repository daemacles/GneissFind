package com.gneissfind.app.audio

import android.media.AudioAttributes
import android.view.Surface
import com.gneissfind.app.camera.PushBroomCameraController
import com.gneissfind.app.inference.RfDetrTwoGraphGpuDetector
import com.gneissfind.app.logging.FieldDataLogger
import com.gneissfind.app.logging.FieldStripMetadata
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.tracking.GroundPlaneRaycaster
import com.google.common.truth.Truth.assertThat
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

@RunWith(JUnit4::class)
class SonifierAndHostBridgeTest {

    @Test
    fun diameterAndConfidenceMappings_followLockedMetalDetectorCurve() {
        assertThat(MetalDetectorSonifier.diameterMmToPitchHz(20f)).isWithin(0.01f).of(780f)
        assertThat(MetalDetectorSonifier.diameterMmToPitchHz(40f)).isWithin(0.01f).of(550f)
        assertThat(MetalDetectorSonifier.diameterMmToPitchHz(60f)).isWithin(0.01f).of(320f)
        assertThat(MetalDetectorSonifier.diameterMmToPitchHz(85f)).isWithin(0.01f).of(320f)

        assertThat(MetalDetectorSonifier.confidenceToPingCadenceHz(0.0f)).isWithin(0.01f).of(2.0f)
        assertThat(MetalDetectorSonifier.confidenceToPingCadenceHz(0.5f)).isWithin(0.01f).of(8.0f)
        assertThat(MetalDetectorSonifier.confidenceToPingCadenceHz(1.0f)).isWithin(0.01f).of(14.0f)

        val burst = MetalDetectorSonifier.synthesizeSinglePingBurstPcm16(diameterMm = 24f)
        assertThat(burst.first().toInt()).isEqualTo(0)
        assertThat(abs(burst.last().toInt())).isLessThan(200)
        assertThat(burst.maxOf { abs(it.toInt()) }).isGreaterThan(15_000)

        assertThat(MetalDetectorSonifier.AUDIO_USAGE).isEqualTo(AudioAttributes.USAGE_MEDIA)
        assertThat(MetalDetectorSonifier.AUDIO_CONTENT_TYPE).isEqualTo(AudioAttributes.CONTENT_TYPE_MUSIC)
    }

    @Test
    fun rfDetrHostBridge_selectsTop100QueriesInDescendingScoreOrder() {
        val numAnchors = 900
        val numClasses = 5
        val hiddenDim = 16
        val topK = 100

        val logits = FloatArray(numAnchors * numClasses) { -4.0f }
        val features = FloatArray(numAnchors * hiddenDim) { idx -> (idx / hiddenDim).toFloat() }

        logits[712 * numClasses + 0] = 9.5f
        logits[42 * numClasses + 2] = 7.2f
        logits[899 * numClasses + 1] = 6.1f

        val outFeatures = FloatArray(topK * hiddenDim)
        val outIndices = IntArray(topK)

        RfDetrTwoGraphGpuDetector.selectTopKQueriesOnHost(
            encoderLogits = logits,
            encoderFeatures = features,
            numAnchors = numAnchors,
            numClasses = numClasses,
            hiddenDim = hiddenDim,
            topK = topK,
            outQueryFeatures = outFeatures,
            outSelectedIndices = outIndices
        )

        assertThat(outIndices[0]).isEqualTo(712)
        assertThat(outIndices[1]).isEqualTo(42)
        assertThat(outIndices[2]).isEqualTo(899)
        assertThat(outFeatures[0]).isEqualTo(712f)
        assertThat(outFeatures[hiddenDim]).isEqualTo(42f)
        assertThat(outFeatures[2 * hiddenDim]).isEqualTo(899f)
    }

    @Test
    fun reversibleLandscapeImuRemapping_preservesPitchAndRollAcross180DegreeFlip() {
        val remapped90 = FloatArray(9)
        val remapped270 = FloatArray(9)

        for (targetPitchDeg in listOf(0f, 30f, 45f, 75f, 90f)) {
            val theta = Math.toRadians(targetPitchDeg.toDouble())
            val c = cos(theta).toFloat()
            val s = sin(theta).toFloat()

            // Construct raw device rotation matrix (device -> ENU world, Row 2 = World Up) for:
            // 1) ROTATION_90 (screen-right = +Y_dev, screen-up = -X_dev, screen-out = +Z_dev)
            //    Level roll=0 => Up·Y_dev = 0, Up·(-X_dev) = cos(theta) => Up·X_dev = -cos(theta), Up·Z_dev = sin(theta)
            val rawRot90 = floatArrayOf(
                0f, 1f, 0f,
                -s, 0f, -c,
                -c, 0f, s
            )

            // 2) ROTATION_270 (screen-right = -Y_dev, screen-up = +X_dev, screen-out = +Z_dev)
            //    Level roll=0 => Up·(-Y_dev) = 0, Up·(+X_dev) = cos(theta), Up·Z_dev = sin(theta)
            val rawRot270 = floatArrayOf(
                0f, -1f, 0f,
                s, 0f, -c,
                c, 0f, s
            )

            GroundPlaneRaycaster.remapLandscapeRotationMatrix(rawRot90, Surface.ROTATION_90, remapped90)
            GroundPlaneRaycaster.remapLandscapeRotationMatrix(rawRot270, Surface.ROTATION_270, remapped270)

            val (pitch90, roll90) = GroundPlaneRaycaster.computePitchAndRollDegrees(remapped90)
            val (pitch270, roll270) = GroundPlaneRaycaster.computePitchAndRollDegrees(remapped270)

            assertThat(pitch90).isWithin(0.05f).of(targetPitchDeg)
            assertThat(pitch270).isWithin(0.05f).of(targetPitchDeg)
            assertThat(roll90).isWithin(0.05f).of(0.0f)
            assertThat(roll270).isWithin(0.05f).of(0.0f)
        }
    }

    @Test
    fun imuPitchToCropYStabilization_tracksGroundPatchAndFlagsPitchLimitAtBounds() {
        val fy = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX

        // 1. At reference pitch (45 deg): centered at cropY = 348, unclamped
        val nominal = GroundPlaneRaycaster.computeStabilizedCropY(
            currentPitchDeg = 45.0f,
            referencePitchDeg = 45.0f,
            verticalFocalLengthPx = fy
        )
        assertThat(nominal.clampedCropY).isEqualTo(StripRoiGeometry.NOMINAL_CROP_Y_PX)
        assertThat(nominal.isPitchClamped).isFalse()

        // 2. Pitching lens DOWN toward boots (+4 deg -> 49 deg): ground patch moves UP in frame (cropY < 348)
        val pitchedDown = GroundPlaneRaycaster.computeStabilizedCropY(
            currentPitchDeg = 49.0f,
            referencePitchDeg = 45.0f,
            verticalFocalLengthPx = fy
        )
        assertThat(pitchedDown.clampedCropY).isLessThan(StripRoiGeometry.NOMINAL_CROP_Y_PX)
        assertThat(pitchedDown.clampedCropY).isGreaterThan(0)
        assertThat(pitchedDown.isPitchClamped).isFalse()

        // 3. Pitching lens UP toward horizon (-4 deg -> 41 deg): ground patch moves DOWN in frame (cropY > 348)
        val pitchedUp = GroundPlaneRaycaster.computeStabilizedCropY(
            currentPitchDeg = 41.0f,
            referencePitchDeg = 45.0f,
            verticalFocalLengthPx = fy
        )
        assertThat(pitchedUp.clampedCropY).isGreaterThan(StripRoiGeometry.NOMINAL_CROP_Y_PX)
        assertThat(pitchedUp.clampedCropY).isLessThan(StripRoiGeometry.MAX_CROP_Y_PX)
        assertThat(pitchedUp.isPitchClamped).isFalse()

        // 4. Exceeding +/- ~6.7 deg authority clamps cleanly at 0 and 696 and illuminates PITCH LIMIT
        val clampedTop = GroundPlaneRaycaster.computeStabilizedCropY(
            currentPitchDeg = 55.0f,
            referencePitchDeg = 45.0f,
            verticalFocalLengthPx = fy
        )
        assertThat(clampedTop.clampedCropY).isEqualTo(0)
        assertThat(clampedTop.isPitchClamped).isTrue()

        val clampedBottom = GroundPlaneRaycaster.computeStabilizedCropY(
            currentPitchDeg = 35.0f,
            referencePitchDeg = 45.0f,
            verticalFocalLengthPx = fy
        )
        assertThat(clampedBottom.clampedCropY).isEqualTo(StripRoiGeometry.MAX_CROP_Y_PX)
        assertThat(clampedBottom.isPitchClamped).isTrue()

        // 5. User SYNC PITCH setpoint at 52 deg recenters cropY to 348
        val synced = GroundPlaneRaycaster.computeStabilizedCropY(
            currentPitchDeg = 52.0f,
            referencePitchDeg = 52.0f,
            verticalFocalLengthPx = fy
        )
        assertThat(synced.clampedCropY).isEqualTo(StripRoiGeometry.NOMINAL_CROP_Y_PX)
        assertThat(synced.isPitchClamped).isFalse()
    }

    @Test
    fun raycasterAndFieldMetadataJsonl_matchLandscape2xSwathGeometry() {
        val centerRay = GroundPlaneRaycaster.raycastTrigonometric(
            stripCenterXPx = 896f,
            stripCenterYPx = 192f,
            stripWidthPx = 1792,
            forwardPitchDeg = 45f,
            waistHeightMeters = 1.06f,
            zoomRatio = 2.0f
        )

        assertThat(centerRay.slantRangeMeters).isWithin(0.02f).of(1.50f)
        assertThat(centerRay.worldPoint.zMeters).isWithin(0.02f).of(1.06f)
        assertThat(centerRay.worldPoint.xMeters).isWithin(0.001f).of(0.0f)
        assertThat(centerRay.gsdMmPerPx).isWithin(0.02f).of(0.59f)

        val jsonLine = FieldDataLogger.buildMetadataJsonLine(
            sessionId = "20261114_100000",
            frameIndex = 1,
            fileName = "strip_000001.png",
            metadata = FieldStripMetadata(
                timestampNs = 123456789L,
                wallTimeMs = 1794650400000L,
                stripWidthPx = 1792,
                stripHeightPx = 384,
                tileCount = 5,
                zoomRatio = 2.0f,
                exposureTimeNs = 2_000_000L,
                sensorIso = 320,
                shutterClamped = true,
                exposureBiasEv = -0.5f,
                imuPitchDeg = 45.0f,
                referencePitchDeg = 45.0f,
                cropY = 348,
                isPitchClamped = false,
                slantRangeMeters = centerRay.slantRangeMeters,
                gsdMmPerPx = centerRay.gsdMmPerPx,
                horizontalSpeedMps = 0.65f
            )
        )
        val parsed = JSONObject(jsonLine)
        assertThat(parsed.getInt("strip_width")).isEqualTo(1792)
        assertThat(parsed.getInt("strip_height")).isEqualTo(384)
        assertThat(parsed.getInt("tile_count")).isEqualTo(5)
        assertThat(parsed.getDouble("zoom_ratio")).isWithin(0.01).of(2.0)
        assertThat(parsed.getLong("exposure_ns")).isAtMost(2_000_000L)
        assertThat(parsed.getInt("crop_y")).isEqualTo(348)
        assertThat(parsed.getBoolean("is_pitch_clamped")).isFalse()
        assertThat(parsed.getDouble("slant_range_m")).isWithin(0.02).of(1.50)
        assertThat(parsed.getDouble("gsd_mm_px")).isWithin(0.02).of(0.59)
    }
}
