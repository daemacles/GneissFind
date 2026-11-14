package com.gneissfind.app.audio

import com.gneissfind.app.inference.RfDetrTwoGraphGpuDetector
import com.gneissfind.app.logging.FieldDataLogger
import com.gneissfind.app.logging.FieldStripMetadata
import com.gneissfind.app.tracking.GroundPlaneRaycaster
import com.google.common.truth.Truth.assertThat
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import kotlin.math.abs

@RunWith(JUnit4::class)
class SonifierAndHostBridgeTest {

    @Test
    fun diameterAndConfidenceMappings_followLockedMetalDetectorCurve() {
        // 20 mm pebble -> 780 Hz; 40 mm -> 550 Hz; 60 mm+ cobble -> 320 Hz
        assertThat(MetalDetectorSonifier.diameterMmToPitchHz(20f)).isWithin(0.01f).of(780f)
        assertThat(MetalDetectorSonifier.diameterMmToPitchHz(40f)).isWithin(0.01f).of(550f)
        assertThat(MetalDetectorSonifier.diameterMmToPitchHz(60f)).isWithin(0.01f).of(320f)
        assertThat(MetalDetectorSonifier.diameterMmToPitchHz(85f)).isWithin(0.01f).of(320f)

        // Confidence 0..1 -> 2..14 Hz ping cadence
        assertThat(MetalDetectorSonifier.confidenceToPingCadenceHz(0.0f)).isWithin(0.01f).of(2.0f)
        assertThat(MetalDetectorSonifier.confidenceToPingCadenceHz(0.5f)).isWithin(0.01f).of(8.0f)
        assertThat(MetalDetectorSonifier.confidenceToPingCadenceHz(1.0f)).isWithin(0.01f).of(14.0f)

        // Tone burst synthesis has click-free start/end envelope and strong peak energy
        val burst = MetalDetectorSonifier.synthesizeSinglePingBurstPcm16(diameterMm = 24f)
        assertThat(burst.first().toInt()).isEqualTo(0)
        assertThat(abs(burst.last().toInt())).isLessThan(200)
        assertThat(burst.maxOf { abs(it.toInt()) }).isGreaterThan(15_000)

        // Sonifier routes through Media stream (USAGE_MEDIA / CONTENT_TYPE_MUSIC) so hardware volume rocker controls volume
        assertThat(MetalDetectorSonifier.AUDIO_USAGE).isEqualTo(android.media.AudioAttributes.USAGE_MEDIA)
        assertThat(MetalDetectorSonifier.AUDIO_CONTENT_TYPE).isEqualTo(android.media.AudioAttributes.CONTENT_TYPE_MUSIC)
    }

    @Test
    fun rfDetrHostBridge_selectsTop100QueriesInDescendingScoreOrder() {
        val numAnchors = 900
        val numClasses = 5
        val hiddenDim = 16
        val topK = 100

        val logits = FloatArray(numAnchors * numClasses) { -4.0f }
        val features = FloatArray(numAnchors * hiddenDim) { idx -> (idx / hiddenDim).toFloat() }

        // Plant 3 unmistakable high-scoring anchors at indices 712, 42, and 899
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
    fun raycasterAndFieldMetadataJsonl_matchWaistHeight45DegreeGeometry() {
        val centerRay = GroundPlaneRaycaster.raycastTrigonometric(
            stripCenterXPx = 544f,
            stripCenterYPx = 192f,
            forwardPitchDeg = 45f,
            waistHeightMeters = 1.06f
        )

        // h = 1.06 m at 45 deg -> R ~ 1.50 m slant range, Z ~ 1.06 m ahead of boots, X = 0.0 m
        assertThat(centerRay.slantRangeMeters).isWithin(0.02f).of(1.50f)
        assertThat(centerRay.worldPoint.zMeters).isWithin(0.02f).of(1.06f)
        assertThat(centerRay.worldPoint.xMeters).isWithin(0.001f).of(0.0f)
        assertThat(centerRay.gsdMmPerPx).isWithin(0.02f).of(0.85f)

        val jsonLine = FieldDataLogger.buildMetadataJsonLine(
            sessionId = "session_20261114_100000",
            frameIndex = 1,
            fileName = "strip_000001.png",
            metadata = FieldStripMetadata(
                timestampNs = 123456789L,
                wallTimeMs = 1794650400000L,
                imuPitchDeg = 45.0f,
                slantRangeMeters = centerRay.slantRangeMeters,
                gsdMmPerPx = centerRay.gsdMmPerPx,
                horizontalSpeedMps = 0.65f,
                exposureTimeNs = 2_000_000L,
                sensorIso = 320,
                shutterClamped = true,
                exposureBiasEv = -0.5f
            )
        )
        val parsed = JSONObject(jsonLine)
        assertThat(parsed.getInt("strip_width")).isEqualTo(1088)
        assertThat(parsed.getInt("strip_height")).isEqualTo(384)
        assertThat(parsed.getLong("exposure_time_ns")).isAtMost(2_000_000L)
        assertThat(parsed.getDouble("slant_range_m")).isWithin(0.02).of(1.50)
    }
}
