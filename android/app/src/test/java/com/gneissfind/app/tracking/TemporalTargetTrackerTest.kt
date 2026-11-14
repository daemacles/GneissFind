package com.gneissfind.app.tracking

import com.gneissfind.app.model.ScanSpeedState
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripDetection
import com.gneissfind.app.model.WorldPoint3D
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class TemporalTargetTrackerTest {

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
}
