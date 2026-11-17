/*
 * Unit tests for [PatrolDutyCycleController] verifying the 5.0-second PATROL_IDLE duty-cycling
 * transition and instant (< 50 ms / 0-frame) wake-up on >= 0.30 candidate score, touch, or Lock-On.
 *
 * Main types:
 * - [PatrolDutyCycleControllerTest]: JUnit4 test suite.
 */
package com.gneissfind.app.tracking

import com.gneissfind.app.model.AcquisitionStage
import com.gneissfind.app.model.OperatingMode
import com.gneissfind.app.model.PatrolPowerState
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

/**
 * Verifies [PatrolDutyCycleController] thermal/OLED state transitions across quiet sweeps,
 * `>= 0.30` candidate wake-ups, touch interactions, and mode/stage overrides.
 */
@RunWith(JUnit4::class)
class PatrolDutyCycleControllerTest {

    @Test
    fun entersPatrolIdleAfterFiveQuietSeconds_andWakesImmediatelyOnCandidateOrTouch() {
        val controller = PatrolDutyCycleController()
        val t0 = 10_000_000_000L

        // Initial quiet sweep starts the 5.0 s timer in PATROL_ACTIVE (30 FPS / 120 Hz)
        val s0 = controller.onSweepCompleted(
            maxAlertableScore = 0.12f,
            operatingMode = OperatingMode.SCAN,
            acquisitionStage = AcquisitionStage.PATROL,
            nowNs = t0
        )
        assertThat(s0).isEqualTo(PatrolPowerState.PATROL_ACTIVE)
        assertThat(s0.targetCameraFps).isEqualTo(30)
        assertThat(s0.preferredRefreshRateHz).isEqualTo(120f)

        // At +4.95 s with only < 0.30 background noise, still PATROL_ACTIVE
        val sBefore5s = controller.onSweepCompleted(
            maxAlertableScore = 0.29f,
            operatingMode = OperatingMode.SCAN,
            acquisitionStage = AcquisitionStage.PATROL,
            nowNs = t0 + 4_950_000_000L
        )
        assertThat(sBefore5s).isEqualTo(PatrolPowerState.PATROL_ACTIVE)

        // At +5.00 s with zero alertable candidates >= 0.30 -> transitions to PATROL_IDLE (15 FPS / 30 Hz)
        val sIdle = controller.onSweepCompleted(
            maxAlertableScore = 0.18f,
            operatingMode = OperatingMode.SCAN,
            acquisitionStage = AcquisitionStage.PATROL,
            nowNs = t0 + 5_000_000_000L
        )
        assertThat(sIdle).isEqualTo(PatrolPowerState.PATROL_IDLE)
        assertThat(sIdle.targetCameraFps).isEqualTo(15)
        assertThat(sIdle.preferredRefreshRateHz).isEqualTo(30f)

        // First candidate at score == 0.30 wakes immediately on that exact sweep
        val sWakeByCandidate = controller.onSweepCompleted(
            maxAlertableScore = 0.30f,
            operatingMode = OperatingMode.SCAN,
            acquisitionStage = AcquisitionStage.PATROL,
            nowNs = t0 + 5_066_000_000L
        )
        assertThat(sWakeByCandidate).isEqualTo(PatrolPowerState.PATROL_ACTIVE)

        // Re-enter PATROL_IDLE after another 5.0 s quiet period, then wake via user screen tap
        val sIdleAgain = controller.onSweepCompleted(
            maxAlertableScore = 0.0f,
            operatingMode = OperatingMode.SCAN,
            acquisitionStage = AcquisitionStage.PATROL,
            nowNs = t0 + 10_100_000_000L
        )
        assertThat(sIdleAgain).isEqualTo(PatrolPowerState.PATROL_IDLE)

        val sWakeByTap = controller.onUserInteraction(nowNs = t0 + 10_120_000_000L)
        assertThat(sWakeByTap).isEqualTo(PatrolPowerState.PATROL_ACTIVE)
    }

    @Test
    fun neverEntersPatrolIdleDuringLogDataOrLockOnModes() {
        val controller = PatrolDutyCycleController()
        val t0 = 1_000_000_000L

        controller.onSweepCompleted(0f, OperatingMode.LOG_DATA, AcquisitionStage.PATROL, t0)
        val logAfter10s = controller.onSweepCompleted(
            maxAlertableScore = 0f,
            operatingMode = OperatingMode.LOG_DATA,
            acquisitionStage = AcquisitionStage.PATROL,
            nowNs = t0 + 10_000_000_000L
        )
        assertThat(logAfter10s).isEqualTo(PatrolPowerState.PATROL_ACTIVE)

        val lockOnAfter20s = controller.onSweepCompleted(
            maxAlertableScore = 0f,
            operatingMode = OperatingMode.SCAN,
            acquisitionStage = AcquisitionStage.LOCK_ON_NEURAL,
            nowNs = t0 + 20_000_000_000L
        )
        assertThat(lockOnAfter20s).isEqualTo(PatrolPowerState.PATROL_ACTIVE)
    }
}
