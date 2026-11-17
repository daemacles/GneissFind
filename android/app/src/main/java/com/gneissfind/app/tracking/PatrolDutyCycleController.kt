/*
 * Sensor and display thermal duty-cycling state machine for Stage 1 (PATROL) scan mode.
 *
 * Pipeline position: driven by ScannerViewModel on user touch interactions, mode changes, and
 * inference sweep completions; outputs PatrolPowerState to throttle Camera2 capture rate (30 FPS ->
 * 15 FPS) and display refresh rate (120 Hz -> 30 Hz) after 5.0 seconds without alertable stones.
 *
 * Main types:
 * - PatrolDutyCycleController: thread-safe state machine managing PATROL_ACTIVE <-> PATROL_IDLE transitions.
 */
package com.gneissfind.app.tracking

import com.gneissfind.app.model.AcquisitionStage
import com.gneissfind.app.model.OperatingMode
import com.gneissfind.app.model.PatrolPowerState

/**
 * Deterministic thermal and OLED power state machine governing Stage 1 (`PATROL`) camera and
 * display duty-cycling (`.agents/rules/android-camera-ar.md §2`).
 *
 * Behavior:
 * - Starts in [PatrolPowerState.PATROL_ACTIVE] (`30 FPS` camera, `120 Hz` display).
 * - Transitions to [PatrolPowerState.PATROL_IDLE] (`15 FPS` camera, `30 Hz` display, optional
 *   `90%` `#E6000000` OLED veil) after [idleTimeoutNs] (`5.0 s` default) of continuous `PATROL`
 *   operation with zero alertable candidate detections (`score >= wakeConfidenceThreshold`) and no
 *   screen touch events.
 * - Wakes instantaneously (`0` sweep delay, `< 50 ms` end-to-end) back to
 *   [PatrolPowerState.PATROL_ACTIVE] upon any alertable candidate (`score >= 0.30`), screen touch,
 *   switch to [OperatingMode.LOG_DATA], or entry into `LOCK-ON` / `PINPOINT`.
 *
 * Thread safety: all state transitions are `@Synchronized` on this instance.
 *
 * @param idleTimeoutNs Continuous quiet duration in nanoseconds before entering [PatrolPowerState.PATROL_IDLE].
 * @param wakeConfidenceThreshold Minimum raw query confidence (`0.30f` default) on an alertable
 *   mineral class required to reset the idle timer or wake from [PatrolPowerState.PATROL_IDLE].
 */
class PatrolDutyCycleController(
    private val idleTimeoutNs: Long = DEFAULT_IDLE_TIMEOUT_NS,
    private val wakeConfidenceThreshold: Float = DEFAULT_WAKE_CONFIDENCE_THRESHOLD
) {
    private var currentState: PatrolPowerState = PatrolPowerState.PATROL_ACTIVE
    private var lastActivityNs: Long = 0L

    /** Current [PatrolPowerState] (`PATROL_ACTIVE` or `PATROL_IDLE`). */
    val state: PatrolPowerState
        @Synchronized get() = currentState

    /**
     * Records a user UI/viewport touch at [nowNs] and wakes immediately to [PatrolPowerState.PATROL_ACTIVE].
     *
     * @param nowNs Monotonic timestamp in nanoseconds (`System.nanoTime()`).
     * @return Updated [PatrolPowerState] (always [PatrolPowerState.PATROL_ACTIVE]).
     */
    @Synchronized
    fun onUserInteraction(nowNs: Long): PatrolPowerState {
        if (nowNs > lastActivityNs || lastActivityNs == 0L) {
            lastActivityNs = nowNs
        }
        currentState = PatrolPowerState.PATROL_ACTIVE
        return currentState
    }

    /**
     * Evaluates mode or acquisition-stage changes at [nowNs], forcing [PatrolPowerState.PATROL_ACTIVE]
     * whenever field logging (`LOG_DATA`) or target Lock-On / Pinpoint is active.
     *
     * @param operatingMode Active top-level [OperatingMode].
     * @param acquisitionStage Active 3-stage [AcquisitionStage].
     * @param nowNs Monotonic timestamp in nanoseconds.
     * @return Updated [PatrolPowerState].
     */
    @Synchronized
    fun onContextChanged(
        operatingMode: OperatingMode,
        acquisitionStage: AcquisitionStage,
        nowNs: Long
    ): PatrolPowerState {
        if (nowNs > lastActivityNs || lastActivityNs == 0L) {
            lastActivityNs = nowNs
        }
        if (operatingMode != OperatingMode.SCAN || acquisitionStage != AcquisitionStage.PATROL) {
            currentState = PatrolPowerState.PATROL_ACTIVE
        }
        return currentState
    }

    /**
     * Updates the duty-cycle state after one completed `PATROL` sweep at [nowNs].
     *
     * @param maxAlertableScore Peak confidence score in `[0.0, 1.0]` among enabled user-alertable
     *   [com.gneissfind.app.model.StoneClass] candidates observed in this sweep.
     * @param operatingMode Active [OperatingMode].
     * @param acquisitionStage Active [AcquisitionStage].
     * @param nowNs Monotonic timestamp in nanoseconds (`System.nanoTime()`).
     * @return Updated [PatrolPowerState].
     */
    @Synchronized
    fun onSweepCompleted(
        maxAlertableScore: Float,
        operatingMode: OperatingMode,
        acquisitionStage: AcquisitionStage,
        nowNs: Long
    ): PatrolPowerState {
        if (operatingMode != OperatingMode.SCAN || acquisitionStage != AcquisitionStage.PATROL) {
            lastActivityNs = nowNs
            currentState = PatrolPowerState.PATROL_ACTIVE
            return currentState
        }

        if (maxAlertableScore >= wakeConfidenceThreshold) {
            lastActivityNs = nowNs
            currentState = PatrolPowerState.PATROL_ACTIVE
            return currentState
        }

        if (lastActivityNs == 0L || nowNs < lastActivityNs) {
            lastActivityNs = nowNs
            currentState = PatrolPowerState.PATROL_ACTIVE
            return currentState
        }

        if (nowNs - lastActivityNs >= idleTimeoutNs) {
            currentState = PatrolPowerState.PATROL_IDLE
        }
        return currentState
    }

    /** Resets the controller back to [PatrolPowerState.PATROL_ACTIVE] and clears the idle timer. */
    @Synchronized
    fun reset() {
        currentState = PatrolPowerState.PATROL_ACTIVE
        lastActivityNs = 0L
    }

    companion object {
        /** Quiet patrol duration (`5.0 s`) before throttling to `15 FPS` / `30 Hz`. */
        const val DEFAULT_IDLE_TIMEOUT_NS: Long = 5_000_000_000L

        /** Minimum alertable candidate confidence (`0.30`) that wakes `PATROL_IDLE` to `PATROL_ACTIVE`. */
        const val DEFAULT_WAKE_CONFIDENCE_THRESHOLD: Float = 0.30f
    }
}
