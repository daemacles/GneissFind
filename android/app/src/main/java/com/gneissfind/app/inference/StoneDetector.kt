package com.gneissfind.app.inference

import com.gneissfind.app.model.DetectorBackend
import com.gneissfind.app.model.TileDetection
import com.gneissfind.app.vision.PushBroomSweepBuffers

/**
 * Per-sweep inference output and stage-level latency breakdown (in milliseconds/sweep).
 */
data class SweepInferenceResult(
    val tileDetections: List<TileDetection>,
    val part1LatencyMs: Float,
    val hostTopKBridgeLatencyMs: Float,
    val part2LatencyMs: Float,
    val totalSweepLatencyMs: Float,
    val rollingAvgLatencyMs: Float,
    val isHardwareGraphActive: Boolean
)

/**
 * Modular on-device stone detection interface operating on 3 overlapping 384x384 push-broom tiles.
 * Implementations MUST be invoked from a single confined background analyzer thread.
 */
interface StoneDetector : AutoCloseable {
    val backend: DetectorBackend
    val isHardwareReady: Boolean
    val statusSummary: String

    fun detectSweep(sweepBuffers: PushBroomSweepBuffers): SweepInferenceResult

    override fun close()
}
