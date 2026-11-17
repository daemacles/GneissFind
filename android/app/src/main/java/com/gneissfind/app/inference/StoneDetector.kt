/*
 * Detector abstraction: the boundary between the camera/cropping pipeline and the ML model.
 *
 * PushBroomCropper fills a [PushBroomSweepBuffers] with one sweep's tiles -> a [StoneDetector]
 * returns raw per-tile detections ([SweepInferenceResult]) -> PushBroomCropper merges them across
 * tiles. Implementations: RfDetrTwoGraphGpuDetector (on-device model) and SyntheticStoneDetector
 * (simulated detections for testing without a model).
 */
package com.gneissfind.app.inference

import com.gneissfind.app.model.DetectorBackend
import com.gneissfind.app.model.TileDetection
import com.gneissfind.app.vision.PushBroomSweepBuffers

/**
 * Result of running the detector on one sweep: raw detections plus timing telemetry.
 *
 * @property tileDetections Detections in tile-local pixel coordinates, each tagged with its tile
 *   index. Not yet merged across overlapping tiles.
 * @property part1LatencyMs Time for model stage 1 (Graph A: backbone/encoder), ms.
 * @property hostTopKBridgeLatencyMs Time for the CPU-side top-K selection between the two model
 *   stages, ms (AGENTS.md §5: done on the host to avoid slow GPU fallbacks).
 * @property part2LatencyMs Time for model stage 2 (Graph B: decoder head), ms.
 * @property totalSweepLatencyMs Wall time for the whole sweep, ms.
 * @property rollingAvgLatencyMs Smoothed average of recent sweep latencies, ms.
 * @property isHardwareGraphActive True if this result came from the real on-device model rather
 *   than a simulation.
 * @property maxAlertableCandidateScore Highest raw confidence score in `[0.0, 1.0]` observed on
 *   any user-alertable [com.gneissfind.app.model.StoneClass] query during this sweep (including
 *   sub-threshold `0.30..0.35` candidates used to wake `PATROL_IDLE`).
 */
data class SweepInferenceResult(
    val tileDetections: List<TileDetection>,
    val part1LatencyMs: Float,
    val hostTopKBridgeLatencyMs: Float,
    val part2LatencyMs: Float,
    val totalSweepLatencyMs: Float,
    val rollingAvgLatencyMs: Float,
    val isHardwareGraphActive: Boolean,
    val maxAlertableCandidateScore: Float = 0f
)

/**
 * Contract for a swappable stone detector: given one sweep's image tiles, return the stones
 * found in each tile.
 *
 * Input is the active push-broom strip already cut into overlapping square tiles; how many
 * depends on the active [com.gneissfind.app.model.SwathTileMode]. Cross-tile merging (NMS) is
 * not the detector's job; it happens downstream in PushBroomCropper.
 *
 * Threading: implementations must be called from a single, dedicated background analyzer thread.
 * The GPU implementation binds its model context to the thread that initializes it
 * (see RfDetrTwoGraphGpuDetector). Call [close] to release the implementation's resources.
 */
interface StoneDetector : AutoCloseable {
    /** Which implementation this is. */
    val backend: DetectorBackend

    /**
     * True when [detectSweep] can produce detections: for the GPU implementation, once its model
     * graphs have loaded. The synthetic implementation is always ready.
     */
    val isHardwareReady: Boolean

    /** Short human-readable status line. ScannerViewModel copies it into UI state as `detectorStatus`. */
    val statusSummary: String

    /**
     * Runs detection on every active tile of one sweep.
     *
     * @param sweepBuffers Preallocated buffers holding this sweep's RGB float tiles. Only the first
     *   [PushBroomSweepBuffers.activeTileCount] tiles are valid. The buffers are reused for the next
     *   sweep, so do not keep references to them.
     * @return Per-tile detections and latency telemetry; see [SweepInferenceResult].
     */
    fun detectSweep(sweepBuffers: PushBroomSweepBuffers): SweepInferenceResult

    override fun close()
}
