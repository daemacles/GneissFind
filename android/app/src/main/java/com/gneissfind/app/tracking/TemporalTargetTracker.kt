package com.gneissfind.app.tracking

import com.gneissfind.app.model.ConfirmedTarget
import com.gneissfind.app.model.ScanSpeedState
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripDetection
import com.gneissfind.app.model.WorldPoint3D

/**
 * Candidate stone observation projected into 3D ground coordinates for a single sweep.
 */
data class ProjectedCandidate(
    val stripDetection: StripDetection,
    val worldPosition: WorldPoint3D,
    val slantRangeMeters: Float,
    val refinedDiameterMm: Float
)

/**
 * Output of a single sweep update in [TemporalTargetTracker].
 */
data class TrackerSweepOutput(
    val confirmedTargets: List<ConfirmedTarget>,
    val speedState: ScanSpeedState,
    val totalCandidateTracks: Int
)

/**
 * World-locked multi-frame spatial voter & ARCore/VIO walk-speed gate:
 *  - Maintains a rolling window of `M = 6` sweeps (`windowSize = 6`).
 *  - Associates detections falling within a `40 mm` (`0.040 m`) 3D ground radius.
 *  - Promotes a candidate stone to [ConfirmedTarget] ONLY after `N >= 3` hits within the rolling
 *    6-sweep window (`3-of-6` temporal voting).
 *  - Monitors horizontal ground translation speed `v_xy`: triggers [ScanSpeedState.SpeedWarning]
 *    (`"SLOW SWEEP"`) whenever `v_xy > 1.10 m/s`.
 */
class TemporalTargetTracker(
    private val minVotesToConfirm: Int = DEFAULT_MIN_VOTES,
    private val rollingWindowSweeps: Int = DEFAULT_WINDOW_SWEEPS,
    private val clusterRadiusMeters: Float = DEFAULT_CLUSTER_RADIUS_METERS,
    private val maxWalkSpeedMps: Float = MAX_SWEEP_SPEED_MPS
) {
    private class TrackEntry(
        val trackId: Long,
        var stoneClass: StoneClass,
        var worldPosition: WorldPoint3D,
        var stripLeftPx: Float,
        var stripTopPx: Float,
        var stripRightPx: Float,
        var stripBottomPx: Float,
        var smoothedConfidence: Float,
        var smoothedDiameterMm: Float,
        var slantRangeMeters: Float,
        var lastTimestampNs: Long,
        val hitSweepIds: ArrayDeque<Long> = ArrayDeque()
    )

    private val activeTracks = ArrayList<TrackEntry>()
    private var nextTrackId = 1L
    private var currentSweepSeq = 0L

    private var prevCameraPos: WorldPoint3D? = null
    private var prevPoseTimestampNs: Long = 0L

    /**
     * Updates world-locked spatial clusters for a new push-broom sweep.
     *
     * @param candidates Raycasted strip detections for this sweep (already post-NMS and confuser-filtered).
     * @param currentSpeedMps Instantaneous horizontal ground speed `v_xy` in m/s (if `null`, computed
     *   directly from consecutive [cameraWorldPos] and [timestampNs]).
     */
    @Synchronized
    fun updateSweep(
        candidates: List<ProjectedCandidate>,
        timestampNs: Long,
        cameraWorldPos: WorldPoint3D? = null,
        currentSpeedMps: Float? = null
    ): TrackerSweepOutput {
        val sweepId = ++currentSweepSeq
        val minValidSweepId = sweepId - rollingWindowSweeps + 1

        // 1. Determine horizontal ground speed v_xy
        val measuredSpeedMps = currentSpeedMps ?: computeHorizontalSpeedFromPose(cameraWorldPos, timestampNs)
        val speedState = if (measuredSpeedMps > maxWalkSpeedMps) {
            ScanSpeedState.SpeedWarning(measuredSpeedMps)
        } else {
            ScanSpeedState.Normal(measuredSpeedMps)
        }

        // 2. Evict expired sweep votes outside the rolling 6-sweep window
        for (track in activeTracks) {
            while (track.hitSweepIds.isNotEmpty() && track.hitSweepIds.first() < minValidSweepId) {
                track.hitSweepIds.removeFirst()
            }
        }

        // 3. Match incoming 3D candidates within 40 mm (0.040 m) spatial radius
        val matchedTrackIdsThisSweep = HashSet<Long>(candidates.size)

        for (cand in candidates) {
            val det = cand.stripDetection
            if (!det.stoneClass.isUserAlertable) continue

            var bestTrack: TrackEntry? = null
            var bestDistM = clusterRadiusMeters

            for (track in activeTracks) {
                if (track.trackId in matchedTrackIdsThisSweep) continue
                if (track.stoneClass != det.stoneClass) continue

                val distM = track.worldPosition.distanceMetersTo(cand.worldPosition)
                if (distM <= bestDistM) {
                    bestDistM = distM
                    bestTrack = track
                }
            }

            if (bestTrack != null) {
                matchedTrackIdsThisSweep.add(bestTrack.trackId)
                // Exponential moving average smoothing of 3D ground coordinate, confidence, and metric diameter
                bestTrack.worldPosition = WorldPoint3D(
                    xMeters = bestTrack.worldPosition.xMeters * 0.65f + cand.worldPosition.xMeters * 0.35f,
                    yMeters = bestTrack.worldPosition.yMeters * 0.65f + cand.worldPosition.yMeters * 0.35f,
                    zMeters = bestTrack.worldPosition.zMeters * 0.65f + cand.worldPosition.zMeters * 0.35f
                )
                bestTrack.stripLeftPx = det.leftPx
                bestTrack.stripTopPx = det.topPx
                bestTrack.stripRightPx = det.rightPx
                bestTrack.stripBottomPx = det.bottomPx
                bestTrack.smoothedConfidence = bestTrack.smoothedConfidence * 0.60f + det.confidence * 0.40f
                bestTrack.smoothedDiameterMm = bestTrack.smoothedDiameterMm * 0.60f + cand.refinedDiameterMm * 0.40f
                bestTrack.slantRangeMeters = cand.slantRangeMeters
                bestTrack.lastTimestampNs = timestampNs
                if (bestTrack.hitSweepIds.lastOrNull() != sweepId) {
                    bestTrack.hitSweepIds.addLast(sweepId)
                }
            } else {
                val newId = nextTrackId++
                matchedTrackIdsThisSweep.add(newId)
                val entry = TrackEntry(
                    trackId = newId,
                    stoneClass = det.stoneClass,
                    worldPosition = cand.worldPosition,
                    stripLeftPx = det.leftPx,
                    stripTopPx = det.topPx,
                    stripRightPx = det.rightPx,
                    stripBottomPx = det.bottomPx,
                    smoothedConfidence = det.confidence,
                    smoothedDiameterMm = cand.refinedDiameterMm,
                    slantRangeMeters = cand.slantRangeMeters,
                    lastTimestampNs = timestampNs
                )
                entry.hitSweepIds.addLast(sweepId)
                activeTracks.add(entry)
            }
        }

        // 4. Prune stale tracks with zero hits remaining in the rolling 6-sweep window
        activeTracks.removeAll { it.hitSweepIds.isEmpty() }

        // 5. Promote tracks meeting N-of-M (3-of-6) spatial votes
        val confirmed = activeTracks
            .filter { it.hitSweepIds.size >= minVotesToConfirm }
            .map { track ->
                ConfirmedTarget(
                    trackId = track.trackId,
                    stoneClass = track.stoneClass,
                    worldPosition = track.worldPosition,
                    stripLeftPx = track.stripLeftPx,
                    stripTopPx = track.stripTopPx,
                    stripRightPx = track.stripRightPx,
                    stripBottomPx = track.stripBottomPx,
                    smoothedConfidence = track.smoothedConfidence,
                    estimatedDiameterMm = track.smoothedDiameterMm,
                    slantRangeMeters = track.slantRangeMeters,
                    hitCountInWindow = track.hitSweepIds.size,
                    windowSize = rollingWindowSweeps,
                    lastUpdatedTimestampNs = track.lastTimestampNs
                )
            }
            .sortedWith(
                compareByDescending<ConfirmedTarget> { it.hitCountInWindow }
                    .thenByDescending { it.smoothedConfidence }
            )

        return TrackerSweepOutput(
            confirmedTargets = confirmed,
            speedState = speedState,
            totalCandidateTracks = activeTracks.size
        )
    }

    @Synchronized
    fun reset() {
        activeTracks.clear()
        currentSweepSeq = 0L
        prevCameraPos = null
        prevPoseTimestampNs = 0L
    }

    private fun computeHorizontalSpeedFromPose(
        cameraWorldPos: WorldPoint3D?,
        timestampNs: Long
    ): Float {
        if (cameraWorldPos == null) return 0f
        val prev = prevCameraPos
        val prevTs = prevPoseTimestampNs
        prevCameraPos = cameraWorldPos
        prevPoseTimestampNs = timestampNs

        if (prev == null || prevTs == 0L || timestampNs <= prevTs) return 0f
        val dtSec = (timestampNs - prevTs) * 1e-9f
        if (dtSec <= 1e-5f || dtSec > 2.0f) return 0f

        val dx = cameraWorldPos.xMeters - prev.xMeters
        val dz = cameraWorldPos.zMeters - prev.zMeters
        return kotlin.math.sqrt(dx * dx + dz * dz) / dtSec
    }

    companion object {
        const val DEFAULT_MIN_VOTES = 3
        const val DEFAULT_WINDOW_SWEEPS = 6
        const val DEFAULT_CLUSTER_RADIUS_METERS = 0.040f // 40 mm 3D radius
        const val MAX_SWEEP_SPEED_MPS = 1.10f            // 1.1 m/s speed gate
    }
}
