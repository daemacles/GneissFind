/*
 * Temporal target tracking: turns single-sweep stone detections into stable, confirmed targets.
 *
 * Pipeline position (called once per sweep by ScannerViewModel):
 *   PushBroomCropper (strip detections, cross-tile NMS)
 *     -> GroundPlaneRaycaster (each detection placed in world coordinates)
 *     -> TemporalTargetTracker (this file: N-of-M voting + walking-speed check)
 *     -> MetalDetectorSonifier (audio pings) and the LCARS UI (reticles)
 *
 * Defines the tracker's input type [ProjectedCandidate], its output type [TrackerSweepOutput],
 * and [TemporalTargetTracker] itself.
 */
package com.gneissfind.app.tracking

import com.gneissfind.app.model.ConfirmedTarget
import com.gneissfind.app.model.ScanSpeedState
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripDetection
import com.gneissfind.app.model.WorldPoint3D

/**
 * One detection from the current sweep after [GroundPlaneRaycaster] has placed it in the world.
 * This is the tracker's unit of input.
 *
 * @property stripDetection The detection in push-broom strip pixel coordinates, with class and
 *   confidence. Its box is passed through to [ConfirmedTarget] so the UI can draw it.
 * @property worldPosition Where the detection's ray meets the ground, in world coordinates
 *   (meters; y is vertical, as in [GroundPlaneRaycaster]). The tracker compares these positions to
 *   decide whether two detections are the same stone.
 * @property slantRangeMeters Straight-line camera-to-stone distance, in meters.
 * @property refinedDiameterMm Stone size estimate in millimeters: the detection box's longer side
 *   (px) times the ground sample distance (mm/px) at this stone's range (see ScannerViewModel).
 */
data class ProjectedCandidate(
    val stripDetection: StripDetection,
    val worldPosition: WorldPoint3D,
    val slantRangeMeters: Float,
    val refinedDiameterMm: Float
)

/**
 * What [TemporalTargetTracker.updateSweep] reports after each sweep.
 *
 * @property confirmedTargets Tracks currently at or above the vote threshold, strongest first
 *   (most votes, then highest smoothed confidence). Empty when nothing is confirmed.
 * @property speedState The user's horizontal walking speed and whether it exceeds the limit.
 * @property totalCandidateTracks Number of tracks alive after this sweep, confirmed or not.
 */
data class TrackerSweepOutput(
    val confirmedTargets: List<ConfirmedTarget>,
    val speedState: ScanSpeedState,
    val totalCandidateTracks: Int
)

/**
 * Turns noisy per-sweep detections into confirmed stone targets by requiring each stone to be
 * re-detected at the same place on the ground in several recent sweeps ("N-of-M voting",
 * AGENTS.md §5). Also classifies the user's walking speed against the over-speed limit that
 * drives the "SLOW SWEEP" warning.
 *
 * State: a list of tracks, one per suspected stone. Each track remembers its smoothed world
 * position, class, confidence, diameter, latest strip box, and the recent sweeps it was seen in.
 * A track that stops being re-detected ages out on its own once its votes leave the window.
 *
 * Votes are counted in sweeps, not seconds, so the window's wall-clock length depends on the
 * sweep rate.
 *
 * Thread safety: [updateSweep] and [reset] are synchronized on this instance.
 *
 * @param minVotesToConfirm N: hits within the window needed for a track to be reported as confirmed.
 * @param rollingWindowSweeps M: how many of the most recent sweeps count toward votes.
 * @param clusterRadiusMeters Maximum 3D world-space distance (m) at which a detection and an
 *   existing track are treated as the same stone.
 * @param maxWalkSpeedMps Horizontal speed (m/s) above which [updateSweep] reports
 *   [ScanSpeedState.SpeedWarning].
 */
class TemporalTargetTracker(
    private val minVotesToConfirm: Int = DEFAULT_MIN_VOTES,
    private val rollingWindowSweeps: Int = DEFAULT_WINDOW_SWEEPS,
    private val clusterRadiusMeters: Float = DEFAULT_CLUSTER_RADIUS_METERS,
    private val maxWalkSpeedMps: Float = MAX_SWEEP_SPEED_MPS
) {
    /**
     * Mutable state for one suspected stone. Internal; copied into a [ConfirmedTarget] for output.
     *
     * - [stoneClass] never changes after creation: detections only match tracks of their own class.
     * - [worldPosition], [smoothedConfidence] and [smoothedDiameterMm] are exponential moving
     *   averages over the track's hits.
     * - The strip box (`stripLeftPx`..`stripBottomPx`) and [slantRangeMeters] are copied from
     *   the latest hit, not smoothed.
     * - [hitSweepIds] holds the sweep numbers in which this track was matched, oldest first, at
     *   most one per sweep. Its size is the track's vote count.
     */
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

    /** All live tracks, confirmed or not. */
    private val activeTracks = ArrayList<TrackEntry>()

    /** Id for the next new track. Never reset, so ids stay unique for this tracker's lifetime. */
    private var nextTrackId = 1L

    /** Number of the most recent sweep; incremented at the start of each [updateSweep]. */
    private var currentSweepSeq = 0L

    /** Previous camera position and its timestamp; used only by [computeHorizontalSpeedFromPose]. */
    private var prevCameraPos: WorldPoint3D? = null
    private var prevPoseTimestampNs: Long = 0L

    /**
     * Processes one sweep: matches its detections to tracks, ages out old votes, and returns the
     * currently confirmed targets and the walking-speed state.
     *
     * Call exactly once per sweep, including sweeps with no detections. Votes age by sweep count,
     * so skipped calls make old votes last longer.
     *
     * @param candidates This sweep's detections in world coordinates, already merged across tiles
     *   and filtered upstream. Candidates whose class is not [StoneClass.isUserAlertable] are
     *   ignored here as well.
     * @param timestampNs Capture time of the sweep's camera frame, in nanoseconds. Recorded on
     *   every track this sweep updates, and used for pose-derived speed.
     * @param cameraWorldPos Camera position in world coordinates (m). Used only when
     *   [currentSpeedMps] is null.
     * @param currentSpeedMps Horizontal walking speed in m/s, if the caller already has it
     *   (ScannerViewModel passes the pose tracker's speed). If null, speed is estimated from the
     *   change in [cameraWorldPos] since the previous call.
     * @return Confirmed targets, speed state and live-track count; see [TrackerSweepOutput].
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

        // 2. Drop votes older than the rolling window
        for (track in activeTracks) {
            while (track.hitSweepIds.isNotEmpty() && track.hitSweepIds.first() < minValidSweepId) {
                track.hitSweepIds.removeFirst()
            }
        }

        // 3. Associate candidates with tracks. Greedy nearest neighbor in input order: each
        //    candidate claims the closest unclaimed same-class track within clusterRadiusMeters,
        //    otherwise it starts a new track. An earlier candidate can therefore claim a track
        //    that a later candidate was closer to.
        // Tracks already claimed this sweep, so that each track gets at most one vote per sweep.
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
                // Exponential moving average of world position, confidence and diameter.
                // RATIONALE-UNKNOWN: how were these smoothing weights chosen? Not recorded in
                // AGENTS.md or docs/.
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
                // No match: start a new track with this detection as its first vote. It is marked
                // as claimed, so a second same-class candidate within the radius in this sweep
                // starts its own track instead of adding a second vote.
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

        // 4. Remove tracks that have no votes left in the window
        activeTracks.removeAll { it.hitSweepIds.isEmpty() }

        // 5. Report every track with at least minVotesToConfirm votes. Confirmation is not
        //    latched: a track drops out of the output when its votes age below the threshold.
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

    /**
     * Forgets all tracks, the sweep counter, and the previous camera pose, e.g. when scanning
     * restarts. Track ids keep counting up, so ids from before the reset are never reused.
     */
    @Synchronized
    fun reset() {
        activeTracks.clear()
        currentSweepSeq = 0L
        prevCameraPos = null
        prevPoseTimestampNs = 0L
    }

    /**
     * Estimates horizontal walking speed (m/s) from the camera's movement since the previous call.
     * Horizontal means the x-z plane, because y is vertical in the world frame.
     *
     * Side effect: stores [cameraWorldPos] and [timestampNs] as the previous pose for the next call.
     * Called only when [updateSweep] gets no speed from the caller, so the stored pose can be old.
     * The time-gap guard below discards such stale pairs.
     *
     * @return Speed in m/s, or 0 when it cannot be estimated: no position, first call, timestamps
     *   not increasing, or a time gap outside the accepted range. 0 therefore means either
     *   "standing still" or "unknown".
     */
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
        /** Default N for N-of-M voting (AGENTS.md §5: default 3 of 6 sweeps). */
        const val DEFAULT_MIN_VOTES = 3

        /** Default M for N-of-M voting (AGENTS.md §5). */
        const val DEFAULT_WINDOW_SWEEPS = 6

        /**
         * Default association radius in meters, in 3D world space.
         * RATIONALE-UNKNOWN: how was this radius chosen? Not specified in AGENTS.md or docs/.
         */
        const val DEFAULT_CLUSTER_RADIUS_METERS = 0.040f

        /** Walking speed (m/s) above which the "SLOW SWEEP" warning is raised (AGENTS.md §5). */
        const val MAX_SWEEP_SPEED_MPS = 1.10f
    }
}
