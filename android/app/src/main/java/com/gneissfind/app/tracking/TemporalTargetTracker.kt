/*
 * Temporal target tracking: turns single-sweep stone detections into stable, confirmed targets.
 *
 * Pipeline position (called once per sweep by ScannerViewModel):
 *   PushBroomCropper (strip detections, cross-tile NMS)
 *     -> GroundPlaneRaycaster (each detection placed in world coordinates + 3D IMU world ray)
 *     -> TemporalTargetTracker (this file: world-bearing association, N-of-M voting + hysteresis, speed check)
 *     -> MetalDetectorSonifier (audio pings) and the LCARS UI (120 Hz ego-motion-locked reticles)
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
import com.gneissfind.app.model.WorldRay3D
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * One detection from the current sweep after [com.gneissfind.app.vision.GroundPlaneRaycaster] has placed it in the world.
 * This is the tracker's unit of input.
 *
 * @property stripDetection The detection in push-broom strip pixel coordinates, with class and
 *   confidence. Its box is passed through to [ConfirmedTarget] so the UI can draw it.
 * @property worldPosition Where the detection's ray meets the ground, in world coordinates
 *   (meters; y is vertical, as in [com.gneissfind.app.vision.GroundPlaneRaycaster]).
 * @property slantRangeMeters Straight-line camera-to-stone distance, in meters.
 * @property refinedDiameterMm Stone size estimate in millimeters: the detection box's longer side
 *   (px) times the ground sample distance (mm/px) at this stone's range (see ScannerViewModel).
 * @property worldRay Optional unit 3D bearing vector in the `TYPE_GAME_ROTATION_VECTOR` world
 *   frame (`R_world_from_scr(t_capture) * d_scr`). When present, track association is invariant
 *   to handheld pitch, yaw, and roll between sweeps, and [ConfirmedTarget] reticles can be
 *   reprojected at 120 Hz.
 * @property angularWidthRad Horizontal angular width (`widthPx / focalLengthPx`, in radians).
 * @property angularHeightRad Vertical angular height (`heightPx / focalLengthPx`, in radians).
 */
data class ProjectedCandidate(
    val stripDetection: StripDetection,
    val worldPosition: WorldPoint3D,
    val slantRangeMeters: Float,
    val refinedDiameterMm: Float,
    val worldRay: WorldRay3D? = null,
    val angularWidthRad: Float = 0f,
    val angularHeightRad: Float = 0f
)

/**
 * What [TemporalTargetTracker.updateSweep] reports after each sweep.
 *
 * @property confirmedTargets Tracks currently confirmed (by N-of-M voting, fast consecutive
 *   world-ray lock, or confirmation hysteresis), strongest first (most votes, then highest
 *   smoothed confidence). Empty when nothing is confirmed.
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
 * re-detected at the same world bearing / ground location across recent sweeps ("N-of-M voting",
 * `AGENTS.md` §5). Also classifies the user's walking speed against the over-speed limit that
 * drives the "SLOW SWEEP" warning.
 *
 * State: a list of tracks, one per suspected stone. Each track remembers its smoothed world
 * position, 3D world bearing ray, angular box size, class, confidence, diameter, latest strip box,
 * confirmation hysteresis latch, and the recent sweeps it was seen in. A track that stops being
 * re-detected ages out on its own once its votes leave the window.
 *
 * Thread safety: [updateSweep] and [reset] are synchronized on this instance.
 *
 * @param minVotesToConfirm N: hits within the window needed for a track to promote to confirmed ([DEFAULT_MIN_VOTES]).
 * @param rollingWindowSweeps M: how many of the most recent sweeps count toward votes ([DEFAULT_WINDOW_SWEEPS]).
 * @param clusterRadiusMeters Baseline 3D world-space distance (m) at which a detection and an
 *   existing track are treated as the same stone ([DEFAULT_CLUSTER_RADIUS_METERS]).
 * @param maxWalkSpeedMps Horizontal speed (m/s) above which [updateSweep] reports
 *   [ScanSpeedState.SpeedWarning] ([MAX_SWEEP_SPEED_MPS]).
 * @param maxAngularDistanceDeg Maximum world-frame angular separation (degrees) between two
 *   [WorldRay3D] vectors to associate them as the same physical target across camera rotations ([DEFAULT_MAX_ANGULAR_DISTANCE_DEG]).
 * @param enableFastConsecutiveLock When true, a world-ray-anchored target detected on [FAST_LOCK_CONSECUTIVE_HITS]
 *   consecutive sweeps (`sweepId - 1` and `sweepId`) with smoothed confidence `>=` [FAST_LOCK_MIN_CONFIDENCE]
 *   promotes immediately so ~2.2 Hz GPU sweeps lock in ~0.4 s instead of ~1.2 s while still
 *   rejecting single-frame false positives.
 */
class TemporalTargetTracker(
    private val minVotesToConfirm: Int = DEFAULT_MIN_VOTES,
    private val rollingWindowSweeps: Int = DEFAULT_WINDOW_SWEEPS,
    private val clusterRadiusMeters: Float = DEFAULT_CLUSTER_RADIUS_METERS,
    private val maxWalkSpeedMps: Float = MAX_SWEEP_SPEED_MPS,
    private val maxAngularDistanceDeg: Float = DEFAULT_MAX_ANGULAR_DISTANCE_DEG,
    private val enableFastConsecutiveLock: Boolean = true
) {
    /**
     * Mutable state for one suspected stone. Internal; copied into a [ConfirmedTarget] for output.
     *
     * - [stoneClass] never changes after creation: detections only match tracks of their own class.
     * - [worldPosition], [worldRay], [angularWidthRad], [angularHeightRad], [smoothedConfidence],
     *   and [smoothedDiameterMm] are exponential moving averages over the track's hits.
     * - [isConfirmedLatch] provides confirmation hysteresis: once a track promotes (at
     *   `>= minVotesToConfirm` or via [FAST_LOCK_CONSECUTIVE_HITS] consecutive high-confidence world-ray hits), it stays
     *   confirmed as long as it retains at least `minVotesToMaintain` ([DEFAULT_MIN_VOTES_TO_MAINTAIN]) hits in the window.
     * - [hitSweepIds] holds the sweep numbers in which this track was matched, oldest first, at
     *   most one per sweep. Its size is the track's vote count.
     */
    private class TrackEntry(
        val trackId: Long,
        var stoneClass: StoneClass,
        var worldPosition: WorldPoint3D,
        var worldRay: WorldRay3D?,
        var angularWidthRad: Float,
        var angularHeightRad: Float,
        var stripLeftPx: Float,
        var stripTopPx: Float,
        var stripRightPx: Float,
        var stripBottomPx: Float,
        var smoothedConfidence: Float,
        var smoothedDiameterMm: Float,
        var slantRangeMeters: Float,
        var lastTimestampNs: Long,
        var isConfirmedLatch: Boolean = false,
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

        // 3. Associate candidates with tracks. When 3D IMU world rays are present, match by
        //    world-frame angular bearing (<= maxAngularDistanceDeg) or range-scaled 3D ground
        //    distance so handheld wrist pitch/yaw rotation never splits a stationary object into
        //    disconnected tracks. When worldRay is null (synthetic/unit tests), match strictly
        //    within clusterRadiusMeters.
        val matchedTrackIdsThisSweep = HashSet<Long>(candidates.size)

        for (cand in candidates) {
            val det = cand.stripDetection
            if (!det.stoneClass.isUserAlertable) continue

            var bestTrack: TrackEntry? = null
            var bestNormalizedScore = 1.0f

            for (track in activeTracks) {
                if (track.trackId in matchedTrackIdsThisSweep) continue
                if (track.stoneClass != det.stoneClass) continue

                val candRay = cand.worldRay
                val trackRay = track.worldRay
                if (candRay != null && trackRay != null) {
                    val angDeg = trackRay.angularDistanceDegTo(candRay)
                    val distM = track.worldPosition.distanceMetersTo(cand.worldPosition)
                    val rangeScaledRadiusM = max(
                        clusterRadiusMeters,
                        cand.slantRangeMeters * RANGE_SCALED_CLUSTER_RATIO
                    )
                    if (angDeg <= maxAngularDistanceDeg || distM <= rangeScaledRadiusM) {
                        val normScore = min(
                            angDeg / maxAngularDistanceDeg.coerceAtLeast(0.1f),
                            distM / rangeScaledRadiusM.coerceAtLeast(0.01f)
                        )
                        if (normScore <= bestNormalizedScore) {
                            bestNormalizedScore = normScore
                            bestTrack = track
                        }
                    }
                } else {
                    val distM = track.worldPosition.distanceMetersTo(cand.worldPosition)
                    if (distM <= clusterRadiusMeters) {
                        val normScore = distM / clusterRadiusMeters.coerceAtLeast(1e-4f)
                        if (normScore <= bestNormalizedScore) {
                            bestNormalizedScore = normScore
                            bestTrack = track
                        }
                    }
                }
            }

            if (bestTrack != null) {
                matchedTrackIdsThisSweep.add(bestTrack.trackId)
                val posWeight = if (cand.worldRay != null) 0.55f else 0.35f
                val posKeep = 1.0f - posWeight
                bestTrack.worldPosition = WorldPoint3D(
                    xMeters = bestTrack.worldPosition.xMeters * posKeep + cand.worldPosition.xMeters * posWeight,
                    yMeters = bestTrack.worldPosition.yMeters * posKeep + cand.worldPosition.yMeters * posWeight,
                    zMeters = bestTrack.worldPosition.zMeters * posKeep + cand.worldPosition.zMeters * posWeight
                )
                if (cand.worldRay != null) {
                    bestTrack.worldRay = bestTrack.worldRay
                        ?.blendAndNormalize(cand.worldRay, WORLD_RAY_NEW_WEIGHT)
                        ?: cand.worldRay.normalized()
                }
                if (cand.angularWidthRad > 0f && cand.angularHeightRad > 0f) {
                    bestTrack.angularWidthRad = if (bestTrack.angularWidthRad > 0f) {
                        bestTrack.angularWidthRad * 0.50f + cand.angularWidthRad * 0.50f
                    } else {
                        cand.angularWidthRad
                    }
                    bestTrack.angularHeightRad = if (bestTrack.angularHeightRad > 0f) {
                        bestTrack.angularHeightRad * 0.50f + cand.angularHeightRad * 0.50f
                    } else {
                        cand.angularHeightRad
                    }
                }
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
                    worldRay = cand.worldRay?.normalized(),
                    angularWidthRad = cand.angularWidthRad,
                    angularHeightRad = cand.angularHeightRad,
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

        // 5. Update confirmation latch with hysteresis:
        //    - Promotes when hits >= minVotesToConfirm (3-of-6) OR on 2 consecutive high-confidence
        //      world-ray-anchored hits.
        //    - Once promoted, stays confirmed while hits >= minVotesToMaintain (2) so a momentary
        //      missed sweep during camera motion does not drop an active lock.
        val minVotesToMaintain = min(DEFAULT_MIN_VOTES_TO_MAINTAIN, minVotesToConfirm)
        for (track in activeTracks) {
            updateTrackConfirmationLatch(track, sweepId, minVotesToMaintain)
        }

        // 6. Track-Level Spatial NMS (.agents/rules/android-camera-ar.md §3):
        //    Merge/suppress duplicate tracks of the same StoneClass when within <= 3.2 deg world-ray
        //    separation, <= 75 mm 3D ground distance, or strip box IoU >= 0.35 / IoM >= 0.60 so a
        //    single physical target never spawns overlapping reticles.
        suppressAndMergeDuplicateTracks(sweepId, minVotesToMaintain)

        val confirmed = activeTracks
            .filter { it.isConfirmedLatch }
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
                    lastUpdatedTimestampNs = track.lastTimestampNs,
                    worldRay = track.worldRay,
                    angularWidthRad = track.angularWidthRad,
                    angularHeightRad = track.angularHeightRad,
                    isVisibleInStrip = true
                )
            }
            .sortedWith(
                compareByDescending<ConfirmedTarget> { it.hitCountInWindow }
                    .thenByDescending { it.smoothedConfidence }
                    .thenBy { it.trackId }
            )

        return TrackerSweepOutput(
            confirmedTargets = confirmed,
            speedState = speedState,
            totalCandidateTracks = activeTracks.size
        )
    }

    /** Updates [TrackEntry.isConfirmedLatch] according to N-of-M voting, fast 2-hit lock, and hysteresis. */
    private fun updateTrackConfirmationLatch(
        track: TrackEntry,
        sweepId: Long,
        minVotesToMaintain: Int
    ) {
        val hits = track.hitSweepIds.size
        if (hits >= minVotesToConfirm) {
            track.isConfirmedLatch = true
        } else if (
            enableFastConsecutiveLock &&
            track.worldRay != null &&
            hits >= FAST_LOCK_CONSECUTIVE_HITS &&
            track.smoothedConfidence >= FAST_LOCK_MIN_CONFIDENCE &&
            hasTwoConsecutiveRecentHits(track.hitSweepIds, sweepId)
        ) {
            track.isConfirmedLatch = true
        } else if (track.isConfirmedLatch && hits < minVotesToMaintain) {
            track.isConfirmedLatch = false
        }
    }

    /**
     * Runs greedy Track-Level Spatial NMS across [activeTracks], keeping confirmed, higher-vote,
     * higher-confidence, older-ID tracks and absorbing duplicate same-class tracks within
     * [TRACK_NMS_MAX_ANGULAR_DEG], [TRACK_NMS_MAX_DISTANCE_METERS], or same-pose strip
     * [TRACK_NMS_MIN_IOU] / [TRACK_NMS_MIN_IOM].
     */
    private fun suppressAndMergeDuplicateTracks(
        sweepId: Long,
        minVotesToMaintain: Int
    ) {
        if (activeTracks.size <= 1) return

        activeTracks.sortWith(
            compareByDescending<TrackEntry> { it.isConfirmedLatch }
                .thenByDescending { it.hitSweepIds.size }
                .thenByDescending { it.smoothedConfidence }
                .thenBy { it.trackId }
        )

        val survivors = ArrayList<TrackEntry>(activeTracks.size)
        for (candidate in activeTracks) {
            var mergedInto: TrackEntry? = null
            for (winner in survivors) {
                if (winner.stoneClass != candidate.stoneClass) continue
                if (areTracksSpatialDuplicates(winner, candidate)) {
                    mergedInto = winner
                    break
                }
            }
            if (mergedInto != null) {
                mergeDuplicateIntoWinner(mergedInto, candidate, sweepId, minVotesToMaintain)
            } else {
                survivors.add(candidate)
            }
        }

        if (survivors.size != activeTracks.size) {
            activeTracks.clear()
            activeTracks.addAll(survivors)
        }
    }

    /**
     * Returns `true` when [a] and [b] represent the same physical target under any of the three
     * Milestone 2C Track-Level Spatial NMS criteria:
     * 1. World-ray angular separation `<= 3.2 deg`,
     * 2. 3D ground-plane distance `<= 75 mm` (`0.075 m`), or
     * 3. Strip bounding-box overlap `IoU >= 0.35` or `IoM >= 0.60` (evaluated when at least one
     *    track lacks a 3D world ray or when both strip boxes come from the same sweep / nearby
     *    world bearing so camera rotation across sweeps cannot falsely merge far-apart stones).
     */
    private fun areTracksSpatialDuplicates(a: TrackEntry, b: TrackEntry): Boolean {
        val rayA = a.worldRay
        val rayB = b.worldRay
        if (rayA != null && rayB != null) {
            if (rayA.angularDistanceDegTo(rayB) <= TRACK_NMS_MAX_ANGULAR_DEG) {
                return true
            }
        }
        if (a.worldPosition.distanceMetersTo(b.worldPosition) <= TRACK_NMS_MAX_DISTANCE_METERS) {
            return true
        }

        val allowStripBoxCheck = rayA == null ||
            rayB == null ||
            a.lastTimestampNs == b.lastTimestampNs ||
            rayA.angularDistanceDegTo(rayB) <= DEFAULT_MAX_ANGULAR_DISTANCE_DEG * 1.5f
        if (allowStripBoxCheck) {
            val (iou, iom) = computeTrackBoxIouAndIom(a, b)
            if (iou >= TRACK_NMS_MIN_IOU || iom >= TRACK_NMS_MIN_IOM) {
                return true
            }
        }
        return false
    }

    /** Computes `(IoU, IoM)` between the strip-space bounding boxes of [a] and [b]. */
    private fun computeTrackBoxIouAndIom(a: TrackEntry, b: TrackEntry): Pair<Float, Float> {
        val interL = max(a.stripLeftPx, b.stripLeftPx)
        val interT = max(a.stripTopPx, b.stripTopPx)
        val interR = min(a.stripRightPx, b.stripRightPx)
        val interB = min(a.stripBottomPx, b.stripBottomPx)
        val interW = (interR - interL).coerceAtLeast(0f)
        val interH = (interB - interT).coerceAtLeast(0f)
        val interArea = interW * interH
        if (interArea <= 0f) return 0f to 0f

        val areaA = (a.stripRightPx - a.stripLeftPx).coerceAtLeast(0f) *
            (a.stripBottomPx - a.stripTopPx).coerceAtLeast(0f)
        val areaB = (b.stripRightPx - b.stripLeftPx).coerceAtLeast(0f) *
            (b.stripBottomPx - b.stripTopPx).coerceAtLeast(0f)
        val union = areaA + areaB - interArea
        val minArea = min(areaA, areaB)
        val iou = if (union > 0f) (interArea / union).coerceIn(0f, 1f) else 0f
        val iom = if (minArea > 0f) (interArea / minArea).coerceIn(0f, 1f) else 0f
        return iou to iom
    }

    /** Absorbs [duplicate]'s unique sweep votes and fresher pose/confidence into [winner]. */
    private fun mergeDuplicateIntoWinner(
        winner: TrackEntry,
        duplicate: TrackEntry,
        sweepId: Long,
        minVotesToMaintain: Int
    ) {
        if (duplicate.hitSweepIds.isNotEmpty()) {
            val mergedSweeps = ArrayList<Long>(winner.hitSweepIds.size + duplicate.hitSweepIds.size)
            mergedSweeps.addAll(winner.hitSweepIds)
            for (s in duplicate.hitSweepIds) {
                if (s !in mergedSweeps) {
                    mergedSweeps.add(s)
                }
            }
            mergedSweeps.sort()
            winner.hitSweepIds.clear()
            val startIdx = (mergedSweeps.size - rollingWindowSweeps).coerceAtLeast(0)
            for (i in startIdx until mergedSweeps.size) {
                winner.hitSweepIds.addLast(mergedSweeps[i])
            }
        }
        winner.smoothedConfidence = max(winner.smoothedConfidence, duplicate.smoothedConfidence)
        if (duplicate.lastTimestampNs > winner.lastTimestampNs) {
            winner.stripLeftPx = duplicate.stripLeftPx
            winner.stripTopPx = duplicate.stripTopPx
            winner.stripRightPx = duplicate.stripRightPx
            winner.stripBottomPx = duplicate.stripBottomPx
            winner.lastTimestampNs = duplicate.lastTimestampNs
        }
        winner.isConfirmedLatch = winner.isConfirmedLatch || duplicate.isConfirmedLatch
        updateTrackConfirmationLatch(winner, sweepId, minVotesToMaintain)
    }

    /** Returns `true` when the two most recent sweep IDs in [hitSweepIds] are `currentSweepId - 1` and [currentSweepId]. */
    private fun hasTwoConsecutiveRecentHits(hitSweepIds: ArrayDeque<Long>, currentSweepId: Long): Boolean {
        if (hitSweepIds.size < 2) return false
        val last = hitSweepIds.last()
        val prev = hitSweepIds[hitSweepIds.size - 2]
        return last == currentSweepId && prev == currentSweepId - 1L
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
        return sqrt(dx * dx + dz * dz) / dtSec
    }

    companion object {
        /** Default N for N-of-M voting (`AGENTS.md` §5: default 3 of 6 sweeps). */
        const val DEFAULT_MIN_VOTES = 3

        /** Minimum votes in the rolling window required to maintain an already-confirmed lock. */
        const val DEFAULT_MIN_VOTES_TO_MAINTAIN = 2

        /** Default M for N-of-M voting (`AGENTS.md` §5). */
        const val DEFAULT_WINDOW_SWEEPS = 6

        /** Baseline association radius in meters in 3D world space (`AGENTS.md` §5: `40 mm`). */
        const val DEFAULT_CLUSTER_RADIUS_METERS = 0.040f

        /**
         * Maximum world-frame angular separation (degrees) between two 3D bearing rays to associate
         * them as the same target across handheld pitch/yaw/roll movements (`docs/ENGINEERING_LEDGER.md` Milestone 2 post-video fix).
         */
        const val DEFAULT_MAX_ANGULAR_DISTANCE_DEG = 4.5f

        /** Range-proportional ground association ratio (`0.08 * R`, e.g., `120 mm` at `1.5 m`). */
        private const val RANGE_SCALED_CLUSTER_RATIO = 0.08f

        /** EMA blend weight applied to a new hit's world-frame unit ray (`65%` new, `35%` prior). */
        private const val WORLD_RAY_NEW_WEIGHT = 0.65f

        /** Consecutive hits required for high-confidence fast lock on live camera sweeps. */
        const val FAST_LOCK_CONSECUTIVE_HITS = 2

        /** Minimum smoothed confidence required for 2-sweep consecutive fast lock. */
        const val FAST_LOCK_MIN_CONFIDENCE = 0.60f

        /** Walking speed (m/s) above which the "SLOW SWEEP" warning is raised (`AGENTS.md` §5). */
        const val MAX_SWEEP_SPEED_MPS = 1.10f

        /** Maximum world-ray separation (degrees) at which two same-class tracks merge in Track-Level Spatial NMS. */
        const val TRACK_NMS_MAX_ANGULAR_DEG = 3.2f

        /** Maximum 3D ground distance (meters, `75 mm`) at which two same-class tracks merge in Track-Level Spatial NMS. */
        const val TRACK_NMS_MAX_DISTANCE_METERS = 0.075f

        /** Minimum strip bounding-box Intersection-over-Union at which two same-class tracks merge. */
        const val TRACK_NMS_MIN_IOU = 0.35f

        /** Minimum strip bounding-box Intersection-over-Minimum-Area at which two same-class tracks merge. */
        const val TRACK_NMS_MIN_IOM = 0.60f
    }
}
