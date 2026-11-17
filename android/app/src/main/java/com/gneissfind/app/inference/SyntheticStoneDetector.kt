/*
 * Deterministic synthetic stone detector for bench and emulator verification.
 *
 * Consumes pre-allocated push-broom sweep buffers from PushBroomCropper and emits
 * synthetic TileDetection trajectories across the active 5-tile or 3-tile strip.
 * Consumed by ScannerViewModel when DetectorBackend.SYNTHETIC_DEMO is selected.
 *
 * Main types:
 * - SyntheticTargetProfile: selectable simulation scenarios for HUD, NMS, voting, and audio verification.
 * - SyntheticStoneDetector: StoneDetector implementation generating deterministic tile detections.
 */
package com.gneissfind.app.inference

import com.gneissfind.app.model.DetectorBackend
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.TileDetection
import com.gneissfind.app.vision.PushBroomSweepBuffers
import kotlin.math.sin

/**
 * Simulation profile for bench and emulator verification of the 5-tile LCARS HUD, cross-tile NMS,
 * 3-of-6 temporal voting, and `AudioTrack` pitch/cadence scaling prior to Milestone 3 `.tflite` weights.
 *
 * @property label Human-readable scenario summary displayed in [SyntheticStoneDetector.statusSummary] and the LCARS source drawer.
 * @property lcarsCode Compact uppercase identifier displayed on LCARS profile selector pills.
 */
enum class SyntheticTargetProfile(val label: String, val lcarsCode: String) {
    /** Cycles through all four target mineral classes while continuously injecting a muted [StoneClass.HARD_CONFUSER]. */
    MIXED_GRAVEL_WALK("MIXED GRAVEL SIM", "SIM-MIX"),

    /** Places a small ~21 mm pebble across the Tile 0 / Tile 1 overlap seam to exercise cross-tile NMS and high-pitch sonification. */
    PEBBLE_20MM_HIGH_PITCH("21MM AGATE (765HZ)", "SIM-20MM"),

    /** Places a large ~57 mm cobble in the center tile to exercise low-pitch sonification. */
    COBBLE_55MM_LOW_PITCH("56MM JASPER (365HZ)", "SIM-55MM"),

    /** Emits only [StoneClass.HARD_CONFUSER] detections to verify that confusers never trigger user alerts or audio pings. */
    QUIET_GRAVEL_CONFUSERS("CONFUSERS (MUTED)", "SIM-CONF")
}

/**
 * Deterministic synthetic [StoneDetector] emitting multi-sweep trajectories across the active
 * 5-tile (`1792x384`) or 3-tile (`1088x384`) push-broom strip at `0.59 mm/px` GSD (`2.0x` crop).
 *
 * State:
 * - [profile]: volatile active [SyntheticTargetProfile] switchable from the UI thread.
 * - [sweepCounter]: monotonically increasing sweep counter driving periodic trajectory phases.
 * - [rollingLatencyMs]: exponential moving average of synthetic sweep execution time in milliseconds.
 *
 * Threading:
 * - [profile] may be written from any thread (`@Volatile`); [detectSweep] is called sequentially from the camera analysis thread.
 *
 * @property profile Active simulation scenario controlling which synthetic [TileDetection]s are emitted each sweep.
 */
class SyntheticStoneDetector(
    @Volatile var profile: SyntheticTargetProfile = SyntheticTargetProfile.MIXED_GRAVEL_WALK
) : StoneDetector {

    override val backend: DetectorBackend = DetectorBackend.SYNTHETIC_DEMO
    override val isHardwareReady: Boolean = true
    override val statusSummary: String
        get() = "SYNTHETIC (${profile.label})"

    /** Monotonically increasing index of sweeps processed since instantiation. */
    private var sweepCounter: Long = 0L

    /** Exponential moving average (`0.85` prior / `0.15` current) of [detectSweep] wall-clock duration in milliseconds. */
    private var rollingLatencyMs: Float = 1.4f

    /**
     * Synthesizes a list of [TileDetection]s for the current [profile] and [sweepCounter] phase
     * without reading the pixel contents of [sweepBuffers] (only [PushBroomSweepBuffers.activeTileCount] is inspected).
     */
    override fun detectSweep(sweepBuffers: PushBroomSweepBuffers): SweepInferenceResult {
        val startNs = System.nanoTime()
        val sweepIdx = sweepCounter++
        val tileCount = sweepBuffers.activeTileCount.coerceAtLeast(3)
        val detections = ArrayList<TileDetection>(8)

        when (profile) {
            SyntheticTargetProfile.PEBBLE_20MM_HIGH_PITCH -> {
                // 36 px box -> ~21.2 mm at 0.59 mm/px GSD -> ~766 Hz high-pitched pebble chirp.
                // Placed inside Tile 0 / Tile 1 32px overlap seam (tile0 x=348..384, tile1 x=0..34,
                // strip x=348..386) with IoU = 32/38 = 0.84 >= 0.45 to continuously exercise Cross-Tile NMS.
                val conf = 0.89f
                detections.add(
                    TileDetection(
                        tileIndex = 0,
                        classId = StoneClass.CHALCEDONY_AGATE.id,
                        confidence = conf,
                        leftPx = 348f,
                        topPx = 172f,
                        rightPx = 384f,
                        bottomPx = 208f
                    )
                )
                detections.add(
                    TileDetection(
                        tileIndex = 1,
                        classId = StoneClass.CHALCEDONY_AGATE.id,
                        confidence = conf - 0.04f,
                        leftPx = 0f,
                        topPx = 172f,
                        rightPx = 34f,
                        bottomPx = 208f
                    )
                )
            }

            SyntheticTargetProfile.COBBLE_55MM_LOW_PITCH -> {
                // 96 px box -> ~56.6 mm at 0.59 mm/px GSD -> ~359 Hz deep cobble ping.
                val centerTile = if (tileCount >= 5) 2 else 1
                detections.add(
                    TileDetection(
                        tileIndex = centerTile,
                        classId = StoneClass.JASPER_RED_YELLOW.id,
                        confidence = 0.94f,
                        leftPx = 144f,
                        topPx = 144f,
                        rightPx = 240f,
                        bottomPx = 240f
                    )
                )
            }

            SyntheticTargetProfile.QUIET_GRAVEL_CONFUSERS -> {
                // Wet orange alder leaf / specular water glint (StoneClass.HARD_CONFUSER) — strictly muted.
                detections.add(
                    TileDetection(
                        tileIndex = (tileCount - 1).coerceAtLeast(0),
                        classId = StoneClass.HARD_CONFUSER.id,
                        confidence = 0.96f,
                        leftPx = 110f,
                        topPx = 120f,
                        rightPx = 190f,
                        bottomPx = 200f
                    )
                )
            }

            SyntheticTargetProfile.MIXED_GRAVEL_WALK -> {
                // 120-sweep (~15 s at 8 Hz) repeating cycle with 3-sweep dead gaps between mineral phases
                // so TemporalTargetTracker exercises track acquisition, maintenance, and expiration.
                val cycle = (sweepIdx % 120L).toInt()
                val jitter = (sin(sweepIdx * 0.25) * 2.0).toFloat()
                when {
                    cycle in 0..28 -> {
                        // Phase A: ~22 mm Carnelian Agate pebble in Tile 0 (38 px @ 0.59 mm/px)
                        val conf = (0.60f + (cycle / 28f) * 0.34f).coerceIn(0.45f, 0.95f)
                        detections.add(
                            TileDetection(
                                tileIndex = 0,
                                classId = StoneClass.CHALCEDONY_AGATE.id,
                                confidence = conf,
                                leftPx = 170f + jitter,
                                topPx = 172f,
                                rightPx = 208f + jitter,
                                bottomPx = 210f
                            )
                        )
                    }
                    cycle in 32..60 -> {
                        // Phase B: ~54 mm Red Jasper cobble in center tile (92 px @ 0.59 mm/px)
                        val centerTile = if (tileCount >= 5) 2 else 1
                        val conf = (0.64f + ((cycle - 32) / 28f) * 0.31f).coerceIn(0.50f, 0.96f)
                        detections.add(
                            TileDetection(
                                tileIndex = centerTile,
                                classId = StoneClass.JASPER_RED_YELLOW.id,
                                confidence = conf,
                                leftPx = 146f + jitter,
                                topPx = 146f,
                                rightPx = 238f + jitter,
                                bottomPx = 238f
                            )
                        )
                    }
                    cycle in 64..88 -> {
                        // Phase C: ~38 mm Dallasite green gem rock in Tile 3 (or Tile 2 in 3-tile mode)
                        val tileIdx = if (tileCount >= 5) 3 else 2
                        detections.add(
                            TileDetection(
                                tileIndex = tileIdx,
                                classId = StoneClass.GREEN_GEM_ROCK.id,
                                confidence = 0.85f,
                                leftPx = 150f + jitter,
                                topPx = 160f,
                                rightPx = 214f + jitter,
                                bottomPx = 224f
                            )
                        )
                    }
                    cycle in 92..114 -> {
                        // Phase D: ~44 mm Petrified Wood in rightmost Tile 4 (or Tile 2 in 3-tile mode)
                        val tileIdx = (tileCount - 1).coerceAtLeast(0)
                        detections.add(
                            TileDetection(
                                tileIndex = tileIdx,
                                classId = StoneClass.PETRIFIED_WOOD.id,
                                confidence = 0.83f,
                                leftPx = 140f + jitter,
                                topPx = 154f,
                                rightPx = 215f + jitter,
                                bottomPx = 229f
                            )
                        )
                    }
                }

                // Continuously inject Class 4 hard_confuser to verify muting in live scanning
                detections.add(
                    TileDetection(
                        tileIndex = (tileCount - 1).coerceAtLeast(0),
                        classId = StoneClass.HARD_CONFUSER.id,
                        confidence = 0.91f,
                        leftPx = 260f,
                        topPx = 88f,
                        rightPx = 330f,
                        bottomPx = 158f
                    )
                )
            }
        }

        val elapsedMs = ((System.nanoTime() - startNs) / 1_000_000.0).toFloat()
        rollingLatencyMs = rollingLatencyMs * 0.85f + elapsedMs * 0.15f

        val maxAlertableScore = detections.fold(0f) { acc, det ->
            if (StoneClass.fromId(det.classId)?.isUserAlertable == true && det.confidence > acc) {
                det.confidence
            } else {
                acc
            }
        }

        return SweepInferenceResult(
            tileDetections = detections,
            // Synthetic split proportions (55% Part 1 / 15% Top-K bridge / 30% Part 2) for HUD telemetry display.
            part1LatencyMs = elapsedMs * 0.55f,
            hostTopKBridgeLatencyMs = elapsedMs * 0.15f,
            part2LatencyMs = elapsedMs * 0.30f,
            totalSweepLatencyMs = elapsedMs,
            rollingAvgLatencyMs = rollingLatencyMs,
            isHardwareGraphActive = false,
            maxAlertableCandidateScore = maxAlertableScore
        )
    }

    override fun close() = Unit
}
