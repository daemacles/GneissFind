package com.gneissfind.app.inference

import com.gneissfind.app.model.DetectorBackend
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.TileDetection
import com.gneissfind.app.vision.PushBroomSweepBuffers
import kotlin.math.sin

/**
 * Simulation profile for bench & emulator verification of the 5-tile LCARS HUD, cross-tile NMS,
 * 3-of-6 temporal voting, and `AudioTrack` pitch/cadence scaling prior to Milestone 3 `.tflite` weights.
 */
enum class SyntheticTargetProfile(val label: String, val lcarsCode: String) {
    MIXED_GRAVEL_WALK("MIXED GRAVEL SIM", "SIM-MIX"),
    PEBBLE_20MM_HIGH_PITCH("21MM AGATE (765HZ)", "SIM-20MM"),
    COBBLE_55MM_LOW_PITCH("56MM JASPER (365HZ)", "SIM-55MM"),
    QUIET_GRAVEL_CONFUSERS("CONFUSERS (MUTED)", "SIM-CONF")
}

/**
 * Deterministic synthetic [StoneDetector] emitting multi-sweep trajectories across the active
 * 5-tile (`1792x384`) or 3-tile (`1088x384`) push-broom strip at `0.59 mm/px` GSD (`2.0x` crop).
 */
class SyntheticStoneDetector(
    @Volatile var profile: SyntheticTargetProfile = SyntheticTargetProfile.MIXED_GRAVEL_WALK
) : StoneDetector {

    override val backend: DetectorBackend = DetectorBackend.SYNTHETIC_DEMO
    override val isHardwareReady: Boolean = true
    override val statusSummary: String
        get() = "SYNTHETIC (${profile.label})"

    private var sweepCounter: Long = 0L
    private var rollingLatencyMs: Float = 1.4f

    override fun detectSweep(sweepBuffers: PushBroomSweepBuffers): SweepInferenceResult {
        val startNs = System.nanoTime()
        val sweepIdx = sweepCounter++
        val tileCount = sweepBuffers.activeTileCount.coerceAtLeast(3)
        val detections = ArrayList<TileDetection>(8)

        when (profile) {
            SyntheticTargetProfile.PEBBLE_20MM_HIGH_PITCH -> {
                // 36 px box -> ~21.2 mm at 0.59 mm/px GSD -> ~766 Hz high-pitched pebble chirp!
                // Placed inside Tile 0 / Tile 1 32px overlap seam (tile0 x=348..384, tile1 x=0..32)
                // with IoU = 32/36 = 0.89 >= 0.45 to continuously exercise Cross-Tile NMS.
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
                // 96 px box -> ~56.6 mm at 0.59 mm/px GSD -> ~359 Hz deep cobble ping!
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
                // Wet orange alder leaf / specular water glint (classId == 4) — strictly muted!
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

        return SweepInferenceResult(
            tileDetections = detections,
            part1LatencyMs = elapsedMs * 0.55f,
            hostTopKBridgeLatencyMs = elapsedMs * 0.15f,
            part2LatencyMs = elapsedMs * 0.30f,
            totalSweepLatencyMs = elapsedMs,
            rollingAvgLatencyMs = rollingLatencyMs,
            isHardwareGraphActive = false
        )
    }

    override fun close() = Unit
}
