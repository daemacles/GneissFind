package com.gneissfind.app.inference

import com.gneissfind.app.model.DetectorBackend
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.TileDetection
import com.gneissfind.app.vision.PushBroomSweepBuffers
import kotlin.math.sin

/**
 * Simulation profile for immediate Pixel 10 Pro HUD + AudioTrack pitch/cadence verification
 * before Milestone 3 `.tflite` weights are trained.
 */
enum class SyntheticTargetProfile(val label: String) {
    MIXED_GRAVEL_WALK("Mixed Gravel Walk"),
    PEBBLE_20MM_HIGH_PITCH("20mm Agate (780 Hz)"),
    COBBLE_55MM_LOW_PITCH("55mm Jasper (320 Hz)"),
    QUIET_GRAVEL_CONFUSERS("Confusers Only (Muted)")
}

/**
 * Deterministic synthetic [StoneDetector] emitting multi-sweep trajectories across the 3-tile
 * push-broom strip, including cross-tile overlap duplicates and Class 4 `hard_confuser` boxes
 * to exercise Cross-Tile NMS, 3-of-6 spatial voting, and audio pitch/cadence scaling.
 */
class SyntheticStoneDetector(
    @Volatile var profile: SyntheticTargetProfile = SyntheticTargetProfile.MIXED_GRAVEL_WALK
) : StoneDetector {

    override val backend: DetectorBackend = DetectorBackend.SYNTHETIC_DEMO
    override val isHardwareReady: Boolean = true
    override val statusSummary: String
        get() = "Synthetic Demo (${profile.label})"

    private var sweepCounter: Long = 0L
    private var rollingLatencyMs: Float = 1.2f

    override fun detectSweep(sweepBuffers: PushBroomSweepBuffers): SweepInferenceResult {
        val startNs = System.nanoTime()
        val sweepIdx = sweepCounter++
        val detections = ArrayList<TileDetection>(6)

        when (profile) {
            SyntheticTargetProfile.PEBBLE_20MM_HIGH_PITCH -> {
                // ~24 px box -> ~20.4 mm at 0.85 mm/px GSD -> ~780 Hz high-pitched chirp!
                // Placed inside Tile 0/Tile 1 overlap seam (strip x ~ 368) to exercise Cross-Tile NMS.
                val conf = 0.88f
                detections.add(
                    TileDetection(
                        tileIndex = 0,
                        classId = StoneClass.CHALCEDONY_AGATE.id,
                        confidence = conf,
                        leftPx = 354f,
                        topPx = 176f,
                        rightPx = 378f,
                        bottomPx = 200f
                    )
                )
                detections.add(
                    TileDetection(
                        tileIndex = 1,
                        classId = StoneClass.CHALCEDONY_AGATE.id,
                        confidence = conf - 0.04f,
                        leftPx = 2f,
                        topPx = 176f,
                        rightPx = 26f,
                        bottomPx = 200f
                    )
                )
            }

            SyntheticTargetProfile.COBBLE_55MM_LOW_PITCH -> {
                // ~64 px box -> ~54.4 mm at 0.85 mm/px GSD -> 320 Hz low-pitched cobble tone!
                detections.add(
                    TileDetection(
                        tileIndex = 1,
                        classId = StoneClass.JASPER_RED_YELLOW.id,
                        confidence = 0.93f,
                        leftPx = 160f,
                        topPx = 150f,
                        rightPx = 224f,
                        bottomPx = 214f
                    )
                )
            }

            SyntheticTargetProfile.QUIET_GRAVEL_CONFUSERS -> {
                // Wet orange alder leaf / specular water glint (classId == 4) — MUST remain muted!
                detections.add(
                    TileDetection(
                        tileIndex = 2,
                        classId = StoneClass.HARD_CONFUSER.id,
                        confidence = 0.95f,
                        leftPx = 110f,
                        topPx = 120f,
                        rightPx = 165f,
                        bottomPx = 175f
                    )
                )
            }

            SyntheticTargetProfile.MIXED_GRAVEL_WALK -> {
                val cycle = (sweepIdx % 90L).toInt()
                val jitter = (sin(sweepIdx * 0.25) * 2.0).toFloat()
                when {
                    cycle in 0..28 -> {
                        // Phase A: 22 mm Carnelian Agate pebble in Tile 0 (~26 px -> ~22.1 mm -> ~750 Hz)
                        val conf = (0.58f + (cycle / 28f) * 0.36f).coerceIn(0.45f, 0.95f)
                        detections.add(
                            TileDetection(
                                tileIndex = 0,
                                classId = StoneClass.CHALCEDONY_AGATE.id,
                                confidence = conf,
                                leftPx = 180f + jitter,
                                topPx = 180f,
                                rightPx = 206f + jitter,
                                bottomPx = 206f
                            )
                        )
                    }
                    cycle in 32..60 -> {
                        // Phase B: 52 mm Red Jasper cobble in Tile 1 (~61 px -> ~51.9 mm -> 320 Hz)
                        val conf = (0.62f + ((cycle - 32) / 28f) * 0.32f).coerceIn(0.50f, 0.96f)
                        detections.add(
                            TileDetection(
                                tileIndex = 1,
                                classId = StoneClass.JASPER_RED_YELLOW.id,
                                confidence = conf,
                                leftPx = 155f + jitter,
                                topPx = 160f,
                                rightPx = 216f + jitter,
                                bottomPx = 221f
                            )
                        )
                    }
                    cycle in 64..84 -> {
                        // Phase C: 36 mm Dallasite green gem rock in Tile 2 (~42 px -> ~35.7 mm -> ~539 Hz)
                        val conf = 0.82f
                        detections.add(
                            TileDetection(
                                tileIndex = 2,
                                classId = StoneClass.GREEN_GEM_ROCK.id,
                                confidence = conf,
                                leftPx = 140f + jitter,
                                topPx = 172f,
                                rightPx = 182f + jitter,
                                bottomPx = 214f
                            )
                        )
                    }
                }

                // Always inject an auxiliary Class 4 hard_confuser (wet alder leaf) to continuously
                // verify confuser muting in live scanning.
                detections.add(
                    TileDetection(
                        tileIndex = 2,
                        classId = StoneClass.HARD_CONFUSER.id,
                        confidence = 0.91f,
                        leftPx = 270f,
                        topPx = 90f,
                        rightPx = 318f,
                        bottomPx = 138f
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

    override fun close() {
        // No native buffers owned by SyntheticStoneDetector.
    }
}
