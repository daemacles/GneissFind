---
trigger: glob
globs: "android/**"
description: Hardware-validated Classic 2360s TNG LCARS 6-role semantic color taxonomy, non-occluding split-screen layout, source drawer, and telemetry rules.
---

# GneissFind Classic 2360s TNG LCARS UI Rules (`8707282` Validated)

## 1. Non-Negotiable Visual & Layout Invariants
1. **Pure `#000000` OLED Canvas & `4.dp` Black Gutters:** Zero translucent Material cards, borders, gradients, or modal sheets over the top camera strip.
2. **Silent UI Chrome:** `isSoundEffectsEnabled = false` on all LCARS controls; visual press feedback is an `80 ms` `#FFFFFF` flash. Audio output (`USAGE_MEDIA`, `STREAM_MUSIC`) is strictly for stone pings and overspeed warning.
3. **Preserved 6-Role Semantic Color Taxonomy (`LcarsSemanticColors` in `Theme.kt`):**
   Keep `LcarsColorTuning.valueStepDelta = 0.05f` (`MINUS` -5%, `BASE` 0%, `PLUS` +5%) and exact role mappings:
   - **On/Off Toggles (`#FF9900` Golden Amber; `#596480` Slate Blue when OFF):** `OPT` (`2.0X CROP`), `EXP` (`1/500S`), `EIS` (`EIS LOCK`), `SIM` (`SPEED SIM`), and the 4 `PROSPECT MATRIX` mineral toggles.
   - **Multi-State Mode Toggles (`#9999FF` Periwinkle):** `MOD` (`SCAN / LOG`), `SND` (`AUDIO / MUTE`), `SW` (`5-TILE / 3-TILE`).
   - **Action / Drawer Triggers (`#FFCC66` TNG Gold):** `CAL` (`SYNC PITCH`), `SRC` (`OPEN SOURCE DRAWER`), and `-` / `+` EV step endcaps.
   - **Realtime Telemetry Readouts (`#99CCFF` Ice Blue):** `PITCH`, `VEL`, `FPS` (sweep Hz), `Y-PX`, and `SWEEP` (live detector inference latency in `MS`).
   - **Static / Config State Readouts (`#FFCC99` Peach):** `RANGE • RESOLUTION`, `OPTICS • SHUTTER • ISO`, `REC` counter, and swath code (`47-1792` / `47-1088`).
   - **Critical / Limit Alerts (`#CC4444` Mars Red):** `PITCH LIMIT`, `SLOW SWEEP`, and active `LOG` indicator.
4. **Preserved Milestone 1B Layout Affordances:**
   - Keep `LcarsPitchTravelGauge` on the left side of the camera viewport inside the left elbow.
   - Keep `rememberHighWaterSegmentWidth` + `fontFeatureSettings = "tnum"` on live numeric bars so telemetry (`SWEEP XX MS`, `FPS X.X HZ`) never jitters horizontally.
   - Do NOT rename the 4 `PROSPECT MATRIX` mineral buttons (`Agate / Carnelian`, `Red / Yellow Jasper`, `Dallasite / Nephrite`, `Petrified Wood`) during Milestone 2—the COCO stand-in objects map transparently underneath them.

## 2. Milestone 2 `SRC` Drawer Additions (`LcarsSourceDrawerOverlay`)
- Register the live LiteRT GPU detector as the primary hardware source in `LcarsSourceDrawerOverlay` (e.g., `RF-DETR NANO GPU [COCO PROXY]`), defaulting to active whenever the exported `.tflite` + `rf_detr_coco_metadata.json` assets are present, while retaining `MIXED_GRAVEL_WALK`, `AGATE_PEBBLE_20MM`, and `JASPER_COBBLE_55MM` synthetic modes for deterministic regression testing.
- Surface live GPU inference time clearly in the existing Ice Blue `SWEEP` telemetry bar (e.g., `SWEEP 74MS • GPU FP16`) so Milestone 2 performance (`<= 125 ms`) is directly observable on-screen and via Logcat.
