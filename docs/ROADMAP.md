# GneissFind — Execution Roadmap & Acceptance Criteria (`docs/ROADMAP.md`)

## Milestone 1 — Baseline Push-Broom Skeleton *(Completed)*
- [x] Initial Android project (`com.gneissfind.app`), CameraX + `Camera2Interop` `<= 1/500 s` shutter clamp with floating ISO verified on Pixel 10 Pro (`1/502s`).
- [x] Simulated target stream, temporal vote counter, speed gate, and `AudioTrack` sonifier scaffold.

---

## Milestone 1B — Reversible Landscape, Classic TNG LCARS, 5-Tile Swath & IMU Pitch Stabilization *(Current Active Task)*

### Copy-Paste Prompt for Antigravity in Android Studio Canary
Paste the block below directly into the Antigravity Agent pane:

```text
Read @AGENTS.md, @.agents/rules/android-camera-ar.md, @.agents/rules/lcars-ui.md, and @docs/ROADMAP.md in full—the project specification has been updated for Milestone 1B.

Refactor the Android app in `android/` to implement every Milestone 1B requirement:
1. Lock `MainActivity` to reversible `sensorLandscape` (`ROTATION_90` and `ROTATION_270`), synchronizing CameraX `Preview` + `ImageAnalysis` targetRotation, ARCore display geometry, and `SensorManager.remapCoordinateSystem` so pitch and roll stay accurate when flipping landscape 180 degrees.
2. Default `2.0x` lossless center sensor crop (`ZoomRatio = 2.0f`) AND `<= 1/500 s` shutter clamp to ON at startup while preserving the working `Camera2Interop` exposure clamp + floating ISO behavior.
3. Upgrade the push-broom strip cropper on the `1920x1080` landscape stream to 5 horizontally overlapping `384x384` tiles (`xOffsets = [64, 416, 768, 1120, 1472]`, spanning `1792x384 px` = 93.3% width with 32 px overlap and cross-tile NMS), with an inline `5-TILE / 3-TILE` toggle.
4. Implement IMU-driven ROI pitch stabilization (`Sensor.TYPE_GAME_ROTATION_VECTOR`) that dynamically shifts `cropY` within `[0..696]` around the `45 deg` reference pitch (or user `SYNC PITCH` setpoint) so the scan strip tracks the same ~1.5m ground patch despite wrist pitch wobble, lighting a `PITCH LIMIT` warning when clamped at `0` or `696`.
5. Replace the portrait card UI and floating setup sheet with the split-screen Classic 2360s TNG LCARS interface specified in `.agents/rules/lcars-ui.md`: pure `#000000` OLED canvas with `4.dp` black gutters, custom swept `LcarsElbow` shapes with concave inner fillets, bundled offline `Antonio` (or `Barlow Condensed`) uppercase tabular-numeral typography, top ~48% panoramic camera + stabilized strip HUD (with black-halo high-contrast stone brackets), and bottom ~52% non-occluding tactical control deck. Keep all LCARS UI button taps silent (`isSoundEffectsEnabled = false`); reserve audio strictly for stone pings and overspeed warning.
6. Update unit tests (covering 5-tile coordinate math, cross-tile NMS, and IMU pitch-to-`cropY` clamping), verify `./gradlew assembleDebug testDebugUnitTest` passes cleanly, and make atomic one-sentence git commits per `@AGENTS.md`.
```

### Milestone 1B Acceptance Tests (Oracle)
1. **Build & Unit Tests:** `./gradlew assembleDebug testDebugUnitTest` succeeds with zero errors, and `git log -n 5 --oneline` shows clean one-sentence atomic commits.
2. **Reversible `sensorLandscape`:** Flipping the Pixel 10 Pro $180^\circ$ between left-landscape and right-landscape smoothly rotates both UI and camera preview upright, and tilting the lens downward increases `PITCH` in both orientations (never swapping pitch with roll).
3. **Default Startup Optics:** Cold-launching the app starts immediately at **`2.0x` zoom** and **`<= 1/500 s` shutter clamp** (`GSD ~ 0.59 mm/px` at `R ~ 1.50 m`).
4. **5-Tile + IMU Pitch Stabilization:** Gently pitching the wrist up/down by $\pm 6^\circ$ visibly shifts the `1792x384` scan strip opposite the wrist motion so the strip stays anchored on the same living-room floor / gravel target; tilting past the frame edge clamps cleanly and lights `PITCH LIMIT`.
5. **Authentic Non-Occluding LCARS UI:** Pure `#000000` background, `4.dp` black gutters, swept `LcarsElbow` chrome in TNG Golden Amber / Peach / Lavender / Periwinkle, tabular numerals without horizontal jitter, zero modal sheets covering the camera feed, and silent UI taps.

---

## Milestone 2 — T7920 Auto-Labeling & Hard-Negative Copy-Paste Pipeline (`ml-pipeline/`)
- Set up `uv` / Docker Python environment on Ubuntu 26.10.
- Build `SAM 2` video propagation scripts pinned to `GPU 1 & 2` (`2x RTX 4000 Ada`) to auto-segment salted gravel-bar strip captures into COCO/YOLO 5-class bounding boxes (`4 target classes + hard_confuser`).
- Build Copy-Paste instance compositor pasting segmented raw PNW stones onto unsalted local hard-negative gravel frames with realistic water-glint / exposure jitter.

---

## Milestone 3 — `RF-DETR-Nano` Fine-Tuning & Two-Graph `litert-torch` GPU Export
- Train `RF-DETR-Nano` (`384x384`, Apache-2.0) and optional `RF-DETR-Large` teacher on `GPU 0` (`RTX 4090 24GB`).
- Export `RF-DETR-Nano` into the 2-graph FP16-compatible `.tflite` bundle (Graph A backbone/encoder + Kotlin `TopK` + Graph B decoder) and wire into `LiteRtStoneDetector` in `android/` for live `5 x 384x384` strip inference on Pixel 10 Pro GPU.

---

## Milestone 4 — Field Calibration, Watershed Presets & Hard-Negative Mining
- Add per-class confidence thresholds and regional presets (Puget Sound beaches, Western Cascade rivers, Bellingham / Skagit bars).
- One-tap false-positive flagger in `LOG DATA` to feed field confusers straight back into Milestone 2/3 retraining.

---

## Milestone 5 — Late-Stage Audio Polish & V2 Ergonomics
- Refine polyphonic/harmonic `Oboe` / `AudioTrack` stone sonification curves, optional tactile haptics, and optional toggleable LCARS trills.
- Evaluate hands-free chest-mount geometry and battery/thermal duty-cycling for multi-hour river hikes.
