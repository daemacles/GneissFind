---
trigger: glob
globs: "android/**"
description: Classic 2360s TNG LCARS 6-role color taxonomy, GneissFind + git/date shoulders, Replay SRC entries, and right-edge-safe full-screen Lock-On HUD rules.
---

# GneissFind Classic 2360s TNG LCARS UI Rules (Milestone 2D)

## 1. Preserved Visual, Semantic & Lock-Mode Invariants (`8707282`, `5d13646`)
1. **Pure `#000000` OLED Canvas & `4.dp` Gutters:** Zero translucent cards, drop shadows, or modal popups over the camera viewport.
2. **Silent UI Chrome:** `isSoundEffectsEnabled = false` on all LCARS controls (`80 ms` `#FFFFFF` tap flash only).
3. **Preserved Shoulders & Central Actuator (`e8dc825`):**
   - `TopLeft` shoulder: **`GNEISSFIND`**.
   - `BottomLeft` shoulder: **`BuildConfig.GIT_COMMIT_HASH_6 • BuildConfig.BUILD_DATE_ISO`**.
   - Central ambidextrous button adjacent to deck divider: **`LOCK NEAREST`** (`#596480` Slate when 0 targets, `#FFCC66` TNG Gold when `>= 1` target) <-> **`BREAK LOCK`** (`#CC4444` Mars Red while locked).
   - Full-screen `16:9` Lock Mode (`5d13646`): collapses lower 2-pane Patrol deck so `ClippingCameraViewportContainer` fills the inner crook alongside `LockOnSideConsole` (`MODEL CONF`, `3D ANGLES`, `TEXTURE PTS` / `HOLD TIMER`, and `88.dp` Mars Red `BREAK LOCK`).

## 2. `SRC` Drawer & `MOD: LOG` Telemetry Updates (Milestone 2D)
- **`LcarsSourceDrawerOverlay` (`SRC` Drawer) Organization:**
  1. **Live Camera Hardware Profiles (`S-01` .. `S-05`):** Default `S-01: RF-DETR 256 B=3 W8-DYN` through `S-05: RF-DETR 384 B=5 BASE`.
  2. **Recorded Sensor-Trace Replays (`REPLAY: <session>`):** Dynamically list all valid `<session>/clip.mp4` + `trace.jsonl` bundles discovered in `Android/media/com.gneissfind.app/replays/` (newest first, showing duration, e.g., `REPLAY: 20261117_0930 [28S]`). While a replay is active, show a `Periwinkle` `REPLAY` badge in the header rail so the operator always knows live camera is paused.
  3. **Single Synthetic Reference (`SIM: MIXED GRAVEL WALK`):** Retain **only** `MIXED_GRAVEL_WALK` as the 1 built-in synthetic demo at the bottom of `SRC` (remove `AGATE_PEBBLE_20MM` and `JASPER_COBBLE_55MM` from the UI drawer while keeping them in `SyntheticStoneDetector` for unit tests).
- **`MOD: LOG` (`30 FPS MP4 + 120 Hz IMU`) Readout:**
  - When `MOD` is toggled to `LOG`, display elapsed recording duration and encoded frame count in the `REC` readout (`e.g., REC 00:14 • 420F`) with `MarsRed` active recording indicator.

## 3. Right-Edge & Off-Screen `DrawScope` Crash Immunity (`ScannerScreen.kt`)
- In `LockOnOverlay` / `OffScreenChevron` rendering:
  - Clamp every canvas coordinate (`x`, `y`) to `[0f, canvasSize.width]` and `[0f, canvasSize.height]` after guarding `!value.isNaN() && value.isFinite()`.
  - When rendering the off-screen directional chevron and angular badge near the **right** (`edgeXNorm -> 1.0f`) or **bottom** (`edgeYNorm -> 1.0f`) viewport border, shift the badge box inward (`left = (anchorX - badgeW - pad).coerceIn(0f, (canvasW - badgeW).coerceAtLeast(0f))`) and coerce `Size(w.coerceAtLeast(0f), h.coerceAtLeast(0f))` so `DrawScope.drawRoundRect` / `drawPath` can NEVER receive negative width/height or throw `IllegalArgumentException` when panning off-screen right!
