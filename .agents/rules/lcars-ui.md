---
trigger: glob
globs: "android/**"
description: Classic 2360s TNG LCARS 6-role color taxonomy, GneissFind + git-hash/date shoulders, central LOCK NEAREST / BREAK LOCK actuator, and full-screen Lock-On HUD symbology.
---

# GneissFind Classic 2360s TNG LCARS UI Rules (Milestone 2C)

## 1. Preserved Visual & Semantic Invariants (`8707282`)
1. **Pure `#000000` OLED Canvas & `4.dp` Gutters:** Zero translucent cards, drop shadows, or modal popups over the camera viewport.
2. **Silent UI Chrome:** `isSoundEffectsEnabled = false` on all LCARS controls (`80 ms` `#FFFFFF` tap flash only).
3. **Preserved 6-Role `LcarsSemanticColors` (`valueStepDelta = 0.05f`):**
   - **On/Off Toggles (`#FF9900` Golden Amber; `#596480` Slate Blue when OFF):** `OPT` (`2.0X CROP`), `EXP` (`1/500S`), `EIS` (`EIS LOCK`), `EYE` (`EYES-UP` idle patrol OLED dim toggle), and the 4 `PROSPECT MATRIX` mineral pills.
   - **Multi-State Mode Toggles (`#9999FF` Periwinkle):** `MOD` (`SCAN / LOG`), `SND` (`AUDIO / MUTE`), `SW` (`5-TILE / 3-TILE`).
   - **Action / Drawer Triggers (`#FFCC66` TNG Gold):** `CAL` (`SYNC PITCH`), **`LOCK NEAREST`** (when active target available), `SRC` (`MODEL SELECTOR`), and `-` / `+` EV steppers.
   - **Realtime Telemetry (`#99CCFF` Ice Blue):** `PITCH`, `VEL`, `FPS`, `Y-PX`, and `SWEEP` (`MS` + mode badge).
   - **Static / Config State (`#FFCC99` Peach):** `RANGE • RESOLUTION`, `OPTICS • SHUTTER • ISO`, `REC` counter, swath/lock status code, and build provenance shoulder.
   - **Critical / Limit / Break Alerts (`#CC4444` Mars Red):** **`BREAK LOCK`** (active Lock-On release), `PITCH LIMIT`, `SLOW SWEEP`, and active `LOG`.

## 2. Mandatory LCARS Shoulder Updates (Build Provenance & Identity)
1. **Top-Left Shoulder (`TopLeft` `LcarsElbow`):**
   - Replace the `"GEO-47"` shoulder label with **`GNEISSFIND`** (rendered cleanly in bundled `Antonio` bold uppercase).
2. **Bottom-Left Shoulder (`BottomLeft` `LcarsElbow`):**
   - Display the **6-character Git commit hash** AND **ISO build date** injected via Gradle `BuildConfig` (`BuildConfig.GIT_COMMIT_HASH_6` + `BuildConfig.BUILD_DATE_ISO`), formatted compactly with tabular numerals so it fits the shoulder bar/stem cleanly without clipping (e.g., `A0FE74 • 2026-11-16` along the bottom-left elbow rail or stacked `A0FE74` / `26-11-16` if placed in the narrow `68.dp` vertical stem—pick whichever renders at crisp legible `>= 11.sp` without truncation!).

## 3. Central Ambidextrous `LOCK NEAREST` / `BREAK LOCK` Button & `SRC` Swap
- **Remove `SPEED SIM` (`SIM`)** from the main tactical control deck (synthetic overspeed tests remain accessible via unit tests or `SRC` synthetic profiles).
- **Single Stateful Lock Button (`LOCK NEAREST` <-> `BREAK LOCK`):**
  - Occupies the former `SRC` slot **immediately adjacent to the horizontal center divider** of the landscape control deck so it is centrally reachable by either left or right thumb!
  - Move **`SRC` (`MODEL SELECTOR`)** to the outer right slot formerly occupied by `SPEED SIM`.
  - **Visual States of the Central Lock Button:**
    1. `PATROL` with `0` visible confirmed targets: `#596480` Slate Blue, label **`LOCK NEAREST`** (tapping does nothing or flashes subtle standby).
    2. `PATROL` with `>= 1` visible confirmed target: `#FFCC66` TNG Gold, label **`LOCK NEAREST`** (tapping locks onto the nearest visible target).
    3. `LOCK-ON` or `PINPOINT` active (including during the `15s` lost-track grace countdown): `#CC4444` Mars Red, label **`BREAK LOCK`** (or `BREAK LOCK [12S]` when counting down lost track; tapping immediately returns to `PATROL`).

## 4. Top Viewport Symbology in `LOCK-ON` & `PINPOINT` Modes
- **Tap-to-Lock Hit Target:** In `PATROL` mode, tapping any visible `ConfirmedTarget` reticle in the top viewport (`>= 48.dp` minimum touch target around reticle center) immediately enters `LOCK-ON` on that target.
- **Full-Screen `16:9` Homing Overlay (`LOCK-ON` & `PINPOINT`):**
  - Hide the `384 px` horizontal push-broom strip lines and tile seam ticks while locked on; show the full uncropped `16:9` camera view.
  - Draw a high-contrast mineral-colored target bracket + **shrinking dashed uncertainty circle** representing `sigma3dMm` projected into pixels (`solid double-ring crosshair` once `PINPOINT` `AKAZE` constellation guidance engages at `sigma3dMm < 0.5 * D`).
  - Telemetry badge on reticle: `[01 AGATE 96% • 0.64M • ±8MM • AKAZE]` (or `RF-DETR B=1`).
  - **Off-Screen Re-Acquisition Chevron:** If camera motion moves the 3D target outside the `16:9` viewport during the `15s` grace period, draw a pulsing `GoldenAmber` / mineral-colored directional chevron clamped to the nearest viewport edge pointing directly toward the 3D anchor with angular offset (`e.g., <- 18° AGATE [11S]`).
- **Async Model Compilation Banner:** When switching models in `SRC`, never freeze the UI thread; keep preview live and show a clean inline LCARS banner `COMPILING GPU SHADERS...` (`< 0.5 s` on cached warm loads).
