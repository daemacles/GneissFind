---
trigger: glob
globs: "android/**"
description: Classic 2360s Star Trek TNG LCARS visual grammar, color tokens, swept elbow shapes, typography, and split-screen landscape layout for GneissFind.
---

# GneissFind Classic 2360s TNG LCARS UI Specification

## 1. Core Architectural Mandates ("Geological Tricorder")
1. **Pure `#000000` OLED Canvas:** Every non-camera background pixel MUST be pure `#000000`. Never use dark grey cards, drop shadows, glassmorphism/blur, gradients, or thin `1.dp` Material card borders.
2. **Solid Color Blocks + `4.dp` Black Gutters:** Separate adjacent LCARS bars, elbows, and pill actuators strictly with crisp `4.dp` solid `#000000` gaps.
3. **Split-Screen Landscape (`sensorLandscape`, Zero Camera Occlusion):**
   - **Upper ~48% Height (`Geological Viewport`):** Framed on the left by a top-left swept `LcarsElbow` (`GEO-47 // PUSH-BROOM SWATH`) and top segmented status bar. Inside the frame sits the wide panoramic CameraX `PreviewView` aligned so the full width of the `1920x1080` camera feed fills the horizontal viewport without obscuring popup sheets.
   - **Lower ~52% Height (`Tactical Control Deck`):** Framed on the left by a bottom-left swept `LcarsElbow` (`SYS-01 // SENSOR & TARGET MATRIX`), housing all mode switches, live tabular telemetry, inline optics/pitch stabilization controls, and mineral class toggles simultaneously. **Eliminate modal bottom sheets or popup dialogs that cover the camera viewport.**
4. **Silent UI Chrome:** Set `View.isSoundEffectsEnabled = false` on the root Compose view / custom LCARS actuators. Tapping LCARS pills produces an immediate **`80 ms` pure `#FFFFFF` visual flash**, with **zero UI click sound** (audio output is strictly reserved for stone detection pings and overspeed alert).

---

## 2. Classic 2360s TNG Color Palette (`LcarsColors`)
Define an immutable `LcarsColors` object with exact hex values:
- `CanvasBlack = Color(0xFF000000)` — Root background, all `4.dp` gutters, text inside bright pills
- `GoldenAmber = Color(0xFFFF9900)` — Primary `LcarsElbow` structural chrome, active primary latch
- `ApricotPeach = Color(0xFFFFCC99)` — Secondary horizontal header/footer rails, numeric telemetry labels
- `WarmSand = Color(0xFFFFCC66)` — Standard interactive actuators (`SYNC PITCH`, `SIM`, EV step buttons)
- `Lavender = Color(0xFFCC99CC)` — Optics & camera hardware control group (`2.0X CROP`, `1/500S CLAMP`)
- `Periwinkle = Color(0xFF9999FF)` — IMU pitch stabilization & swath geometry bars (`EIS LOCK`, `5-TILE`)
- `IceBlue = Color(0xFF99CCFF)` — Secondary readouts, active scan state indicator
- `MutedSlate = Color(0xFF555577)` — Disabled / toggled-OFF pill state (with `#000000` or `#CCCCDD` text for readability)
- `MarsRed = Color(0xFFCC4444)` — `LOG DATA` active recording state, `"SLOW SWEEP"` overspeed alert, `"PITCH LIMIT"` clamp alert
- `AlertWhite = Color(0xFFFFFFFF)` — `80 ms` press flash state & high-priority text

### High-Contrast Video Target Overlay Palette (Anti-Camouflage)
Because warm LCARS chrome sits below the viewport, bounding brackets drawn *over* live gravel video MUST use high-luminance colors backed by a **`2.dp` `#000000` outer keyline halo** so orange/red carnelian and jasper stones are never visually camouflaged:
- `chalcedony_agate`: `#00FFFF` (Electric Cyan corner reticles + black keyline)
- `jasper_red_yellow`: `#FFFF00` (High-Vis Solar Yellow corner reticles + black keyline)
- `green_gem_rock`: `#00FF88` (Spring Mint corner reticles + black keyline)
- `petrified_wood`: `#FFAA00` (Bright Amber-Gold corner reticles + black keyline)
- `hard_confuser`: **Never rendered in normal `SCAN` mode** (optional faint `#777777` dotted box only when a developer diagnostic toggle is active).

---

## 3. Swept `LcarsElbow` & Pill Geometry in Jetpack Compose
Implement reusable Compose primitives in `ui/lcars/LcarsComponents.kt`:
- **`LcarsElbow` Custom `Shape` / `Canvas` `Path`:**
  - Parameters: `orientation` (`TopLeft`, `BottomLeft`, `TopRight`, `BottomRight`), `stemWidth: Dp = 68.dp`, `barHeight: Dp = 22.dp`, `outerRadius: Dp = 32.dp`, `innerRadius: Dp = 14.dp`.
  - Geometry (`TopLeft` example): Outer top-left corner sweeps smoothly with `outerRadius = 32.dp`; vertical left stem has width `stemWidth = 68.dp`; horizontal top rail has height `barHeight = 22.dp`; where the inner vertical edge (`x = stemWidth`) meets the inner horizontal edge (`y = barHeight`), draw a smooth **concave inner fillet arc** of `innerRadius = 14.dp` so the camera viewport or data grid tucks cleanly into the elbow's crook.
- **Hemispherical End-Caps & Pill Actuators (`LcarsPillButton`):**
  - Use `RoundedCornerShape(percent = 50)` for standalone capsule buttons, or flat-left / hemispherical-right (`RoundedCornerShape(topStart = 0.dp, bottomStart = 0.dp, topEnd = 50.dp, bottomEnd = 50.dp)`) when docked flush (after a `4.dp` black gutter) next to a vertical rail.
  - All button text sits **bottom-right or center-right aligned**, uppercase, bold `#000000` on solid LCARS fill when active, or `#FFCC99` on `MutedSlate` when inactive, with a crisp numeric prefix or status dot.

---

## 4. Typography & Tabular Telemetry (`LcarsTypography`)
- Bundle **`Antonio`** (or **`Barlow Condensed`** Bold/SemiBold via downloadable/bundled Google Font `.ttf` in `android/app/src/main/res/font/`) as the primary display & telemetry typeface, with `FontFamily.SansSerif` compressed fallback if font asset loading is unavailable offline (bundle the actual `.ttf` file in `res/font/` so it works 100% offline!).
- **Strict Formatting Rules:**
  - All labels and headers MUST be `UPPERCASE` with slight letter-spacing (`0.06.em`).
  - Every live numeric readout (`PITCH`, `SLANT R`, `GSD`, `VEL`, `SWEEP HZ`, `SHUTTER`, `ISO`, `CROP Y`) MUST set `fontFeatureSettings = "tnum"` (**tabular numerals**) so digits never shift layout width at 12-30 Hz.
  - Include authentic LCARS numeric sub-block identifiers along horizontal rails (e.g., `47-1792`, `OPT-2X`, `EIS-SYNC`).

---

## 5. Landscape Screen Zone Map (`sensorLandscape`)
1. **Top Viewport Zone (~48% screen height):**
   - Left `68.dp` `GoldenAmber` `TopLeft` `LcarsElbow` containing stacked status codes (`SCAN` / `REC`, FPS/Hz readout, mute toggle pill).
   - Top `22.dp` segmented header rail (`ApricotPeach` + `Periwinkle` segments separated by `4.dp` black gaps) displaying live tabular telemetry:
     `PITCH 45° [SYNCED]  |  R 1.50M  |  0.59 MM/PX  |  VEL 0.42 M/S  |  1/500S  ISO 640`
   - Wide CameraX `PreviewView` filling the remaining horizontal width:
     - Draw the active **`1792 x 384` (5-tile)** or **`1088 x 384` (3-tile)** push-broom scan strip boundary as a crisp `Periwinkle` / `IceBlue` bracket that moves vertically in real time as `cropY` stabilizes pitch!
     - Draw subtle `1.dp` vertical tick marks at the 5 tile seams (`x = 64, 416, 768, 1120, 1472`, `1856`) and a right-edge **vertical pitch-stabilization travel gauge** showing `cropY` within `[0..696]` (turning `MarsRed` if `isPitchClamped`).
2. **Bottom Tactical Deck (~52% screen height):**
   - Left `68.dp` `Lavender` `BottomLeft` `LcarsElbow` housing primary mode pills: `SCAN`, `LOG DATA`, `SIM`, `AUDIO`.
   - **Column 1 — Optics & IMU Pitch Stabilization (`OPTICS // EIS`):**
     - `2.0X SENSOR CROP [ON]` pill (default ON)
     - `<= 1/500S SHUTTER [ON]` pill (default ON)
     - `EIS PITCH LOCK [ON]` + `SYNC PITCH (45°)` actuator pill
     - `SWATH: 5-TILE (1792PX)` / `3-TILE` toggle pill
     - Segmented LCARS bar-graph slider for `CANOPY EV BIAS (-2.0 .. +1.0 EV)`
   - **Column 2 — Target Mineral Matrix & Sim/Gate Controls (`PROSPECT MATRIX`):**
     - 2x2 grid of wide hemispherical LCARS target toggle pills (`01 AGATE / CARNELIAN`, `02 RED / YELLOW JASPER`, `03 DALLASITE / NEPHRITE`, `04 PETRIFIED WOOD`) showing active latch state and colour key.
     - Inline Simulation scenario cycler (`LIVE CAM` / `MIXED GRAVEL SIM` / `OVERSPEED TEST`) so testing never opens a modal sheet.
