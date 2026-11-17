---
trigger: glob
globs: "android/**"
description: Standing Classic 2360s TNG LCARS design language, semantic color roles, tabular stability, ambidextrous ergonomics, and outdoor AR overlay principles.
---

# Classic 2360s TNG LCARS UI & Field UX Principles

## 1. Framing, Geometry & OLED Discipline
- **Pure `#000000` Canvas & Crisp Gutters:** Anchor all layouts on true `#000000` black with uniform `4.dp` black gutters between bars, elbows, and pills. Never use drop shadows, gradients, blurs, translucent cards, or modal dialogs over the camera viewport.
- **Architectural Framing:** Frame viewports and control decks with swept LCARS elbows, slim telemetry rails, and capsule/flat-ended pills. Reserve outer shoulders for permanent app identity and compile-time build provenance.
- **OLED Longevity:** Support low-luminance viewport dimming during prolonged idle Eyes-Up operation, waking immediately on touch or target activity.

## 2. Semantic Color Roles
Color communicates **control behavior and system state**, never arbitrary decoration (use subtle value steps to separate adjacent blocks of the same role):
- **Binary Toggles:** `#FF9900` Golden Amber when active; `#596480` Slate Blue when inactive/disabled.
- **Multi-State Selectors:** `#9999FF` Periwinkle for cycling between discrete operating modes.
- **Actions & Drawers:** `#FFCC66` TNG Gold for one-shot triggers, steppers, and expandable selectors.
- **Live Telemetry:** `#99CCFF` Ice Blue for continuously updating sensor and timing readouts.
- **Configuration & Identity:** `#FFCC99` Peach for static state, optics configuration, and provenance Chrome.
- **Alerts & Release Actions:** `#CC4444` Mars Red for critical limits, active data capture, and immediate lock-release / abort controls.
- **1:1 Gauge Correlation:** Any bar or meter must share the exact semantic color of its label and numeric value so the relationship is unmistakable at a glance.

## 3. Typography, Layout Stability & Non-Redundancy
- **Tabular Uppercase Display:** Use condensed uppercase `Antonio` typography with tabular numerals (`tnum`) on all counters, timers, angles, and rates.
- **Zero-Jitter Readouts:** Live numbers and HUD badges must never resize their containers or shift neighboring elements as digit counts change (use fixed-width slots or monotonic high-water-mark widths).
- **Single Source of Truth:** Display each metric in one canonical location. Avoid repeating the same range, accuracy, or size readout across headers, side panels, and viewport overlays.

## 4. Ambidextrous Ergonomics & Silent Chrome
- **Two-Handed Landscape Reach:** Place high-frequency field actions near the horizontal center of landscape layouts so they are equally reachable by either left or right thumb.
- **Field Touch Targets:** Maintain `>= 48.dp` effective hit bounds on interactive controls and tap-selectable viewport reticles for outdoor, walking, or gloved use.
- **Acoustic Exclusivity:** Keep all UI touches silent (`isSoundEffectsEnabled = false`, using brief `#FFFFFF` tap-flash feedback instead) so interface chrome never competes with audio sonification.

## 5. Viewport Continuity & Anti-Camouflage Symbology
- **Persistent Camera Surface:** Transitions between scanning and target-inspection modes must re-frame the existing camera view in place without tearing down or rebinding the underlying preview surface.
- **Task-Focused Chrome:** Collapse non-essential configuration controls during focused inspection/homing modes to maximize live video area while retaining LCARS structural framing and an unmistakable exit action.
- **High-Contrast Keylined Overlays:** Every reticle, bracket, crosshair, directional indicator, and ROI boundary drawn over live video must carry a `#000000` outer keyline halo for legibility against bright or wet gravel.
- **Honest Visual Lock & Uncluttered Targets:** Reticle geometry (e.g., solid vs. dashed crosshair) must reflect actual live tracking lock rather than nominal mode. Keep tracked subjects free of floating text labels that flicker or occlude the target, and keep background work (such as shader compilation) non-blocking via compact inline status banners.
