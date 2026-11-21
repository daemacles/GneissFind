---
trigger: glob
globs: "android/**/ui/**"
description: Classic 2360s Star Trek TNG LCARS visual grammar, 8707282 6-role semantic color taxonomy, 120 Hz RenderNode EIS viewport, and inline BEAM MCAP control.
---

# GneissFind Classic 2360s TNG LCARS UI Rules (`INV-UI-001`, `INV-UI-002`)

## 1. Preserve Existing Milestone 1B–2D LCARS Invariants
Do NOT redesign or regress the working LCARS layout:
1. **Pure `#000000` OLED Canvas & `4.dp` Gutters:** Zero translucent cards, drop shadows, gradients, or Material borders.
2. **6-Role Semantic Color Taxonomy (`8707282`):** Preserve `LcarsSemanticColors` (`0.05f` HSV value-stepping), custom swept `LcarsElbow` concave fillets, `Antonio` / `Barlow Condensed` uppercase typography with `fontFeatureSettings = "tnum"` (tabular numerals), and silent UI taps (`isSoundEffectsEnabled = false`, `80 ms` `#FFFFFF` press flash).
3. **Viewport & 120 Hz RenderNode EIS (`fc80785`):** Keep `ClippingCameraViewportContainer` unconditional child layout and `120 Hz` `RenderNode` `previewView.translationY` pitch decoupling completely intact.
4. **Anti-Camouflage Video Brackets:** Keep `2.dp` `#000000` outer keyline halos around high-luminance target brackets (`#00FFFF`, `#FFFF00`, `#00FF88`, `#FFAA00`), and keep `HardConfuser` strictly muted in `SCAN` mode.

## 2. Minimal Milestone 2.5 Additions (Inline Only — Zero Modal Sheets)
1. **`LOG DATA` -> `.mcap` Telemetry:** When `LOG DATA` is latched active (`MarsRed`), display live `.mcap` bag size (`MB`) and recorded sweep count in tabular numerals, plus an inline **`FLAG CONFUSER`** pill that writes a `/user/confuser_flag` bookmark event into the active Rust `.mcap` stream.
2. **Inline `BEAM MCAP` Control (Inside Existing `SRC` Drawer or `LOG` Sub-Row):** Add a compact LCARS pill (`BEAM MCAP`) + status readout (`IDLE` / `SYNCING XX MB/S` / `COMPLETE`) triggering `EngineClient.syncPendingMcaps(endpointTicketOrHost)` so field logs transfer wirelessly to the Dell T7920 without opening a modal popup or touching a wet USB-C port.
