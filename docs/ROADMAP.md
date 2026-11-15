# GneissFind — Execution Roadmap & Acceptance Criteria (`docs/ROADMAP.md`)

## Milestone 1 — Baseline Push-Broom Skeleton *(Completed — `3f1dee0`)*
- [x] Initial Android project (`com.gneissfind.app`), `<= 1/500 s` shutter clamp + floating ISO, synthetic gravel simulator, 3-of-6 voter, speed gate, and `AudioTrack` sonifier.

## Milestone 1B — Reversible Landscape, TNG LCARS, 5-Tile Swath & 120 Hz IMU EIS *(Completed — `fc80785`)*
- [x] Validated on Pixel 10 Pro XL: reversible `sensorLandscape`, default `2.0x` sensor crop + dynamic `Camera2CameraControl` exposure/EV servo, `5 x 384x384` (`1792x384`) push-broom cropper, 120 Hz GPU RenderNode IMU pitch stabilization, and 6-role TNG LCARS UI.

---

## Milestone 2 — RF-DETR-Nano COCO Stand-In Benchmark & Integration *(Current Active Task)*

### Goal
Validate the real end-to-end `RF-DETR-Nano` (`384x384`, Apache-2.0) LiteRT GPU FP16 inference pipeline inside GneissFind on the Pixel 10 Pro XL using baseline MS-COCO weights and 4 household stand-in classes (`cup`, `apple`, `toothbrush`, `remote`) mapped transparently onto the existing 4 mineral buttons—retiring mobile ViT latency, seam-splitting, GC, and thermal risks before field data collection.

### Copy-Paste Prompt for Antigravity
Paste this prompt into Antigravity:

```text
Read @AGENTS.md, @docs/ENGINEERING_LEDGER.md, @.agents/rules/android-camera-ar.md, @.agents/rules/lcars-ui.md, @.agents/rules/litert-inference.md, and @docs/ROADMAP.md in full—Milestone 2 (RF-DETR-Nano COCO Stand-In Vertical Slice) is now active.

Execute Milestone 2 end-to-end without regressing any `fc80785` or `8707282` hardware/UI invariants:
1. Set up `ml-pipeline/` with `uv` (pinned to `CUDA_VISIBLE_DEVICES=0` on the RTX 4090 where GPU is used) and create a reproducible export + verification script for pretrained `RF-DETR-Nano` (`384x384` MS-COCO object detection). Fuse `NHWC -> NCHW` permute and ImageNet `(x - mean)/std` normalization onto `[B, 384, 384, 3]` float32 `[0..1]` inputs matching `PushBroomSweepBuffers`.
2. Empirically verify the exported `.tflite` against PyTorch on real/sample test images containing `cup`, `apple`, `toothbrush`, and `remote` (`pytest` parity check) so 80- vs 91-logit COCO IDs cannot mismatch, and write `android/app/src/main/assets/rf_detr_coco_metadata.json` mapping `cup -> CHALCEDONY_AGATE`, `apple -> JASPER_RED_YELLOW`, `toothbrush -> GREEN_GEM_ROCK`, `remote -> PETRIFIED_WOOD`, and other confident COCO classes -> muted `HARD_CONFUSER`.
3. Benchmark 1-graph vs 2-graph LiteRT GPU FP16 execution AND static `B=5` batched vs `5 x B=1` sequential tile dispatch on the connected Pixel 10 Pro XL (`adb`), keeping whichever achieves the fastest steady-state 5-tile latency under the `<= 125 ms/sweep` (`>= 8.0 Hz`) pass floor (with shader disk caching and zero per-frame heap allocations on a dedicated inference thread).
4. Upgrade cross-tile seam deduplication in `PushBroomCropper` to fuse same-class boxes across the `32 px` seam using `IoU >= 0.45f OR IoM >= 0.65f` (union box) so elongated `toothbrush` and `remote` stand-ins spanning two tiles merge into one target, covered by unit tests.
5. Register the live LiteRT GPU detector as the default source in `LcarsSourceDrawerOverlay` (keeping all LCARS mineral UI labels untouched and displaying live GPU inference `MS` in the `SWEEP` telemetry readout), verify `./gradlew -p android assembleDebug testDebugUnitTest` and `uv run pytest` pass, make atomic one-sentence git commits, append full benchmark results (`1-graph vs 2-graph`, `B=5 vs 5xB=1`, p50/p95 ms, sustained Hz) to `docs/ENGINEERING_LEDGER.md`, and run `./scripts/pack_context.sh`.
```

### Milestone 2 Acceptance Criteria (Oracle)
1. **Python Parity Tests (`ml-pipeline/`):** `uv run pytest` passes PyTorch-vs-LiteRT numerical/box parity and empirically verifies `cup`, `apple`, `toothbrush`, and `remote` class indices in `rf_detr_coco_metadata.json`.
2. **Pixel 10 Pro XL Latency Floor:** `FIVE_TILE` (`5 x 384x384`) GPU FP16 inference achieves steady-state **`<= 125 ms/sweep` (`>= 8.0 Hz`)** (`p50`) over a 60-second continuous indoor scan while maintaining smooth 120 Hz RenderNode EIS and zero GC pause spikes.
3. **Indoor Household Stand-In Walk Test:** Walking past a `cup`, `apple`, `toothbrush`, and `remote` on the living-room floor at `~1.5 m` slant range highlights each with its corresponding mineral bracket + 3-of-6 acoustic stone ping (fusing elongated `toothbrush`/`remote` across tile seams), while ignoring chairs/rugs/books (`HARD_CONFUSER` muted).
4. **Build, Commits & Ledger:** `./gradlew -p android assembleDebug testDebugUnitTest` passes, atomic commits are recorded, `docs/ENGINEERING_LEDGER.md` logs the `B=5` vs `5xB=1` benchmark table, and `/tmp/gneissfind-sync.md` is generated.

---

## Milestone 3 — T7920 Auto-Labeling & Hard-Negative Copy-Paste Pipeline (`ml-pipeline/`)
- Build `SAM 2` video propagation scripts pinned to `GPU 1 & 2` (`2x RTX 4000 Ada`) to auto-segment `FieldDataLogger` salted gravel-bar PNG strips into 5-class bounding boxes (`4 PNW stone targets + hard_confuser`).
- Build Copy-Paste instance compositor pasting segmented raw PNW stones onto unsalted local hard-negative gravel frames.

## Milestone 4 — `RF-DETR-Nano` PNW Stone Fine-Tuning & Production Weight Swap
- Fine-tune `RF-DETR-Nano` (`384x384`, Apache-2.0) and optional `RF-DETR-Large` teacher on `GPU 0` (`RTX 4090 24GB`) using real + synthetic PNW gravel strips.
- Re-run the Milestone 2 `litert-torch` exporter to swap COCO stand-in weights + `metadata.json` for production 5-class PNW mineral weights with zero Android architecture rework.

## Milestone 5 — Field Calibration, Watershed Presets & Hard-Negative Mining
- Per-class confidence sliders, regional presets (Puget Sound beaches, Western Cascade rivers, Bellingham / Skagit bars), and one-tap field confuser flagging.

## Milestone 6 — Late-Stage Audio Polish & V2 Ergonomics
- Polyphonic/harmonic sonification curves, tactile haptics, optional LCARS trills, and hands-free chest-mount evaluation.
