# GneissFind — Execution Roadmap & Acceptance Criteria (`docs/ROADMAP.md`)

## Milestone 1 — Baseline Push-Broom Skeleton *(Completed — `3f1dee0`)*
- [x] Initial Android project (`com.gneissfind.app`), `<= 1/500 s` shutter clamp + floating ISO, synthetic gravel simulator, 3-of-6 voter, speed gate, and `AudioTrack` sonifier.

## Milestone 1B — Reversible Landscape, TNG LCARS, 5-Tile Swath & 120 Hz IMU EIS *(Completed — `fc80785`)*
- [x] Validated on Pixel 10 Pro XL: reversible `sensorLandscape`, default `2.0x` sensor crop + dynamic `Camera2CameraControl` exposure/EV servo, `5 x 384x384` (`1792x384`) push-broom cropper, 120 Hz GPU RenderNode IMU pitch stabilization, and 6-role TNG LCARS UI.

---

## Milestone 2 — RF-DETR-Nano COCO Stand-In Benchmark & Integration *(Completed & Hardware-Benchmarked — see [`docs/ENGINEERING_LEDGER.md §3.5`](ENGINEERING_LEDGER.md#35-milestone-2--rf-detr-nano-litert-gpu-export-litert_cl-operator-engineering--pixel-10-pro-xl-benchmark-4825ef5-965d027-head))*

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
- [x] **Python Parity Tests (`ml-pipeline/`):** `uv run pytest` passes PyTorch-vs-LiteRT numerical/box parity (`IoU = 1.0000`, `max_diff <= 8.2e-3`) and empirically verifies `cup=47`, `apple=53`, `toothbrush=90`, and `remote=75` class indices in `rf_detr_coco_metadata.json`.
- [x] **Pixel 10 Pro XL 4-Way LiteRT GPU FP16 Benchmark (`58110DLCQ00003`):** Eliminated all `TOPK_V2`, `GATHER_ND`, `BROADCAST_TO`, `FLOOR_DIV`, `RELU_0_TO_1`, and rank $\ge 5$ ops so `2-Graph B=5` compiles 100% into single `LITERT_CL` GPU partitions (`909/909` nodes in `Part 1`, `350/350` nodes in `Part 2`), achieving **`75.37 ms/tile` (`13.27 tiles/sec`)** and **`376.86 ms/sweep` (`p50`) / `363.01 ms/sweep` (`min`)** across `5` tiles (`148.51 GFLOPs` at `394 GFLOP/s` sustained) while maintaining locked 120 Hz RenderNode EIS and zero GC pause spikes.
- [x] **Cross-Tile Seam Union Merging & Confuser Muting:** `PushBroomCropper.mergeCrossTileDetections` fuses elongated `toothbrush`/`remote` fragments across the `32 px` overlap band (`IoU >= 0.45f OR IoM >= 0.65f OR seam-straddling with vertical overlap >= 0.60f`) into union boxes, while routing unmapped COCO classes to muted `HARD_CONFUSER`.
- [x] **3D IMU World-Ray AR Tracking & 120 Hz Ego-Motion Reprojection:** `GroundPlaneRaycaster` + `TemporalTargetTracker` anchor detections as rotation-invariant 3D world rays ($\Delta\theta \le 3.8^\circ$), lock in 2 consecutive high-confidence ($\ge 0.72$) sweeps with 2-of-6 maintenance hysteresis, and forward-reproject confirmed reticles at 120 Hz while suppressing off-strip ghost brackets and audio pings.
- [x] **Build, Annotations, Commits & Ledger:** `./gradlew -p android assembleDebug testDebugUnitTest` (`18/18` unit tests), `uv run pytest` (`6/6` tests), and `comment_diff_check.py --strict` (`19/19` Kotlin files) pass, `docs/ENGINEERING_LEDGER.md` and `docs/INTERNALS.md` are updated, and `/tmp/gneissfind-sync.md` is generated.

---

## Milestone 2B — Achieving >= 5.0 Hz Sustained Full-Swath Detection on Pixel 10 Pro XL *(Completed & 180s Thermal-Soak Verified — see [`docs/M2B_OVERNIGHT_REPORT.md`](M2B_OVERNIGHT_REPORT.md))*

### Goal
Overcome the live-app GPU bandwidth and thermal bottleneck (`420–577 ms/sweep`, `1.5–2.4 Hz` in `RF-DETR-Nano 384 B=5`) to achieve **`>= 5.0 FPS` (`<= 200 ms` sweep period; stretch `>= 8.0 FPS`) sustained** across a **180-second live CameraX + 120 Hz LCARS thermal soak** on the Pixel 10 Pro XL (`Tensor G5`) while preserving the full `1792 px` (`93.3%` landscape width) horizontal FOV and small-pebble recall (`28–45 px` targets representing `20–26 mm` stones at `0.59 mm/px` GSD).

### Milestone 2B Acceptance Criteria & Empirical Results
- [x] **Anti-Cheating Dual-Scale Strip Recall Oracle (`ml-pipeline/tests/eval_small_proxy_strip.py`):** Synthesizes deterministic `1792x384` gravel-textured evaluation strips with `4` medium targets (`120–200 px`, including a cross-tile seam target) and `12` small pebble targets (`28–45 px` minor-axis width) and evaluates candidate detectors end-to-end with `IoU >= 0.45 OR IoM >= 0.65` cross-tile union merging.
- [x] **LiteRT `2.3.2` Upgrade, Zero-Copy `encMemory`, & Lock-Free 3-Slot Async Pipeline:** Upgraded to `com.google.ai.edge.litert:litert:2.3.2` (`litert-gpu:2.3.2`, `litert-npu-runtime-google-tensor:2.3.2`), eliminated the `5.9 MB` `part1Outputs[0] -> part2Inputs[0]` host round-trip copy via `part2ZeroCopyInputs`, and decoupled CameraX `ImageAnalysis` (`~3 ms` YUV extraction + immediate `imageProxy.close()`) from a dedicated `GneissFind-Inference` thread via a 3-slot lock-free `PreparedSweepSlot` ring buffer.
- [x] **`W8-Dyn` (`dynamic_wi8_afp32`) GPU Weight Quantization + Top-K=`100` Surgery:** Compressed `RF-DETR-Nano Part 1 + Part 2` by **`3.7x` (`104.5 MB -> 28.1 MB`)** with zero recall loss using `dynamic_wi8_afp32` and `allowSrcQuantizedFcConvOps = true` on the Tensor G5 GPU, while trimming Stage-2 decoder queries from `K=300` to `K=100` (`-46%` Part 2 latency, `0.0%` recall loss).
- [x] **Tensor G5 NPU Postmortem & 27-Configuration Pareto Bake-Off:** Documented why `Accelerator.NPU` on standalone APKs hits `kLiteRtStatusErrorNotFound` (AICore dynamic NPU provider requirement), contends with Camera HAL `G3ACoordinator` on `/dev/edgetpu`, and suffers `W8A8` ViT activation collapse (`100% -> 12.5%` small recall), whereas `W8-Dyn` on GPU FP16 delivers a `28–33%` speedup with `100%` FP32 recall parity. Also benchmarked 1-Graph `YOLO26s` / `YOLO26n` (`384/320/256 B=5`, `76.1–55.7 ms` / `13.1–18.0 FPS`).
- [x] **180-Second Continuous Live-App Thermal Soaks (`t = 150..180s`):**
  - **`S-01` (`RF-DETR 256 B=3 W8-DYN`, Default Apache-2.0):** **`108.2 ms` (`9.24 FPS`) cold** (beating the `>= 8.0 FPS` stretch goal) and **`152.2 ms` (`6.57 FPS` batch / priority target tile, `3.29 Hz` full 5-tile swath) sustained at `t = 150..180s`**.
  - **`S-02` (`RF-DETR 320 B=3 W8-DYN`, `100%` Small `38-45px` Recall Apache-2.0):** **`141.5 ms` (`7.07 FPS`) cold** and **`187.9–204.2 ms` (`4.90–5.32 FPS`) under `Thermal Status: 1`**.
  - **`S-03` (`RF-DETR 256 B=5 W8-DYN`, All 5 Tiles Every Sweep Apache-2.0):** **`158.0 ms` (`6.33 FPS`) cold** and **`206.0 ms` (`4.85 FPS` across all 5 tiles simultaneously) at `t = 150..180s`** (`3.2x` faster than the `577 ms` baseline).
  - **`S-04` (`YOLO26-S 384 B=5 GPU`, AGPL-3.0 Reference):** **`76.1 ms` (`13.15 FPS`) cold** and **`128.3 ms` (`7.80 FPS` across all 5 tiles) at `t = 150..180s`**.
- [x] **Multi-Model LCARS `SOURCE` Drawer & Report:** Bundled the finalist `.tflite` models into `android/app/src/main/assets/`, wired runtime hot-swapping in `LcarsSourceDrawerOverlay` (`DetectorSourceOption`), and published [`docs/M2B_OVERNIGHT_REPORT.md`](M2B_OVERNIGHT_REPORT.md).

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
