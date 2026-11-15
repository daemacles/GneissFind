---
trigger: glob
globs: "{ml-pipeline/**,android/app/src/main/java/com/gneissfind/app/inference/**}"
description: Milestone 2 RF-DETR-Nano LiteRT GPU FP16 export, 1-graph vs 2-graph and B=5 vs B=1 benchmarking, COCO stand-in index verification, and zero-GC Android runtime rules.
---

# LiteRT GPU Inference & RF-DETR-Nano Export Rules (Milestone 2)

## 1. Python Environment & GPU Pinning (`ml-pipeline/`)
- Manage Python dependencies reproducibly inside `ml-pipeline/` with `uv` (`pyproject.toml` + `uv.lock`, Python 3.11/3.12 on Ubuntu 26.10).
- Pin all PyTorch / `rfdetr` / `litert-torch` (`ai-edge-torch`) export & verification jobs strictly to `CUDA_VISIBLE_DEVICES=0` (`RTX 4090 24GB`) or CPU where required by the LiteRT converter.

## 2. Fused Input Wrapper & Tensor Contract
- `PushBroomCropper` produces `FloatArray(384 * 384 * 3)` in **`NHWC` layout, RGB channel order, `[0.0, 1.0]` float32** per tile.
- Wrap pretrained `RFDETRNano` (`384x384` resolution, pretrained MS-COCO weights) in a clean `torch.nn.Module` export wrapper that:
  1. Accepts `nhwc_rgb_01: torch.Tensor` of static shape `[B, 384, 384, 3]` (`dtype=torch.float32`).
  2. Permutes `(0, 3, 1, 2)` -> `[B, 3, 384, 384]` (`NCHW`).
  3. Applies ImageNet normalization inside the graph: `(x - mean) / std` with `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]`.
  4. Outputs normalized center-format boxes `boxes_cxcywh` in `[0.0, 1.0]` (`[B, num_queries, 4]`) and `scores` (`sigmoid(logits)`, `[B, num_queries, num_classes]`).
- Export with `FP32` weights targeting LiteRT GPU OpenCL/WebGPU (`Fp16` precision enabled in Android LiteRT GPU options so the Pixel 10 Pro XL GPU executes all supported ops natively in FP16 without PTQ INT8 bounding-box collapse).

## 3. Graph Topology & Batch Strategy Benchmark Matrix
Run a disciplined bake-off before locking the shipped Milestone 2 runtime:
1. **1-Graph vs. 2-Graph Check:**
   - First export the **single-graph** `RF-DETR-Nano` `.tflite` (`rfdetr` `litert-torch` path). Inspect operator compatibility and measure on-device LiteRT GPU delegate execution on the Pixel 10 Pro XL.
   - If `GATHER_ND` / `TOPK` / deformable or query-selection ops cause LiteRT GPU partition splits or CPU fallback costing `> 15 ms` (or pushing 5-tile latency near `125 ms`), export the **2-graph split** matching `RfDetrTwoGraphGpuDetector` (`part1` encoder/proposals + Kotlin `selectTopKQueries` + `part2` decoder).
2. **`B=5` Batched vs. `5 x B=1` Sequential Benchmark:**
   - Export both static **`B=5` (`[5, 384, 384, 3]`)** (single JNI/GPU dispatch; zero-pad tiles 3 & 4 when in `THREE_TILE` mode) and static **`B=1` (`[1, 384, 384, 3]`)** (looped `activeTileCount` times).
   - Measure steady-state warm latency over `>= 100` sweeps on the physical Pixel 10 Pro XL via `adb` benchmark / instrumentation logging (`p50`, `p95`, and sustained 60-second rate).
   - Select the fastest combination that comfortably clears the **`<= 125 ms/sweep` (`>= 8.0 Hz`)** Milestone 2 pass gate, and record the full comparison table in `docs/ENGINEERING_LEDGER.md`.

## 4. Mandatory COCO Stand-In Class Index Oracle (`rf_detr_coco_metadata.json`)
`rfdetr` checkpoints vary between **80 contiguous (0-indexed)** and **91 sparse MS-COCO category IDs** (notably, index `47` is `apple` in 80-class contiguous but `cup` in 91-class sparse!). Never hardcode guessed integer IDs in Kotlin:
- Write `ml-pipeline/scripts/verify_coco_export.py` (covered by `pytest`) that runs both the PyTorch wrapper and the exported `.tflite` interpreter on validation images containing **`cup`**, **`apple`**, **`toothbrush`**, and **`remote`**, asserts box IoU `>= 0.85` and score parity `< 1e-2` between PyTorch and `.tflite`, empirically confirms the exact integer class index activated by each of the 4 objects, and generates `android/app/src/main/assets/rf_detr_coco_metadata.json`:
  ```json
  {
    "model_name": "rf_detr_nano_coco",
    "input_resolution": 384,
    "batch_size": 5,
    "topology": "single_graph_or_two_graph",
    "num_queries": 300,
    "num_classes": 91,
    "score_threshold": 0.35,
    "stand_in_map": {
      "CHALCEDONY_AGATE": {"coco_label": "cup", "class_id": 47},
      "JASPER_RED_YELLOW": {"coco_label": "apple", "class_id": 53},
      "GREEN_GEM_ROCK": {"coco_label": "toothbrush", "class_id": 90},
      "PETRIFIED_WOOD": {"coco_label": "remote", "class_id": 75}
    },
    "unmapped_confident_to_hard_confuser": true
  }
  ```
  *(Note: exact `class_id` numbers above are illustrative for 91-class; the verification script MUST populate the empirically verified indices from the actual `rfdetr` checkpoint!)*
- Any detection with `score >= score_threshold` whose `class_id` is NOT in `stand_in_map` maps to `HARD_CONFUSER` (exercising silent clutter rejection on indoor chairs, books, keyboards, etc.).

## 5. Zero-Allocation Android GPU Runtime
- Pre-allocate direct `ByteBuffer`s (`ByteBuffer.allocateDirect(...).order(ByteOrder.nativeOrder())`) and reusable output arrays once during detector init; copy `PushBroomSweepBuffers` float tiles into the direct buffer without per-frame allocations.
- Initialize the LiteRT GPU delegate / `CompiledModel` (`GpuDelegateOptions` with `isPrecisionLossAllowed = true` / FP16 enabled, serialized shader cache enabled in `context.codeCacheDir` for fast warm startups) on the dedicated inference thread that invokes `.run(...)` (OpenCL/EGL thread-affinity safety!).
- Always provide graceful fallback to `SyntheticStoneDetector` with a logged warning if model assets are absent in unit-test environments.
