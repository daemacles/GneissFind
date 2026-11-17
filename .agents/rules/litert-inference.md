---
trigger: glob
globs: "{ml-pipeline/**,android/app/src/main/java/com/gneissfind/app/inference/**}"
description: LiteRT 2.3.2 W8-Dyn GPU inference invariants, per-profile OpenCL shader disk caching, async model switching, and B=1 320x320 Lock-On export rules.
---

# LiteRT 2.3.2 GPU Inference, Shader Caching & `B=1` Lock-On Rules

## 1. Validated Milestone 2B Runtime Invariants (`1bfdda0` — Preserve!)
- **LiteRT `2.3.2` GPU `W8-Dyn` (`dynamic_wi8_afp32`):** Always enable `allowSrcQuantizedFcConvOps = true` and `FP16` precision in `CompiledModel.GpuOptions` so `28 MB` 8-bit weights stream directly into FP16 OpenCL shaders on the Tensor G5 GPU with `0.0%` small-pebble recall loss. Do NOT use static `W8A8` (`static_wi8_ai8`) on `RF-DETR-Nano` (collapses ViT recall) or `Accelerator.NPU` (blocked by AICore registry + Camera HAL `G3ACoordinator` contention).
- **Zero-Copy `encMemory` Passthrough:** Keep `part1Outputs[0]` bound directly as `part2Inputs[0]` (`part2ZeroCopyInputs`) so the `5.9 MB` encoder memory tensor never copies to CPU RAM.
- **Host `TopK=100` Bridge:** Keep zero-allocation `selectTopKProposalsFromMaxScores` (`K=100` of `256`/`400`/`576` tokens, `< 0.8 ms`).

## 2. Mandatory Per-Profile OpenCL Shader Disk Caching & Async Loading
- **Root Cause of 9-20s Model-Switch Delays:** First-run OpenCL `clBuildProgram` compiles hundreds of kernels per unique input batch/resolution shape (`256 B=3`, `320 B=3`, `256 B=5`, `320 B=1`, `YOLO26 384 B=5`).
- **Fix Requirements:**
  1. Configure LiteRT GPU kernel serialization (`serializationDir = File(context.codeCacheDir, "litert_gpu_cache").apply { mkdirs() }.absolutePath` and a deterministic per-profile `modelToken = "${profile.id}_litert232_v1"`) on EVERY `CompiledModel` (`Part 1` and `Part 2` get distinct tokens: `"${profile.id}_p1_v1"` and `"${profile.id}_p2_v1"`!).
  2. Perform all `CompiledModel` creation + `1` dummy warmup pass asynchronously on `GneissFind-Inference` (never blocking the main UI thread or freezing `PreviewView`), surfacing `isCompilingModel = true` in UI state so `ScannerScreen` displays `INITIALIZING GPU KERNELS...` cleanly while switching.
  3. Verify via Logcat benchmark that switching back to an already-cached `LiveModelProfile` completes in **`<= 500 ms`** (`< 0.5 s`).

## 3. Foveated `B=1` `320x320` `W8-Dyn` Lock-On Export (`ml-pipeline/`)
- Extend `ml-pipeline/` export scripts to generate **`rf_detr_nano_320_b1_w8dyn`** (`Part 1` + `Part 2`, `batch_size = 1`, `input_resolution = 320`, `K = 100`, `dynamic_wi8_afp32` `W8-Dyn`, `coco_91_sparse` metadata).
- Why `320x320` `B=1`:
  - `320x320` achieved **`100.0%` Medium and `100.0%` Small `38-45px` recall** (`91.7%` `28-45px`) in the Milestone 2B Dual-Scale Oracle while executing only `19.8 GFLOPs` for a single tile (`~45-60 ms` = **`16-22 FPS`** on Tensor G5 GPU!).
  - Keep the `B=1` `320x320` `CompiledModel` warm in memory alongside the active `PATROL` model (`~28 MB` total weights) OR share weights cleanly so tapping a stone transitions from `PATROL` (`B=3` / `B=5`) into `LOCK-ON` (`B=1`) in **`< 50 ms`** with zero shader compilation pause!
- Verify `rf_detr_nano_320_b1_w8dyn` parity in `uv run pytest` before bundling in `android/app/src/main/assets/`.
