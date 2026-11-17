---
trigger: glob
globs: "{ml-pipeline/**,android/app/src/main/java/com/gneissfind/app/inference/**}"
description: LiteRT 2.3.2 W8-Dyn GPU inference invariants, 3-slot warm SRC LRU, disk shader cache, and B=1 320x320 Lock-On runtime rules.
---

# LiteRT 2.3.2 GPU Inference, Warm LRU & Replay Compatibility Rules

## 1. Validated Milestone 2B/2C Runtime Invariants (DO NOT REGRESS)
- **LiteRT `2.3.2` GPU `W8-Dyn` (`dynamic_wi8_afp32`):** Keep `allowSrcQuantizedFcConvOps = true` and `FP16` precision enabled on all `RF-DETR-Nano` (`256 B=3`, `320 B=3`, `256 B=5`, `320 B=1`) `CompiledModel` instances.
- **Zero-Copy `encMemory` Passthrough:** Keep `part1Outputs[0]` bound directly as `part2Inputs[0]` (`part2ZeroCopyInputs`).
- **Disk Shader Cache & 3-Slot Warm `SRC` LRU (`8d8be41`, `c307819`):** Preserve per-profile `part1`/`part2` `modelCacheKey` serialization in `context.codeCacheDir/litert_gpu_cache` and the 3-slot `WarmPatrolSlot` LRU (`~0.8-1.5 ms` warm switches) alongside the resident `rf_detr_nano_320_b1_w8dyn` Lock-On runner.

## 2. Identical GPU Execution Under `ReplayCameraSource`
- When `ReplayCameraSource` feeds decoded `1920x1080` `YUV_420_888` / `NV12` frames from `clip.mp4`, route them through the **exact same `PushBroomCropper` -> `PreparedSweepSlot` -> `RfDetrTwoGraphGpuDetector` (`PATROL` + `B=1` `LOCK-ON`) GPU pipeline** as live CameraX frames. Do NOT mock or bypass neural GPU inference during `.mp4` replay—replay exists specifically to reproduce real model scores, jitter, specular drops, and GPU latency deterministically!
