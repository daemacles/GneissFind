# GneissFind Sacred Invariants & Algorithm Specs (`qms/reqs/03_invariants_and_algos.md`)

## `INV-CAM-001`: `fc80785` Dynamic Camera2 Shutter Clamp & Cold-Start 2.0x Crop
- **Parents:** UN-001
- **Status:** Active
- **Oracle:** `2.0x` zoom crop (`ZoomRatio = 2.0f`) and `<= 1/500 s` (`2,000,000 ns`) exposure clamp engage at cold start via dynamic `Camera2CameraControl` only (never static `ALWAYS_OVERRIDE`), surviving zero-luma dark-room conditions without crash.

## `INV-CAM-002`: `1bfdda0` Zero-Copy `< 2.0 ms` `ImageProxy` Ring-Buffer Hold
- **Parents:** SYS-PERF-001
- **Status:** Active
- **Oracle:** `ImageProxy` `DirectByteBuffer` read completes and calls `imageProxy.close()` in `< 2.0 ms` (`p95`) prior to GPU inference so Camera2 HAL ring buffers never starve.

## `INV-UI-001`: `fc80785` + `8707282` Non-Occluding 120 Hz Classic TNG LCARS
- **Parents:** UN-002
- **Status:** Active
- **Oracle:** `ClippingCameraViewportContainer` unconditional child layout, `120 Hz` `RenderNode` `previewView.translationY` EIS, `#000000` canvas with `4.dp` gutters, 6-role `LcarsSemanticColors`, tabular numerals (`tnum`), and silent UI taps remain intact.

## `INV-GPU-001`: `1e813e2` + `1bfdda0` 2-Graph `W8-Dyn` `LITERT_CL` GPU Pipeline
- **Parents:** SYS-PERF-001
- **Status:** Active
- **Oracle:** Android detector preserves 2-Graph rank-`<=4` `W8-Dyn` (`allowSrcQuantizedFcConvOps = true`) `LITERT_CL` `Image2D` execution with native Rust `select_top_k_queries` bridge.

## `ALG-DET-001`: Workstation `ort` FP16 <-> `LiteRT` `W8-Dyn` Quantization Parity Gate
- **Parents:** UN-003, INV-GPU-001
- **Status:** Active
- **Oracle:** Automated desktop check (`cli_runner verify-parity`) confirms 2-Graph `W8-Dyn` `.tflite` detections match `OrtCudaBackend` on calibration tiles (`mean matched IoU >= 0.90`, zero silent class-index skew across `cup`, `apple`, `toothbrush`, `remote`).

## `ALG-TRK-001`: Multi-Frame World-Ray Tracking, N-of-M Voter & Acoustic Curve
- **Parents:** UN-001, UN-002
- **Status:** Active
- **Oracle:** Projects detections to local `WorldGroundFrame` (`nalgebra`/`faer`/`factrs`), accumulates N-of-M persistence votes (`default 3 of 6`), mutes `HardConfuser`, maps stone metric diameter (`20..60 mm`) to `[780..320 Hz]` and vote persistence to `[2..14 Hz]`, and raises `overspeed_warning` above `1.10 m/s`.

## `ALG-TUNE-001`: Open M2C/M2D Empirical Tuning Constants Registry (`RATIONALE-UNKNOWN`)
- **Parents:** ALG-TRK-001
- **Status:** Open-M3-Sweep
- **Oracle:** Centralizes all 6 `RATIONALE-UNKNOWN` tuning constants from `ENGINEERING_LEDGER.md` into a single serializable `PerceptionTuningConfig` struct in `crates/core_types` so Milestone 3 T7920 `.mcap` parameter sweeps calibrate and lock them empirically.
