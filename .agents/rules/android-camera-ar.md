---
trigger: glob
globs: "android/**"
description: Kotlin Android shell boundaries, fc80785 Camera2 invariants, zero-copy YUV JNI bridge (<2ms hold), synthetic HIL injection for dark/stationary testing, and Iroh sync hook.
---

# Android Shell, Camera2 Invariants & Zero-Copy JNI Rules

## 1. Strict Shell Responsibility (Dropbox Owl Guardrail)
Kotlin in `android/` acts strictly as the **Hardware HAL & LCARS Presentation Shell**:
- **Keep in Kotlin:** `MainActivity` (`sensorLandscape`), `CameraX` + `Camera2CameraControl` (`fc80785` shutter clamp `<= 1/500 s`, `2.0x` lossless zoom crop, EV-scaled luma servo), `ClippingCameraViewportContainer` + `120 Hz` `RenderNode` `previewView.translationY` visual EIS, `SensorManager` (`TYPE_GAME_ROTATION_VECTOR` + `remapCoordinateSystem`), `AudioTrack` continuous phase-coherent PCM generator, and Jetpack Compose LCARS UI.
- **Delegate to Rust (`EngineClient`):** YUV strip cropping, pitch-to-`cropY` computation, `TopK` + cross-tile IoU/IoM NMS, homography/`WorldRay3D` multi-frame tracking, N-of-M voting, acoustic parameter calculation (`freqHz`, `pulseRateHz`, `overspeedAlert`), `.mcap` session recording, and LAN bag transfer.

## 2. Zero-Copy Sensor Hot Path (`INV-CAM-002`)
1. **IMU Stream (`200 Hz`):** On `SensorEvent` (`TYPE_GAME_ROTATION_VECTOR`), remap axes for `ROTATION_90` / `ROTATION_270` (or pass raw quaternion + `displayRotation` enum so Rust remaps deterministically!) along with `event.timestamp` (`CLOCK_BOOTTIME` ns) via a lightweight primitive JNI call `nativePushImu(timestampNs, qx, qy, qz, qw, displayRotation)`. Zero object allocations per event.
2. **Camera Frame Stream (`YUV_420_888`):**
   - Inside `ImageAnalysis.Analyzer`, extract `imageProxy.planes[0..2].buffer` (`DirectByteBuffer`), `rowStride`, `pixelStride`, and `imageProxy.imageInfo.timestamp`.
   - Call `nativeExtractStrip(timestampNs, yBuf, uBuf, vBuf, width, height, yRowStride, uvRowStride, uvPixelStride)` which copies/converts **only** the `1792x384` stabilized strip into Rust's pre-allocated ring slot in `< 1.8 ms`, then **immediately call `imageProxy.close()`** in `finally` BEFORE running GPU inference! This guarantees `1bfdda0` Camera2 HAL ring-buffer safety (`< 2.0 ms` hold).

## 3. Dark-Room / Stationary Hardware-in-the-Loop (HIL) Verification Mode (`SYS-HIL-001`)
Because automated overnight builds often run with a USB-connected Pixel 10 Pro sitting stationary in a dark room (`luma ~ 0`, `angular_velocity ~ 0`):
1. **Dark-Frame Camera2 Survival:** Ensure the dynamic `Camera2CameraControl` EV-scaled luma servo (`fc80785`) clamps `SENSOR_SENSITIVITY` cleanly at `SENSOR_INFO_SENSITIVITY_RANGE.upper` without integer overflow, divide-by-zero on zero-luma histograms, or endless request thrashing when the room is pitch black.
2. **On-Device Synthetic Stimulus Injector (`InstrumentationTest` / `SIM` Parity Mode):**
   - Implement an `androidTest` suite (`RustPipelineParityInstrumentedTest`) AND a debug `adb` intent / `SIM` fixture mode that feeds deterministic synthetic & COCO stand-in `YUV_420_888` buffers (`cup`, `apple`, `toothbrush`, `remote` rendered with padded `rowStride = 1984` + synthetic `CLOCK_BOOTTIME` IMU walk trajectory) through the **real on-device Rust JNI + Pixel 10 Pro `LITERT_CL` 2-Graph `W8-Dyn` GPU pipeline**!
   - Measure and log on-device 5-tile sweep latency (`p50`, `p95` over 100 sweeps; pass gate `<= 125.0 ms`), native heap stability (`Debug.getNativeHeapAllocatedSize()` delta `< 5 MB` over 300 sweeps), live dark-frame Camera2 + stationary IMU `.mcap` recording, and LAN/loopback `BEAM MCAP` transfer.
