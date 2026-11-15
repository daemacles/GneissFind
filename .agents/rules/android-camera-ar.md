---
trigger: glob
globs: "android/**"
description: Hardware-validated CameraX/Camera2 exposure invariants, 120 Hz RenderNode IMU pitch stabilization, 5-tile push-broom cropper, and IoU+IoM cross-tile seam merging.
---

# Android Camera, 120 Hz EIS & Push-Broom Rules

## 1. Hardware-Validated Camera & Viewport Invariants (`fc80785`)
1. **`ClippingCameraViewportContainer` (`ScannerScreen.kt`):**
   - Always execute standard `onMeasure` and unconditional `onLayout` sizing on `previewView` so CameraX's asynchronous `TextureView` child acquires non-zero bounds immediately and never stalls `SurfaceRequest`.
   - Apply 120 Hz `SENSOR_DELAY_GAME` pitch stabilization strictly via `previewView.translationY = t.frameTopPx` on the GPU RenderNode—never call `requestLayout()` per IMU tick.
2. **Dynamic `Camera2CameraControl` AE & EV Servo (`PushBroomCameraController.kt`):**
   - Keep `Camera2Interop.Extender` strictly limited to `setSessionCaptureCallback` for telemetry (`SENSOR_EXPOSURE_TIME`, `SENSOR_SENSITIVITY`, `LENS_INTRINSIC_CALIBRATION`).
   - Drive `<= 1/500 s` (`2,000,000 ns`) shutter clamp + floating ISO exclusively via `Camera2CameraControl.setCaptureRequestOptions(...)`, and call `clearCaptureRequestOptions()` when `shutterClampEnabled == false`.
   - Scale `targetLuma` by `2^(0.5 * exposureBiasEv)` (clamped to `[42, 210]`) inside the closed-loop Y-plane luma servo so Canopy EV bias persists.
   - Read `CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM` synchronously after `bindToLifecycle` so default `2.0x` lossless sensor crop succeeds on cold launch.
3. **Reversible `sensorLandscape` (`ROTATION_90` & `ROTATION_270`):**
   - Maintain both `DisplayManager.DisplayListener` and `OrientationEventListener` so direct 180-degree landscape flips update CameraX `targetRotation`, `GroundPlaneRaycaster`, and ARCore `Session.setDisplayGeometry`.

## 2. 5-Tile Push-Broom Cropper & Elongated-Object Seam Merging (`PushBroomCropper.kt`)
- **Pre-Allocated Sweep Contract:**
  - `FIVE_TILE`: `5` tiles of `384x384`, `xOffsets = [64, 416, 768, 1120, 1472]`, `32 px` overlap, `1792x384` strip.
  - `THREE_TILE`: `3` tiles of `384x384`, `xOffsets = [416, 768, 1120]`, `32 px` overlap, `1088x384` strip.
  - `PushBroomSweepBuffers` provides reusable `FloatArray(384 * 384 * 3)` RGB `[0..1]` NHWC buffers.
- **Upgraded Cross-Tile Seam Deduplication (`IoU` OR `IoM` Union Merge):**
  - At `0.59 mm/px` GSD, elongated Milestone 2 stand-in objects (`toothbrush` ~190 mm, `remote` ~185 mm) and large petrified wood limbs (`>100 mm`) frequently span two adjacent `384x384` tiles across the `32 px` seam, producing two partial boxes whose standard `IoU` is `< 0.45f`.
  - Upgrade cross-tile suppression in unified strip coordinates: when two detections share the same class and either:
    1. `IoU(boxA, boxB) >= 0.45f`, OR
    2. `IoM(boxA, boxB) = intersectionArea / min(areaA, areaB) >= 0.65f`, OR
    3. Both boxes touch/straddle the same `[xOffset .. xOffset + 32]` seam band (within `8 px` margin) with vertical projection overlap `>= 0.60f`,
    **fuse** them into a single unified bounding box (`minLeft, minTop, maxRight, maxBottom`) with `score = max(scoreA, scoreB)` rather than discarding the non-overlapping half. Add unit tests in `PushBroomCropperTest.kt` verifying cross-seam fusion on wide (`300 px`) synthetic boxes.

## 3. Asynchronous Inference Decoupling
- Run `StoneDetector.detectSweep(...)` on a dedicated single-threaded high-priority coroutine/executor decoupled from the main UI thread and 120 Hz IMU callback, keeping `STRATEGY_KEEP_ONLY_LATEST` backpressure clean so camera preview and RenderNode EIS stay locked at 60/120 FPS even if a GPU sweep takes `60-120 ms`.
