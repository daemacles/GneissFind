---
trigger: glob
globs: "android/**"
description: CameraX/Camera2Interop shutter clamping, 2.0x lossless sensor crop, reversible landscape IMU pitch stabilization, 5-tile push-broom cropper, and ARCore speed gating.
---

# Android Camera, IMU Pitch Stabilization & Push-Broom Rules

## 1. Reversible Landscape (`sensorLandscape`) Geometry Synchronization
- `AndroidManifest.xml` MUST set `android:screenOrientation="sensorLandscape"` on `MainActivity`.
- Register a `DisplayManager.DisplayListener` (or `WindowMetrics` + rotation observer) tracking `Display.rotation` (`Surface.ROTATION_90` vs `Surface.ROTATION_270`):
  1. Update `Preview.targetRotation = rotation` and `ImageAnalysis.targetRotation = rotation` so CameraX delivers landscape-oriented frames (`1920 x 1080`) upright in both landscape directions.
  2. Update ARCore session geometry via `arSession.setDisplayGeometry(rotation, width, height)`.
  3. Remap IMU axes BEFORE computing pitch and roll:
     - `Surface.ROTATION_90`: `SensorManager.remapCoordinateSystem(R_raw, SensorManager.AXIS_Y, SensorManager.AXIS_MINUS_X, R_remapped)`
     - `Surface.ROTATION_270`: `SensorManager.remapCoordinateSystem(R_raw, SensorManager.AXIS_MINUS_Y, SensorManager.AXIS_X, R_remapped)`
  4. Verify sign convention: tilting the phone lens downward toward the user's boots increases pitch angle (`90 deg` = straight down nadir; `0 deg` = horizontal horizon; `45 deg` = nominal forward rockhounding slant).

## 2. CameraX + `Camera2Interop` Optical Controls
- **Default Startup State:**
  - `isTwoXCropEnabled = true` (`cameraControl.setZoomRatio(2.0f)` applied immediately when camera binds so Pixel 10 Pro engages its lossless 2x center sensor crop on the 50 MP wide sensor).
  - `isShutterClampEnabled = true` (exposure time clamped `<= 2,000,000 ns` = `1/500 s`).
- **Shutter Clamp + Floating ISO Implementation:**
  - Preserve working `Camera2Interop` capture request / capture callback control so `SENSOR_EXPOSURE_TIME` never exceeds `2,000,000 ns` when clamp is ON, while ISO (`SENSOR_SENSITIVITY`) dynamically adjusts to maintain target scene luminance offset by the user's Canopy Exposure Bias (`-2.0 EV` to `+1.0 EV`).
  - Read actual `CaptureResult.SENSOR_EXPOSURE_TIME`, `CaptureResult.SENSOR_SENSITIVITY`, `LENS_INTRINSIC_CALIBRATION`, and `SCALER_CROP_REGION` via `Camera2Interop.Extender.setSessionCaptureCallback` to feed accurate telemetry (`1/500s`, `ISO`, `fy`, and live `GSD mm/px`) to the LCARS HUD.

## 3. 5-Tile Cross-Track Swath Cropper (`1792 x 384` in `1920 x 1080`)
- Request `ResolutionSelector` targeting `1920x1080` (16:9 aspect ratio) on `ImageAnalysis` with `OUTPUT_IMAGE_FORMAT_YUV_420_888` and `STRATEGY_KEEP_ONLY_LATEST`.
- **Tile X Coordinates (`384 x 384` tiles, `32 px` overlap, `352 px` stride):**
  - **5-TILE Mode (Default):** `xOffsets = intArrayOf(64, 416, 768, 1120, 1472)`
    - Tile 0: `[64 .. 448)`
    - Tile 1: `[416 .. 800)`
    - Tile 2: `[768 .. 1152)`
    - Tile 3: `[1120 .. 1504)`
    - Tile 4: `[1472 .. 1856)`
    - Total strip width = `1792 px` (`93.3%` of `1920 px` width; `64 px` side margins).
  - **3-TILE Mode (Optional toggle):** `xOffsets = intArrayOf(416, 768, 1120)` (`1088 px` strip width).
- **Zero-Allocation Fast Path:** Reuse pre-allocated direct `ByteBuffer` / `IntArray` / GPU texture buffers per frame. Convert ONLY the active strip (`1792 x 384`, not the full `1920 x 1080` frame) from `YUV_420_888` to RGB float/uint8 tensors, then slice or stride into the 5 tiles. Always call `imageProxy.close()` in a `finally` block.
- **Cross-Tile Seam NMS:** Map detections from tile-local `[0..384]` coordinates into unified strip coordinates `(x_tile + xOffset, y_tile)` and run class-aware Non-Maximum Suppression (`IoU >= 0.45`) across adjacent tiles so stones sitting in the `32 px` overlap zone produce a single unified bounding box.

## 4. IMU-Driven ROI Pitch Stabilization (Electronic Image Stabilization)
- **Sensor Choice:** Use `Sensor.TYPE_GAME_ROTATION_VECTOR` sampled at `SENSOR_DELAY_GAME`. Do NOT use magnetometer-fused `TYPE_ROTATION_VECTOR` (wet Cascade basalt cobbles often contain magnetite that causes compass drift).
- **Stabilization Math:**
  - Reference pitch `thetaRef`: defaults to `45.0 deg` (`Math.toRadians(45.0)`), updatable anytime via the LCARS `"SYNC PITCH"` button (`thetaRef = currentPitch`).
  - Smooth raw pitch with a fast low-pass filter (`alpha = 0.25f`) only for UI display; for `cropY` computation, match the IMU timestamp as closely as possible to `imageProxy.imageInfo.timestamp`.
  - Compute vertical pixel focal length `fy` from `LENS_INTRINSIC_CALIBRATION` scaled to the `1920x1080` `2.0x`-cropped stream (fallback: compute from vertical FOV `~20.5 deg` at `2.0x` zoom -> `fy = (1080.0 / 2.0) / tan(toRadians(10.25))` `~ 2985 px`, or tunable calibration factor so `+/- 6.5 deg` maps cleanly across `[0, 696]`).
  - Compute stabilized strip top coordinate:
    `rawCropY = 348 + (fy * tan(currentPitchRad - thetaRefRad)).roundToInt()`
    *(Verify sign on device: if the user pitches the phone slightly DOWN toward their feet (`currentPitch > thetaRef`), the ground target moves UP toward the top of the image frame (`cropY` decreases toward `0`); if the user pitches slightly UP toward the horizon (`currentPitch < thetaRef`), the ground target moves DOWN (`cropY` increases toward `696`). Ensure sign keeps the strip locked onto the fixed ground patch!)*
  - Clamp: `cropY = rawCropY.coerceIn(0, 696)` (since `1080 - 384 = 696`).
  - Set `isPitchClamped = (rawCropY < 0 || rawCropY > 696)` to illuminate the LCARS `"PITCH LIMIT"` indicator when wrist tilt exceeds the frame buffer margin.

## 5. Slant Range, GSD Telemetry & Field Data Logger (`LOG DATA`)
- **Live Geometry Estimates:**
  - Given assumed waist height `h = 1.06 m` (or ARCore plane hit distance when available) and live `currentPitch`:
    `slantRangeM = h / sin(currentPitchRad.coerceAtLeast(toRadians(10.0)))`
  - Compute live cross-track `gsdMmPerPx` from `slantRangeM`, sensor horizontal FOV, and active zoom (`1.0x` vs `2.0x`).
- **Field Strip Logger (`LOG DATA` mode):**
  - Captures stabilized `1792 x 384` strip frames at configurable cadence (`2-5 Hz`) to app-scoped external storage (`Android/media/com.gneissfind.app/sessions/<timestamp>/`) as lossless PNG or `Q=95` JPEG alongside `metadata.jsonl` recording `{frame_id, timestamp_ns, zoom_ratio, exposure_ns, iso, pitch_deg, ref_pitch_deg, crop_y, is_pitch_clamped, slant_range_m, gsd_mm_px, ground_speed_mps}`.
