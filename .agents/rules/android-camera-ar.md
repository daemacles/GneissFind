---
trigger: glob
globs: "android/**"
description: CameraX/Camera2 invariants, Patrol 15 FPS duty-cycling, Track-level spatial NMS, Tap-to-Lock B=1 ROI tracking, and OpenCV AKAZE planar constellation terminal guidance.
---

# Android Camera, Patrol Duty-Cycling & OpenCV Lock-On Funnel Rules

## 1. Preserved Hardware Invariants (`fc80785`, `0cad9b2`, `1bfdda0`)
- Keep standard `onMeasure` and unconditional `onLayout` in `ClippingCameraViewportContainer` so `PreviewView`'s `TextureView` never starves at `0x0`.
- Keep all AE/shutter/ISO/FPS overrides on dynamic `Camera2CameraControl` (never static `ALWAYS_OVERRIDE` on `Camera2Interop.Extender`).
- Keep `imageProxy` hold time `< 3 ms` via lock-free ring buffer slots (`imageProxy.close()` in `finally`).
- Keep reversible `sensorLandscape` (`ROTATION_90` & `ROTATION_270`) IMU remapping (`Sensor.TYPE_GAME_ROTATION_VECTOR`).

## 2. Stage 1: `PATROL` Sensor & Display Duty-Cycling
- Maintain a state machine with two thermal/power sub-states while in `PATROL` scan mode:
  1. **`PATROL_ACTIVE`:** Engaged on cold start, screen touch, or within `< 50 ms` of any alertable candidate tile detection (`confidence >= 0.30f`).
     - Camera2 `CONTROL_AE_TARGET_FPS_RANGE = Range(30, 30)` (preserving `SENSOR_EXPOSURE_TIME <= 2,000,000 ns` when `1/500s` shutter clamp is ON!).
     - `window.attributes.preferredRefreshRate = 120f` (full 120 Hz RenderNode EIS).
  2. **`PATROL_IDLE`:** Engaged automatically after **`5.0 seconds`** with zero alertable candidates (`confidence >= 0.30f`) and no UI touch:
     - Set `CONTROL_AE_TARGET_FPS_RANGE = Range(15, 15)` via `Camera2CameraControl` (merged cleanly into the active exposure capture request options so `<= 1/500 s` shutter clamp and floating ISO remain active without re-opening the camera session!).
     - Set `window.attributes.preferredRefreshRate = 30f` to halve SurfaceFlinger/GPU memory bandwidth and hold the Pixel 10 Pro XL in `Thermal Status: 0`.
     - If the user's `"EYES-UP"` (`eyesUpPatrolEnabled`) toggle is `ON`, render a `90%` black (`#E6000000`) OLED-saving veil over the camera viewport with a minimal dim `EYES-UP PATROL • 15 FPS` indicator; clear it instantaneously (`< 50 ms`) upon candidate detection or tap.

## 3. Track-Level Spatial NMS (`TemporalTargetTracker.kt`)
- Eliminate duplicate overlapping reticles on a single physical object: after updating/promoting tracks each sweep, run **Track-Level Spatial NMS** across active `ConfirmedTarget` tracks of the same `StoneClass`:
  - Merge/suppress lower-vote (or lower-smoothed-confidence) duplicate tracks when `worldRay.angularDistanceDegTo(other.worldRay) <= 3.2f`, OR 3D `worldPosition.distanceMetersTo(other.worldPosition) <= 0.075f` (`75 mm`), OR current strip box `IoU >= 0.35f` / `IoM >= 0.60f`.

## 4. Official OpenCV Dependency & 16 KB Page Alignment (`build.gradle.kts`)
- Add the official Maven Central OpenCV AAR (`org.opencv:opencv:4.12.0` or `4.13.0`, Apache-2.0 — verify exact artifact resolution and `16 KB` ELF segment alignment for Pixel 10 Pro XL Android 17).
- Ensure `packaging { jniLibs { useLegacyPackaging = false } }` is configured in `android/app/build.gradle.kts` and initialize natively once via `OpenCVLoader.initLocal()`.
- Use OpenCV's C++ JNI wrappers (`org.opencv.features2d.AKAZE`, `ORB`, `BFMatcher`, `org.opencv.calib3d.Calib3d`, `org.opencv.video.Video`) directly on pre-allocated single-channel `Mat(height, width, CvType.CV_8UC1)` populated from the `YUV_420_888` luminance (`Y`) plane—do NOT write bespoke feature detectors or homography solvers from scratch.

## 5. Stage 2 (`LOCK-ON`) & Stage 3 (`PINPOINT`) Acquisition Funnel
### Entry & Exit Semantics
- **Entry:**
  1. Tapping directly on (or within `32.dp` touch slop of) any visible `ConfirmedTarget` reticle inside the push-broom viewport locks onto that specific `trackId`.
  2. Tapping the central **`LOCK NEAREST`** LCARS button locks onto the visible `ConfirmedTarget` with the **smallest `slantRangeMeters`** (tie-breaker: smallest horizontal distance from viewport center).
- **Viewport Transition:**
  - Expand the top `PanoramicViewportZone` from push-broom strip zoom to the **full uncropped `16:9` (`1920x1080`) landscape camera frame** and freeze EIS (`previewView.translationY = 0f`) so the user has full peripheral visual context while walking/crouching toward the stone.
- **Stage 2 — Foveated `B=1` Neural Lock-On (`LOCK_ON_NEURAL`):**
  - Crop a single `384x384` or native **`320x320`** RGB `[0..1]` patch from `1920x1080` centered on the target's current reprojected pixel `(u, v)` (clamped to frame bounds `[0..1920] x [0..1080]`) and run **`rf_detr_nano_320_b1_w8dyn`** (`B=1`, `~15-22 FPS`).
  - **Multi-View Parallax Triangulation:** Estimate metric inter-frame camera translation + local ground plane from OpenCV sparse KLT / homography (`Video.calcOpticalFlowPyrLK` / `Calib3d.findHomography` on background gravel keypoints outside the target box, scaled by initial waist-height range `R_0` and `TYPE_GAME_ROTATION_VECTOR` `R_world_from_cam`) and triangulate target rays (`Calib3d.triangulatePoints` / recursive 3D ray-intersection covariance) to compute 3D target coordinates `P_3d` and **sphere-of-confusion radius** `sigma3dMm` (95% confidence / RMS ray-bundle spread in mm).
- **Stage 3 — OpenCV `AKAZE` Gravel-Constellation Terminal Guidance (`PINPOINT_CONSTELLATION`):**
  - **Handover Trigger:** Once `sigma3dMm < 0.5f * target.estimatedDiameterMm` (with `>= 4` valid multi-view observations and parallax angle `>= 3.0 deg`), automatically transition to `PINPOINT_CONSTELLATION`.
  - **Gravel-Constellation Collar (`3x` Context Patch):** Because smooth agates have wet specular glints and sparse internal corners, extract `AKAZE` keypoints & binary descriptors on the `Y` luma plane across a **`3.0x` target-diameter context patch** (`256x256` to `448x448` px) centered on the stone, recording the stone center in local constellation plane coordinates.
  - Match live frames against the reference (+ rolling mid-approach scale keyframe updated every `1.8x` scale growth as the user crouches from `1.5m -> 0.25m`) using `BFMatcher.create(Core.NORM_HAMMING)` + Lowe ratio test (`0.78f`) + `Calib3d.findHomography(..., Calib3d.USAC_MAGSAC, 3.5)` at full `30 FPS` camera rate (`< 5 ms` CPU).
  - **1 Hz Neural Confidence Refresh:** While `PINPOINT_CONSTELLATION` holds lock (`inliers >= 12`), fire `B=1` `rf_detr_nano_320_b1_w8dyn` only once every **`1.0 second`** centered on the homography-projected stone box to update `StoneClass` probability as pixel resolution triples on approach. If inliers dip `< 12`, immediately fall back to full-rate `B=1` neural re-acquisition around the 3D-reprojected `WorldRay3D` / `P_3d` locus.
- **15-Second Lost-Track Grace Window & Exit:**
  - If the stone leaves the `16:9` FOV or is occluded, project its 3D world anchor into screen space to draw an edge **re-acquisition chevron** and maintain Lock-On standby for up to **`15.0 seconds`**.
  - If neither OpenCV constellation matching nor `B=1` neural search re-acquires the target within **`15.0 continuous seconds`**, automatically exit back to `PATROL`. Tapping **`BREAK LOCK`** exits to `PATROL` immediately (`0 ms`).
