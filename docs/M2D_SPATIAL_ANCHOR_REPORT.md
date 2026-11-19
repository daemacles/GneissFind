# Milestone 2D-2 — Overnight Spatial Anchor, `MOD: LOG` Audio Silence & UX Hardening Report

> **Session:** Overnight Unattended Deep Debug (`2026-11-19`)  
> **Reference Clip:** `20261119_011006` (`R-01`, `896` frames, `30.0 s`, `1920x1080` H.264 + `200 Hz` IMU, indoor room walk-around with multiple distractors + off-screen wall/doorframe pans) + user screen grab `gneissfind screen grab 13.mp4`  
> **Target Hardware:** Google Pixel 10 Pro XL (`mustang`, Tensor G5 ARM64, `16 KB` pages, Android 16)

---

## 1. Executive Summary

Forensic analysis of `20261119_011006` and `screen grab 13.mp4` uncovered **four interacting root causes** that broke spatial anchoring and caused audible detector pings during `MOD: LOG`, plus **six UX honesty/clarity gaps**. Replacing the single-keyframe unverified `AKAZE` + flat-plane VO with a **Unified 3-Tier Spatial Anchor** (`640x360` IMU-predicted Local `PyrLK` Similarity + Rotation-Known Sparse 3D Mini-VIO + 4-Gate Multi-Keyframe `ORB` Bank), fixing the API 30+ Camera2 `CONTROL_ZOOM_RATIO` focal-length bug (`2.0x` under-rotation), disabling OEM EIS video lag, and sealing the `MOD: LOG` in-flight sweep/sonifier race achieved:

- **Offline 896-Frame `20261119_011006` Benchmark (ZERO neural detector help after Frame 0):**
  - Median tracking error dropped from **`927.8 px` (`13.7%` `< 30 px`)** to **`8.8 px` (`90.4%` `< 30 px`, `94.5%` `< 60 px`)**.
  - **272-Frame Adversarial Blank Beige Wall + Doorframe Stress Test:** **`0 / 272` (`0.0%`) false locks**, re-acquiring the mug in **`1–2 frames`** upon return (`98.0%` `< 30 px`).
  - **9-Second Central Blackout Stress Test (`F200..350`, `F550..670`):** **`9.7 px`** visible median, **`98.0%` `< 30 px`**, re-acquiring in **`1–2 frames`**.
- **On-Device Pixel 10 Pro XL Verification (`58110DLCQ00003`):**
  - Live `CAM2_INTRINSICS` confirmed API-30 root cause directly on hardware: `zoom=2.00 capZoom=2.0 fySensor=2907.95 crop=4080x3072 active=4080x3072 -> computedFy=2736.9 ois=1 videoStab=0` (previously misreported as `1368.4 px`, 2.0x wrong!).
  - Live 4s `MOD: LOG` (`--ei debug_record_seconds 4`): **`112 frames` (`27.84 avg FPS` incl. cold camera start, `~30 FPS` steady), `discarded_sweeps=0`, `audio_pings=0`**.
  - End-to-end `20261119_011006` replay (`--es debug_replay_session 20261119_011006 --ei debug_auto_lock_frame 12 --ez debug_loop_replay false`):
    ```json
    GNEISSFIND_REPLAY_SUMMARY {"session":"20261119_011006","frames":896,"locks":1,"handovers":1,"lost_locks":0,"distractor_jumps":0,"distractors_rejected":38,"exceptions":0}
    ```
  - ARM64 OpenCV 4.13 instrumented benchmarks on Pixel 10 Pro XL:
    - 3-Tier (Tier 1 + Tier 2 + Pyramid upload): **`5.34–7.24 ms`** median CPU, **`0.66 px`** final drift over 25-frame pan (`25/25` tracked).
    - Blank wall + doorframe adversarial test: **`0 / 30` (`0.0%`) false locks**, first-frame re-acquisition at **`0.66 px` (`ncc=0.939`, `61` inliers)**.
    - Tier 3 `ORB` collar: **`4.10 ms`** warm median; `480x270` `ORB` relocalization: **`8.53–9.50 ms`**.
- **Full Test Suite:** `50/50` JVM unit tests, `8/8` Pixel 10 Pro XL instrumented tests, `7/7` `ml-pipeline` Python `pytest` tests.

---

## 2. Forensic Root-Cause Findings

### 2.1. Camera2 API 30+ `CONTROL_ZOOM_RATIO` Halved Focal Length (`2.0x` Rotation Mismatch)
- **Root Cause:** On Android 11+ (API 30+), CameraX `setZoomRatio(2.0f)` applies digital zoom via `CaptureRequest.CONTROL_ZOOM_RATIO = 2.0f` while leaving `SCALER_CROP_REGION` equal to the full `SENSOR_INFO_ACTIVE_ARRAY_SIZE` (`4080x3072`). `PushBroomCameraController.computeStreamVerticalFocalLengthPx` computed `fy` solely from `cropRegion`, yielding **`1368–1389 px`** instead of **`2737–2778 px`** at `2.0x` zoom.
- **Impact:** Every IMU rotation prediction (`R_rel * K^{-1}`) and 3D world-ray projection rotated pixels by **half** the true camera motion, causing Lock-On boxes to drag off targets during pitch/yaw pans and leaving off-screen coasting boxes stuck inside the viewport on walls/doorframes.
- **Fix:**
  - `PushBroomCameraController.computeStreamVerticalFocalLengthPx[FromDims]` now inspects `TotalCaptureResult.CONTROL_ZOOM_RATIO` and multiplies `fy` whenever `cropRegion >= 98%` of `activeArray` (never double-scaling legacy HALs that already shrink `SCALER_CROP_REGION`).
  - Explicitly set `CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE = OFF` in `Camera2CameraControl` (preserving the `fc80785` dynamic-only invariant) to eliminate `-30..-60 ms` OEM EIS smoothing lag.
  - Added `SO(3)` linear interpolation + Gram-Schmidt orthonormalization and `cameraImuLagCompensationNs` in `GroundPlaneRaycaster.lookupRotationMatrixAtTimestampNs`.
  - Added automatic legacy `trace.jsonl` focal-length repair (`ReplayCameraSource.resolveReplayVerticalFocalLengthPx`) for pre-M2D-2 `>= 1.5x` clips (`900..1750 px` -> `fy * zoomRatio`), leaving calibrated `2.0x` and `1.0x` clips untouched.

### 2.2. Why the Legacy Spatial Anchor Failed (`med = 927.8 px`, Blank-Wall False Locks)
1. **Unverified 1-Keyframe Homography:** Legacy `AKAZE` + `USAC_MAGSAC` accepted degenerate homographies (collapsed singular value ratio, negative determinant, or 1D vertical doorframe edge matches) without checking inlier spread, Jacobian conditioning, IMU roll consistency, or target appearance (NCC), causing false locks on the beige hallway wall/doorframe.
2. **Stage 2 Visual Blindness:** In Stage 2 (`LOCK_ON_NEURAL`), zero frame-to-frame visual tracking ran between `~10 Hz` `B=1` GPU sweeps, allowing `100 ms` motion jumps to destabilize initial triangulation.
3. **Flat Ground-Plane (`z = 0`) VO Violation:** Projecting full-frame `PyrLK` correspondences onto `z = 0` corrupted `camPosEnu` whenever walls, furniture, chairs, or table edges entered the frame (common indoors and on terraced riverbanks/boulders).

### 2.3. `MOD: LOG` Audible Sonifier Pings During Recording
- **Root Cause:** Switching to `MOD: LOG` set `operatingMode = LOG_DATA`, but an already in-flight `RF-DETR` sweep on `GneissFind-Inference` or a queued `120 Hz` IMU callback could finish milliseconds later and call `sonifier.updateState(...)`, firing audible audio pings even though UI detection boxes were hidden.
- **Fix:** Added lock-free `isLoggingDataActive: AtomicBoolean` set immediately at entry to `setOperatingMode(OperatingMode.LOG_DATA)`, guarded `onRealtimeImuPitchListener`, `processYuvFrameFromSource`, `drainLatestSweepOnInferenceThread`, `runInferenceSweepForSlot` (entry, post-inference, and `_uiState.update`), and `runLockOnNeuralInferenceForSlot`, and added `discarded_sweeps` & `audio_pings` telemetry to `GNEISSFIND_RECORD_SUMMARY`.

---

## 3. Quantitative CV Benchmark (896 Frames of `20261119_011006`, Zero Detector After Frame 0)

All trackers were initialized **only at Frame 0** (no neural detector assistance on Frames `1..895`) and evaluated against `RF-DETR` ground truth across the full 30-second walk-around:

| Tracker Architecture | Visible Median Error | Visible P90 Error | `< 30 px` Accuracy | `< 60 px` Accuracy | Blank-Wall False Locks | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **M2D Baseline** (`f=1390`, 1-KF unverified) | `927.8 px` | `> 1000 px` | `13.7%` | `18.2%` | High | Catastrophic drift & wall hijacks |
| **Unified 3-Tier (Uncalibrated `f=1390, lag=0`)** | `9.9 px` | `31.4 px` | `89.0%` | `93.8%` | `0 / 272 (0.0%)` | Resilient even on legacy uncalibrated traces |
| **Unified 3-Tier (Calibrated `f=2778` — Shipped)** | **`8.8 px`** | **`29.7 px`** | **`90.4%`** | **`94.5%`** | **`0 / 272 (0.0%)`** | `4.72 ms` Python / **`5.34–7.24 ms` ARM64** |
| **Stress Test A: 9s Central Blackout (`F200..350`, `F550..670`)** | **`9.7 px`** | **`18.6 px`** | **`98.0%`** | **`99.1%`** | `0` | Re-acquires in **`1–2 frames`** |
| **Stress Test B: 272-Frame Blank Beige Wall + Doorframe** | **`9.7 px`** | **`19.8 px`** | **`98.0%`** | **`99.0%`** | **`0 / 272 (0.0%)`** | Zero false locks; re-acquires in **`1–2 frames`** |

### Shipped Unified 3-Tier Architecture (`OpenCvConstellationTracker.kt` + `LockOnPinpointController.kt`)
1. **Tier 1 (`30 FPS`, Stage 2 & Stage 3 from Frame 1):** Shared `640x360` `beginTrackingFrame` pyramid -> IMU-predicted `OPTFLOW_USE_INITIAL_FLOW` local `PyrLK` -> Forward-Backward `< 1.5 px` -> 4-DOF Similarity RANSAC -> IMU in-plane roll gate (`<= 12°`) -> `32x32` zero-mean NCC (`>= 0.25f`, blank wall `std < 3.5` rejected).
2. **Tier 2 (`30 FPS`, Continuous Ego-Motion):** Rotation-Known Sparse 3D Mini-VIO on `640x360` (~90 persistent 3D ENU landmarks, IMU-predicted `PyrLK` + FB `< 1.5 px`, 3-iter Huber IRLS 3-DOF camera-center solver with adaptive Tikhonov prior toward `C_prev`, online 2-view midpoint triangulation, **zero `z = 0` ground-plane assumption**).
3. **Tier 3 (`~5 Hz` Drift Audit / `10 Hz` Re-Entry Relocalization):** Multi-Keyframe `ORB` Bank (up to `6` collar keyframes at `192x192` + up to `4` full-frame `480x270` reloc keyframes; Keyframe 0 permanently preserved) protected by the **4-Gate Verifier**:
   - **Gate 1:** Inlier spatial spread `>= 6%` of search width (rejects 1D doorframe/edge clusters).
   - **Gate 2:** Homography `2x2` Jacobian determinant `[0.16..6.25]`, SVD ratio `<= 2.4`, perspective bound `< 0.45`.
   - **Gate 3:** IMU in-plane roll consistency `<= 25°`.
   - **Gate 4:** `32x32` zero-mean unit-L2 NCC `>= 0.40f` (rejects flat walls `std < 3.5`).

---

## 4. UX Lead Audit & Shipped Improvements (`ScannerScreen.kt` + `StoneModels.kt`)

| UX Finding | Before (`screen grab 13`) | After (M2D-2 Shipped & Verified on Device) |
| :--- | :--- | :--- |
| **1. `MOD: LOG` Viewfinder & Footer** | Dark letterbox veil cropped top/bottom; footer showed stale `SWEEP MS` & active `SONIFIER ARMED` | Full uncropped `16:9` Mars Red recording frame, subtle dashed strip guide, `● REC MM:SS • 1080P30 • DETECTOR PAUSED • AUDIO MUTED` pill, Mars Red `REC-30P` footer, muted `AUDIO PAUSED`. |
| **2. Honest Visual Lock vs Coasting** | Solid turquoise `PINPOINT` reticle remained drawn over blank walls while coasting | `LockOnHudState.isVisuallyLocked` splits solid mineral `PINPOINT` brackets from translucent `48%` dashed GoldenAmber `COAST` reticle (`TRK COAST` / `COASTING` in side console). |
| **3. Honest Range Before 3D Triangulation** | Displayed confident `RANGE 1.42 M • ±18 MM` even before parallax converged | `LockOnHudState.hasMetric3dDepth` displays `RANGE ~1.42 M (EST)` until metric 3D triangulation converges, then `RANGE %.2f M • ±%.0f MM`. |
| **4. Consistent `30 HZ` Tracker Cadence** | Lock-On HUD flickered between `10 HZ` neural rate and `30 HZ` visual rate | Publishes steady `30 HZ` cadence whenever Tier 1/Tier 3 visual lock is active (`PIN 3-TIER` / `LOCK KLT30`). |
| **5. Side Console Stage 2 Gauge** | `TEXTURE PTS` showed `WAIT 3D` with 0% bar throughout Stage 2 | Renamed `VISUAL PTS` and reflects live Tier 1 KLT inlier count immediately from Frame 1 (`COASTING` when visually unanchored). |
| **6. Off-Screen Directional Chevrons** | Legacy `2x` focal error left coasting boxes stranded on walls instead of pointing off-screen | With calibrated `fy = 2737–2778 px`, off-screen pans cleanly transition to pulsing edge chevrons (`28° • 14S`, `29° • 14S`) pointing directly back to the target. |
