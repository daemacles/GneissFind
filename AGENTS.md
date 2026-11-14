# GneissFind — Workspace Architecture & Engineering Rules (`AGENTS.md`)

## 1. Project Identity & Monorepo Layout
- **Project Name:** `GneissFind` ("Visual Metal Detector" for PNW rockhounding)
- **Android Application ID / Namespace:** `com.gneissfind.app` (primary domain: `gneissfind.com`; `gneissfind.net` reserved for future model CDN/telemetry)
- **Target Hardware:** Google Pixel 10 Pro (Android 16, flagship GPU + Tensor G5)
- **Training Workstation:** Ubuntu 26.10 on Dell Precision T7920 (96 GB RAM, 2x 24-core Xeon Platinum, `1x RTX 4090 24GB` + `2x RTX 4000 Ada 20GB`)
- **Repository Structure:**
  - `android/` — Native Kotlin + Jetpack Compose app (`minSdk = 34`, `targetSdk = 36`), CameraX + `Camera2Interop`, ARCore VIO, LiteRT GPU FP16 inference, Classic 2360s TNG LCARS UI.
  - `ml-pipeline/` — Python (`uv` / Docker) pipeline for `SAM 2` video auto-labeling, hard-negative Copy-Paste augmentation, `RF-DETR` training, and 2-graph `litert-torch` `.tflite` export.
  - `docs/` — Milestone roadmap, acceptance criteria, and field test protocols.
  - `.agents/rules/` — Scoped engineering rules for Camera/AR/IMU (`android-camera-ar.md`) and LCARS UI (`lcars-ui.md`).

---

## 2. Git Workflow Mandate (Test-Gated Atomic Commits)
- **Mandatory Rule:** Whenever you complete a functional change in code, first verify that it compiles cleanly without error and passes unit tests:
  - For `android/` changes: run `./gradlew -p android assembleDebug testDebugUnitTest`
  - For `ml-pipeline/` changes: run `pytest` (or syntax/smoke check if tests are being bootstrapped)
- **Immediate Atomic Commit:** Once compilation and tests pass, immediately stage the relevant files and create a Git commit with a **brief, single-sentence imperative description** of the functional change (e.g., `git commit -m "Add 5-tile landscape push-broom cropper with IMU pitch stabilization"`).
- **Never Commit:** Broken builds, `build/` directories, `.gradle/`, `local.properties`, raw field video captures, or large model binaries (`.pt`, `.pth`, `.onnx`, `.tflite` > 10 MB) unless explicitly tracked via Git LFS. Ensure `.gitignore` enforces this.

---

## 3. Locked Optical & Push-Broom Geometry (Landscape v2)
- **Screen Orientation:** Reversible landscape (`android:screenOrientation="sensorLandscape"` supporting both `Surface.ROTATION_90` and `Surface.ROTATION_270`) so the user can hold the Pixel 10 Pro with the camera bar + clip-on Circular Polarizer (CPL) filter free of hand occlusion, while maximizing horizontal sensor usage.
- **Split-Screen Non-Occluding Layout:**
  - **Top ~48% of screen:** Live wide camera viewport + stabilized push-broom strip overlay, seam markers, target reticles, and pitch-travel gauge.
  - **Bottom ~52% of screen:** Persistent Classic 2360s TNG LCARS tactical control deck on pure `#000000` OLED canvas. **Zero modal bottom sheets or floating cards may ever cover the live camera feed.**
- **Camera Sensor & Stream:**
  - Rear 50 MP Main Wide camera (`f/1.68`) with **`2.0x` lossless in-sensor center crop (`ZoomRatio = 2.0f` / `SCALER_CROP_REGION`) DEFAULTED TO ON (`true`)** at startup.
  - CameraX `ImageAnalysis` configured for `1920x1080` (`YUV_420_888`, `STRATEGY_KEEP_ONLY_LATEST`) in landscape buffer coordinates.
  - `Camera2Interop` shutter clamp: **<= 1/500 s (`<= 2,000,000 ns`)** DEFAULTED TO ON (`true`) with floating ISO + Canopy EV bias control (`-2.0 EV` to `+1.0 EV`) to eliminate motion blur under overcast PNW skies and tree canopy.
- **5-Tile Cross-Track Swath Packing (`1792 x 384 px`):**
  - Default **5-TILE mode**: 5 horizontally overlapping `384 x 384` native-resolution patches with `32 px` seam overlap (`stride = 352 px`), spanning `1792 x 384 px` (`93.3%` of horizontal `1920 px` frame width, `x_left = [64, 416, 768, 1120, 1472]`).
  - Optional **3-TILE fast mode** toggle (`1088 x 384 px`, `x_left = [416, 768, 1120]`) for higher sweep rate testing.
  - At nominal `45 deg` pitch from waist height (`h ~ 1.06 m`, slant range `R ~ 1.50 m`) with `2.0x` crop ON: cross-track swath is `~1.05 m` wide x `~0.25 m` deep at `~0.59 mm/px` GSD (`20 mm` minimum target stone = `~34 px`).
- **IMU-Driven ROI Pitch Stabilization (Software EIS):**
  - Uses `Sensor.TYPE_GAME_ROTATION_VECTOR` (magnetometer-free, immune to magnetic basalt) remapped via `SensorManager.remapCoordinateSystem` for `ROTATION_90` and `ROTATION_270`.
  - Dynamically shifts the strip's vertical top coordinate `cropY` inside `[0, 696]` (`1080 - 384 = 696`, nominal center `cropY = 348` at `45 deg` reference pitch or user `"SYNC PITCH"` setpoint) using vertical pixel focal length `fy` (`deltaY = fy * tan(pitch - refPitch)`), keeping the strip pointed at the same `~1.50 m` ground patch despite wrist pitch wobble (`~ +/-7 deg` stabilization authority).
  - When `cropY` hits `0` or `696`, clamp cleanly and illuminate an LCARS `"PITCH LIMIT"` warning pill.

---

## 4. Target Taxonomy (PNW Gravel Bars & Beaches)
Minimum target stone size: **`20 mm`**. Model outputs 5 classes (`0..3` active prospect targets, `4` muted auxiliary hard negative):
0. `chalcedony_agate` — Translucent carnelian, clear/banded agate
1. `jasper_red_yellow` — Opaque red, yellow/mustard, and orbicular jasper
2. `green_gem_rock` — Opaque green Dallasite breccia and nephrite jade
3. `petrified_wood` — Silicified wood exhibiting grain or bark structure
4. `hard_confuser` — *(Internal auxiliary negative supervision ONLY; **strictly muted in UI & audio**)* Wet orange alder leaves, iron-stained quartz, red brick/clay fragments, and specular water glints.

---

## 5. On-Device Inference, Speed Gate & Sonification
- **Detector Interface (`StoneDetector`):** Pluggable Kotlin interface taking a batch/list of `384x384` RGB tiles and returning normalized tile detections merged via cross-tile Non-Maximum Suppression (NMS) in strip space (`1792x384`).
- **Primary Model Architecture (Apache-2.0):** `RF-DETR-Nano` (`384x384`) exported as a **2-graph LiteRT GPU FP16** pipeline (Graph A backbone/encoder + Kotlin `TopK` bridge + Graph B decoder head) avoiding `GATHER_ND`/`TOPK` GPU fallback bottlenecks and avoiding Android 16 sideloaded EdgeTPU allowlist blocks. Optional `YOLO26s` single-graph GPU runner supported behind the same interface.
- **Temporal Voting & ARCore Speed Gate:**
  - Detections project onto ground/slant plane and accumulate N-of-M multi-frame votes (`default 3 of 6 sweeps`) before highlighting or chirping.
  - Monitor ground speed via ARCore VIO (with IMU/optical-flow fallback when ARCore tracking initializes). Flash Mars Red `#CC4444` `"SLOW SWEEP"` warning if speed exceeds **`1.10 m/s`**.
- **Audio Rules:**
  - **LCARS UI button taps are SILENT** (`SoundEffectConstants` disabled on custom LCARS pills; UI sound tweaks deferred to Milestone 5).
  - Low-latency `AudioTrack` synthesizer is strictly reserved for **stone detection pings** (ping rate `2-14 Hz` scaled to vote persistence; frequency `780 Hz` for `20-30 mm` pebbles down to `320 Hz` for `50 mm+` cobbles based on metric diameter) and the overspeed warning tone.

---

## 6. Dell T7920 Multi-GPU Split (`ml-pipeline/`)
Never combine the `RTX 4090` and `RTX 4000 Ada` cards in synchronous PyTorch `DDP` (prevents straggler bottlenecks):
- **`GPU 0` (`RTX 4090 24GB`):** Primary `RF-DETR-Nano` student training, `RF-DETR-Large` teacher distillation, and `litert-torch` 2-graph `.tflite` export.
- **`GPU 1 & GPU 2` (`2x RTX 4000 Ada 20GB`):** Parallel `SAM 2` video mask propagation on salted-bar captures, Copy-Paste synthetic compositing onto empty PNW hard-negative gravel frames, and evaluation sweeps.
