# GneissFind — Execution Roadmap & Acceptance Criteria (`docs/ROADMAP.md`)

## Milestone 1 & 1B — Push-Broom Optical Skeleton, 120 Hz EIS & TNG LCARS *(Completed — `fc80785`)*
- [x] Reversible `sensorLandscape`, default `2.0x` sensor crop + `<= 1/500 s` dynamic shutter clamp, `5 x 384x384` (`1792x384`) strip cropper, 120 Hz RenderNode EIS, and 6-role TNG LCARS UI.

## Milestone 2 & 2B — RF-DETR-Nano COCO Vertical Slice & `W8-Dyn` >= 6.5 Hz GPU Optimization *(Completed — `b892dd7`)*
- [x] LiteRT `2.3.2` `W8-Dyn` (`3.7x` smaller weights, `0%` recall loss), zero-copy `encMemory`, 3-slot lock-free async pipeline, `K=100` decoder surgery, and `B=3` priority interleaving (`9.24 FPS` cold / `6.57 FPS` 180s soaked).

## Milestone 2C — Shader Cache, Patrol Duty-Cycling & Initial OpenCV `AKAZE` Funnel *(Completed — `5d13646`, Post-Field Bugs Identified)*
- [x] `0.8–1.5 ms` warm `SRC` LRU + disk shader cache, `15 FPS` `"EYES-UP"` Patrol duty-cycling, `B=1` `320x320` Lock-On export, `16 KB` OpenCV `4.13.0` integration, and LCARS `GNEISSFIND` + Git/date shoulders.
- [!] Live handheld walk/pan testing exposed: (1) reproducible crash when a locked target exits the **right** viewport edge, (2) distractor hijacking onto adjacent cups/screens during `640x640` `B=1` search, (3) `AKAZE <-> B=1` thrashing on clear/reflective targets, and (4) lost 3D anchor translation during occlusion/off-screen pans because `5d13646` froze VO off-screen.

---

## Milestone 2D — Right-Edge Crash Fix, `30 FPS` MP4+IMU Logger & Replay Source, Innovation Gate & Two-Tier Occlusion Tracking *(Completed)*

### Goal
Eliminate the off-screen-right Lock-On crash, replace the legacy `4 Hz` PNG logger with `30 FPS` `1080p` H.264 `.mp4` + `120 Hz` IMU `trace.jsonl` recording and on-device **`ReplayCameraSource`** (retaining 1 synthetic demo in `SRC` and adding `adb` automation hooks so Antigravity can run deterministic video regression tests autonomously), and harden Lock-On/Pinpoint with a **3D Bearing Innovation Gate**, **`300 ms` `AKAZE` coasting hysteresis**, and **continuous full-frame IMU-derotated KLT VO + `480x270` context relocalization** across occlusions.

### Milestone 2D Acceptance Criteria (Oracle)
- [x] **Zero Off-Screen Crashes (All 4 Edges):** Fixed `LockOnPinpointController.kt` `coerceIn` crash when `targetXPx >= 1920` (`min > max`) and hardened `PushBroomCropper` UV stride bounds, `OpenCvConstellationTracker.computeSafeSubmatRect`, and `ScannerScreen` Canvas chevron/badge bounds; verified across `168+` 4-edge sweep frames (`x = -400..2320`, `y = -400..1480`) in unit + instrumented tests and live `m2d_validation_v1` off-screen-right pan (`x = 2383.4 px`).
- [x] **Distractor & Glint Immunity + Occlusion Memory:** Enforced 3D Bearing (`<= 4.5 deg` on-screen, widening `+0.75 deg/s` to `<= 9.0 deg` off-screen) & Scale (`[0.45x..2.20x]`) + Class Innovation Gate, `300 ms` (`9`-frame) `AKAZE` coasting hysteresis, continuous `640x360` IMU-derotated `PyrLK` VO (`1.52 ms` warm median CPU), and `480x270` full-frame `AKAZE` relocalization (`9.07 ms` at `~5 Hz`); verified `distractor_jumps: 0, distractors_rejected: 83, lost_locks: 0, exceptions: 0` on Pixel 10 Pro XL.
- [x] **`MOD: LOG` MP4+IMU & `ReplayCameraSource`:** Hardware `MediaCodec` (`video/avc`, `1920x1080`, `30 FPS`, `14 Mbps`) + `MediaMuxer` fed directly from the existing `ImageAnalysis` `YUV_420_888` stream (2 CameraX surfaces) + `120 Hz` IMU `trace.jsonl`, paired with `ReplayCameraSource`, dynamic `REPLAY: <session>` `SRC` drawer entries (1 synthetic demo `MIXED_GRAVEL_WALK`), and `MainActivity` `adb shell am start` headless hooks (`GNEISSFIND_REPLAY_SUMMARY`).

---

## Milestone 3 — Gravel-Bar Video Capture, Patrol FPS Replay Simulator & T7920 `SAM 2` Auto-Labeling (`ml-pipeline/`) *(Next)*
- Record indoor stand-in and outdoor salted/unsalted PNW gravel-bar `clip.mp4 + trace.jsonl` sessions using the Milestone 2D `MOD: LOG` recorder.
- Build `ml-pipeline/scripts/simulate_patrol_fps.py` to replay `clip.mp4 + trace.jsonl` walks (`<= 1.1 m/s`) at decimated rates (`15, 10, 7.5, 6, 5, 4, 3, 2 FPS`) to find the lowest power-saving `PATROL` FPS ($X	ext{ FPS}$) maintaining `>= 98%` stone lock probability.
- Run `SAM 2` video propagation + Copy-Paste hard-negative synthesis on `GPU 1 & 2` (`2x RTX 4000 Ada`).

## Milestone 4 — `RF-DETR-Nano` PNW Stone Fine-Tuning, Replay Validation & Production Weight Swap
- Fine-tune `RF-DETR-Nano` (`256` & `320`, `B=3`/`B=5`/`B=1`, `W8-Dyn`) on `GPU 0` (`RTX 4090 24GB`), validate on held-out `ReplayCameraSource` gravel clips, and swap production 5-class PNW mineral weights into `android/app/src/main/assets/`.

## Milestone 5 — Field Calibration, Watershed Presets & Hard-Negative Mining
- Per-class confidence sliders, regional presets (Puget Sound beaches, Western Cascade rivers, Bellingham / Skagit bars), and one-tap field confuser flagging.

## Milestone 6 — Late-Stage Audio Polish & V2 Ergonomics
- Polyphonic/harmonic sonification curves, tactile haptics, optional LCARS trills, and hands-free chest-mount evaluation.
