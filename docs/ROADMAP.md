# GneissFind — Execution Roadmap & Acceptance Criteria (`docs/ROADMAP.md`)

## Milestone 1 & 1B — Push-Broom Optical Skeleton, 120 Hz EIS & TNG LCARS *(Completed — `fc80785`)*
- [x] Reversible `sensorLandscape`, default `2.0x` sensor crop + `<= 1/500 s` dynamic shutter clamp, `5 x 384x384` (`1792x384`) strip cropper, 120 Hz RenderNode EIS, and 6-role TNG LCARS UI.

## Milestone 2 & 2B — RF-DETR-Nano COCO Vertical Slice & `W8-Dyn` >= 6.5 Hz GPU Optimization *(Completed — `b892dd7`)*
- [x] LiteRT `2.3.2` `W8-Dyn` (`3.7x` smaller weights, `0%` recall loss), zero-copy `encMemory`, 3-slot lock-free async pipeline, `K=100` decoder surgery, and `B=3` priority interleaving (`9.24 FPS` cold / `6.57 FPS` 180s soaked).

## Milestone 2C — Shader Cache, Patrol Duty-Cycling & Initial OpenCV `AKAZE` Funnel *(Completed — `5d13646`, Post-Field Bugs Identified)*
- [x] `0.8–1.5 ms` warm `SRC` LRU + disk shader cache, `15 FPS` `"EYES-UP"` Patrol duty-cycling, `B=1` `320x320` Lock-On export, `16 KB` OpenCV `4.13.0` integration, and LCARS `GNEISSFIND` + Git/date shoulders.
- [!] Live handheld walk/pan testing exposed: (1) reproducible crash when a locked target exits the **right** viewport edge, (2) distractor hijacking onto adjacent cups/screens during `640x640` `B=1` search, (3) `AKAZE <-> B=1` thrashing on clear/reflective targets, and (4) lost 3D anchor translation during occlusion/off-screen pans because `5d13646` froze VO off-screen.

---

## Milestone 2D — Right-Edge Crash Fix, `30 FPS` MP4+IMU Logger & Replay Source, Innovation Gate & Two-Tier Occlusion Tracking *(Current Active Task)*

### Goal
Eliminate the off-screen-right Lock-On crash, replace the legacy `4 Hz` PNG logger with `30 FPS` `1080p` H.264 `.mp4` + `120 Hz` IMU `trace.jsonl` recording and on-device **`ReplayCameraSource`** (retaining 1 synthetic demo in `SRC` and adding `adb` automation hooks so Antigravity can run deterministic video regression tests autonomously), and harden Lock-On/Pinpoint with a **3D Bearing Innovation Gate**, **`300 ms` `AKAZE` coasting hysteresis**, and **continuous full-frame IMU-derotated KLT VO + `480x270` context relocalization** across occlusions.

### Copy-Paste Prompt for Antigravity
Start a fresh session (`+`), select **`Gemini 4 Carbon` · `High`** (or `/plan` on `Carbon · High` -> execute on `Hydrogen · High`), and paste:

```text
Read @AGENTS.md, @docs/ENGINEERING_LEDGER.md, @.agents/rules/android-camera-ar.md, @.agents/rules/lcars-ui.md, @.agents/rules/litert-inference.md, and @docs/ROADMAP.md in full—Milestone 2D is now active.

Execute Milestone 2D systematically without regressing `fc80785`, `8707282`, `1bfdda0`, or `c307819` invariants:
1. **Forensic Crash Pull & Right-Edge Crash Elimination (`x >= 1920`):**
   - First, query the connected Pixel 10 Pro XL (`adb logcat -b crash -d` and `adb shell dumpsys activity exit-info com.gneissfind.app`) to capture the exact exception/tombstone from the reproducible Lock-On off-screen-right crash. Record the exact root cause in `docs/ENGINEERING_LEDGER.md`.
   - Audit and harden all three boundary paths per `.agents/rules/android-camera-ar.md §1` and `lcars-ui.md §3`: (a) `PushBroomCropper` `YUV_420_888` semi-planar UV `pixelStride=2` last-pixel `buffer.limit()` overrun at `x=1919, y=1079`, (b) OpenCV `Rect`/`Mat.submat` `x + w <= cols` & `y + h <= rows` clamping in `OpenCvConstellationTracker`, and (c) Compose `DrawScope` right/bottom off-screen chevron & badge `Size >= 0` / `NaN` guards in `ScannerScreen`.
   - Add deterministic unit + instrumented boundary tests sweeping a locked target across all 4 edges (`x = -400..2320`, `y = -400..1480`) for 100+ frames with zero crashes.
2. **Upgrade `MOD: LOG` to Hardware `MediaCodec` `30 FPS` `1080p` MP4 + `120 Hz` IMU JSONL:**
   - Replace the legacy `4 Hz` PNG strip logger in `FieldDataLogger` with a background hardware `MediaCodec` (`video/avc` H.264, `1920x1080`, `30 FPS`, `14 Mbps`) + `MediaMuxer` encoder fed directly from the existing `ImageAnalysis` `1920x1080` `YUV_420_888` stream (**NEVER bind CameraX `VideoCapture` as a 3rd camera surface!**), paired with `trace.jsonl` (`header`, `120 Hz` `imu` rotation matrices, and per-`frame` timestamps/optics) in `Android/media/com.gneissfind.app/replays/<YYYYMMDD_HHMMSS>/`.
3. **Implement `ReplayCameraSource` (`SRC` Drawer + `adb` Headless Test Hooks):**
   - Build `ReplayCameraSource` decoding `clip.mp4` (`MediaExtractor` + `MediaCodec` at `30 FPS`) + synchronized `trace.jsonl` `120 Hz` IMU poses into the top viewport AND the exact production `GroundPlaneRaycaster` -> `PushBroomCropper` -> `RfDetrTwoGraphGpuDetector` -> `TemporalTargetTracker` -> `LockOnPinpointController` pipeline.
   - Update `LcarsSourceDrawerOverlay` (`SRC` drawer) to dynamically list discovered `REPLAY: <session>` folders, retain **only 1** synthetic demo (`MIXED_GRAVEL_WALK`), and support `MainActivity` `adb shell am start` extras (`debug_replay_session`, `debug_auto_lock_frame`, `debug_loop_replay`) logging `GNEISSFIND_REPLAY_SUMMARY` to Logcat.
4. **Two-Tier Occlusion Tracking, Innovation Gate & Anti-Thrash Hysteresis (`LockOnPinpointController` + `OpenCvConstellationTracker`):**
   - Keep the `3.0x` target-diameter `192x192` `AKAZE` collar for coplanar pinpoint accuracy, AND unfreeze/upgrade full-frame ego-motion VO (`640x360` IMU-derotated `PyrLK` on `80-120` corners across the full `16:9` frame, `< 2 ms` CPU) so `camPosEnu` and the 3D triangulated point `P_3d` stay tracked **while the target is occluded or off-screen**, backed by a `480x270` full-frame `AKAZE` relocalization snapshot matched at `~5 Hz` during post-occlusion re-entry to reset drift.
   - Enforce a strict **3D Bearing & Scale Innovation Gate** (`<= 4.5 deg` on-screen max step, `[0.45x..2.2x]` range-adjusted diameter gate) so `640x640` `B=1` wide search never jumps onto adjacent cups/teapots/screens, plus **`300 ms` (`9-frame`) `AKAZE` coasting hysteresis** so brief specular/refractive glints never thrash back to `B=1`.
5. **Closed-Loop On-Device Replay Validation:**
   - Generate or capture a validation `.mp4 + trace.jsonl` test clip (including a multi-object distractor scene and a full off-screen-right pan + occlusion return), run it end-to-end on the connected Pixel 10 Pro XL via `debug_replay_session` + `adb exec-out screencap -p` inspection, verify `./gradlew -p android assembleDebug testDebugUnitTest connectedDebugAndroidTest` and `uv run pytest`, make atomic git commits, update `docs/ENGINEERING_LEDGER.md`, and run `./scripts/pack_context.sh`.
```

### Milestone 2D Acceptance Criteria (Oracle)
1. **Zero Off-Screen Crashes (All 4 Edges):** Panning a locked target rapidly off-screen **right, left, up, and down** (and holding off-screen right for `> 5 s` before returning) produces zero crashes, clean edge chevrons, and accurate 3D re-entry.
2. **Distractor & Glint Immunity + Occlusion Memory:** In a scene with two adjacent same-class objects (e.g., glass/mug next to teapot) or brief hand occlusion + camera translation, Lock-On never jumps to the distractor, holds `PINPOINT` steady without `AKAZE <-> B=1` thrashing, and remembers the 3D triangulated target location across occlusion/off-screen translation.
3. **`MOD: LOG` MP4+IMU & `ReplayCameraSource`:** Recording a `10s` clip in `MOD: LOG` creates a valid `1080p 30 FPS` `clip.mp4` + `trace.jsonl` that immediately appears in `SRC` (`REPLAY: ...`) and replays deterministically through the live GPU detector + tracker (both via UI tap and `adb shell am start` extras), with only `1` synthetic demo (`MIXED_GRAVEL_WALK`) remaining in `SRC`.

---

## Milestone 3 — Gravel-Bar Video Capture, Patrol FPS Replay Simulator & T7920 `SAM 2` Auto-Labeling (`ml-pipeline/`)
- Record indoor stand-in and outdoor salted/unsalted PNW gravel-bar `clip.mp4 + trace.jsonl` sessions using the Milestone 2D `MOD: LOG` recorder.
- Build `ml-pipeline/scripts/simulate_patrol_fps.py` to replay `clip.mp4 + trace.jsonl` walks (`<= 1.1 m/s`) at decimated rates (`15, 10, 7.5, 6, 5, 4, 3, 2 FPS`) to find the lowest power-saving `PATROL` FPS ($X	ext{ FPS}$) maintaining `>= 98%` stone lock probability.
- Run `SAM 2` video propagation + Copy-Paste hard-negative synthesis on `GPU 1 & 2` (`2x RTX 4000 Ada`).

## Milestone 4 — `RF-DETR-Nano` PNW Stone Fine-Tuning, Replay Validation & Production Weight Swap
- Fine-tune `RF-DETR-Nano` (`256` & `320`, `B=3`/`B=5`/`B=1`, `W8-Dyn`) on `GPU 0` (`RTX 4090 24GB`), validate on held-out `ReplayCameraSource` gravel clips, and swap production 5-class PNW mineral weights into `android/app/src/main/assets/`.

## Milestone 5 — Field Calibration, Watershed Presets & Hard-Negative Mining
- Per-class confidence sliders, regional presets (Puget Sound beaches, Western Cascade rivers, Bellingham / Skagit bars), and one-tap field confuser flagging.

## Milestone 6 — Late-Stage Audio Polish & V2 Ergonomics
- Polyphonic/harmonic sonification curves, tactile haptics, optional LCARS trills, and hands-free chest-mount evaluation.
