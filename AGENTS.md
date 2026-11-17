# GneissFind — Workspace Architecture & Engineering Rules (`AGENTS.md`)

## 1. Project Identity & Monorepo Layout
- **Project Name:** `GneissFind` ("Visual Metal Detector" for PNW rockhounding)
- **Android Application ID / Namespace:** `com.gneissfind.app` (`gneissfind.com` primary; `gneissfind.net` reserved)
- **Validated Target Hardware:** Google Pixel 10 Pro XL (`Tensor G5`, Android 17 / SDK 36, `16 KB` page size)
- **Training Workstation:** Ubuntu 26.10 on Dell Precision T7920 (96 GB RAM, 2x 24-core Xeon Platinum, `GPU 0: RTX 4090 24GB`, `GPU 1 & 2: 2x RTX 4000 Ada 20GB`)
- **Repository Structure:**
  - `android/` — Native Kotlin + Jetpack Compose app (`minSdk = 34`, `targetSdk = 36`), CameraX + `Camera2CameraControl`, hardware `MediaCodec` `30 FPS` `1080p` logger & deterministic `ReplayCameraSource`, 120 Hz RenderNode IMU EIS, LiteRT `2.3.2` GPU `W8-Dyn` FP16 inference, OpenCV `4.13.0` (`AKAZE` + full-frame IMU-derotated KLT VO), Classic 2360s TNG LCARS UI.
  - `ml-pipeline/` — Python (`uv`) pipeline for `RF-DETR-Nano` LiteRT `W8-Dyn` export (`B=3`, `B=5`, `B=1`), offline frame-decimated video replay (`simulate_patrol_fps.py`), `SAM 2` video auto-labeling, Copy-Paste augmentation, and PNW stone fine-tuning.
  - `docs/ROADMAP.md` — Forward-looking milestones and acceptance criteria.
  - `docs/ENGINEERING_LEDGER.md` — **Canonical append-only hardware & postmortem ledger.** Read before modifying camera/UI/inference/tracking code; append milestone findings, crash root causes, and commit hashes upon completion.
  - `scripts/pack_context.sh` — Context sync bundler (`/tmp/gneissfind-sync.md`).

---

## 2. Git, Build Provenance & Ledger Workflow Mandate
- **Test-Gated Atomic Commits:** Every functional code change MUST compile cleanly and pass unit tests before committing:
  - `android/`: `./gradlew -p android assembleDebug testDebugUnitTest`
  - `ml-pipeline/`: `uv run pytest`
  - Preserve `docs/COMMENTING.md` standards on Kotlin sources (`comment_diff_check.py` where applicable).
- **Immediate Atomic Commit:** Stage relevant files and commit with a **single-sentence imperative summary**.
- **Compile-Time Build Provenance:** `android/app/build.gradle.kts` injects `BuildConfig.GIT_COMMIT_HASH_6` (`git rev-parse --short=6 HEAD`) and `BuildConfig.BUILD_DATE_ISO` (`YYYY-MM-DD`), displayed in the bottom-left LCARS shoulder (`BottomLeft` elbow); top-left shoulder (`TopLeft` elbow) displays `GNEISSFIND`.
- **Ledger Sync:** Append crash postmortems, benchmark tables, hardware discoveries, and commit hashes to `docs/ENGINEERING_LEDGER.md` and run `./scripts/pack_context.sh` at milestone completion.
- **Never Commit:** `build/`, `.gradle/`, `.venv/`, `local.properties`, raw `.mp4` field captures (`> 10 MB`), or un-LFS'd large model binaries.

---

## 3. Hardware-Validated Android & LiteRT Invariants (DO NOT REGRESS)
1. **`PreviewView` / Replay Surface Unconditional Child Layout (`fc80785`):** `ClippingCameraViewportContainer` MUST run standard `onMeasure` and unconditional `onLayout` child sizing so `PreviewView`'s `TextureView` never stalls at `0x0`. Drive 120 Hz EIS in `PATROL` strictly via RenderNode `translationY`; set `translationY = 0f` in full-screen `16:9` `LOCK-ON` / `PINPOINT` modes.
2. **Dynamic-Only `Camera2CameraControl` & Two-Surface CameraX Limit (`fc80785`):** Keep `Camera2Interop.Extender` strictly limited to `setSessionCaptureCallback`. Drive `<= 1/500 s` (`2,000,000 ns`) shutter clamp, EV-scaled auto-ISO, and `PATROL` `Range(15, 15)` / `Range(30, 30)` FPS throttling via `Camera2CameraControl`. **NEVER bind CameraX `VideoCapture` as a 3rd surface** alongside `Preview` + `ImageAnalysis (1920x1080)`—encode `MOD: LOG` `.mp4` video directly from the `ImageAnalysis` `1920x1080` `YUV_420_888` buffer via hardware `MediaCodec`.
3. **Cold-Start `2.0x` Zoom Bound (`fc80785`):** Query `SCALER_AVAILABLE_MAX_DIGITAL_ZOOM` synchronously via `Camera2CameraInfo` right after `bindToLifecycle`.
4. **3-Slot Lock-Free Ring Buffer & Zero-Copy `encMemory` (`1bfdda0`):** Hold `ImageProxy` `< 3 ms` and call `imageProxy.close()` in `finally`; keep `part1Outputs[0]` bound zero-copy to `part2Inputs[0]` (`allowSrcQuantizedFcConvOps = true` on `W8-Dyn` graphs) and maintain the 3-slot warm `SRC` LRU + disk shader cache (`8d8be41`, `c307819`).
5. **6-Role LCARS Semantic Color Taxonomy (`8707282`, `5d13646`):** Preserve `LcarsSemanticColors`, full-screen `16:9` Lock-On side console (`LockOnSideConsole`), central `LOCK NEAREST` / `BREAK LOCK` button, and silent UI chrome (`isSoundEffectsEnabled = false`).

---

## 4. 3-Stage Acquisition Funnel & Two-Tier Post-Occlusion Tracking
1. **Stage 1 — `PATROL`:** Scans `1792 x 384 px` pitch-stabilized strip (`Default S-01: RF-DETR 256 B=3 W8-Dyn K=100` Interleaved), dropping to `15 FPS` / `30 Hz` (+ optional `90%` `"EYES-UP"` OLED veil) after `5.0 s` idle, waking in `< 50 ms`.
2. **Stage 2 — `LOCK-ON` (`B=1` `320x320` + Innovation Gate):** Tapping a reticle or `LOCK NEAREST` opens full-screen `16:9` view, tracks via `RF-DETR 320 B=1 W8-Dyn` gated by a **3D Bearing Innovation Gate** (`<= 4.5 deg` max single-step jump + `[0.5x..2.0x]` diameter ratio gate so wide `640x640` search never hijacks onto adjacent objects!), and triangulates 3D `sigma3dMm`.
3. **Stage 3 — `PINPOINT` (Two-Tier Local Collar + Full-Frame Ego-Motion & Relocalization):**
   - **Local Coplanar Pinpoint (`3.0x` Collar `192x192` `AKAZE`):** Sub-centimeter target lock (`< 5 ms` CPU) immune to wide-FOV 3D depth relief skew, with **`300 ms` (`8-frame`) coasting hysteresis** so brief specular/refractive glints never thrash back to `B=1`.
   - **Continuous Full-Frame IMU-Derotated KLT Odometry (`640x360` luma, `< 2 ms` CPU) + `480x270` Context Keyframe:** Runs across the entire `16:9` frame **even while the target is occluded or off-screen** so `camPosEnu` and the 3D triangulated stone anchor `P_3d` stay locked in world space across occlusions, paired with a `480x270` full-frame `AKAZE` context snapshot matched at `~5 Hz` during post-occlusion re-entry to eliminate cumulative drift.

---

## 5. Field Recording (`MOD: LOG`) & Deterministic Sensor-Trace Replay (`SRC`)
- **`MOD: LOG` (`30 FPS 1080p MP4` + `120 Hz IMU JSONL`):** Replaces the legacy `4 Hz` PNG strip logger. Encodes `1920x1080` `30 FPS` H.264 (`clip.mp4`, `~14 Mbps`) via hardware `MediaCodec` + `MediaMuxer` alongside `trace.jsonl` (`120 Hz` IMU rotation matrices + per-frame `timestamp_ns`, `pts_us`, `exposure_ns`, `iso`, `zoom_ratio`) under `Android/media/com.gneissfind.app/replays/<timestamp>/`.
- **`ReplayCameraSource`:** `SRC` drawer retains **1** synthetic demo (`MIXED_GRAVEL_WALK`) plus all discovered `replays/<session>/` clips (and supports `adb shell am start` intent extras for automated agent replay & screenshot verification).

<!-- dexiotites:begin (managed by dexiotites install.py: rerun it to update; edits inside this block are overwritten) -->
## Code Comments (Purpose-First Documentation)
Full standard with examples: **`docs/COMMENTING.md`**. Read it before writing or editing comments. Core rules:
- **Reader:** an experienced engineer who can read any language but should not have to read function bodies to understand purpose. No explanations of language features; no analogies to other languages.
- **Required:** a header on every file; a doc comment on every non-trivial type (purpose, state, lifecycle, threading); a doc comment on every non-trivial function (one-line purpose, each parameter with units/coordinate frame/valid range/meaning of `null`, return value, plus side effects, failure behavior, and calling contract where present/non-obvious). Omit `"None"` boilerplate on pure, infallible, unrestricted helpers. `override` and trait-impl methods inherit interface docs; add only implementation-specific behavior or constraints. Test files need a file header and scenario/invariant comments, not full function contracts on self-explanatory `@Test`s.
- **Prohibited:** restating the code; copying constant values into prose (link the constant instead); change history; spec slogans.
- **Voice:** the doc comment holds the contract (what it is for, how to use it correctly); how it works goes inline beside the code. First sentence in plain words, no code names; at most two code names per sentence; no parentheticals (`docs/COMMENTING.md` §10).
- **Never invent rationale:** cite the source (`AGENTS.md §N`, `docs/…`), derive it from the code, or write `RATIONALE-UNKNOWN: <question>`.
- **Keep comments true:** any code change updates the comments describing it in the same edit.
- **Doc linter gate:** before finishing, run this project's doc linter (`docs/COMMENTING.md` §8) and fix every finding in the files you changed with real documentation, not placeholders or suppressions.
- **Retrofitting existing code:** use the `annotate-code` skill (`.agents/skills/annotate-code/`). Change comments only, and verify with its `comment_diff_check.py`.
<!-- dexiotites:end -->
