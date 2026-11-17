# GneissFind — Workspace Architecture & Engineering Rules (`AGENTS.md`)

## 1. Project Identity & Monorepo Layout
- **Project Name:** `GneissFind` ("Visual Metal Detector" for PNW rockhounding)
- **Android Application ID / Namespace:** `com.gneissfind.app` (`gneissfind.com` primary; `gneissfind.net` reserved)
- **Validated Target Hardware:** Google Pixel 10 Pro XL (`Tensor G5`, Android 17 / SDK 36)
- **Training Workstation:** Ubuntu 26.10 on Dell Precision T7920 (96 GB RAM, 2x 24-core Xeon Platinum, `GPU 0: RTX 4090 24GB`, `GPU 1 & 2: 2x RTX 4000 Ada 20GB`)
- **Repository Structure:**
  - `android/` — Native Kotlin + Jetpack Compose app (`minSdk = 34`, `targetSdk = 36`), CameraX + `Camera2CameraControl`, 120 Hz RenderNode IMU EIS, LiteRT `2.3.2` GPU `W8-Dyn` FP16 inference, OpenCV `4.12.0+` (`AKAZE` planar constellation homography & triangulation), Classic 2360s TNG LCARS UI.
  - `ml-pipeline/` — Python (`uv`) pipeline for `RF-DETR-Nano` LiteRT `W8-Dyn` export (`B=3`, `B=5`, and `B=1` Lock-On), offline frame-decimated video replay (`simulate_patrol_fps.py`), `SAM 2` video auto-labeling, Copy-Paste augmentation, and PNW stone fine-tuning.
  - `docs/ROADMAP.md` — Forward-looking milestones and acceptance criteria.
  - `docs/ENGINEERING_LEDGER.md` — **Canonical append-only hardware & postmortem ledger.** Read before modifying camera/UI/inference/tracking code; append milestone findings and commit hashes upon milestone completion.
  - `scripts/pack_context.sh` — Context sync bundler (`/tmp/gneissfind-sync.md`).

---

## 2. Git, Build Provenance & Ledger Workflow Mandate
- **Test-Gated Atomic Commits:** Every functional code change MUST compile cleanly and pass unit tests before committing:
  - `android/`: `./gradlew -p android assembleDebug testDebugUnitTest`
  - `ml-pipeline/`: `uv run pytest`
  - Preserve `docs/COMMENTING.md` standards on Kotlin sources (`comment_diff_check.py` where applicable).
- **Immediate Atomic Commit:** Stage relevant files and commit with a **single-sentence imperative summary**.
- **Compile-Time Build Provenance:** `android/app/build.gradle.kts` MUST inject `BuildConfig.GIT_COMMIT_HASH_6` (`git rev-parse --short=6 HEAD`) and `BuildConfig.BUILD_DATE_ISO` (`YYYY-MM-DD`) at build time, displayed permanently in the bottom-left LCARS shoulder (`BottomLeft` elbow), while the top-left LCARS shoulder (`TopLeft` elbow) displays `GNEISSFIND`.
- **Ledger Sync:** Append benchmark metrics, hardware discoveries, and commit hashes to `docs/ENGINEERING_LEDGER.md` and run `./scripts/pack_context.sh` at milestone completion.
- **Never Commit:** `build/`, `.gradle/`, `.venv/`, `local.properties`, raw field video captures, or un-LFS'd large model binaries unless explicitly required for reproducible local builds per repo `.gitignore`.

---

## 3. Hardware-Validated Android & LiteRT Invariants (DO NOT REGRESS)
1. **`PreviewView` Unconditional Child Layout (`fc80785`):** `ClippingCameraViewportContainer` MUST run standard `onMeasure` and unconditional `onLayout` child sizing so CameraX's internal `TextureView` never stalls at `0x0`. Drive 120 Hz EIS in `PATROL` mode strictly via `previewView.translationY` on the RenderNode; set `translationY = 0f` and expand to full `16:9` frame in `LOCK-ON` / `PINPOINT` modes.
2. **Dynamic-Only `Camera2CameraControl` Exposure & FPS Duty-Cycling (`fc80785`):** Keep `Camera2Interop.Extender` strictly limited to `setSessionCaptureCallback`. Drive `<= 1/500 s` (`2,000,000 ns`) shutter clamp, EV-scaled auto-ISO, and `PATROL` idle frame-rate throttling (`Range(15, 15)` vs `Range(30, 30)`) exclusively via `Camera2CameraControl`.
3. **Cold-Start `2.0x` Zoom Bound (`fc80785`):** Query `SCALER_AVAILABLE_MAX_DIGITAL_ZOOM` synchronously via `Camera2CameraInfo` right after `bindToLifecycle`.
4. **3-Slot Lock-Free Producer-Consumer Queue & Zero-Copy `encMemory` (`1bfdda0`):** Always copy strips/ROIs out of `ImageProxy` in `< 3 ms` and call `imageProxy.close()` immediately on `GneissFind-Analyzer`; keep `part1Outputs[0]` bound zero-copy to `part2Inputs[0]` (`allowSrcQuantizedFcConvOps = true` on `W8-Dyn` graphs).
5. **6-Role LCARS Semantic Color Taxonomy (`8707282`):** Preserve `LcarsSemanticColors` (`valueStepDelta = 0.05f`), left-side `LcarsPitchTravelGauge`, monotonic high-water-mark telemetry widths, and silent UI taps (`isSoundEffectsEnabled = false`).

---

## 4. 3-Stage Target Acquisition Funnel (`PATROL` -> `LOCK-ON` -> `PINPOINT`)
1. **Stage 1 — `PATROL` (Wide Push-Broom Search + Idle Duty-Cycling):**
   - Scans the `1792 x 384 px` (`93.3%` width) pitch-stabilized strip (`Default S-01: RF-DETR 256 B=3 W8-Dyn K=100` Priority Interleaved).
   - **Idle Patrol Throttle:** After `5.0 s` with zero alertable candidates (`score >= 0.30`), steps Camera2 down to `15 FPS`, UI refresh to `30 Hz`, and (when `"EYES-UP"` toggle is ON) dims the camera viewport by `90%` (`#E6000000` OLED saver veil) until a candidate stone blips or the user taps the screen (`< 50 ms` wake to `30 FPS` / `120 Hz`).
2. **Stage 2 — `LOCK-ON` (Tap Target or Central `LOCK NEAREST` Button -> Full-Screen `B=1` Tracking):**
   - Switches top viewport to **uncropped full-screen `16:9` camera view**, disables EIS (`cropY` lock off), and runs **`RF-DETR 320 B=1 W8-Dyn K=100`** (`~15-22 FPS`) on a single `320x320` crop dynamically centered on the tracked stone.
   - Accumulates multi-view camera rays + ground-patch parallax via official OpenCV (`org.opencv:opencv:4.12.0` / `4.13.0` AAR, Apache-2.0) + IMU rotation to estimate 3D stone position and **3D sphere-of-confusion radius** `sigma3dMm`.
3. **Stage 3 — `PINPOINT` (OpenCV `AKAZE` Gravel-Constellation Terminal Guidance):**
   - Automatically engages once `sigma3dMm < 0.5 * estimatedDiameterMm` (with sufficient parallax baseline): OpenCV `AKAZE` + `USAC_MAGSAC` planar homography across a **`3x` gravel context collar** on the `Y` luma plane tracks the stone at full `30 FPS` camera rate down to pickup distance (`~0.25 m`), while throttling `B=1` `RF-DETR 320` to **`1.0 Hz`** to refine mineral ID confidence as resolution increases.
   - **Exit:** Tapping the central **`BREAK LOCK`** button immediately returns to `PATROL`, or auto-exits after **`15.0 seconds`** of continuous lost track (drawing directional edge chevrons during the `15s` re-acquisition window if the target leaves the FOV).

---

## 5. Taxonomy & T7920 Multi-GPU Split
- **Production Classes (`0..3` alertable, `4` muted):** `0: CHALCEDONY_AGATE` (M2 proxy: `cup`), `1: JASPER_RED_YELLOW` (`apple`), `2: GREEN_GEM_ROCK` (`toothbrush`), `3: PETRIFIED_WOOD` (`remote`), `4: HARD_CONFUSER` (all other confident COCO detections, muted).
- **T7920 Split:** `GPU 0` (`RTX 4090`) = `RF-DETR` training/export & FPS replay simulation; `GPU 1,2` (`2x RTX 4000 Ada`) = `SAM 2` video auto-labeling & Copy-Paste synthesis.

<!-- dexiotites:begin (managed by dexiotites install.py: rerun it to update; edits inside this block are overwritten) -->
## Code Comments (Purpose-First Documentation)
Full standard with examples: **`docs/COMMENTING.md`**. Read it before writing or editing comments. Core rules:
- **Reader:** an experienced engineer who can read any language but should not have to read function bodies to understand purpose. No explanations of language features; no analogies to other languages.
- **Required:** a header on every file; a doc comment on every non-trivial type (purpose, state, lifecycle, threading); a doc comment on every non-trivial function (one-line purpose, each parameter with units/coordinate frame/valid range/meaning of `null`, return value, plus side effects, failure behavior, and calling contract where present/non-obvious). Omit `"None"` boilerplate on pure, infallible, unrestricted helpers. `override` and trait-impl methods inherit interface docs; add only implementation-specific behavior or constraints. Test files need a file header and scenario/invariant comments, not full function contracts on self-explanatory `@Test`s.
- **Prohibited:** restating the code; copying constant values into prose (link the constant instead); change history; spec slogans.
- **Never invent rationale:** cite the source (`AGENTS.md §N`, `docs/…`), derive it from the code, or write `RATIONALE-UNKNOWN: <question>`.
- **Keep comments true:** any code change updates the comments describing it in the same edit.
- **Doc linter gate:** before finishing, run this project's doc linter (`docs/COMMENTING.md` §8) and fix every finding in the files you changed with real documentation, not placeholders or suppressions.
- **Retrofitting existing code:** use the `annotate-code` skill (`.agents/skills/annotate-code/`). Change comments only, and verify with its `comment_diff_check.py`.
<!-- dexiotites:end -->
