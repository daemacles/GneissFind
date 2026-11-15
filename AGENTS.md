# GneissFind — Workspace Architecture & Engineering Rules (`AGENTS.md`)

## 1. Project Identity & Monorepo Layout
- **Project Name:** `GneissFind` ("Visual Metal Detector" for PNW rockhounding)
- **Android Application ID / Namespace:** `com.gneissfind.app` (`gneissfind.com` primary; `gneissfind.net` reserved)
- **Validated Target Hardware:** Google Pixel 10 Pro XL (Android 16, flagship GPU + Tensor G5)
- **Training Workstation:** Ubuntu 26.10 on Dell Precision T7920 (96 GB RAM, 2x 24-core Xeon Platinum, `GPU 0: RTX 4090 24GB`, `GPU 1 & 2: 2x RTX 4000 Ada 20GB`)
- **Repository Structure:**
  - `android/` — Native Kotlin + Jetpack Compose app (`minSdk = 34`, `targetSdk = 36`), CameraX + `Camera2CameraControl`, 120 Hz RenderNode IMU EIS, LiteRT GPU FP16 inference, Classic 2360s TNG LCARS UI.
  - `ml-pipeline/` — Python (`uv`) pipeline for `RF-DETR-Nano` LiteRT export/benchmarking, `SAM 2` video auto-labeling, hard-negative Copy-Paste augmentation, and PNW stone fine-tuning.
  - `docs/ROADMAP.md` — Forward-looking milestones and acceptance criteria.
  - `docs/ENGINEERING_LEDGER.md` — **Canonical append-only hardware & postmortem ledger.** Read before touching camera/UI/inference code; append milestone findings and commit hashes upon milestone completion.
  - `scripts/pack_context.sh` — Context sync bundler (`/tmp/gneissfind-sync.md`) for external architecture reviews.

---

## 2. Git & Ledger Workflow Mandate (Test-Gated Atomic Commits)
- **Test Before Commit:** Every functional code change MUST compile cleanly and pass unit tests before committing:
  - `android/`: `./gradlew -p android assembleDebug testDebugUnitTest`
  - `ml-pipeline/`: `pytest` (using `uv run pytest`)
- **Immediate Atomic Commit:** Stage relevant files and commit with a **single-sentence imperative summary** (e.g., `git commit -m "Wire LiteRT GPU RF-DETR-Nano detector with COCO proxy mapping"`).
- **Ledger Sync:** At the close of each milestone, append benchmark numbers, hardware discoveries, and commit hashes to `docs/ENGINEERING_LEDGER.md` and run `./scripts/pack_context.sh`.
- **Never Commit:** `build/`, `.gradle/`, `.venv/`, `local.properties`, raw field captures, or model weights (`*.pt`, `*.pth`, `*.onnx`, `*.tflite` > 10 MB) unless tracked via Git LFS or explicitly ignored with a reproducible export command in `ml-pipeline/`.

---

## 3. Hardware-Validated Android Invariants (DO NOT REGRESS)
1. **`PreviewView` Unconditional Child Layout (`fc80785`):** `ClippingCameraViewportContainer` MUST run standard `onMeasure` and unconditional `onLayout` child sizing so CameraX's asynchronous internal `TextureView` never starves at `0x0`. Drive 120 Hz IMU EIS strictly via `previewView.translationY = t.frameTopPx` on the RenderNode.
2. **Dynamic-Only `Camera2CameraControl` Exposure (`fc80785`):** NEVER attach static AE request options (`CONTROL_AE_MODE_OFF`, `SENSOR_EXPOSURE_TIME`, `SENSOR_SENSITIVITY`) to `Camera2Interop.Extender` (`ALWAYS_OVERRIDE`). Keep only `setSessionCaptureCallback` on `Camera2Interop.Extender` and apply/clear `<= 1/500 s` shutter clamp + EV-scaled auto-ISO exclusively via `Camera2CameraControl`.
3. **Cold-Start `2.0x` Zoom Bound (`fc80785`):** Query `SCALER_AVAILABLE_MAX_DIGITAL_ZOOM` synchronously via `Camera2CameraInfo` right after `bindToLifecycle` so cold-start `2.0x` lossless sensor crop never clamps to `1.0x`.
4. **6-Role LCARS Semantic Color Taxonomy (`8707282`):** Preserve `LcarsSemanticColors` (`valueStepDelta = 0.05f`), left-side `LcarsPitchTravelGauge`, single-action controls, high-water-mark telemetry widths, silent UI chrome (`isSoundEffectsEnabled = false`), and the inline `LcarsSourceDrawerOverlay` (`SRC` drawer).

---

## 4. Push-Broom Geometry & Target Taxonomy
- **Landscape Stream & Swath (`sensorLandscape`):** `1920x1080` `YUV_420_888` at `2.0x` sensor crop (`0.59 mm/px` GSD at `R ~ 1.50 m`, `45 deg` pitch).
  - `FIVE_TILE` (default): `5 x 384x384` tiles (`xOffsets = [64, 416, 768, 1120, 1472]`, `32 px` overlap, `1792x384` strip = `93.3%` width).
  - `THREE_TILE`: `3 x 384x384` tiles (`xOffsets = [416, 768, 1120]`, `1088x384` strip).
  - Pre-allocated `PushBroomSweepBuffers`: `FloatArray(384 * 384 * 3)` RGB `[0.0..1.0]` NHWC per active tile.
- **Production PNW Taxonomy (`TargetMineral` + `HARD_CONFUSER`):**
  0. `CHALCEDONY_AGATE` (`01 Agate / Carnelian`)
  1. `JASPER_RED_YELLOW` (`02 Red / Yellow Jasper`)
  2. `GREEN_GEM_ROCK` (`03 Dallasite / Nephrite`)
  3. `PETRIFIED_WOOD` (`04 Petrified Wood`)
  4. `HARD_CONFUSER` *(Internal negative class; strictly muted in UI reticles & audio)*
- **Milestone 2 Temporary COCO Stand-In Mapping (UI labels unchanged):**
  - `CHALCEDONY_AGATE` <- COCO **`cup`**
  - `JASPER_RED_YELLOW` <- COCO **`apple`**
  - `GREEN_GEM_ROCK` <- COCO **`toothbrush`**
  - `PETRIFIED_WOOD` <- COCO **`remote`**
  - Other high-confidence COCO detections (`person`, `chair`, `tv`, `book`, `bottle`, `keyboard`, `laptop`, etc.) -> **`HARD_CONFUSER`** (muted in UI/audio to test real indoor clutter rejection).

---

## 5. Inference Performance Target & Dell T7920 GPU Split
- **Milestone 2 Pixel 10 Pro XL Pass Floor:** `FIVE_TILE` (`5 x 384x384`) end-to-end LiteRT GPU FP16 sweep latency **`<= 125 ms/sweep` (`>= 8.0 Hz`)** with zero per-frame heap allocations causing GC stutter and 120 Hz EIS smoothness preserved.
- **T7920 Multi-GPU Pinning (`ml-pipeline/`):** Never mix `RTX 4090` and `RTX 4000 Ada` in synchronous `DDP`:
  - `CUDA_VISIBLE_DEVICES=0` (`RTX 4090 24GB`): `RF-DETR` training, distillation, and `litert-torch` export.
  - `CUDA_VISIBLE_DEVICES=1,2` (`2x RTX 4000 Ada 20GB`): `SAM 2` video auto-labeling, Copy-Paste synthetic gravel generation, and evaluation sweeps.

---

## 6. Code Comments (Purpose-First Documentation)
Full standard with examples: **`docs/COMMENTING.md`**. Read it before writing or editing comments. Core rules:
- **Reader:** an experienced engineer who can read any language but should not have to read function bodies to understand purpose. No explanations of language features; no analogies to other languages.
- **Required:** a header on every file; a doc comment on every non-trivial type (purpose, state, lifecycle, threading); a doc comment on every non-trivial function (one-line purpose, each parameter with units/coordinate frame/valid range/meaning of `null`, return value, plus side effects, failure behavior, and calling contract where present/non-obvious). Omit `"None"` boilerplate on pure, infallible, unrestricted helpers. `override` and trait-impl methods inherit interface docs; add only implementation-specific behavior or constraints. Test files need a file header and scenario/invariant comments, not full function contracts on self-explanatory `@Test`s.
- **Prohibited:** restating the code; copying constant values into prose (link the constant instead); change history; spec slogans.
- **Never invent rationale:** cite the source (`AGENTS.md §N`, `docs/…`), derive it from the code, or write `RATIONALE-UNKNOWN: <question>`.
- **Keep comments true:** any code change updates the comments describing it in the same edit.
- **Retrofitting existing code:** use the `annotate-code` skill (`.agents/skills/annotate-code/`). Change comments only, and verify with its `comment_diff_check.py`.
