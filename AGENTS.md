# GneissFind — Workspace Architecture & Engineering Rules (`AGENTS.md`)

## 1. Project Identity & Polyglot Monorepo Layout
- **Project Name:** `GneissFind` ("Visual Metal Detector" for PNW rockhounding)
- **Android Application ID / Namespace:** `com.gneissfind.app` (`gneissfind.com`)
- **Target Mobile Hardware:** Google Pixel 10 Pro / Pixel 10 Pro XL (Android 16, Tensor G5, OpenCL `LITERT_CL` GPU)
- **Workstation / V&V Rig:** Ubuntu 26.10 on Dell Precision T7920 (96 GB RAM, 2x 24-core Xeon Platinum, `GPU 0: RTX 4090 24GB`, `GPU 1 & 2: 2x RTX 4000 Ada 20GB`)
- **Dual-Plane Architecture (Strangler-Fig Boundary):**
  - **Kotlin Android Shell (`android/`):** Owns Android OS lifecycle, `CameraX` + `Camera2CameraControl` dynamic exposure/zoom (`fc80785`), `SensorManager` (`TYPE_GAME_ROTATION_VECTOR`), `120 Hz` `RenderNode` `previewView.translationY` pitch EIS, `AudioTrack` PCM waveform synthesis, and Classic 2360s TNG LCARS Jetpack Compose UI (`8707282`). **Never rewrite Android OS/Camera/UI lifecycles in Rust or Flutter.**
  - **Rust "Non-ROS" Core (`crates/` + `cli_runner/`):** Owns 100% of deterministic heavy lift—phantom-typed coordinate geometry (`nalgebra`), stride-safe 5-tile YUV->RGB cropping (`kornia-rs`), `TopK` 2-graph query selection, cross-tile IoU/IoM NMS, homography & `WorldRay3D` factor-graph tracking (`faer`/`factrs`), N-of-M temporal voting, acoustic target parameter math, `.mcap` + `rkyv` high-rate bag recording, `iroh` P2P LAN sync, `rerun` 3D desktop visualization, and `ort` CUDA/TensorRT workstation batch V&V.
  - **ML Pipeline (`ml-pipeline/`):** Python (`uv`) scripts for `SAM 2` auto-labeling (GPUs 1-2), Copy-Paste synthesis, `RF-DETR` training, and dual ONNX + 2-Graph `litert-torch` `W8-Dyn` export (GPU 0).
  - **Lightweight QMS (`qms/` + `scripts/qms_verify.py`):** Git-native Trace-as-Code linking `UN-*`, `SYS-*`, `ALG-*`, and `INV-*` requirements to Rust `#[doc = " @req ..."]` / `#[trace_req(...)]` and Kotlin `@TracesReq(...)` symbols and automated V&V tests.

---

## 2. Git, Ledger Compaction & QMS Gate Mandate
1. **Branch Safety:** Execute major architectural spikes on a dedicated Git branch (e.g., `git checkout -b rust-core-m2.5`) so `main` stays a clean reference oracle until parity passes.
2. **Test- & QMS-Gated Atomic Commits:** After completing each logical work item, verify:
   - Rust workspace: `cargo test --workspace` (or `cargo nextest run --workspace` if installed)
   - Android shell: `./gradlew -p android assembleDebug testDebugUnitTest`
   - QMS traceability: `python3 scripts/qms_verify.py` (or `just qms-check`)
   Once green, commit immediately with a **single-sentence imperative summary** referencing key requirement IDs when applicable.
3. **N - 1 Ledger Compaction Rule:** Keep `docs/ENGINEERING_LEDGER.md` `<= 16 KB`. Retain active hardware invariants (`§2`), 1-line dead-end tombstones (`§3`), and full detail only for milestones `M_{N-1}` and `M_N` (`§4`), archiving older chronological write-ups to `docs/archive/` (excluded from `scripts/pack_context.sh`).
4. **Never Commit:** `target/`, `build/`, `.gradle/`, `.mcap` field captures `> 15 MB`, or `.onnx`/`.tflite`/`.pt` weights `> 10 MB` (except tiny synthetic test fixtures under `qms/fixtures/`).

---

## 3. Sacred Hardware, Camera2, LiteRT & LCARS Invariants (`INV-*`)
Every rule below was earned empirically on physical Pixel 10 Pro hardware. Do NOT regress them:
- **`INV-CAM-001` (`fc80785` Unconditional Viewport & Dynamic Camera2):**
  - `ClippingCameraViewportContainer` must lay out its child unconditionally.
  - Apply `<= 1/500 s` (`2,000,000 ns`) shutter clamp and floating ISO/EV luma servo ONLY via dynamic `Camera2CameraControl` capture request options—NEVER static `ALWAYS_OVERRIDE` on `Camera2Interop.Extender`.
  - Read `SCALER_AVAILABLE_MAX_DIGITAL_ZOOM` synchronously on cold start and engage `2.0x` lossless sensor crop (`ZoomRatio = 2.0f`) by default.
- **`INV-CAM-002` (`1bfdda0` Ring-Buffer Hold `< 2.0 ms` & `CLOCK_BOOTTIME`):**
  - Hold CameraX `ImageProxy` for `< 2.0 ms` total: pass `DirectByteBuffer` plane addresses (`Y`, `U`, `V`), explicit `rowStride` / `pixelStride`, and hardware `imageInfo.timestamp` (`CLOCK_BOOTTIME` ns) into Rust JNI to extract only the stabilized `1792x384` strip, then close `ImageProxy` immediately in `finally`.
- **`INV-UI-001` (`fc80785` + `8707282` Non-Occluding 120 Hz LCARS):**
  - Preserve `120 Hz` `RenderNode` `previewView.translationY` pitch EIS view decoupling, pure `#000000` OLED canvas with `4.dp` gutters, 6-role `LcarsSemanticColors` (`0.05f` HSV value-stepping), inline `SRC` drawer, tabular numerals (`tnum`), and silent UI taps.
- **`INV-GPU-001` (`1e813e2` + `1bfdda0` 2-Graph `W8-Dyn` `LITERT_CL` `Image2D`):**
  - Do NOT replace Android on-device inference with `ort` NNAPI/CPU, static `W8A8` (collapses small-pebble recall to `12.5%`), or sideloaded `Accelerator.NPU` (`/dev/edgetpu` HAL contention).
  - Keep `RF-DETR-Nano` on Pixel 10 Pro running as the **2-Graph `W8-Dyn` (`allowSrcQuantizedFcConvOps = true`) `LITERT_CL` `Image2D` GPU** pipeline (rank `<= 4` tensors, `BATCH_MATMUL` bilinear grid-sample, no `BROADCAST_TO`), invoked from Rust `hal_android` (or thin C-API/JNI bridge during Strangler-Fig transition) with zero JVM float-array copy between Part 1 and Part 2.

---

## 4. Target Taxonomy & M2/M2.5 COCO Parity Stand-In Mapping
- **5-Class Output Schema (`0..3` active targets, `4` muted hard negative):**
  - `0: ChalcedonyAgate` (`01 AGATE / CARNELIAN`) <- COCO Stand-In: **`cup`**
  - `1: JasperRedYellow` (`02 RED / YELLOW JASPER`) <- COCO Stand-In: **`apple`**
  - `2: GreenGemRock` (`03 DALLASITE / NEPHRITE`) <- COCO Stand-In: **`toothbrush`**
  - `3: PetrifiedWood` (`04 PETRIFIED WOOD`) <- COCO Stand-In: **`remote`**
  - `4: HardConfuser` *(Muted in UI & audio)* <- Non-target high-confidence COCO classes (`person`, `chair`, `tv`, `book`, `keyboard`, etc.)

---

## 5. Dell T7920 Multi-GPU Split
- **`GPU 0` (`CUDA_VISIBLE_DEVICES=0`, `RTX 4090 24GB`):** `cli_runner` `OrtCudaBackend` high-speed `.mcap` batch V&V (`>= 500 sweeps/s`), `RF-DETR` fine-tuning, and dual ONNX + `litert-torch` 2-graph export.
- **`GPU 1 & 2` (`CUDA_VISIBLE_DEVICES=1,2`, `2x RTX 4000 Ada 20GB`):** `SAM 2` auto-labeling, Copy-Paste hard-negative compositing, and parallel hyperparameter sweeps over `cli_runner` (`ALG-TUNE-001..006`).
