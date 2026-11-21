---
trigger: glob
globs: "{crates/**,cli_runner/**,Cargo.toml,justfile,android/app/build.gradle.kts}"
description: Rust Non-ROS workspace architecture, crate selection, phantom coordinate frames, stride-safe YUV cropping, dual-backend ML trait, MCAP/Rerun/Iroh plumbing, and Android FFI rules.
---

# Rust "Non-ROS" Core Architecture & Implementation Rules

## 1. Workspace Layout (`Cargo.toml` at Repo Root)
Create a clean Cargo workspace at the repository root so `cargo test --workspace` compiles and passes natively on Ubuntu 26.10 in `< 3 seconds` with **zero** Android SDK/NDK requirement for core crates:

```text
gneissfind/
├── Cargo.toml                  # Workspace root (resolver = "2")
├── justfile                    # Unified commands: test-rust, build-android, qms-check, hil-darkroom
├── crates/
│   ├── core_types/             # Phantom-typed spaces, MineralClass, Clock trait, HUD/Acoustic/QMS structs
│   ├── perception/             # Pure-Rust cropper, EIS crop_y, TopK, IoU/IoM NMS, homography/factrs tracker, voter
│   ├── recorder_sync/          # .mcap + rkyv writer/reader, deterministic replay cursor, iroh QUIC P2P sender/receiver
│   ├── hal_desktop/            # Desktop OrtCudaBackend (ort v2) + LiteRT Linux parity runner + Rerun.io sink
│   ├── hal_android/            # Android JNI DirectByteBuffer entrypoints + LiteRT 2-Graph GPU adapter (cfg(target_os="android"))
│   └── app_bridge/             # Thin UniFFI control-plane API exposed to Kotlin (`EngineClient`)
└── cli_runner/                 # x86_64 binary: `replay`, `verify-parity`, `validate`, `receive-mcap`, `synth-mcap`
```

### Strict Dependency Boundary Rules
1. **`crates/core_types` and `crates/perception` MUST NOT depend on `uniffi`, `jni`, `ndk`, `ort`, or `opencv`.** They must compile cleanly on any Rust target (`x86_64-unknown-linux-gnu` and `aarch64-linux-android`) using 100% safe, pure Rust.
2. **Quarantine C++ OpenCV:** Do NOT link `opencv` or C++ `libopencv_java4.so` into the Android Rust build (`hal_android`). Use `kornia-rs` (`kornia-image`, `kornia-imgproc`, `kornia-icp`/feature ops) + `imageproc` + `faer` SVD homography + `factrs` in `crates/perception`. If C++ `opencv` reference comparison is needed on Ubuntu, gate it strictly behind an optional desktop-only feature in `hal_desktop` (`pc-opencv-ref`, default OFF).

---

## 2. Recommended Crate Versions & API Guardrails
Verify latest compatible minor versions with `cargo add` or pin to tested 2026 releases; do NOT mix legacy `ort 1.x` or `uniffi 0.25` UDL boilerplate:
- **Math & Optimization:** `nalgebra = "0.33"`, `faer = "0.20"` (dense SVD/QR for homography & triangulation), `factrs = "0.2"` (factor-graph pose/landmark smoothing; fallback to pure `nalgebra`/`faer` sliding-window Gauss-Newton if a `factrs` trait conflicts with `no_std`/ARM64 flags).
- **Vision:** `kornia-image` / `kornia-imgproc` (or `image` + `imageproc` + custom NEON/SIMD-auto-vectorized stride-aware YUV->RGB kernel). Note: write the stride-aware `YUV_420_888` BT.601/BT.709 full-range strip converter explicitly in `crates/perception/src/yuv.rs` with unit tests so UV chroma subsampling and arbitrary `row_stride` / `uv_pixel_stride` (`1` planar I420 vs `2` semi-planar NV12/NV21) are 100% deterministic!
- **Serialization & Bags:** `serde = { version = "1.0", features = ["derive"] }`, `rkyv = "0.8"`, `mcap = "0.9"`.
- **Desktop Inference (`hal_desktop` only):** `ort = { version = "2.0.0-rc.9", features = ["cuda", "tensorrt"] }` (gracefully falling back to CPU execution provider in `OrtBackend` if CUDA libraries are absent in unit-test sandboxes, while logging active EP).
- **Visual Debugging (`hal_desktop` / `cli_runner`):** `rerun = "0.22"` (gate behind a `--rerun` CLI flag or save `.rrd` recording files headless via `RecordingStreamBuilder::new("gneissfind").save(path)` so overnight headless runs never block waiting for a GUI window!).
- **Wireless Sync (`recorder_sync`):** `iroh = "0.32"` (or `quinn` + `rustls` + `blake3` direct LAN QUIC stream if `iroh` relay discovery adds unnecessary dependencies on Android; verify `cargo check --target aarch64-linux-android` succeeds cleanly!).
- **Android FFI (`app_bridge` & `hal_android`):** `uniffi = { version = "0.29", features = ["cli"] }` (proc-macro `#[derive(uniffi::Record, uniffi::Enum, uniffi::Object)]` — zero `.udl` files needed!) + `jni = "0.21"`.
- **Testing:** `proptest = "1.6"`, `approx = "0.5"`.

---

## 3. Phantom-Typed Coordinate Frames (`SYS-COORD-001`)
Prevent M2C's `5d13646` coordinate-space mixup at compile time in `crates/core_types/src/coords.rs`:
- Define zero-sized marker types:
  - `pub struct SensorFrame;` (`1920 x 1080` post-2x-crop landscape buffer)
  - `pub struct StripFrame;` (`1792 x 384` in 5-tile mode, or `1088 x 384` in 3-tile mode; origin `(64, crop_y)` or `(416, crop_y)` in `SensorFrame`)
  - `pub struct TileFrame;` (`384 x 384` patch; `tile_idx in 0..5`, `x_offsets = [64, 416, 768, 1120, 1472]`)
  - `pub struct NormalizedViewportFrame;` (`[0.0..1.0]` UI overlay coordinates accounting for `crop_y`)
  - `pub struct WorldGroundFrame;` (metric meters `(x_right_m, y_forward_m, z_up_m)` relative to local ground plane)
- Wrap 2D points and bounding boxes in `Coord2D<Space>` and `BBox2D<Space>` with explicit, tested conversion methods:
  - `BBox2D<TileFrame>::to_strip(tile_x_offset: u32) -> BBox2D<StripFrame>`
  - `BBox2D<StripFrame>::to_sensor(strip_x_origin: u32, crop_y: u32) -> BBox2D<SensorFrame>`

---

## 4. Stride-Safe `YUV_420_888` Strip Cropper (`INV-CAM-002`, `SYS-OPT-002`)
Android camera ISP planes on Pixel 10 Pro frequently have `y_row_stride > width` (e.g., `1920` or `1984` or `2048`) and interleaved chroma (`uv_pixel_stride == 2`).
- Signature in `crates/perception/src/yuv.rs`:
  ```rust
  pub struct Yuv420View<'a> {
      pub y_plane: &'a [u8],
      pub u_plane: &'a [u8],
      pub v_plane: &'a [u8],
      pub width: u32,          // 1920
      pub height: u32,         // 1080
      pub y_row_stride: usize,
      pub uv_row_stride: usize,
      pub uv_pixel_stride: usize,
  }
  ```
- Convert **only** rows `[crop_y .. crop_y + 384)` and columns `[strip_x0 .. strip_x0 + strip_w)` directly into pre-allocated reusable `TileBatchBuffer` (`[5][384 * 384 * 3]` `f32` NHWC `[0.0..1.0]` or fused ImageNet-normalized tensor matching existing `rf_detr_nano_part1` input contract!).
- Write mandatory unit tests with `width = 1920, y_row_stride = 1984, uv_row_stride = 1984, uv_pixel_stride = 2` to prove zero chroma shear or out-of-bounds slice panic.

---

## 5. Dual-Backend `TileBatchDetector` Trait (`INV-GPU-001`, `ALG-DET-001`)
Define in `crates/perception/src/detector.rs`:
```rust
pub trait TileBatchDetector: Send {
    /// Runs 3 or 5 tiles (384x384 RGB) and returns raw per-tile candidate boxes + class logits.
    fn detect_tiles(&mut self, tiles: &TileBatchBuffer, active_tiles: usize) -> Result<Vec<TileRawOutput>, DetectorError>;
}
```
1. **Shared Rust Post-Processing (100% identical on PC and Pixel):**
   - If using 2-graph split (`Part 1` -> `TopK` -> `Part 2`), implement `select_top_k_queries` in pure Rust (`crates/perception/src/topk.rs`) so both Linux parity tests and Android use the exact same query gather math!
   - Sigmoid activation, COCO stand-in -> `MineralClass` + `HardConfuser` mapping, cross-tile seam fusion (`IoU >= 0.45 || IoM >= 0.65`), homography/`WorldRay3D` projection, N-of-M temporal voting (`default 3 of 6`), and acoustic target synthesis (`AcousticCommand { active: bool, freq_hz: f32, pulse_rate_hz: f32, overspeed_warning: bool }`) ALL live inside `crates/perception::PipelineEngine::step(...)`.
2. **Android Backend (`hal_android`):**
   - Pragmatic Strangler-Fig Option A (Preferred if C headers for `libLiteRt.so` are present in Gradle cache): Direct Rust FFI to `LiteRT` C API (`LiteRtCompiledModel`) keeping buffers in native memory.
   - Pragmatic Strangler-Fig Option B (Zero-blocker fallback for overnight build reliability): Rust `PipelineEngine` owns YUV->RGB DirectByteBuffer filling, `select_top_k_queries` in native memory, NMS, tracking, voting, and `.mcap` logging, while invoking the already-working `RfDetrTwoGraphGpuDetector` `LITERT_CL` `Image2D` runner via pre-allocated `DirectByteBuffer`s (`encMemory` zero-copy `1bfdda0` invariant preserved!). Pick whichever compiles cleanly and hits `<= 125 ms/sweep` on the connected Pixel 10 Pro overnight—document the exact choice and latency in `ENGINEERING_LEDGER.md`.
3. **Desktop Backends (`hal_desktop`):**
   - `OrtCudaBackend`: Runs exported full/split ONNX model on `GPU 0` (`RTX 4090`).
   - `LiteRtPythonOrCppParityRunner`: Runs the exact `.tflite` 2-graph `W8-Dyn` pair on Ubuntu x86_64 against 32 synthetic/COCO calibration tiles and verifies `IoU >= 0.90` & class agreement against `OrtCudaBackend`.

---

## 6. `.mcap` + `RobotClock` + Headless Rerun (`SYS-LOG-001`, `SYS-LOG-002`)
- **`RobotClock` Trait:** Never call `std::time::Instant::now()` inside `crates/perception`. Pass `timestamp_boottime_ns: u64` explicitly into `ingest_imu(...)` and `process_frame(...)`.
- **`.mcap` Channels (`crates/recorder_sync`):**
  - `/camera/strip_jpeg` (Q=92 JPEG of stabilized `1792x384` strip + `crop_y` header in `rkyv` envelope, keeping data rate `~4-8 MB/s` at `12.5 Hz`)
  - `/imu/game_rot_vec` (`200 Hz` quaternion + accuracy, `rkyv`)
  - `/camera/optics_meta` (`exposure_ns`, `iso`, `zoom_ratio`, `fy_px`, `slant_range_m`)
  - `/perception/hud_snapshot` (tracks, votes, `AcousticCommand`)
  - `/user/confuser_flag` (one-tap false-positive bookmark timestamp)
- **Headless Rerun `.rrd` Output:** `cli_runner replay --mcap <bag.mcap> --save-rrd <out.rrd>` writes 3D ground plane, camera frustum, 2D strip boxes, and `acoustic/freq_hz` time-series to `.rrd` without opening an X11/Wayland window during unattended runs.

---

## 7. Gradle + `cargo-ndk` + `uniffi-bindgen` Atomic Wiring
- Install `cargo-ndk` (`cargo install cargo-ndk`) and `rustup target add aarch64-linux-android` if not already installed on Ubuntu 26.10.
- Add a Gradle task `buildRustAndroid` in `android/app/build.gradle.kts` wired as a dependency of `preBuild`:
  1. Compiles `--release` (for realistic SIMD/NEON vision speed even in Android `debug` APK!) for `arm64-v8a` into `android/app/src/main/jniLibs/arm64-v8a/libgneissfind_bridge.so`.
  2. Runs `cargo run -p app_bridge --bin uniffi-bindgen generate ...` into `android/app/build/generated/source/uniffi/kotlin` so `.so` and `.kt` checksums can NEVER drift out of sync!
