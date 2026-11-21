# GneissFind — Execution Roadmap & Acceptance Criteria (`docs/ROADMAP.md`)

## Completed Foundation
- [x] **Milestone 1 & 1B:** Reversible `sensorLandscape`, Classic TNG LCARS (`8707282`), `fc80785` `<= 1/500 s` shutter clamp + `2.0x` crop, `120 Hz` `RenderNode` EIS, 5-tile swath.
- [x] **Milestone 2, 2B, 2C & 2D:** `RF-DETR-Nano` 2-Graph `W8-Dyn` `LITERT_CL` `Image2D` GPU pipeline (`1e813e2`, `1bfdda0`), COCO stand-ins (`cup`, `apple`, `toothbrush`, `remote`), `WorldRay3D` projection (`0cad9b2`), and ledger compaction.

---

## Milestone 2.5 — Kotlin Shell + Modern Rust "Non-ROS" Core, `.mcap` Sync & Lightweight QMS *(Active Overnight Run)*

### Overnight Operating Conditions (Dark-Room Stationary Pixel 10 Pro HIL)
- **Workstation:** Dell T7920 (Ubuntu 26.10, `GPU 0: RTX 4090`, `GPU 1 & 2: 2x RTX 4000 Ada`).
- **Connected Phone:** Pixel 10 Pro / Pro XL connected via USB (`adb`), stationary in a dark room (`lights off`, `zero physical motion`).
- **Implication for Verification:** Live camera frames will contain dark sensor noise (`luma ~ 0`) and live `TYPE_GAME_ROTATION_VECTOR` IMU samples will show static gravity with zero angular velocity. Therefore, you MUST verify:
  1. Live dark-frame Camera2 + stationary 200 Hz IMU plumbing (real hardware `rowStride`, `< 2.0 ms` `ImageProxy` hold, zero luma-servo crash, `CLOCK_BOOTTIME` monotonic alignment, live `.mcap` recording & LAN/loopback sync), AND
  2. Full algorithmic, GPU detection, NMS, tracking, and voting accuracy via **Synthetic / COCO Stimulus Injection** both on T7920 (`cli_runner` + `ort` + Linux `LiteRT`) and on the physical Pixel 10 Pro (`connectedDebugAndroidTest` feeding synthetic/COCO `YUV_420_888` frames with padded strides + simulated walk IMU trajectories through the real JNI + `LITERT_CL` 2-Graph `W8-Dyn` GPU pipeline!).

---

### Copy-Paste Overnight Prompt for Antigravity (`Gemini 4 Carbon · High` or `Orchestrator (Carbon)`)

```text
Read `@AGENTS.md`, `@.agents/rules/rust-non-ros-core.md`, `@.agents/rules/android-camera-ar.md`, `@.agents/rules/lcars-ui.md`, `@.agents/rules/qms-traceability.md`, `@qms/reqs/01_user_needs.md`, `@qms/reqs/02_system_reqs.md`, `@qms/reqs/03_invariants_and_algos.md`, and `@docs/ENGINEERING_LEDGER.md` in full.

Context for this unattended overnight run:
- My Pixel 10 Pro is connected via USB (`adb`) in a dark room with zero motion. Everything EXCEPT optical/kinematic checks requiring variable real-world light or physical phone motion can and MUST be verified end-to-end tonight.
- First create and switch to branch `git checkout -b rust-core-m2.5` from current passing `HEAD` so `main` stays untouched until parity passes.

Execute Milestone 2.5 end-to-end across all 4 stages below, making test-gated atomic single-sentence Git commits after each stage:

### Stage 1 — Pure-Rust "Non-ROS" Workspace & QMS Bootstrap (`HIR-0`)
1. Initialize the root Cargo workspace (`crates/core_types`, `crates/perception`, `crates/recorder_sync`, `crates/hal_desktop`, `crates/hal_android`, `crates/app_bridge`, `cli_runner`) using pure-Rust math/CV crates (`nalgebra`, `faer`, `factrs` [or pure `nalgebra`/`faer` Gauss-Newton if needed for ARM64], `kornia-rs`/`imageproc`, `mcap`, `rkyv`, `uniffi` v0.29 proc-macros). Ensure zero C++ OpenCV dependency in Android crates.
2. Centralize all 6 `RATIONALE-UNKNOWN` constants from `docs/ENGINEERING_LEDGER.md` into `PerceptionTuningConfig` (`@TracesReq(ALG-TUNE-001)`).
3. Implement phantom-typed coordinate spaces (`TileFrame`, `StripFrame`, `SensorFrame`, `WorldGroundFrame` — `@TracesReq(SYS-COORD-001)`), IMU pitch-to-`crop_y` stabilization (`@TracesReq(SYS-OPT-001)`), stride-safe `YUV_420_888` 5-tile/3-tile cropper (`@TracesReq(SYS-OPT-002, INV-CAM-002)`), `select_top_k_queries`, cross-tile `IoU >= 0.45 || IoM >= 0.65` seam NMS (`@TracesReq(SYS-SEAM-001)`), `WorldRay3D` homography/factor tracker + N-of-M voter + acoustic curve generator (`@TracesReq(ALG-TRK-001)`), and `CLOCK_BOOTTIME`-indexed `.mcap` + `rkyv` recorder/reader (`@TracesReq(SYS-LOG-001)`).
4. Write comprehensive unit + `proptest` suites (including padded `y_row_stride = 1984, uv_pixel_stride = 2` chroma tests, elongated seam-straddling boxes, and deterministic `.mcap` round-trip replay) and verify `cargo test --workspace` passes in `< 5s`.

### Stage 2 — Dell T7920 `ort` CUDA Engine, `LiteRT` Quantization Parity & Headless Rerun (`HIR-1`)
1. In `ml-pipeline/` and `crates/hal_desktop` + `cli_runner`, wire `OrtCudaBackend` (`ort` v2 pinned to `CUDA_VISIBLE_DEVICES=0` RTX 4090, with clean CPU fallback for sandboxed unit tests) alongside a Linux `LiteRT` 2-graph `W8-Dyn` parity checker (`@TracesReq(ALG-DET-001)`).
2. Add `cli_runner synth-mcap`, `cli_runner verify-parity`, `cli_runner replay --mcap ... --save-rrd ...` (headless Rerun `.rrd` file export — never block on an X11 GUI window!), and `cli_runner receive-mcap` (`iroh` / QUIC P2P receiver writing verified `.mcap` bags into `ml-pipeline/inbox/` — `@TracesReq(SYS-LOG-002)`).
3. Benchmark `cli_runner` replay throughput on RTX 4090 and verify `verify-parity` passes across COCO stand-in test fixtures (`cup`, `apple`, `toothbrush`, `remote`).

### Stage 3 — Android `cargo-ndk` + UniFFI + Zero-Copy JNI + Dark-Room Pixel 10 Pro HIL (`HIR-2`)
1. Configure `cargo-ndk` (`arm64-v8a` `--release`) and `uniffi-bindgen` inside `android/app/build.gradle.kts` so native `.so` and generated Kotlin `EngineClient` bindings stay atomically synced.
2. Implement zero-copy JNI (`GetDirectBufferAddress` on `YUV_420_888` planes releasing `ImageProxy` in `< 2.0 ms` — `@TracesReq(INV-CAM-002)`) and wire on-device 2-Graph `W8-Dyn` `LITERT_CL` `Image2D` execution (`@TracesReq(INV-GPU-001, SYS-PERF-001)`) while strictly preserving all `fc80785` Camera2 (`@TracesReq(INV-CAM-001)`) and `8707282` `120 Hz` LCARS invariants (`@TracesReq(INV-UI-001)`). Add the inline `FLAG CONFUSER` and `BEAM MCAP` pills inside the LCARS `LOG` / `SRC` deck (zero modal sheets).
3. Build and run `connectedDebugAndroidTest` (`@TracesReq(SYS-HIL-001)`) on the USB-connected stationary dark-room Pixel 10 Pro to verify:
   - Live Camera2 binds at `2.0x` zoom + `<= 1/500 s` clamp in pitch darkness (`luma ~ 0`) for 15 seconds with zero crashes, `ImageProxy` hold `< 2.0 ms` (`p95`), and monotonic `CLOCK_BOOTTIME` alignment against live stationary 200 Hz `TYPE_GAME_ROTATION_VECTOR` IMU.
   - Synthetic/COCO `YUV_420_888` stimulus injection (`cup`, `apple`, `toothbrush`, `remote` with padded row stride + simulated walk IMU) through the real on-device Rust JNI + Pixel 10 Pro `LITERT_CL` GPU pipeline achieves `<= 125.0 ms/sweep` (`>= 8.0 Hz`), matches expected stand-in detections, and shows `< 5 MB` native heap growth across 300 sweeps.
   - Live `.mcap` recording on the Pixel 10 Pro writes a valid `.mcap` file and transfers via `BEAM MCAP` (`iroh` / QUIC over LAN or `adb reverse` port forward if Wi-Fi AP isolates clients) to `cli_runner receive-mcap` on T7920 with matching Blake3/SHA-256 hash!

### Stage 4 — Full QMS Verification, Traceability Matrix & Ledger Update
1. Annotate all relevant Rust, Kotlin, and Python symbols and tests with `@TracesReq(...)` and run `python3 scripts/qms_verify.py --write-matrix docs/TRACEABILITY_MATRIX.md` until it exits `0` with **100% PASS** across all active requirements.
2. Update `docs/ENGINEERING_LEDGER.md` (keeping total file size `<= 16 KB` per our `N - 1` rule) with Milestone 2.5 latency, RTX 4090 replay speed, dark-room HIL results, and a concise `Morning 5-Minute Lit/Motion Sanity Checklist` for the human owner.
3. Run `./scripts/pack_context.sh` (if present) and leave the working tree 100% clean on branch `rust-core-m2.5`.
```

---

## Morning 5-Minute Human Sanity Checklist (Post-Overnight Run)
Once lights are on and you pick up the Pixel 10 Pro:
1. **Optical Live Stand-In Check:** Point at a real `cup`, `apple`, `toothbrush`, and `remote` at `~1.5 m` — confirm live bounding brackets + acoustic pitchpings match pre-port M2D feel.
2. **Dynamic Wrist Pitch & `180 deg` Flip:** Tilt wrist `+/- 6 deg` to confirm stabilized `cropY` stays anchored on floor target (`120 Hz` `RenderNode` smooth), and flip phone `180 deg` (`ROTATION_90 <-> ROTATION_270`) to confirm upright UI + downward pitch increase.
3. **Merge to `main`:** Once those two physical checks pass, `git checkout main && git merge --no-ff rust-core-m2.5` — ready for Milestone 3 PNW gravel capture!

---

## Milestone 3 — PNW Gravel `.mcap` Capture, T7920 `SAM 2` Auto-Labeling & `RF-DETR` Fine-Tuning
- Capture salted & unsalted PNW gravel `.mcap` bags, `BEAM MCAP` to T7920, auto-label via `SAM 2` on GPUs 1-2 (`2x RTX 4000 Ada`), fine-tune `RF-DETR-Nano` on GPU 0 (`RTX 4090`), and sweep `PerceptionTuningConfig` (`ALG-TUNE-001`) against `.mcap` bags at `500+ sweeps/s`.

## Milestone 4 — Regional Watershed Presets & Hard-Negative Loop
- Per-watershed thresholds + 1-command retraining from `/user/confuser_flag` `.mcap` bookmarks.

## Milestone 5 — Audio Ergonomics & V2 Chest-Mount Polish
- Polyphonic/harmonic sonification curves, haptics, and multi-hour thermal duty-cycling.
