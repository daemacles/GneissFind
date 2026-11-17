# GneissFind — Execution Roadmap & Acceptance Criteria (`docs/ROADMAP.md`)

## Milestone 1 & 1B — Push-Broom Optical Skeleton, 120 Hz EIS & TNG LCARS *(Completed — `fc80785`)*
- [x] Reversible `sensorLandscape`, default `2.0x` sensor crop + `<= 1/500 s` dynamic shutter clamp, `5 x 384x384` (`1792x384`) strip cropper, 120 Hz RenderNode EIS, and 6-role TNG LCARS UI.

## Milestone 2 — RF-DETR-Nano COCO Stand-In Integration & 3D World-Ray Tracking *(Completed — `0cad9b2`)*
- [x] 100% single-partition `LITERT_CL` 2-Graph `RF-DETR-Nano` export, verified 91-logit COCO stand-in oracle (`cup`, `apple`, `toothbrush`, `remote`), `IoU+IoM` cross-tile seam merge, and 120 Hz 3D `WorldRay3D` ego-motion reprojection.

## Milestone 2B — >= 5.0 Hz Sustained Full-Swath GPU Optimization *(Completed & Merged to `main` — `b892dd7`)*
- [x] LiteRT `2.3.2` + `W8-Dyn` (`dynamic_wi8_afp32`, `3.7x` weight compression, `0%` recall loss), zero-copy `encMemory`, 3-slot lock-free async pipeline, `K=100` decoder surgery, and `B=3` priority interleaving (`9.24 FPS` cold / `6.57 FPS` 180s thermal-soaked on `256 B=3 W8-Dyn`; `7.07 FPS` cold / `100%` small recall on `320 B=3 W8-Dyn`).

## Milestone 2C — Idle Patrol Duty-Cycling, Tap-to-Lock `B=1` + OpenCV `AKAZE` Pinpoint Funnel & LCARS Build Provenance *(Completed — `c307819`)*
- [x] Merged `exp/m2b-fps-exploration` into `main` (`b892dd7`).
- [x] Per-profile LiteRT OpenCL disk shader cache (`codeCacheDir/litert_gpu_cache`, `4.05x` faster cold startup) + 3-slot warm `SRC` LRU (`0.78–1.53 ms` warm profile switches) + Track-Level Spatial NMS (`<= 3.2°` ray / `<= 75 mm` 3D / `IoU >= 0.35`, `IoM >= 0.60`).
- [x] `PATROL` `15 FPS` Camera2 + `30 Hz` display thermal duty-cycling (`<= 1/500 s` shutter preserved) + `"EYES-UP"` `90%` `#E6000000` OLED veil with `< 50 ms` candidate/touch wake.
- [x] Exported & parity-verified `rf_detr_nano_320_b1_w8dyn` (`24.11 + 3.71 MiB`, `IoU = 0.982–0.998`) pre-warmed on GPU alongside `16 KB`-aligned `org.opencv:opencv:4.13.0` (`KleidiCV` NEON).
- [x] Implemented 3-stage `PATROL -> LOCK-ON` (full-screen `16:9`, `B=1` `320x320` + OpenCV gravel KLT + `Calib3d.triangulatePoints` `sigma3dMm` handover) `-> PINPOINT` (`3.32–4.83 ms` `192x192` `AKAZE` + `USAC_MAGSAC` `30 FPS` gravel-constellation tracking + `1.0 Hz` neural refresh, rolling `1.45x` scale keyframes, off-screen chevrons, `15.0 s` grace window).
- [x] Updated LCARS shoulders (`GNEISSFIND` + `BuildConfig.GIT_COMMIT_HASH_6 • BuildConfig.BUILD_DATE_ISO`) and placed stateful `LOCK NEAREST` / `BREAK LOCK` next to the center deck divider.

---

## Milestone 3 — Scene/Gravel Video Capture, Patrol FPS Replay Simulator & T7920 `SAM 2` Auto-Labeling (`ml-pipeline/`) *(Next Up)*
- **Full-Frame `30 FPS` Video + IMU Sidecar Capture (`LOG DATA` Upgrade):** Record synchronized `1920x1080` `30 FPS` walkthrough videos + `metadata.jsonl` IMU/optics telemetry—starting with indoor mocked-up stone/stand-in scenes and moving to salted + unsalted PNW gravel bars.
- **Trace-Driven Patrol FPS Simulator (`ml-pipeline/scripts/simulate_patrol_fps.py`):** Replay recorded `<= 1.1 m/s` walks through `RF-DETR` + `TemporalTargetTracker` at decimated frame rates (`15, 10, 7.5, 6, 5, 4, 3, 2 FPS`) to empirically determine the minimum `PATROL` FPS ($X	ext{ FPS}$) that preserves `>= 98%` stone lock probability—unlocking further thermal and battery savings in the field.
- **`SAM 2` + Hard-Negative Copy-Paste (`GPU 1 & 2`, `2x RTX 4000 Ada`):** Propagate `SAM 2` masks across recorded gravel videos into 5-class training boxes (`4 PNW stone classes + hard_confuser`) and composite segmented raw PNW stones onto empty hard-negative gravel frames.

## Milestone 4 — `RF-DETR-Nano` PNW Stone Fine-Tuning, FPS-Simulated Validation & Production Weight Swap
- Fine-tune `RF-DETR-Nano` (`256` & `320`, `B=3`/`B=5`/`B=1`, `W8-Dyn`) and optional `RF-DETR-Large` teacher on `GPU 0` (`RTX 4090 24GB`).
- Validate fine-tuned weights against held-out real gravel-bar walk videos across simulated `PATROL` FPS rates in the Milestone 3 replay harness before locking production field duty-cycle defaults and swapping weights into `android/app/src/main/assets/`.

## Milestone 5 — Field Calibration, Watershed Presets & Hard-Negative Mining
- Per-class confidence sliders, regional presets (Puget Sound beaches, Western Cascade rivers, Bellingham / Skagit bars), and one-tap field confuser flagging.

## Milestone 6 — Late-Stage Audio Polish & V2 Ergonomics
- Polyphonic/harmonic sonification curves, tactile haptics, optional LCARS trills, and hands-free chest-mount evaluation.
