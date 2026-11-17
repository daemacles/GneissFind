# GneissFind — Execution Roadmap & Acceptance Criteria (`docs/ROADMAP.md`)

## Milestone 1 & 1B — Push-Broom Optical Skeleton, 120 Hz EIS & TNG LCARS *(Completed — `fc80785`)*
- [x] Reversible `sensorLandscape`, default `2.0x` sensor crop + `<= 1/500 s` dynamic shutter clamp, `5 x 384x384` (`1792x384`) strip cropper, 120 Hz RenderNode EIS, and 6-role TNG LCARS UI.

## Milestone 2 — RF-DETR-Nano COCO Stand-In Integration & 3D World-Ray Tracking *(Completed — `0cad9b2`)*
- [x] 100% single-partition `LITERT_CL` 2-Graph `RF-DETR-Nano` export, verified 91-logit COCO stand-in oracle (`cup`, `apple`, `toothbrush`, `remote`), `IoU+IoM` cross-tile seam merge, and 120 Hz 3D `WorldRay3D` ego-motion reprojection.

## Milestone 2B — >= 5.0 Hz Sustained Full-Swath GPU Optimization *(Completed on `exp/m2b-fps-exploration` — `a0fe747`)*
- [x] LiteRT `2.3.2` + `W8-Dyn` (`dynamic_wi8_afp32`, `3.7x` weight compression, `0%` recall loss), zero-copy `encMemory`, 3-slot lock-free async pipeline, `K=100` decoder surgery, and `B=3` priority interleaving (`9.24 FPS` cold / `6.57 FPS` 180s thermal-soaked on `256 B=3 W8-Dyn`; `7.07 FPS` cold / `100%` small recall on `320 B=3 W8-Dyn`).

---

## Milestone 2C — Merge 2B, Idle Patrol Duty-Cycling, Tap-to-Lock `B=1` + OpenCV `AKAZE` Pinpoint Funnel & LCARS Build Provenance *(Current Active Task)*

### Goal
Merge `exp/m2b-fps-exploration` into `main`, eliminate 15s model-switch freezes and duplicate reticles, keep the phone cool during eyes-up walking via `15 FPS` / OLED-dimmed `PATROL` duty-cycling, and implement the **3-Stage Target Acquisition Funnel** (`PATROL` -> Full-Screen `B=1` `320x320` `LOCK-ON` -> OpenCV `AKAZE` gravel-constellation `PINPOINT` guidance once 3D triangulation sphere of confusion `< 0.5 * stone diameter`) with ambidextrous LCARS controls and Git build provenance.

### Copy-Paste Prompt for Antigravity
Paste this prompt into Antigravity:

```text
Read @AGENTS.md, @docs/ENGINEERING_LEDGER.md, @.agents/rules/android-camera-ar.md, @.agents/rules/lcars-ui.md, @.agents/rules/litert-inference.md, and @docs/ROADMAP.md in full—Milestone 2C is now active.

Execute Milestone 2C step-by-step without regressing `fc80785`, `8707282`, `0cad9b2`, or `1bfdda0` invariants:
1. Merge `exp/m2b-fps-exploration` cleanly into `main` (and work on `main` or a clean feature branch merged back to `main` once all tests pass).
2. Fix GPU shader caching & track deduplication:
   - Configure per-profile LiteRT GPU disk shader serialization (`part1` and `part2` distinct tokens in `context.codeCacheDir`) and non-blocking async loading (`INITIALIZING GPU KERNELS...` inline banner) so warm `SRC` profile switches complete in `<= 500 ms` without UI freezes.
   - Add Track-Level Spatial NMS in `TemporalTargetTracker` (`<= 3.2 deg` ray separation or `<= 75 mm` 3D distance or strip `IoU >= 0.35 / IoM >= 0.60`) so one object never spawns duplicate brackets.
3. Implement `PATROL` Duty-Cycling + `"EYES-UP"` OLED Dimming:
   - After `5.0 s` with zero alertable candidates (`score >= 0.30`) in `PATROL` mode, drop Camera2 `CONTROL_AE_TARGET_FPS_RANGE` to `Range(15, 15)` via `Camera2CameraControl` (preserving `<= 1/500 s` shutter clamp!) and set `preferredRefreshRate = 30f`. Add an `"EYES-UP"` (`EYE`) toggle pill that applies a `90%` `#000000` OLED-saving veil when `PATROL_IDLE` is active, waking to `30 FPS` / `120 Hz` in `< 50 ms` on first candidate detection or tap.
4. Export `rf_detr_nano_320_b1_w8dyn` (`B=1`, `320x320`, `K=100`, `W8-Dyn`) in `ml-pipeline/` (`uv run pytest` verified) and keep it pre-warmed so Lock-On transitions in `< 50 ms` at `~15-22 FPS`.
5. Integrate official Maven Central OpenCV (`org.opencv:opencv:4.12.0` or `4.13.0`, Apache-2.0, verifying `16 KB` page alignment on Pixel 10 Pro XL) and implement the **Lock-On -> Pinpoint Funnel**:
   - Triggered by tapping any target reticle in the strip OR tapping the central `LOCK NEAREST` LCARS button: switch top viewport to **uncropped full-screen `16:9` camera view**, disable EIS (`translationY = 0f`), and track the target with dynamic `B=1` `320x320` crops while running OpenCV ground-patch parallax + `Calib3d.triangulatePoints` / 3D ray covariance to compute `sigma3dMm` (sphere of confusion).
   - Once `sigma3dMm < 0.5 * estimatedDiameterMm` (with valid multi-view parallax), hand off 30 FPS guidance to OpenCV `AKAZE` + `BFMatcher(NORM_HAMMING)` + `Calib3d.findHomography(..., USAC_MAGSAC)` over a **`3x` gravel-constellation collar** on the `Y` luma plane (with rolling scale re-keyframing on approach from `1.5m -> 0.25m`), throttling `B=1` RF-DETR down to **`1.0 Hz`** for confidence refinement.
   - Maintain a **`15.0 s` lost-track grace window** (with off-screen directional edge chevrons) before auto-returning to `PATROL`, or exit immediately when `BREAK LOCK` is tapped.
6. Apply LCARS UI & Build Provenance updates:
   - Top-left shoulder (`TopLeft` elbow): change `"GEO-47"` to **`GNEISSFIND`**.
   - Bottom-left shoulder (`BottomLeft` elbow): inject `BuildConfig.GIT_COMMIT_HASH_6` (`git rev-parse --short=6 HEAD`) and `BuildConfig.BUILD_DATE_ISO` (`YYYY-MM-DD`) via Gradle and render both cleanly (`e.g., A0FE74 • 2026-11-16`).
   - Replace `SPEED SIM` with the single stateful **`LOCK NEAREST` / `BREAK LOCK`** button and **swap positions with `SRC` (`MODEL SELECTOR`)** so `LOCK NEAREST / BREAK LOCK` sits right next to the center deck divider for ambidextrous left/right-handed use.
7. Add unit + instrumentation/device tests covering Track NMS, Patrol idle state transitions, Lock-On ROI clamping, and OpenCV `AKAZE` homography + triangulation convergence. Verify `uv run pytest` and `./gradlew -p android assembleDebug testDebugUnitTest`, install on Pixel 10 Pro XL, make atomic git commits on `main`, update `docs/ENGINEERING_LEDGER.md`, and run `./scripts/pack_context.sh`.
```

### Milestone 2C Acceptance Criteria (Oracle)
1. **Warm Switch Speed & Zero Duplicate Tracks:** Switching cached models in `SRC` takes `<= 0.5 s` with zero UI freeze, and indoor stand-ins (`cup`, `apple`, `toothbrush`, `remote`) produce strictly 1 reticle per physical object.
2. **Patrol Power/Thermal Throttle:** Leaving the camera aimed at empty floor for `> 5 s` drops Camera2 rate to `15 FPS` and dims the viewport (when `EYES-UP` is ON), waking `< 50 ms` when a `cup` or `remote` enters the strip.
3. **Full-Screen `B=1` Lock-On -> OpenCV `AKAZE` Handover:** Tapping a reticle or the central `LOCK NEAREST` button opens full-screen `16:9` uncropped view at `>= 15 FPS` (`B=1` `320px`), shrinks the uncertainty ring as the phone translates `~15-25 cm` laterally/forward, switches cleanly to `30 FPS` `[AKAZE]` constellation guidance (`1 Hz` neural refresh) down to `0.25 m` close-up, survives brief occlusion with edge chevrons up to `15 s`, and releases immediately on `BREAK LOCK` (which turns from `LOCK NEAREST` TNG Gold to `BREAK LOCK` Mars Red).
4. **LCARS Shoulders & Build:** Top-left elbow reads `GNEISSFIND`; bottom-left elbow displays the live 6-char Git hash + `YYYY-MM-DD` build date; `apk` installs and runs cleanly on `16 KB` page-size Pixel 10 Pro XL.

---

## Milestone 3 — Scene/Gravel Video Capture, Patrol FPS Replay Simulator & T7920 `SAM 2` Auto-Labeling (`ml-pipeline/`)
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
