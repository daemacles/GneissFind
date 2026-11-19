# GneissFind — Engineering & Validation Ledger (`docs/ENGINEERING_LEDGER.md`)

> [!IMPORTANT]
> **Hot-Ledger (`<= 16 KB` cap, `< 14 KB` target, `N - 1` Rolling Window):** Active invariants, dead-end tombstones, rolling write-ups for Milestones `N - 1` & `N`, and current interfaces. Verbatim M1–M2B history lives in [`docs/archive/LEDGER_M1_M2B.md`](archive/LEDGER_M1_M2B.md); forward specs live in [`docs/ROADMAP.md`](ROADMAP.md).

---

## 1. Milestone Status Summary

| Milestone | Commits | Hardware | Summary & Archive Links |
| :--- | :--- | :--- | :--- |
| **M1** | `3f1dee0` | Pixel 10 Pro XL | Push-broom skeleton, `<= 1/500 s` clamp, 3-of-6 voter, speed gate, `USAGE_MEDIA` sonifier ([Archive](archive/LEDGER_M1_M2B.md)). |
| **M1B** | `0e3c81e`..`fc80785` | Pixel 10 Pro XL | Reversible `sensorLandscape`, `2.0x` crop (`0.59 mm/px`), `5x384` strip, 120 Hz RenderNode EIS, 6-role TNG LCARS ([Archive](archive/LEDGER_M1_M2B.md)). |
| **M2** | `96147e5`..`0cad9b2` | Pixel 10 Pro XL (`58110DLCQ00003`) | 2-Graph `RF-DETR-Nano` (`909/909` + `350/350` `LITERT_CL`), COCO oracle, cross-tile IoM merge, 120 Hz `WorldRay3D` ([Archive](archive/LEDGER_M1_M2B.md)). |
| **M2B** | `7d81cbf`..`b892dd7` | Pixel 10 Pro XL (`58110DLCQ00003`) | LiteRT `2.3.2` `W8-Dyn` (`28.1 MB`, `0%` loss), zero-copy `encMemory`, 3-slot ring, `K=100`, `B=3` interleave (`9.24 FPS` cold / `6.57 FPS` 180s soak; [Report](M2B_OVERNIGHT_REPORT.md), [Archive](archive/LEDGER_M1_M2B.md)). |
| **M2C** (`N-1`) | `8d8be41`..`5d13646` | Pixel 10 Pro XL (`16 KB` pages) | OpenCL disk shader cache + 3-slot warm LRU (`0.78–1.53 ms`), Track NMS, `15 FPS` `"EYES-UP"` idle, `B=1` `320` Lock-On, OpenCV `4.13` `AKAZE` (`1.99–4.83 ms`), `16:9` Lock HUD. |
| **M2D** (`N`) | `172c4a5`..HEAD | Pixel 10 Pro XL (`58110DLCQ00003`) | Fixed right/bottom `coerceIn` crash, `1080p30` MP4 + `200 Hz` IMU `MOD: LOG` (audio-silent) + `ReplayCameraSource`, API-30 `CONTROL_ZOOM_RATIO` `fy` fix, **Unified 3-Tier Spatial Anchor** (`5.34–7.24 ms` ARM64, `0%` blank-wall false locks; [Report](M2D_SPATIAL_ANCHOR_REPORT.md)). |

---

## 2. Distilled Hardware, Camera2 & LiteRT Invariants (M1–M2B)

- **`PreviewView`, Camera2 & Reversible `sensorLandscape` (`0e3c81e`, `fc80785`):**
  - `ClippingCameraViewportContainer` MUST execute standard `onMeasure` and unconditional `onLayout` child sizing so `PreviewView`'s async `TextureView` never stalls at `0x0`; drive 120 Hz EIS strictly via `previewView.translationY` (`0f` in `16:9` Lock-On).
  - Keep `Camera2Interop.Extender` limited to `setSessionCaptureCallback` (avoiding `OptionPriority.ALWAYS_OVERRIDE`); drive `<= 1/500 s` (`2,000,000 ns`) shutter clamp, `Range(15, 15)`/`Range(30, 30)` FPS, `CONTROL_VIDEO_STABILIZATION_MODE_OFF` (eliminating `-30..-60 ms` EIS lag), and auto-ISO strictly via `Camera2CameraControl`. Scale `fy` by API-30 `CONTROL_ZOOM_RATIO` whenever `SCALER_CROP_REGION >= 98%` of `activeArray` (verified on Pixel 10 Pro XL: `zoom=2.0 -> computedFy=2736.9 px` vs broken `1368.4 px`). Scale closed-loop luma target `118f` by $2^{0.5 \cdot \text{EV}}$ (`[42, 210]`).
  - Query `SCALER_AVAILABLE_MAX_DIGITAL_ZOOM` synchronously via `Camera2CameraInfo` right after `bindToLifecycle` to prevent the `maxZoomRatio == 1.0f` cold-start race. Bind **at most 2 CameraX surfaces** (`Preview` + `ImageAnalysis 1920x1080`). Wire `DisplayListener` + `OrientationEventListener` for $180^\circ$ `ROTATION_90`/`270` flips and `AudioAttributes.USAGE_MEDIA` (`STREAM_MUSIC`).
- **6-Role TNG LCARS Palette (`8707282`):** `LcarsSemanticColors` with `±5%` HSV stepping (`#FF9900` On/Off, `#9999FF` Mode, `#FFCC66` Action/Drawer, `#99CCFF` Telemetry, `#FFCC99` Config, `#CC4444` Alert), one button per action, monotonic high-water segment widths, `isSoundEffectsEnabled = false`.
- **5 `LITERT_CL` GPU Rules & COCO Oracle (`4825ef5`, `1e813e2`):** (1) Split `Part 1` (`909/909` GPU) and `Part 2` (`350/350` GPU) around host min-heap `TopK` (no `TOPK_V2`/`GATHER_ND`); (2) `BATCH_MATMUL` tent bilinear grid-sample (no FP16 `>2048` 1D index overflow); (3) fuse non-transposed axes so all tensors stay **rank $\le 4$**; (4) `torch.cat` instead of `BROADCAST_TO`; (5) precomputed sine-embed freq + `relu(x) - relu(x - 1)` instead of `FLOOR_DIV`/`RELU_0_TO_1`. Stand-ins (`IoU = 1.0000`): `cup=47`, `apple=53`, `toothbrush=90`, `remote=75`.
- **120 Hz `WorldRay3D` Reprojection (`0cad9b2`):** 128-sample $3\times 3$ `TYPE_GAME_ROTATION_VECTOR` ring buffer back-projects hits at $t_{\text{capture}}$ (`fy = 1475/2950 px`) into unit `WorldRay3D` vectors ($\Delta\theta \le 3.8^\circ$ association, 2-hit `>= 0.72` fast lock, 2-vote hysteresis) and forward-reprojects reticles at `120 Hz`.
- **Async Ring, Zero-Copy `encMemory` & `W8-Dyn` (`1bfdda0`):** Copy YUV in `< 3 ms` into the 3-slot `PreparedSweepSlot` ring and `imageProxy.close()` in `finally`; bind `part1Outputs[0]` directly as `part2Inputs[0]`; quantize weights with `dynamic_wi8_afp32` (`allowSrcQuantizedFcConvOps = true`, `104.5 -> 28.1 MB`, `0%` loss) + `K=100` + `B=3` `{0,2,4}`/`{1,prio,3}` interleaving (`108.2 ms` / `9.24 FPS` cold, `152.2 ms` / `6.57 FPS` 180s soaked on `S-01`).
- **Active `RATIONALE-UNKNOWN` Markers (`docs/COMMENTING.md`):** (1) `PushBroomCropper.DEFAULT_MIN_CONFIDENCE = 0.35f`; (2) `MetalDetectorSonifier` `0.55f / 0.45f` confidence/vote blend; (3) `TONE_BURST_DURATION_SEC = 0.048f`; (4) `320..780 Hz` pitch bounds; (5) `2.0..14.0 Hz` ping cadence bounds; (6) `PushBroomCameraController` `118f` 0-EV luma target & `0.12f` deadband.

---

## 3. Dead-End & Negative-Knowledge Registry

- **Static `W8A8` (`static_wi8_ai8`) on DINOv2 ViT:** Collapses `38–45 px` pebble recall `100% -> 12.5%` (`50%` Med) and segfaults `liblitert_jni.so` under `Options(Accelerator.NPU, Accelerator.CPU)`.
- **Sideloaded `Accelerator.NPU` on Tensor G5:** Fails with `kLiteRtStatusErrorNotFound` without Play Services / AICore `DISPATCH_OP`, while Camera HAL `G3ACoordinator` occupies `/dev/edgetpu` (`dive0`) at 30–60 Hz.
- **Fused 1-Graph `RF-DETR-Nano`:** `TOPK_V2` barrier drops `404` decoder nodes onto `8` XNNPACK CPU partitions (`538.12 ms` vs `376.86 ms` 2-Graph `B=5`; sequential `5xB=1` is `595.97 ms`).
- **OpenCL `BufferStorageType.BUFFER` (SSBO) on PowerVR DXT-48-1536:** `2.25x` slower (`848.90 ms` vs `376.86 ms`) than texture-cached `Image2D`.
- **`1x256x1152` Panoramic Squash:** `1.5x` horizontal compression collapses `38–45 px` pebble recall `100% -> 37.5%`.
- **Zero-Retrain ViT Layer Pruning & `ToMe-2:1`:** `9/12`-layer prune yields `0.0%` recall; `ToMe-2:1` drops small-pebble recall `100% -> 37.5%` (`-7 ms` GPU).

---

## 4. Rolling Recent Milestones (`N - 1` & `N`)

### 4.1. Milestone 2C (`N - 1`) — Shader Disk Cache + Warm LRU, Track NMS, `"EYES-UP"` Idle & OpenCV `AKAZE` Funnel (`8d8be41`..`5d13646`)

1. **OpenCL Disk Cache + 3-Slot `WarmPatrolSlot` LRU (`RfDetrTwoGraphGpuDetector.kt`):** Disk-caching OpenCL programs in `codeCacheDir/litert_gpu_cache` sped up startup `4.05x` (`8,288.8 -> 2,047.7 ms`), and keeping 3 compiled `W8-Dyn` profiles (`~84 MB`) resident cut `SRC` switching from `15 s` to **`0.78–1.53 ms`**.
2. **Track-Level Spatial NMS (`TemporalTargetTracker.kt`):** Deduplicates tracks within `<= 3.2°` world-ray angle, `<= 75 mm` 3D ground distance, or strip `IoU >= 0.35 / IoM >= 0.60`.
3. **`PATROL` `15 FPS` / `30 Hz` Duty-Cycling & `"EYES-UP"` Veil (`PatrolDutyCycleController.kt`):** After `5.0 s` with `score < 0.30`, clamps `CONTROL_AE_TARGET_FPS_RANGE = Range(15, 15)` inside dynamic `CaptureRequestOptions` (preserving `<= 1/500 s` shutter), sets `preferredRefreshRate = 30f`, and applies an optional `90%` `#E6000000` OLED veil (`< 50 ms` wake).
4. **`B=1` `320x320` `W8-Dyn` Export, `16 KB` OpenCV `4.13.0` & Full-Screen `16:9` Lock HUD (`838b1f4`..`5d13646`):** Exported `rf_detr_nano_part1_b1_320_w8dyn.tflite` (`24.11 MiB`, `905/905` GPU) + `part2` (`3.71 MiB`, `324/324` GPU, `IoU = 0.982–0.998`, `75–102 ms` GPU); verified `0x4000` (`16 KB`) ELF alignment + `KleidiCV 0.7.0` in `org.opencv:opencv:4.13.0`; expanded persistent `16:9` Lock viewport (`+91%` height) beside `LockOnSideConsole` + `GNEISSFIND` / `GIT_COMMIT_HASH_6 • BUILD_DATE_ISO` shoulders; unified Lock-On clocks on `System.nanoTime()`, switched `AKAZE` `3.0x` collar (`192x192`) to 4-DOF `estimateAffinePartial2D` (`1.99–4.83 ms` CPU) with `1.28x`/`0.78x` rolling keyframes, and added adaptive `[320..640] -> 320x320` neural cropping. Field walks exposed 4 remaining bugs fixed in M2D: right-edge crash, distractor hijack, glint thrashing, and frozen off-screen VO.

### 4.2. Milestone 2D (`N`) — Right-Edge Crash Fix, `30 FPS` MP4+IMU Replay, Unified 3-Tier Spatial Anchor & UX Hardening (`172c4a5`..HEAD)

1. **Right-Edge Crash Fix (`ae0ea3e`):** `logcat -b crash` captured 9 `Cannot coerce value to an empty range` tombstones at `LockOnPinpointController.onNeuralB1Detections:847` (`1920 < 1925`, `1080 < 1088`). Hardened via `clampBoxToFrame` + `finiteOr`, `PushBroomCropper.safeChromaByteIndex`, `computeSafeSubmatRect`, and `DrawScope` `Size >= 0`/`NaN` guards (`168+` 4-edge sweep frames).
2. **`1080p30` MP4 + `200 Hz` IMU `MOD: LOG` & `ReplayCameraSource` (`820c87e`, `e3886b3`, `8febba8`):** Background `MediaCodec` (`video/avc`, `1080p30`, `14 Mbps`) + `MediaMuxer` (`8`-slot vectorized `NV12`/`I420` pool, **0 extra CameraX surfaces**) writing `clip.mp4` + `trace.jsonl` (`Android/media/com.gneissfind.app/replays/`); `ReplayCameraSource` decodes into `TextureView` + production pipeline with dynamic `SRC` discovery, **1** synthetic demo (`MIXED_GRAVEL_WALK`), EOF wrap reset, and `MainActivity` `adb` hooks (`debug_replay_session`, `debug_auto_lock_frame`, `debug_loop_replay`, `debug_record_seconds`).
3. **Innovation Gate, Hysteresis & Baseline Replay (`eb83918`, `8febba8`):** Enforced `cls == lockedClass`, `<= 4.5°` on-screen (`<= 9.0°` off-screen) ray step, `[0.45x..2.20x]` diameter gate, `300 ms` glint coasting, `>= 1.1 s` sole-candidate center recovery probe, and `leaveApksInstalledAfterRun=true`. `m2d_validation_v1` (`240` frames): `locks=1, handovers=1, lost_locks=0, distractor_jumps=0, distractors_rejected=83, exceptions=0`.
4. **Unified 3-Tier Spatial Anchor, API-30 `fy` Calibration, `MOD: LOG` Audio Silence & Honest UX ([Report](M2D_SPATIAL_ANCHOR_REPORT.md)):**
   - **Camera2 + IMU Calibration:** Fixed API-30 `CONTROL_ZOOM_RATIO` `2x` focal halving (`PushBroomCameraController.computeStreamVerticalFocalLengthPxFromDims`; live Pixel 10 Pro XL `CAM2_INTRINSICS`: `crop=4080x3072 == active, capZoom=2.0 -> computedFy=2736.9 px, videoStab=0`), added `SO(3)` Gram-Schmidt interpolation + `cameraImuLagCompensationNs` in `GroundPlaneRaycaster`, and legacy `>= 1.5x` trace `fy` repair in `ReplayCameraSource.resolveReplayVerticalFocalLengthPx`.
   - **Unified 3-Tier OpenCV 4.13 Engine (`OpenCvConstellationTracker.kt` + `LockOnPinpointController.kt`):** **Tier 1** (`30 FPS`, Stage 2 & 3 from Frame 1): `640x360` IMU-predicted `PyrLK` + FB `< 1.5 px` + 4-DOF Similarity RANSAC + `12°` roll + `32x32` zero-mean NCC; **Tier 2** (`30 FPS`): Rotation-Known Sparse 3D Mini-VIO (~90 ENU landmarks, 3-iter Huber IRLS, zero `z=0` plane assumption); **Tier 3** (`~5 Hz` audit / `10 Hz` re-entry): Multi-Keyframe `ORB` Bank (`6` collar `192x192` + `4` reloc `480x270`, KF0 preserved) behind the **4-Gate Verifier** (spread `>= 6%`, Jacobian det/SVD `<= 2.4`/persp, `25°` IMU roll, `32x32` NCC `>= 0.40` rejecting blank walls `std < 3.5`). Cut 896-frame `20261119_011006` error `927.8 -> 8.8 px` (`90.4%` `< 30 px`, `0/272` blank-wall false locks; ARM64 `5.34–7.24 ms` median, `0/30` wall false locks, `0.66 px` re-acquire).
   - **Audio Silence & Honest LCARS UX:** Sealed `MOD: LOG` in-flight sweep/sonifier race via `isLoggingDataActive: AtomicBoolean` (`frames=112, avg_fps=27.84, discarded_sweeps=0, audio_pings=0`); uncropped `16:9` Mars Red `MOD: LOG` viewfinder + `REC-30P` footer; honest `PINPOINT` vs dashed `48%` GoldenAmber `COAST` reticle (`isVisuallyLocked`), `~EST` range (`hasMetric3dDepth`), and steady `30 HZ` cadence. On-device `20261119_011006`: `frames=896, locks=1, handovers=1, lost_locks=0, distractor_jumps=0, distractors_rejected=38, exceptions=0` (`50/50` JVM, `8/8` instrumented, `7/7` `pytest`).

---

## 5. Current Architecture & Interfaces

- **`PushBroomCropper` (`vision/`):** Extracts `FIVE_TILE` (`1792x384`) or `THREE_TILE` (`1088x384`) `[0..1]` RGB tiles (`IoU >= 0.45 / IoM >= 0.65` seam-union merge), upright `1920x1080` `Y` luma + adaptive `[320..640] -> 320x320` Lock-On crops (`< 3 ms`), and `NV12`/`I420` encoder planes with safe UV stride clamping.
- **`GroundPlaneRaycaster`, `TemporalTargetTracker` & `PatrolDutyCycleController` (`tracking/`):** 128-pose `120 Hz` IMU ring buffer (live sensor or `ReplayCameraSource` injection), `WorldRay3D` association + Track NMS + `1.10 m/s` speed gate, and `5.0 s` `15 FPS` / `30 Hz` `"EYES-UP"` idle controller.
- **`LockOnPinpointController` & `OpenCvConstellationTracker` (`tracking/`):** `PATROL -> LOCK_ON_NEURAL -> PINPOINT_CONSTELLATION` funnel powered by the Unified 3-Tier Spatial Anchor (`640x360` Tier 1 IMU-`PyrLK` Similarity + Tier 2 Sparse 3D Mini-VIO + Tier 3 4-Gate Multi-Keyframe `ORB` collar `192x192` & `480x270` reloc), 3D `<= 4.5°` / `[0.45x..2.20x]` innovation gate, `300 ms` coasting hysteresis, and honest `isVisuallyLocked` / `hasMetric3dDepth` HUD state.
- **`StoneDetector` (`inference/`):** `RfDetrTwoGraphGpuDetector` (`S-01..S-05` `WarmPatrolSlot` LRU + disk shader cache + resident `B=1` `320` `W8-Dyn` runner) and `SyntheticStoneDetector` (`MIXED_GRAVEL_WALK`).
- **`FieldDataLogger` (`logging/`) & `ReplayCameraSource` (`camera/`):** `MOD: LOG` records `1080p30` H.264 `clip.mp4` + `120 Hz` IMU `trace.jsonl`; `ReplayCameraSource` replays `Android/media/` & `Android/data/` sessions via `SRC` (`REPLAY: <session>`) or `adb shell am start --es debug_replay_session <name> --ei debug_auto_lock_frame <N> --ez debug_loop_replay false`.
