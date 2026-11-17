# GneissFind — Engineering & Validation Ledger (`docs/ENGINEERING_LEDGER.md`)

> [!IMPORTANT]
> **Purpose of this file:** This ledger is the canonical, append-only record of implementation decisions, hardware validation findings, UI/UX iterations, and low-level Android/Camera2/Compose postmortems across milestones.
> - **Forward-looking specs and upcoming milestones** live in [`docs/ROADMAP.md`](ROADMAP.md) and [`AGENTS.md`](../AGENTS.md).
> - **Completed work and hardware-discovered invariants** live here so that roadmap restructuring never overwrites hard-won hardware findings or causes spec drift between local and cloud agent sessions.

---

## 1. Milestone Status Summary

| Milestone | Status | Target Hardware Validated | Key Deliverable Summary |
| :--- | :--- | :--- | :--- |
| **Milestone 1** | **Completed** (`3f1dee0`) | Google Pixel 10 Pro XL (Android 16) | Baseline push-broom skeleton, `Camera2Interop` `<= 1/500 s` shutter clamp + floating ISO, synthetic detector, 3-of-6 temporal voter, ARCore/IMU speed gate, and `AudioTrack` sonifier. |
| **Milestone 1B** | **Completed & Hardware-Validated** (`0e3c81e` .. `fc80785`) | Google Pixel 10 Pro XL (Android 16) + API 37 Emulator | Reversible `sensorLandscape`, default `2.0x` lossless sensor crop + `<= 1/500 s` clamp, `5 x 384x384` (`1792x384`) overlapping tile cropper with cross-tile NMS, 120 Hz GPU RenderNode IMU pitch stabilization, Classic 2360s TNG LCARS split-screen UI with 6-role semantic color taxonomy, inline source drawer, and dynamic `Camera2CameraControl` exposure/EV servo. |
| **Milestone 2** | **Completed, Benchmarked & Field-Tuned** (`96147e5` .. `0cad9b2` + `annotate-code`) | Google Pixel 10 Pro XL (`58110DLCQ00003`, Tensor G5, Android 17 `CP3A.261005.005`) | Reproducible `uv` + `litert-torch` `RF-DETR-Nano` (`384x384` MS-COCO) exporter with rank-4 `LITERT_CL` operator rewrites (`909/909` `Part 1` and `350/350` `Part 2` single-partition OpenCL GPU delegation), verified COCO stand-in oracle (`cup=47`, `apple=53`, `toothbrush=90`, `remote=75`, `IoU=1.0000`), cross-tile `IoU + IoM + seam` union box merging, on-device 4-way (`1-Graph vs 2-Graph` $\times$ `B=5 vs 5xB=1`) benchmark harness (`scripts/benchmark_pixel10.sh`), 3D IMU world-ray track anchoring + 120 Hz ego-motion reticle reprojection (post-screengrab field fix), developer onboarding docs (`README.md`, `docs/INTERNALS.md`, `LICENSE`), and full `docs/COMMENTING.md` annotation across all 19 `.kt` sources. |

---

## 2. Commit Ledger (Milestone 1 → Milestone 2)

| Commit | Summary | Primary Files Touched |
| :--- | :--- | :--- |
| `3f1dee0` | Initialize GneissFind workspace specs and Milestone 1 baseline | `AGENTS.md`, `docs/ROADMAP.md`, `.agents/rules/*`, `android/app/src/main/java/com/gneissfind/app/**` |
| `0e3c81e` | Add 5-tile landscape push-broom cropper and reversible IMU pitch stabilization | `model/StoneModels.kt`, `vision/PushBroomCropper.kt`, `tracking/GroundPlaneRaycaster.kt`, `camera/PushBroomCameraController.kt`, `inference/SyntheticStoneDetector.kt`, `inference/RfDetrTwoGraphGpuDetector.kt`, `logging/FieldDataLogger.kt`, `ui/ScannerViewModel.kt`, `PushBroomCropperTest.kt`, `SonifierAndHostBridgeTest.kt` |
| `11009b7` | Implement split-screen 2360s TNG LCARS interface with offline Antonio typography | `MainActivity.kt`, `ui/ScannerScreen.kt`, `ui/lcars/LcarsComponents.kt`, `ui/theme/Theme.kt`, `res/font/antonio_bold.ttf`, `res/font/antonio_regular.ttf` |
| `d724234` | Decouple 120 Hz GPU EIS preview translation from analyzer frame rate | `tracking/GroundPlaneRaycaster.kt`, `camera/PushBroomCameraController.kt`, `ui/ScannerScreen.kt`, `ui/ScannerViewModel.kt` |
| `8707282` | Refine LCARS semantic color taxonomy, source drawer, left pitch gauge, and single-action affordances | `MainActivity.kt`, `ui/ScannerScreen.kt`, `ui/ScannerViewModel.kt`, `ui/lcars/LcarsComponents.kt`, `ui/theme/Theme.kt`, `LcarsGeometryAndViewportTest.kt` |
| `fc80785` | Fix `PreviewView` `TextureView` layout starvation and Camera2 dynamic exposure overrides | `camera/PushBroomCameraController.kt`, `ui/ScannerScreen.kt`, `ui/ScannerViewModel.kt` |
| `6d796fc` | Record Milestone 1 to 1B engineering and hardware validation ledger | `docs/ENGINEERING_LEDGER.md`, `docs/ROADMAP.md` |
| `38d167a` | Install dexiotites commenting standard and annotate-code skill | `docs/COMMENTING.md`, `CLAUDE.md`, `.agents/skills/annotate-code/*` |
| `96147e5` | Update workspace specifications and rules for Milestone 2 RF-DETR-Nano COCO vertical slice | `AGENTS.md`, `docs/ROADMAP.md`, `.agents/rules/*`, `scripts/pack_context.sh` |
| `4825ef5` | Add RF-DETR-Nano LiteRT GPU export pipeline, COCO stand-in oracle, and parity test suite | `ml-pipeline/**`, `android/app/src/main/assets/rf_detr_coco_metadata.json` |
| `965d027` | Wire LiteRT GPU FP16 RF-DETR-Nano detector, COCO stand-in routing, and cross-tile IoM seam union merging | `model/StoneModels.kt`, `vision/PushBroomCropper.kt`, `inference/RfDetrTwoGraphGpuDetector.kt`, `ui/ScannerViewModel.kt`, `ui/ScannerScreen.kt`, `PushBroomCropperTest.kt`, `SonifierAndHostBridgeTest.kt` |
| `1e813e2` | Eliminate `LITERT_CL` CPU fallback ops in RF-DETR-Nano and add Pixel 10 Pro XL benchmark harness | `ml-pipeline/src/gneissfind_ml/wrappers.py`, `ml-pipeline/tests/test_coco_export_parity.py`, `inference/RfDetrTwoGraphGpuDetector.kt`, `scripts/benchmark_pixel10.sh` |
| `cc2496f` | Record Milestone 2 LiteRT GPU operator findings and Pixel 10 Pro XL benchmark table in ledger | `docs/ENGINEERING_LEDGER.md`, `docs/ROADMAP.md` |
| `2d28979` | Add root `README.md`, MIT `LICENSE`, and `docs/INTERNALS.md` developer onboarding guide | `README.md`, `LICENSE`, `docs/INTERNALS.md` |
| `0cad9b2` | Add 3D IMU world-ray anchoring and 120 Hz ego-motion reticle reprojection | `model/StoneModels.kt`, `tracking/GroundPlaneRaycaster.kt`, `tracking/TemporalTargetTracker.kt`, `ui/ScannerViewModel.kt`, `ui/ScannerScreen.kt`, `TemporalTargetTrackerTest.kt`, `docs/ENGINEERING_LEDGER.md`, `docs/INTERNALS.md` |
| *(Working tree)* | Annotate all 19 Kotlin source and test files per `docs/COMMENTING.md` (`comment_diff_check.py --strict` verified) | `android/app/src/main/java/com/gneissfind/app/**`, `android/app/src/test/java/com/gneissfind/app/**` |

---

## 3. Chronological Findings & Architectural Evolution (M1 → M2)

### 3.1. Milestone 1 On-Device Critique (Portrait Living-Room Run on Pixel 10 Pro XL)
Running the initial Milestone 1 portrait UI on a physical Pixel 10 Pro XL surfaced four critical field-usability defects:
1. **Top-Third HUD Clutter & Overlap:** Floating Material 3 cards (`TopControlBar`, `TelemetryBanner`) and the push-broom reticle all competed for the top third of the portrait screen, colliding with each other and obscuring the live camera feed.
2. **Dark-on-Dark Contrast Failure:** Unspecified Compose `Text` colors inside translucent dark cards defaulted to dark `onSurface` tokens, rendering critical telemetry (`Shutter Clamp`, `2.0x Crop`, `Canopy Exposure Bias`) nearly illegible over dark indoor/canopy backgrounds.
3. **Modal Sheet Occlusion:** Opening the optics setup sheet covered the live push-broom viewport, preventing the operator from seeing the effect of exposure/zoom changes while adjusting them.
4. **Hardware Volume Rocker Routing (`USAGE_NOTIFICATION_EVENT` vs. `USAGE_MEDIA`):** The `AudioTrack` sonifier originally used `AudioAttributes.USAGE_NOTIFICATION_EVENT`, binding stone ping volume to the phone's ring/notification stream instead of media volume.
   - **Resolution:** Switched `MetalDetectorSonifier` to `AudioAttributes.USAGE_MEDIA` (`CONTENT_TYPE_SONIFICATION`) and set `volumeControlStream = AudioManager.STREAM_MUSIC` in `MainActivity.onCreate` so the physical volume rocker always controls stone ping volume.

### 3.2. Milestone 1B Core Optical, Sensor & Push-Broom Refactor (`0e3c81e`, `11009b7`)
To maximize cross-track ground coverage and eliminate UI occlusion, Milestone 1B re-architected both the sensor pipeline and the UI:
1. **Reversible `sensorLandscape` (`ROTATION_90` & `ROTATION_270`):**
   - Locked `MainActivity` to `android:screenOrientation="sensorLandscape"`.
   - Wired both `DisplayManager.DisplayListener` and `OrientationEventListener` in `MainActivity` so direct $180^\circ$ landscape flips (which Android suppresses from `onConfigurationChanged` because window dimensions do not change) immediately update `PushBroomCameraController.updateTargetRotation(rotation)`, `GroundPlaneRaycaster.updateDisplayRotation(rotation)`, and ARCore `Session.setDisplayGeometry(...)`.
   - Remapped `Sensor.TYPE_GAME_ROTATION_VECTOR` via `SensorManager.remapCoordinateSystem(rawMatrix, AXIS_Y, AXIS_MINUS_X, ...)` for `ROTATION_90` and `(AXIS_MINUS_Y, AXIS_X)` for `ROTATION_270` so forward downward pitch increases positively ($0^\circ$ horizontal $\rightarrow 45^\circ$ nominal scan $\rightarrow 90^\circ$ nadir) in both landscape orientations without pitch/roll cross-coupling.
2. **5-Tile Cross-Track Swath (`1792 x 384 px`) with `32 px` Overlap & Cross-Tile NMS:**
   - Upgraded `PushBroomCropper` on the `1920 x 1080` (`YUV_420_888`) landscape buffer to extract **5 horizontally overlapping `384 x 384` tiles** (`xOffsets = [64, 416, 768, 1120, 1472]`, `stride = 352 px`, spanning `1792 x 384 px` = `93.3%` of sensor width) alongside an inline toggleable **3-TILE fast mode** (`1088 x 384 px`, `xOffsets = [416, 768, 1120]`).
   - Added cross-tile Non-Maximum Suppression (`CROSS_TILE_NMS_IOU_THRESHOLD = 0.45f`) in unified strip coordinates so stones straddling any of the four `32 px` vertical tile seams are deduplicated before 3D ground-plane raycasting and 3-of-6 temporal voting.
3. **Default `2.0x` Lossless In-Sensor Crop & `<= 1/500 s` Shutter Clamp:**
   - Defaulted both `twoXCropEnabled = true` (`ZoomRatio = 2.0f`) and `shutterClampEnabled = true` (`SENSOR_EXPOSURE_TIME <= 2,000,000 ns`) at cold start.
   - At nominal `45°` pitch from waist height (`h = 1.06 m`, slant range `R ~ 1.50 m`), `2.0x` crop yields **`0.59 mm/px` GSD** (`~34 px` across a minimum `20 mm` target pebble) across a **`~1.05 m` wide $\times$ `~0.25 m` deep** ground swath.
4. **IMU-Driven ROI Pitch Stabilization (Software EIS):**
   - `GroundPlaneRaycaster.computeStabilizedCropY` shifts `cropY` within `[0..696]` around `NOMINAL_CROP_Y_PX = 348` using `deltaY = fy * tan(pitch - refPitch)` (`~ +/-6.7°` pitch authority at `2.0x` zoom, `fy ~ 2985 px`).
   - Clamping at `cropY == 0` or `cropY == 696` illuminates `PITCH LIMIT` in the LCARS header and control deck.
5. **Split-Screen Classic 2360s TNG LCARS Interface:**
   - Replaced all floating cards and modal sheets with a permanent split-screen layout on pure `#000000` OLED canvas (`4.dp` black gutters, custom swept `LcarsElbowShape` with concave inner fillets, bundled offline `Antonio` uppercase typography with tabular numerals `tnum`, and `isSoundEffectsEnabled = false` on all UI controls).
   - **Top ~48% (`PanoramicViewportZone`):** Cropped panoramic camera viewport zoomed so the active `1792 x 384` (or `1088 x 384`) push-broom strip fills the viewport width, with high-contrast black-halo target brackets and tile seam ticks.
   - **Bottom ~52% (`TacticalControlDeckZone`):** Persistent 2-column control deck (`OPT-2X` optics/EIS + `OPT-EV` exposure bias on the left; `PROSPECT MATRIX` 4 mineral toggles + detector source selector + speed gate on the right).

### 3.3. Pixel 10 Pro XL Field Validation #1 — 120 Hz EIS & LCARS Semantic Taxonomy (`d724234`, `8707282`)
Testing the initial Milestone 1B build on the physical Pixel 10 Pro XL identified several UX and motion-smoothness improvements:
1. **120 Hz GPU RenderNode EIS Decoupling (`d724234`):**
   - *Finding:* Updating the camera viewport's vertical crop offset only when `ImageAnalysis` completed a sweep (~10–15 Hz) and calling `View.requestLayout()` on `PreviewView` caused visible stepping during wrist pitch motion.
   - *Solution:* Added a continuous `SENSOR_DELAY_GAME` callback (`GroundPlaneRaycaster.onRealtimeImuUpdate`) that computes unrounded `displayCropYFloat` (`computeStabilizedCropYFloat`) and drives `previewView.translationY = t.frameTopPx` directly on the GPU RenderNode inside `ClippingCameraViewportContainer` at 120 Hz—requiring zero View `requestLayout()` passes during pitch stabilization.
2. **6-Role LCARS Semantic Color Taxonomy & Parameterized Value Stepping (`8707282`):**
   - *Finding:* Buttons and passive telemetry labels shared overlapping palette assignments, making it non-obvious which pills were interactive toggles vs. readouts.
   - *Solution:* Established `LcarsSemanticColors` (`Theme.kt`) with 6 distinct semantic roles and a parameterized HSV value-stepping function (`LcarsColorTuning.valueStepDelta = 0.05f`, providing `MINUS` $-5\%$, `BASE` $0\%$, and `PLUS` $+5\%$ brightness variants so adjacent pills of the same category remain visually distinct):
     - **On/Off Toggles (`#FF9900` Golden Amber):** `OPT` (`2.0X CROP`), `EXP` (`1/500S`), `EIS` (`EIS LOCK`), `SIM` (`SPEED SIM`), and the 4 `PROSPECT MATRIX` mineral toggles (`#596480` Slate Blue when inactive).
     - **Multi-State Mode Toggles (`#9999FF` Periwinkle):** `MOD` (`SCAN / LOG`), `SND` (`AUDIO / MUTE`), `SW` (`5-TILE / 3-TILE`).
     - **Action / Drawer Triggers (`#FFCC66` TNG Gold):** `CAL` (`SYNC PITCH`), `SRC` (`OPEN SOURCE DRAWER`), and `−` / `+` EV step buttons.
     - **Realtime Telemetry Readouts (`#99CCFF` Ice Blue):** `PITCH`, `VEL`, `FPS`, `Y-PX` readout, and `SWEEP` latency bar.
     - **Static / Config State Readouts (`#FFCC99` Peach):** `RANGE • RESOLUTION` bar, `OPTICS • SHUTTER • ISO` bar, `REC` saved-frame counter, and `47-1792` swath code.
     - **Critical / Limit Alerts (`#CC4444` Mars Red):** `PITCH LIMIT`, `SLOW SWEEP`, and active `LOG` recording indicator.
3. **Affordance & Layout Polish (`8707282`):**
   - Enlarged the mineral class indicator dot on `PROSPECT MATRIX` buttons to `16.dp` with a dual black/white contrast ring matching the on-screen reticle color (`Cyan #00E5FF`, `Gold #FFD740`, `Green #00E676`, `Orange #FF9100`).
   - Replaced the tap-to-cycle `SRC` button with an inline scrollable LCARS selection drawer (`LcarsSourceDrawerOverlay`) that expands over the control deck on first tap and selects the source on second tap.
   - Moved the vertical pitch-travel gauge (`LcarsPitchTravelGauge`) to the left side of the camera viewport (immediately inside the left LCARS elbow) so it sits next to the `FPS` and `SND` stem pills.
   - Consolidated redundant controls so there is strictly **one button per user action**: merged `M-01 SCAN` and `M-02 LOG` into a single `MOD` (`SCAN` / `LOG`) toggle above a passive `REC` frame counter; removed tap handlers from the passive `OPT-EV` bar segments so EV is adjusted exclusively via the `−` and `+` endcap buttons; removed duplicate speed-test tap handlers from the top `VEL` readout.
   - Added monotonic high-water-mark width measurement (`rememberHighWaterSegmentWidth`) on dynamic header/footer pills so changing digit counts never shifts neighboring LCARS bar segments horizontally.

### 3.4. Pixel 10 Pro XL Field Validation #2 — CameraX / Camera2 Hardware Postmortem (`fc80785`)
A follow-up screen recording on the physical Pixel 10 Pro XL after the 120 Hz EIS and UI refactor revealed four coupled runtime regressions (black camera viewport, idle simulation, silent sonifier, and unresponsive exposure toggle). Root-cause analysis identified four hardware/framework invariants that must be preserved in all future camera/UI work:

1. **Invariant 1 — `PreviewView` Asynchronous `TextureView` Child Layout:**
   - *Symptom:* Camera viewport was completely black, `FPS` read `0.0HZ`, synthetic simulation produced zero targets, and the sonifier remained silent.
   - *Root Cause:* `ClippingCameraViewportContainer.onLayout` cached `lastMeasuredViewportW` / `lastMeasuredViewportH` during the initial Compose layout pass (before `ProcessCameraProvider.getInstance()` asynchronously bound `Preview` and `ImageAnalysis`) and skipped calling `previewView.measure(...)` and `previewView.layout(...)` on subsequent `onLayout` passes when the outer container size had not changed. When CameraX requested a surface ~50 ms later, `PreviewView` (`ImplementationMode.COMPATIBLE`) dynamically added its internal `TextureView` child and called `requestLayout()`. Because the container skipped measuring/laying out `previewView`, the inner `TextureView` stayed at `0 x 0` pixels, `onSurfaceTextureAvailable` never fired, `SurfaceRequest.provideSurface` never completed, and CameraX's combined `Preview` + `ImageAnalysis` `CameraCaptureSession` stalled indefinitely waiting for the preview surface.
   - *Fix (`ScannerScreen.kt`):* Implemented standard `onMeasure` and unconditional `onLayout` child sizing in `ClippingCameraViewportContainer` (plus `requestLayout()` when `SwathTileMode` changes), while keeping 120 Hz EIS updates strictly inside `updateStabilizationState` via `previewView.translationY = t.frameTopPx`.
2. **Invariant 2 — `Camera2Interop.Extender` `ALWAYS_OVERRIDE` Priority vs. `Camera2CameraControl`:**
   - *Symptom:* Toggling `EXP: 1/500S [ON/OFF]` or stepping Canopy `EV` bias did not change the live shutter speed or ISO.
   - *Root Cause:* Setting `CONTROL_AE_MODE_OFF`, `SENSOR_EXPOSURE_TIME`, and `SENSOR_SENSITIVITY` on `Camera2Interop.Extender(analysisBuilder)` attaches those options to the `ImageAnalysis` use case with CameraX `OptionPriority.ALWAYS_OVERRIDE`. In CameraX 1.4, `ALWAYS_OVERRIDE` options on any bound use case permanently win over dynamic options submitted via `Camera2CameraControl.from(camera.cameraControl).setCaptureRequestOptions(...)` (`OptionPriority.OPTIONAL`). Additionally, switching `shutterClampEnabled` to `false` requires calling `c2Control.clearCaptureRequestOptions()` so Camera2 releases `SENSOR_EXPOSURE_TIME` and `SENSOR_SENSITIVITY` back to native `CONTROL_AE_MODE_ON`.
   - *Fix (`PushBroomCameraController.kt`):* Removed all static AE capture request options from `Camera2Interop.Extender` (retaining only `setSessionCaptureCallback` for `TotalCaptureResult` telemetry) and drove manual `<= 1/500 s` shutter + floating ISO exclusively via `Camera2CameraControl.setCaptureRequestOptions(...)` / `clearCaptureRequestOptions()`.
3. **Invariant 3 — CameraX `zoomState` Cold-Start Race on Pixel 10 Pro XL:**
   - *Symptom:* Calling `camera.cameraInfo.zoomState.value?.maxZoomRatio` immediately after `bindToLifecycle` returns `1.0f` before the Camera2 session opens, clamping the initial `2.0x` sensor crop down to `1.0x`.
   - *Fix (`PushBroomCameraController.kt`):* Synchronously query `CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM` via `Camera2CameraInfo` right after `bindToLifecycle` and use it as the fallback upper bound until `zoomState` populates.
4. **Invariant 4 — EV-Biased Luma Target in Closed-Loop Auto-ISO Servo:**
   - *Symptom:* Under `CONTROL_AE_MODE_OFF` (`<= 1/500 s` clamp), stepping Canopy `EV` multiplied `targetIso` by $2^{\text{EV}}$, but the 350 ms Y-plane luma servo (`maybeServoFloatingIsoFromCenterLuma`) compared against a fixed `targetLuma = 118f` and immediately servoed `clampedFloatingIso` back down to cancel the user's EV offset.
   - *Fix (`PushBroomCameraController.kt`):* Scaled `targetLuma` by $2^{0.5 \cdot \text{exposureBiasEv}}$ (clamped to `[42, 210]`) so the closed-loop ISO servo maintains the user-selected Canopy EV bias.

---

### 3.5. Milestone 2 — `RF-DETR-Nano` LiteRT GPU Export, `LITERT_CL` Operator Engineering & Pixel 10 Pro XL Benchmark (`4825ef5`, `965d027`, `HEAD`)

#### 3.5.1. Empirical MS-COCO Stand-In Class-Index Oracle (`ml-pipeline/`)
`RFDETRNano` (`rfdetr==1.12.1`, `resolution=384`, `num_queries=300`) emits **91 sparse MS-COCO category logits** (`shape [B, 300, 91]`), not contiguous `0..79` indices. Running the empirical verification suite (`uv run python scripts/verify_coco_export.py` and `uv run pytest -v`, `6/6` tests passing) across real `384x384` fixtures verified the exact 0-based logit indices and PyTorch-vs-LiteRT parity:

| GneissFind `StoneClass` | LCARS UI Button (Unchanged) | Household Stand-In | Verified 91-Logit `class_id` | PyTorch Score | LiteRT `.tflite` Score | Top-Box IoU | Max Score $\|\Delta\|$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `0: CHALCEDONY_AGATE` | `01 AGATE / CARNELIAN` | **`cup`** | **`47`** | `0.963` | `0.963` | **`1.0000`** | `0.000002` |
| `1: JASPER_RED_YELLOW` | `02 RED / YEL JASPER` | **`apple`** | **`53`** | `0.959` | `0.959` | **`1.0000`** | `0.008182` |
| `2: GREEN_GEM_ROCK` | `03 DALLASITE / JADE` | **`toothbrush`** | **`90`** | `0.869` | `0.869` | **`1.0000`** | `0.000004` |
| `3: PETRIFIED_WOOD` | `04 PETRIFIED WOOD` | **`remote`** | **`75`** | `0.956` | `0.956` | **`1.0000`** | `0.000005` |
| `4: HARD_CONFUSER` | *(Muted — no bracket / audio)* | All other classes (`!= 47, 53, 90, 75`) | `0..90` | — | — | — | — |

#### 3.5.2. Achieving 100% Single-Partition `LITERT_CL` OpenCL GPU Delegation (`Part 1: 909/909`, `Part 2: 350/350`)
Deploying the initial `litert_torch` exports onto the physical Google Pixel 10 Pro XL (`serial=58110DLCQ00003`, `soc=Tensor G5`, `board=laguna`, PowerVR DXT-48-1536 OpenCL GPU via `libLiteRtClGlAccelerator.so`) exposed five critical `LITERT_CL` hardware/compiler constraints that caused standard `RF-DETR-Nano` exports to either fail compilation or fragment into slow CPU fallback partitions:

1. **Constraint 1 — `TOPK_V2` & `GATHER_ND` Not Supported by `LITERT_CL`:**
   - In the fused 1-Graph export (`rf_detr_nano_1g_b5.tflite`), two-stage query proposal selection (`torch.topk` over `576` tokens $\rightarrow$ `TOPK_V2` + `GATHER_ND`) is rejected by `LITERT_CL`. Even worse, LiteRT's graph partitioner refuses to re-enter the GPU after the mid-graph `TOPK_V2` barrier, delegating only `900 / 1304` nodes (the DINOv2 backbone) to `LITERT_CL` and dropping all remaining `404` decoder nodes (`8` XNNPACK partitions + `30` un-delegated CPU ops) onto the CPU.
   - *Fix (`RfDetrPart1BackboneEncoder` + `host_select_topk_queries` + `RfDetrPart2DecoderHead`):* Splitting the graph immediately after encoder proposal scoring moves `TopK` (`K=300` of `576` tokens) and static `24x24` proposal anchor gathering to a zero-allocation Kotlin min-heap bridge (`selectTopKProposalsFromMaxScores`, taking **`1.5–3.6 ms`** for all 5 tiles) so neither subgraph contains `TOPK_V2` or `GATHER_ND`.
2. **Constraint 2 — `F.grid_sample` Lowers to `GATHER_ND` + Boolean Masks; `pixel_row_grid_sample` Overflow in FP16:**
   - PyTorch's `F.grid_sample` inside `MSDeformAttn` lowers to `GATHER_ND` + `LOGICAL_AND` + `SELECT_V2` (unsupported on `LITERT_CL`), while `rfdetr`'s `pixel_row_grid_sample` flattens `(B * n_heads, H+2, W+2)` into 1D float indices that exceed `float16`'s exact integer limit (`2048`) once `N * 676 > 2048`.
   - *Fix (`bilinear_matmul_grid_sample` in `wrappers.py`):* Computes separable 1D bilinear tent weights `relu(1 - |p - i|)` over `0..23` (exact in FP16) and contracts via `torch.bmm` (`BATCH_MATMUL`), matching `F.grid_sample` within `2e-5` without `GATHER_ND`.
3. **Constraint 3 — `LITERT_CL` Rejects Any Tensor with Rank $\ge 5$ (`Tensor dimensions must be less than 5`):**
   - `WindowedDinov2WithRegistersEmbeddings`, `WindowedDinov2WithRegistersBackbone`, and `MSDeformAttn` reshape tensors into 5D/6D views (`(B, num_windows, num_h_win, num_windows, num_w_win, C)` and `(B, Q, n_heads, n_levels, n_points, 2)`), causing `LITERT_CL` to reject `SLICE`, `TRANSPOSE`, `ADD`, `MUL`, and `DIV` nodes and fragment `Part 1` into `27` partitions!
   - *Fix (`_gpu_rank4_embeddings_forward`, `_gpu_rank4_backbone_forward`, `_gpu_rank4_ms_deform_attn_forward` in `wrappers.py`):* Fused adjacent non-transposed axes (`num_w_win * C`, `B * Q`, `n_points * 2`) during every window/un-window and attention transpose so every tensor in both `Part 1` and `Part 2` has rank $\le 4$.
4. **Constraint 4 — `BROADCAST_TO` Not Supported by `LITERT_CL`:**
   - `.expand(...)` and `.repeat(...)` on `cls_token`, `register_tokens`, `query_feat`, `refpoint_embed`, and `pos_embed` lower to `BROADCAST_TO`, which `LITERT_CL` rejects.
   - *Fix (`_repeat_along_dim0` in `wrappers.py`):* Replaced batch/window expansions with `torch.cat([tensor] * count, dim=0)` (`CONCATENATION`).
5. **Constraint 5 — `FLOOR_DIV` (`gen_sineembed_for_position`) and `RELU_0_TO_1` (`.clamp(0.0, 1.0)`) in `Part 2`:**
   - Tracing `dim_t // 2` on `torch.arange(128)` inside `rfdetr.models.transformer.gen_sineembed_for_position` emitted `FLOOR_DIV`, and `boxes.clamp(0.0, 1.0)` emitted `RELU_0_TO_1`, splitting `Part 2` into `3` partitions (`306/360` nodes on GPU, `54` on CPU).
   - *Fix (`_gpu_friendly_sineembed_for_position` and `_clamp_unit_interval_via_relu` in `wrappers.py`):* Precomputed the 64-element float frequency denominator and replaced `.clamp(0.0, 1.0)` with `F.relu(x) - F.relu(x - 1.0)` (`RELU` + `SUB`), achieving **100% single-partition `LITERT_CL` delegation on both `Part 1` (`909/909` nodes, `1` partition) and `Part 2` (`350/350` nodes, `1` partition)**.

#### 3.5.3. Physical Pixel 10 Pro XL (`58110DLCQ00003`) 4-Way LiteRT GPU (FP16) Benchmark Matrix
Measured on physical **Google Pixel 10 Pro XL** (`serial=58110DLCQ00003`, `soc=Tensor G5`, `board=laguna`, build `google/mustang/mustang:17/CP3A.261005.005/16271449:user/release-keys`, `libLiteRtClGlAccelerator.so` OpenCL FP16 delegate) via `./scripts/benchmark_pixel10.sh --skip-push --warmup 2 --sweeps 12 --cooldown 8` (isolated process per configuration; full JSON report at `ml-pipeline/artifacts/pixel10_benchmark_report.json`):

| Configuration | `LITERT_CL` GPU / CPU Partitions | Stage 1 `p1` Mean (ms) | Host `TopK` Mean (ms) | Stage 2 `p2` Mean (ms) | Sweep Min (ms) | Sweep `p50` (ms) | Sweep `p95` (ms) | Sustained Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2-Graph B=5` Batched (`5-Tile`)** *(Selected Default)* | **`909/909 (1p)` + `350/350 (1p)`** *(100% GPU)* | **294.25** | **3.63** | **78.18** | **363.01** | **376.86** | **388.40** | **2.65 Hz** (`75.37 ms/tile`) |
| **`2-Graph 5xB=1` Sequential (`5-Tile`)** | `905/905 (1p)` + `324/324 (1p)` *(100% GPU)* | 417.14 | 3.31 | 167.78 | 544.17 | **595.97** | 626.45 | **1.68 Hz** (`119.19 ms/tile`) |
| **`1-Graph B=5` Batched (`5-Tile`)** | `900/1304 (2p)` + `XNN:375/405 (8p)` *(Hybrid)* | 544.39 | 0.00 | 0.00 | 500.40 | **538.12** | 610.94 | **1.86 Hz** (`107.62 ms/tile`) |
| **`1-Graph 5xB=1` Sequential (`5-Tile`)** | `900/1268 (2p)` + `XNN:343/369 (6p)` *(Hybrid)* | 822.61 | 0.00 | 0.00 | 771.50 | **799.32** | 895.18 | **1.25 Hz** (`159.86 ms/tile`) |
| **`2-Graph 3xB=1` Sequential (`3-Tile` Narrow)** | `905/905 (1p)` + `324/324 (1p)` *(100% GPU)* | 250.14 | 3.11 | 102.74 | 332.70 | **355.68** | 380.49 | **2.81 Hz** (`118.56 ms/tile`) |
| **`2-Graph B=5` CPU XNNPACK Fallback (`4 threads`)** | `XNN:909/909 (1p)` + `XNN:350/350 (1p)` *(CPU)* | 1320.09 | 2.04 | 310.35 | 1618.40 | **1632.75** | 2041.58 | **0.61 Hz** (`326.55 ms/tile`) |

#### 3.5.4. Roofline & Architectural Conclusions from the Pixel 10 Pro XL Benchmark
1. **Why `2-Graph B=5` Wins Decisively (`376.86 ms` `p50`, `363.01 ms` `min`):**
   - **2-Graph vs. 1-Graph (`1.43x` speedup at `B=5`):** Moving two-stage `TopK` (`3.63 ms` for 5 tiles) to the host CPU eliminates the `TOPK_V2`/`GATHER_ND` partition barrier that forced all `404` decoder nodes in `1-Graph B=5` onto `TfLiteXNNPackDelegate` (`8` CPU partitions), cutting `p50` sweep time by **`161.26 ms`** (`538.12 ms -> 376.86 ms`).
   - **Batched `B=5` vs. Sequential `5 x B=1` (`1.58x` speedup in `2-Graph`):** Dispatching a single `B=5` tensor amortizes weight texture fetches and eliminates 8 redundant OpenCL command-queue flushes and JNI synchronizations per sweep, cutting `p50` sweep time by **`219.11 ms`** (`595.97 ms -> 376.86 ms`).
   - **OpenCL `Image2D` Texture Storage vs. SSBO Buffers (`2.25x` speedup on PowerVR DXT-48-1536):** Forcing `BufferStorageType.BUFFER` on `LITERT_CL` degraded `2-Graph B=5` from `376.86 ms` to `848.90 ms` (`p1_ms = 711.97 ms`) because PowerVR's hardware texture cache (`Image2D`) provides >2x higher effective bandwidth for ViT linear/conv layers than global SSBO buffer loads. Meanwhile, keeping all intermediate tensor dimensions $\le 1200$ (`< CL_DEVICE_IMAGE2D_MAX_WIDTH = 16384`) prevented `clCreateImage: Invalid image size` failures.
2. **Hardware Roofline vs. the `125 ms/sweep` Target:**
   - A single `384x384` tile of `RF-DETR-Nano` (`12`-layer DINOv2-Small backbone + `P4` C2f projector + `2`-layer `300`-query deformable decoder) requires **`29.703 GFLOPs` (`14.851 GMACs`)**: `27.509 GFLOPs` (`92.6%`) in `Part 1` and `2.194 GFLOPs` (`7.4%`) in `Part 2`.
   - Across `5` tiles (`B=5`), each sweep executes **`148.513 GFLOPs` (`74.257 GMACs`)**. At **`376.86 ms` (`p50`) / `363.01 ms` (`min`)**, the Pixel 10 Pro XL GPU sustains **`394–409 GFLOP/s` effective end-to-end** (`467 GFLOP/s` on `Part 1`) while simultaneously rendering the 120 Hz camera `PreviewView` and ARCore VIO session.
   - Single-tile (`B=1` amortized in `B=5`) latency is **`75.37 ms/tile` (`13.27 tiles/sec`)**, which is well below `125 ms` per tile; executing `5` tiles (`148.51 GFLOPs`) in `<= 125 ms` would require `>= 1.19 TFLOP/s` sustained mobile OpenCL throughput (or INT8 / NPU delegation or backbone token pruning in future milestones). Because `ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST` and the 120 Hz GPU RenderNode EIS (`GroundPlaneRaycaster.onRealtimeImuUpdate`) are completely decoupled from the background analyzer executor, the UI and camera preview run at a locked 120 Hz with zero frame drops while `2-Graph B=5` sweeps complete every `~377 ms` (`2.65 Hz`).

#### 3.5.5. On-Device Video Analysis (`gneissfind screen grab 3.mp4`) & 120 Hz AR Ego-Motion Reticle Reprojection (`0cad9b2`)
Live handheld testing on the Pixel 10 Pro XL (`1.0x` zoom, `R-270` reverse landscape, `PITCH 13°–24°` across a desk with `cup -> 01 AGATE` and `remote -> 04 PETRIFIED WOOD`) exposed three UX/tracking issues when running neural sweeps at `~2.2–2.6 Hz` (`~390 ms/sweep`):
1. **Fragile Track Association Under Shallow-Angle Pitch & Yaw Motion:**
   - At `PITCH 20°` (`h = 1.06 m`), forward ground distance $Z = h / \tan\theta$ has pitch sensitivity $|dZ / d\theta| = h / \sin^2(20^\circ) \approx \mathbf{158\text{ mm/deg}}$, and `raycastTrigonometric` previously ignored camera yaw/azimuth. Consequently, a **`> 0.26°` wrist tilt** or **`> 1.0°` horizontal pan** between two sweeps (`~400 ms` apart) shifted `worldPosition` by `> 40 mm` (`DEFAULT_CLUSTER_RADIUS_METERS`), causing `TemporalTargetTracker` to fail association and reset the track to `1/6`.
2. **Slow Initial Lock (`>= 1.2 s` Minimum Under 3-of-6 Voting):**
   - Requiring 3 hits at `2.5 Hz` (`~400 ms/sweep`) imposed a minimum `1.2 s` dwell before any reticle or ping could appear—and any single missed association forced a full restart.
3. **Uncompensated `~390 ms` Inference Latency & Frozen Inter-Sweep Reticles:**
   - `ConfirmedTarget` previously stored static 2D strip pixel coordinates (`stripLeftPx..stripBottomPx`) from $t_{\text{capture}}$ (`~390 ms` before inference finished) and drew them without ego-motion compensation during the `400 ms` between sweeps or the `1.2 s` while an unmatched track coasted down from `[5/6] -> [3/6]`, leaving ghost brackets floating over empty background when the camera tilted away.
4. **3D IMU World-Ray Anchoring & 120 Hz Forward Reprojection Fix (`GroundPlaneRaycaster.kt`, `TemporalTargetTracker.kt`, `ScannerViewModel.kt`, `ScannerScreen.kt`):**
   - **128-Sample $3\times 3$ Rotation Ring Buffer (`GroundPlaneRaycaster`):** Stores full $3\times 3$ remapped landscape rotation matrices $R_{\text{world}\leftarrow\text{scr}}(t)$ (`~1.02 s` history at `125 Hz`). At frame capture time `frameTs`, `raycastStripDetection` looks up the nearest historical rotation matrix (`lookupRotationMatrixAtTimestampNs`) and back-projects the detection center $(u_{\text{cap}}, v_{\text{cap}})$ using calibrated focal length $f_y$ (`1475.0 px` at `1.0x`, `2950.0 px` at `2.0x`) and $R_{\text{world}\leftarrow\text{scr}}(t_{\text{capture}})$ into a rotation-invariant unit 3D bearing vector $\hat{\mathbf{d}}_{\text{world}} \in S^2$ (`WorldRay3D`) plus angular box dimensions $(\Delta\theta_x, \Delta\theta_y)$.
   - **Rotation-Invariant World-Bearing Track Association (`TemporalTargetTracker`):** Associates same-class detections across sweeps when world-ray angular separation $\Delta\theta = \arccos(\hat{\mathbf{d}}_{\text{track}} \cdot \hat{\mathbf{d}}_{\text{cand}}) \le \mathbf{3.8^\circ}$ (`DEFAULT_MAX_RAY_ANGLE_DEG = 3.8f`) or within slant-range-scaled 3D ground radius `clusterRadiusMeters * clamp(R / 1.50, 1.0, 2.5)` (`40 mm` at nominal `1.50 m` scaling up to `100 mm` at shallow pitch angles).
   - **Rapid 2-Sweep High-Confidence Acquisition & Confirmation Hysteresis (`TemporalTargetTracker`):** Promotes a track on **2 consecutive high-confidence ($\ge 0.72$) world-ray hits** (`DEFAULT_FAST_LOCK_MIN_CONFIDENCE = 0.72f`, `DEFAULT_FAST_LOCK_CONSECUTIVE_HITS = 2`, achieving **`~0.4 s` initial lock** at `2.5 Hz`) in addition to standard `3-of-6` voting, and applies confirmation hysteresis (`DEFAULT_MIN_VOTES_TO_MAINTAIN = 2`) so a single missed sweep during camera motion never drops an active lock. Confirmed tracks smooth their world-ray bearing via normalized EMA (`alpha = 0.45f`), angular size (`0.35f`), confidence (`0.40f`), and physical diameter (`0.35f`).
   - **120 Hz Ego-Motion Reprojection & Off-Strip Ghost Suppression (`ScannerViewModel` + `ScannerScreen`):** Both immediately upon sweep completion (eliminating the `~390 ms` GPU inference lag between $t_{\text{capture}}$ and $t_{\text{done}}$) and on every 120 Hz IMU callback (`onRealtimeImuPitchListener`), `reprojectConfirmedTargetsToCurrentPose` projects each target's $\hat{\mathbf{d}}_{\text{world}}$ through $R_{\text{world}\leftarrow\text{scr}}(t_{\text{now}})^T$ into current strip coordinates (`reprojectWorldRayToStripBox`) and sets `isVisibleInStrip = false` (suppressing both the visual reticle in `PushBroomTripwireOverlay` and the acoustic ping in `ScannerViewModel.updateAudioSonifierForVisibleTargets`) whenever camera rotation moves the target outside the `[0..stripWidthPx] x [0..384]` strip ROI.

#### 3.5.6. Codebase-Wide `annotate-code` Pass (`docs/COMMENTING.md`) & Audit Findings
All 19 Kotlin source and test files under `android/app/src` were annotated to conform to `docs/COMMENTING.md` and verified with `comment_diff_check.py --strict` (`19/19` `OK (comment-only)`, `18/18` JUnit4 tests passing). During the audit:
- **Corrected Comment Contradiction:** In `SyntheticStoneDetector.kt` (`PEBBLE_20MM_HIGH_PITCH`), updated the inline seam geometry comment from `tile1 x=0..32` (`IoU = 32/36 = 0.89`) to match the actual emitted box `rightPx = 34f` (`tile1 x=0..34`, `strip x=348..386`, `IoU = 32/38 = 0.84 >= 0.45`).
- **`RATIONALE-UNKNOWN` Markers Logged for Future Tuning:**
  1. `PushBroomCropper.DEFAULT_MIN_CONFIDENCE = 0.35f`: how `0.35` was chosen relative to the COCO metadata threshold.
  2. `MetalDetectorSonifier.updateTargets`: how the `0.55f / 0.45f` blend weights between `smoothedConfidence` and `voteRatio` were selected.
  3. `MetalDetectorSonifier.TONE_BURST_DURATION_SEC = 0.048f`: why `48 ms` was chosen as the nominal tone burst duration.
  4. `MetalDetectorSonifier.MAX_PITCH_HZ = 780.0f` / `MIN_PITCH_HZ = 320.0f`: how the exact `780 Hz` ceiling and `320 Hz` floor frequencies were selected.
  5. `MetalDetectorSonifier.MIN_PING_HZ = 2.0f` / `MAX_PING_HZ = 14.0f`: how the `2.0..14.0 Hz` ping cadence bounds were selected.
  6. `PushBroomCameraController.maybeServoFloatingIsoFromCenterLuma`: why `118f` was chosen as the 0-EV reference Y-luma target and `0.12f` as the deadband threshold.
- **Non-Bug Code Observations Preserved for Reference:**
  - `FieldDataLogger.buildMetadataJsonLine` writes paired compatibility alias keys (`"exposure_ns"` + `"exposure_time_ns"`, `"iso"` + `"sensor_iso"`, `"pitch_deg"` + `"imu_pitch_deg"`, `"gsd_mm_px"` + `"gsd_mm_per_px"`, `"ground_speed_mps"` + `"walk_speed_mps"`).
  - `RfDetrTwoGraphGpuDetector.stageBatchedTileInputs` populates both `batchedInputFloats` and `directInputFloatBuffer` (while `part1Inputs[0].writeFloat(batchedInputFloats)` consumes `batchedInputFloats`).
  - `MetalDetectorSonifier.synthesizeSinglePingBurstPcm16` synthesizes a pure fundamental sine (`0.75 * sin(phase)`), whereas the live `runAudioSynthesisLoop` adds a `0.18` 2nd-harmonic overtone (`0.65 * (sin(phase) + 0.18 * sin(2 * phase))`).

---

### 3.6. Milestone 2B — >= 5.0 Hz Sustained Full-Swath Detection, `W8-Dyn` GPU Quantization & 180s Thermal Soaks (`exp/m2b-fps-exploration`)

Full report, 27-configuration Pareto table, and Tensor G5 NPU postmortem: [`docs/M2B_OVERNIGHT_REPORT.md`](M2B_OVERNIGHT_REPORT.md).

1. **Root Cause of Live-App (`420–577 ms`) vs. CLI (`117.7 ms`) Latency Gap:**
   - Holding CameraX `ImageProxy` (`HardwareBuffer` / Gralloc surface) open on `GneissFind-Analyzer` for the entire `350–500 ms` GPU inference starved the Camera2 `ImageReader` queue and contended with 120 Hz `PreviewView` rendering.
   - `RfDetrTwoGraphGpuDetector` previously performed a synchronous `5.9 MB` GPU->CPU->GPU copy of `part1Outputs[0]` (`encMemory`) into `part2Inputs[0]` every sweep.
   - `104.5 MB` of unquantized model weights saturated LPDDR5X bandwidth when running concurrently with Camera2 ISP (`1656x1240` RAW10 + YUV420) and 120 Hz SurfaceFlinger composition, driving `pixel-thermal` into `Thermal Status: 1` within 50 seconds.
2. **Architectural & Model Optimizations Implemented:**
   - **3-Slot Lock-Free Producer-Consumer Pipeline (`ScannerViewModel.kt`):** `onFrameCaptured` extracts the `1792x384` YUV tiles into a 3-slot `PreparedSweepSlot` ring buffer in `~3 ms` on `GneissFind-Analyzer`, calls `imageProxy.close()` immediately, and dispatches GPU inference asynchronously on a dedicated `GneissFind-Inference` thread (`420.4 ms -> 352.7 ms` on the `384 B=5` baseline alone).
   - **Zero-Copy `encMemory` GPU Buffer Binding (`RfDetrTwoGraphGpuDetector.kt`):** Passes `part1Outputs[0]` directly as `part2Inputs[0]` (`part2ZeroCopyInputs`), eliminating the `5.9 MB` host copy.
   - **`W8-Dyn` (`dynamic_wi8_afp32`) GPU Weight Quantization + LiteRT `2.3.2` (`allowSrcQuantizedFcConvOps = true`):** Compresses `RF-DETR-Nano Part 1 + Part 2` by **`3.7x` (`104.5 MB -> 28.1 MB`)** with **0.0% recall loss** (unlike static `W8A8` which collapses ViT small-pebble recall from `100% -> 12.5%` and triggers `kLiteRtStatusErrorNotFound` / JNI faults under `Accelerator.NPU`), cutting `Part 1` GPU time by `28–33%`.
   - **Top-K=`100` Decoder Surgery + Priority-Aware `B=3` Even/Odd Sub-Swath Interleaving:** Reduces Stage-2 decoder queries from `300 -> 100` (`-46%` Part 2 latency, `0.0%` recall loss) and interleaves `{0, 2, 4}` and `{1, prio, 3}` tiles while always including the highest-priority active target tile on every sweep.
3. **180-Second Continuous Live-App Thermal Soak Results on Pixel 10 Pro XL (`58110DLCQ00003`):**
   - **`S-01` (`RF-DETR 256 B=3 W8-DYN`, Default Apache-2.0):** **`108.2 ms` (`9.24 FPS`) cold** (beating the `>= 8.0 FPS` stretch goal) and **`152.2 ms` (`6.57 FPS` batch / priority target tile, `3.29 Hz` full 5-tile swath) sustained at `t = 150..180s`** (`100%` Med recall, `87.5%` Small `38-45px` recall).
   - **`S-02` (`RF-DETR 320 B=3 W8-DYN`, `100%` Small Recall Apache-2.0):** **`141.5 ms` (`7.07 FPS`) cold** and **`187.9–204.2 ms` (`4.90–5.32 FPS`) at `Thermal Status: 1`** with **`100.0%` Med recall, `100.0%` Small `38-45px` recall, and `91.7%` All-Small `28-45px` recall**.
   - **`S-03` (`RF-DETR 256 B=5 W8-DYN`, All 5 Tiles Every Sweep Apache-2.0):** **`158.0 ms` (`6.33 FPS`) cold** and **`206.0 ms` (`4.85 FPS` across all 5 tiles simultaneously) at `t = 150..180s`** (`3.2x` faster than the `577 ms` baseline).
   - **`S-04` (`YOLO26-S 384 B=5 GPU`, AGPL-3.0 Reference):** **`76.1 ms` (`13.15 FPS`) cold** and **`128.3 ms` (`7.80 FPS` across all 5 tiles) at `t = 150..180s`**.

---

## 4. Current Android Architecture & Interfaces for Milestone 3 / 4 Coordination

> [!TIP]
> For a developer-oriented walkthrough of how runtime configurations are selected, how the 4-stage pipeline fits together, and "if you want to do X, look in file Y" entry points, see [`docs/INTERNALS.md`](INTERNALS.md).

When planning or modifying upcoming milestones in `docs/ROADMAP.md` or `ml-pipeline/`, use these exact contracts from `android/` and `ml-pipeline/`:

- **Push-Broom Tile Extraction & Cross-Tile Seam Merge (`com.gneissfind.app.vision.PushBroomCropper`):**
  - Input frame: `1920 x 1080` landscape `YUV_420_888` (`ImageProxyYuv420Frame`).
  - Active strip modes (`SwathTileMode`):
    - `FIVE_TILE` (default): `5` tiles of `384 x 384 px`, `xOffsets = [64, 416, 768, 1120, 1472]`, `32 px` overlap, unified strip `1792 x 384 px`.
    - `THREE_TILE`: `3` tiles of `384 x 384 px`, `xOffsets = [416, 768, 1120]`, `32 px` overlap, unified strip `1088 x 384 px`.
  - Pre-allocated buffers (`PushBroomSweepBuffers`): holds `activeTileCount` reusable `FloatArray(384 * 384 * 3)` RGB `[0..1]` NHWC buffers (`applyImageNetNormalization = false`, since `(x - mean) / std` is fused into the `.tflite` graph prefix) plus an optional packed `IntArray(1792 * 384)` ARGB strip populated only when `OperatingMode.LOG_DATA` is active.
  - Cross-tile seam deduplication (`mergeAndFilterTileDetections`): clusters same-class detections across tiles when `IoU >= 0.45f`, `IoM >= 0.65f`, or when boxes from adjacent tiles straddle the `[xOffset .. xOffset + 32]` seam band within `8 px` margin with vertical overlap ratio `>= 0.60f`, fusing each cluster into its union bounding box (`minLeft, minTop, maxRight, maxBottom`).
- **3D IMU World-Ray Anchoring & Temporal Tracking (`com.gneissfind.app.tracking.GroundPlaneRaycaster` & `TemporalTargetTracker`):**
  - `GroundPlaneRaycaster`: maintains a 128-entry timestamped ring buffer of $3\times 3$ remapped `TYPE_GAME_ROTATION_VECTOR` landscape rotation matrices (`lookupRotationMatrixAtTimestampNs`), back-projects strip detections into unit `WorldRay3D` vectors (`pixelToWorldRay`), and forward-reprojects confirmed targets at 120 Hz (`reprojectConfirmedTargetsToCurrentPose`).
  - `TemporalTargetTracker`: associates `ProjectedCandidate` hits via world-ray angular bearing ($\Delta\theta \le 3.8^\circ$, `DEFAULT_MAX_RAY_ANGLE_DEG`) and slant-range-scaled 3D ground radius (`40 mm * clamp(R / 1.50, 1.0, 2.5)`), supports 2-sweep consecutive high-confidence (`>= 0.72`) fast acquisition plus 3-of-6 spatial voting with 2-vote maintenance hysteresis (`DEFAULT_MIN_VOTES_TO_MAINTAIN = 2`), and enforces the `1.10 m/s` walk-speed gate (`ScanSpeedState.SpeedWarning`).
- **Detector Interface (`com.gneissfind.app.inference.StoneDetector`):**
  - `fun detectSweep(sweepBuffers: PushBroomSweepBuffers): SweepInferenceResult`
  - Implementations:
    - `RfDetrTwoGraphGpuDetector` *(Default active backend)*: LiteRT `2.3.2` (`com.google.ai.edge.litert:litert:2.3.2`) self-configuring 1-Graph/2-Graph GPU FP16 + `W8-Dyn` (`allowSrcQuantizedFcConvOps = true`) runner supporting `LiveModelProfile` hot-swapping from `LcarsSourceDrawerOverlay` (`RFDETR_NANO_256_B3_INTERLEAVED`, `RFDETR_NANO_320_B3_INTERLEAVED`, `RFDETR_NANO_256_B5_K100`, `YOLO26S_384_B5_GPU`, `RFDETR_NANO_384_BASELINE`), zero-copy `part1Outputs[0]` (`encMemory`) passthrough, 1D precomputed bilinear resampling (`384 -> 320/256`), `80`-class contiguous & `91`-class sparse COCO LUTs, intra-tile greedy NMS (`IoU >= 0.45`), and Priority-Aware `B=3` Even/Odd Sub-Swath Interleaving.
    - `SyntheticStoneDetector`: deterministic moving-gravel simulator (`MIXED_GRAVEL_WALK`, `AGATE_PEBBLE_20MM`, `JASPER_COBBLE_55MM`) selectable from `SourceSelectionDrawer`.
- **Field Data Logger (`com.gneissfind.app.logging.FieldDataLogger`):**
  - When `MOD` is toggled to `LOG` (`OperatingMode.LOG_DATA`), saves lossless `1792 x 384` (or `1088 x 384`) PNG strips at `4.0 Hz` alongside `metadata.jsonl` (`FieldStripMetadata`: timestamp, strip dimensions, tile count, zoom ratio, exposure ns, ISO, shutter clamp state, EV bias, IMU pitch, reference pitch, stabilized `cropY`, slant range `R`, GSD `mm/px`, and horizontal walk speed `m/s`) for Milestone 3 `SAM 2` auto-labeling on the Dell T7920.
