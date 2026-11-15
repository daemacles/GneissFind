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

---

## 2. Commit Ledger (Milestone 1 → Milestone 1B)

| Commit | Summary | Primary Files Touched |
| :--- | :--- | :--- |
| `3f1dee0` | Initialize GneissFind workspace specs and Milestone 1 baseline | `AGENTS.md`, `docs/ROADMAP.md`, `.agents/rules/*`, `android/app/src/main/java/com/gneissfind/app/**` |
| `0e3c81e` | Add 5-tile landscape push-broom cropper and reversible IMU pitch stabilization | `model/StoneModels.kt`, `camera/PushBroomCropper.kt`, `ar/GroundPlaneRaycaster.kt`, `camera/PushBroomCameraController.kt`, `inference/SyntheticStoneDetector.kt`, `inference/RfDetrTwoGraphGpuDetector.kt`, `data/FieldDataLogger.kt`, `ui/ScannerViewModel.kt`, `PushBroomCropperTest.kt`, `SonifierAndHostBridgeTest.kt` |
| `11009b7` | Implement split-screen 2360s TNG LCARS interface with offline Antonio typography | `MainActivity.kt`, `ui/ScannerScreen.kt`, `ui/theme/LcarsComponents.kt`, `ui/theme/Theme.kt`, `res/font/antonio_bold.ttf`, `res/font/antonio_regular.ttf` |
| `d724234` | Decouple 120 Hz GPU EIS preview translation from analyzer frame rate | `ar/GroundPlaneRaycaster.kt`, `camera/PushBroomCameraController.kt`, `ui/ScannerScreen.kt`, `ui/ScannerViewModel.kt` |
| `8707282` | Refine LCARS semantic color taxonomy, source drawer, left pitch gauge, and single-action affordances | `MainActivity.kt`, `ui/ScannerScreen.kt`, `ui/ScannerViewModel.kt`, `ui/theme/LcarsComponents.kt`, `ui/theme/Theme.kt`, `LcarsGeometryAndViewportTest.kt` |
| `fc80785` | Fix `PreviewView` `TextureView` layout starvation and Camera2 dynamic exposure overrides | `camera/PushBroomCameraController.kt`, `ui/ScannerScreen.kt`, `ui/ScannerViewModel.kt` |

---

## 3. Chronological Findings & Architectural Evolution (M1 → M1B)

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

## 4. Current Android Architecture & Interfaces for Milestone 2 / 3 Coordination

When planning or modifying upcoming milestones in `docs/ROADMAP.md` or `ml-pipeline/`, use these exact contracts from `android/`:

- **Push-Broom Tile Extraction (`com.gneissfind.app.camera.PushBroomCropper`):**
  - Input frame: `1920 x 1080` landscape `YUV_420_888` (`ImageProxyYuv420Frame`).
  - Active strip modes (`SwathTileMode`):
    - `FIVE_TILE` (default): `5` tiles of `384 x 384 px`, `xOffsets = [64, 416, 768, 1120, 1472]`, `32 px` overlap, unified strip `1792 x 384 px`.
    - `THREE_TILE`: `3` tiles of `384 x 384 px`, `xOffsets = [416, 768, 1120]`, `32 px` overlap, unified strip `1088 x 384 px`.
  - Pre-allocated buffers (`PushBroomSweepBuffers`): holds `activeTileCount` reusable `FloatArray(384 * 384 * 3)` RGB `[0..1]` NHWC buffers plus an optional packed `IntArray(1792 * 384)` ARGB strip populated only when `OperatingMode.LOG_DATA` is active.
- **Detector Interface (`com.gneissfind.app.inference.StoneDetector`):**
  - `fun detectSweep(sweepBuffers: PushBroomSweepBuffers): DetectorSweepResult`
  - Implementations:
    - `SyntheticStoneDetector`: deterministic moving-gravel simulator (`MIXED_GRAVEL_WALK`, `AGATE_PEBBLE_20MM`, `JASPER_COBBLE_55MM`) generating realistic cross-seam targets and muted `HARD_CONFUSER` instances.
    - `RfDetrTwoGraphGpuDetector`: two-graph LiteRT GPU FP16 runner (`rf_detr_nano_part1_fp16.tflite` + Kotlin `selectTopKQueries` host bridge + `rf_detr_nano_part2_fp16.tflite`) with automatic fallback to `SyntheticStoneDetector` until Milestone 3 `.tflite` artifacts are placed in `android/app/src/main/assets/`.
- **Field Data Logger (`com.gneissfind.app.data.FieldDataLogger`):**
  - When `MOD` is toggled to `LOG` (`OperatingMode.LOG_DATA`), saves lossless `1792 x 384` (or `1088 x 384`) PNG strips at `4.0 Hz` alongside `metadata.jsonl` (`FieldStripMetadata`: timestamp, strip dimensions, tile count, zoom ratio, exposure ns, ISO, shutter clamp state, EV bias, IMU pitch, reference pitch, stabilized `cropY`, slant range `R`, GSD `mm/px`, and horizontal walk speed `m/s`) for Milestone 2 `SAM 2` auto-labeling on the Dell T7920.
