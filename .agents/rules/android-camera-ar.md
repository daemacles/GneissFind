---
trigger: glob
globs: "android/**"
description: Right-edge bounds safety, MediaCodec 30 FPS 1080p + 120 Hz IMU recording, ReplayCameraSource injection, 3D Innovation Gate, and Two-Tier Full-Frame + Collar occlusion tracking.
---

# Android Camera, Sensor-Trace Replay & Two-Tier Occlusion Tracking Rules

## 1. Strict Frame-Boundary & Right-Edge (`x >= 1920`) Memory Safety
Every pixel extractor, OpenCV `Mat`/`Rect` crop, and Compose `DrawScope` overlay MUST survive target coordinates anywhere in `(-10000f .. +10000f)` (including `NaN`/`Infinite` guards):
1. **`YUV_420_888` Semi-Planar Chroma Overrun Guard (`PushBroomCropper.kt`):**
   - On Android `YUV_420_888` (`NV21`/`NV12` semi-planar with `uvPixelStride == 2`), `uBuffer.limit()` / `vBuffer.limit()` is frequently `(width / 2 - 1) * uvPixelStride + 1 + (height / 2 - 1) * uvRowStride` (1 byte shorter than `width / 2 * uvPixelStride` on the final row, `x = 1918..1919, y = 1078..1079`!).
   - Always clamp `uIndex` and `vIndex` against `0 until bufferLimit` (or compute exact `maxValidIndex = buffer.limit() - 1`) before indexing `ByteBuffer.get(idx)` so right/bottom edge crops (`cropLeftPx + cropSizePx == 1920`) NEVER throw `IndexOutOfBoundsException`!
2. **OpenCV JNI `Rect` / `submat` Guard (`OpenCvConstellationTracker.kt`):**
   - Never pass unclamped coordinates to `org.opencv.core.Rect(x, y, w, h)` or `Mat.submat(...)` (which triggers fatal native `cv::Exception` SIGABRT when `x + w > mat.cols`, `y + h > mat.rows`, `x < 0`, `y < 0`, or `w <= 0 || h <= 0`).
   - Helper mandate: `val safeX = x.coerceIn(0, mat.cols - 2); val safeW = w.coerceIn(2, mat.cols - safeX)` (and identically for `y`, `h`, `mat.rows`) before every `submat` call.
3. **Compose `DrawScope` Right/Bottom Edge Guard (`ScannerScreen.kt`):**
   - Guard all `drawRect`, `drawRoundRect`, `drawCircle`, and off-screen chevron badge placements against negative `Size(width, height)` (`width.coerceAtLeast(0f)`, `height.coerceAtLeast(0f)`) and `NaN` floats when clamping badges against the right (`x -> viewportWidth`) or bottom edge.

## 2. `MOD: LOG` — Hardware `MediaCodec` `30 FPS` `1080p` MP4 + `120 Hz` IMU Trace Logger
- **Replace the legacy `4.0 Hz` PNG strip logger** in `FieldDataLogger` with a synchronized video + IMU trace recorder writing to `Android/media/com.gneissfind.app/replays/<YYYYMMDD_HHMMSS>/`:
  - **`clip.mp4` (`1920 x 1080`, `30 FPS`, H.264 `video/avc`, `14 Mbps`):**
    - **CRITICAL:** Do NOT bind CameraX `VideoCapture` (a 3rd camera surface breaks `1920x1080` `ImageAnalysis` stream combinations or overrides `Camera2CameraControl` `<= 1/500 s` shutter clamps).
    - Instead, copy the already-upright `1920x1080` `YUV_420_888` frame on a bounded pool into a background `MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)` (`COLOR_FormatYUV420Flexible` / `COLOR_FormatYUV420SemiPlanar`, `I_FRAME_INTERVAL = 1` sec for fast seeking) + `MediaMuxer`.
  - **`trace.jsonl` (Synchronized Telemetry & 120 Hz IMU Sidecar):**
    - Line 0 (`"type": "header"`): `{schema_version: 1, width: 1920, height: 1080, fps: 30, zoom_ratio, focal_length_px, ref_pitch_deg, display_rotation, git_hash}`.
    - Continuous `120 Hz` IMU lines (`"type": "imu"`): `{type: "imu", t_ns, rot_mat: [r0..r8], pitch_deg}` using the exact same monotonic clock (`System.nanoTime()` aligned offset) as frame timestamps.
    - Per-frame lines (`"type": "frame"`): `{type: "frame", frame_idx, pts_us, t_ns, exposure_ns, iso, zoom_ratio, shutter_clamp, ev_bias}`.
  - Update the LCARS `REC` readout to show recorded seconds/frames (`e.g., REC 00:18 • 540F`) while `MOD == LOG_DATA`.

## 3. `ReplayCameraSource` — Deterministic On-Device Video + IMU Replay (`SRC` Drawer & `adb` CLI)
- Implement `ReplayCameraSource` decoding any `<session>/clip.mp4` + `trace.jsonl` folder under `Android/media/com.gneissfind.app/replays/` (and `filesDir/replays/` fallback for `adb push` convenience):
  1. Unbinds/suspends live CameraX capture while replay is active, decodes `clip.mp4` via hardware `MediaExtractor` + `MediaCodec` paced at `30 FPS` (looping by default in interactive UI, single-pass when launched via test intent), renders each frame onto the top viewport (`Surface` / `Bitmap` / `TextureView` inside `ClippingCameraViewportContainer` preserving 120 Hz EIS in `PATROL` and full `16:9` in `LOCK-ON`), and feeds synchronized `trace.jsonl` `imu` matrices + decoded `1920x1080` `YUV` buffers through the **exact production `GroundPlaneRaycaster`, `PushBroomCropper`, `StoneDetector` (real Tensor G5 GPU!), `TemporalTargetTracker`, and `LockOnPinpointController` pipeline**.
  2. **`SRC` Drawer Integration:** Keep **1** synthetic profile (`MIXED_GRAVEL_WALK`) for zero-file smoke checks, remove the other two synthetic demos from the UI drawer (`AGATE_PEBBLE_20MM`, `JASPER_COBBLE_55MM` stay in JUnit only), and dynamically populate all discovered `REPLAY: <session_name>` clips.
  3. **Agent `adb` Automation Hooks (`MainActivity` Intent Extras):**
     - `--es debug_replay_session "<folder_name_or_abs_path>"` — auto-starts that replay clip immediately on launch.
     - `--ei debug_auto_lock_frame <N>` — automatically invokes `lockNearest()` (or locks synthetic/detected target) at `frame_idx == N`.
     - `--ez debug_loop_replay false` — stops cleanly at end-of-clip and logs a single `GNEISSFIND_REPLAY_SUMMARY {frames, locks, handovers, lost_locks, distractor_jumps, exceptions}` JSON line to Logcat so Antigravity can run closed-loop regression tests and `adb exec-out screencap -p` inspections autonomously.

## 4. Hardened Lock-On & Two-Tier Post-Occlusion Tracking (`LockOnPinpointController` + `OpenCvConstellationTracker`)
### A. Why Both Local `3x` Collar AND Full-Frame Tracking Are Required
- **Keep the `3.0x` Target-Diameter `AKAZE` Collar (`192x192`, `< 5 ms`)** for local coplanar pinpoint accuracy: running `AKAZE` + single-plane affine/homography across the full `1920x1080` frame (`56x` more pixels = `~120 ms` CPU) fails because foreground vs. background **3D depth relief** and non-planar objects skew a single global homography off the stone, while gravel **perceptual aliasing** degrades global `BFMatcher` ratio tests.
- **Unfreeze & Upgrade Full-Frame Ego-Motion Odometry (`640x360` IMU-Derotated `PyrLK`, `< 2 ms` CPU):**
  - Remove the off-screen VO freeze (`allowGroundVoIntegration` must stay **ACTIVE** during occlusion and off-screen excursions!).
  - Every `30 FPS` luma frame during `LOCK_ON_NEURAL` and `PINPOINT_CONSTELLATION`, downsample the full `1920x1080` `Y` plane to `640x360` (`3x` decimation), maintain `~80–120` Shi-Tomasi features (`Imgproc.goodFeaturesToTrack`, masking out only the immediate target interior when visible), track frame-to-frame via `Video.calcOpticalFlowPyrLK`, **subtract exact 3D rotational pixel flow** using `GroundPlaneRaycaster`'s IMU $\Delta R = R_{	ext{cam}}(t_2)^T R_{	ext{cam}}(t_1)$, and compute robust RANSAC/median metric camera translation $\Delta \mathbf{t}_{	ext{ENU}}$ (scaled by ray-ground depth). This keeps `camPosEnu` and the 3D triangulated stone anchor `P_3d` accurately remembered in 3D space even when the user walks/pans while the stone is occluded or off-screen!
- **Full-Frame `480x270` `AKAZE` Relocalization Snapshot (Post-Occlusion Drift Reset):**
  - Capture a single `480x270` (`1/4` scale) full-frame `AKAZE` keyframe (`<= 250` keypoints) once when `PINPOINT_CONSTELLATION` first locks (refreshed at major `1.5x` scale steps while locked).
  - Do NOT match it every frame; match it only at `~5 Hz` when recovering from occlusion or off-screen re-entry (`akazeInliers < 12` and target projected near/inside FOV) via 2D/3D PnP or affine anchor transfer to snap out any accumulated KLT translation drift!

### B. Anti-Hijack Innovation Gate & Anti-Thrash Coasting Hysteresis
1. **3D Bearing & Scale Innovation Gate (Fixes Teapot/Monitor Distractor Hijacking):**
   - When `LOCK_ON_NEURAL` or `640x640` wide re-acquisition evaluates `B=1` detections, REJECT any candidate whose:
     - Class does not match `lockedStoneClass`, OR
     - 3D world-ray angular separation from the current IMU+VO-predicted anchor ray exceeds **`4.5 deg`** while on-screen (widening gradually up to **`9.0 deg` max** after several seconds of off-screen VO drift—never accepting a `15–25 deg` jump across the table onto another cup/stone!), OR
     - Apparent bounding-box major span differs from range-adjusted `estimatedDiameterMm` outside **`[0.45x .. 2.2x]`** (ignoring frame-border-clipped boxes).
2. **`AKAZE` Specular/Refractive Coasting Hysteresis (`300 ms` / `9 Frames`):**
   - When `PINPOINT_CONSTELLATION` is active and `akazeInliers` momentarily dips `< 12` (due to wet specular glare or clear-glass background parallax), do NOT immediately drop out of `PINPOINT_CONSTELLATION` into `RE-ACQUIRING B=1`. Coast smoothly for up to **`300 ms` (`9` consecutive luma frames)** using `120 Hz` IMU + `30 FPS` full-frame KLT VO while attempting local collar + `480x270` context relocalization; only demote to `LOCK_ON_NEURAL` if `akazeInliers < 12` persists continuously for `> 300 ms`.
