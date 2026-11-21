# GneissFind System Requirements (`qms/reqs/02_system_reqs.md`)

## `SYS-COORD-001`: Compile-Time Coordinate Space Separation
- **Parents:** UN-001, UN-003
- **Status:** Active
- **Oracle:** Pure-Rust `Coord2D<S>` / `BBox2D<S>` phantom types separate `TileFrame`, `StripFrame`, `SensorFrame`, and `WorldGroundFrame` with unit-tested round-trip conversions.

## `SYS-OPT-001`: Reversible Landscape & IMU Pitch Stabilization (`crop_y`)
- **Parents:** UN-001, UN-002
- **Status:** Active
- **Oracle:** `sensorLandscape` (`ROTATION_90` and `ROTATION_270`) remaps `TYPE_GAME_ROTATION_VECTOR` so downward tilt increases pitch in both orientations, shifting `crop_y` within `[0..696]` around `theta_ref` (`45 deg` default) and flagging `is_pitch_clamped` at limits.

## `SYS-OPT-002`: Stride-Safe 5-Tile Push-Broom YUV Cropper
- **Parents:** UN-001, UN-003
- **Status:** Active
- **Oracle:** Extracts 5 overlapping `384x384` tiles (`x_offsets = [64, 416, 768, 1120, 1472]`, `1792x384` strip; or 3 tiles `[416, 768, 1120]`) from `YUV_420_888` planes with arbitrary `row_stride >= width` and `uv_pixel_stride in {1, 2}` with zero chroma shear.

## `SYS-SEAM-001`: Cross-Tile Elongated Target Seam Fusion
- **Parents:** UN-001
- **Status:** Active
- **Oracle:** Adjacent-tile detections of the same class merge into a single unified `StripFrame` bounding box when `IoU >= 0.45 OR IoM >= 0.65`, preventing double-counting on elongated objects (`toothbrush`, `remote`, petrified wood).

## `SYS-PERF-001`: Pixel 10 Pro On-Device Sweep Latency & Heap Stability
- **Parents:** UN-001, UN-002
- **Status:** Active
- **Oracle:** 5-tile push-broom sweep (`YUV extract + 2-Graph GPU + NMS + Tracker + Voter`) executes in `<= 125.0 ms` (`>= 8.0 Hz`) on Pixel 10 Pro with `< 5 MB` native heap growth across 300 consecutive sweeps.

## `SYS-HIL-001`: Dark-Room / Stationary Synthetic Stimulus Verification
- **Parents:** UN-003
- **Status:** Active
- **Oracle:** Instrumented/HIL parity runner injects synthetic padded-stride `YUV_420_888` COCO stand-in frames and `CLOCK_BOOTTIME` IMU samples on a stationary dark-room device, verifying GPU detections, Camera2 zero-luma stability, and `.mcap` recording.

## `SYS-LOG-001`: Deterministic `CLOCK_BOOTTIME` `.mcap` Recorder & Replay
- **Parents:** UN-003
- **Status:** Active
- **Oracle:** Records strip frames, 200 Hz IMU, optics metadata, HUD snapshots, and user confuser flags into `.mcap` indexed by `CLOCK_BOOTTIME` ns; replaying in `cli_runner` reproduces identical track IDs and vote counts deterministically.

## `SYS-LOG-002`: Wireless P2P `.mcap` Sync & Headless Rerun Export
- **Parents:** UN-003
- **Status:** Active
- **Oracle:** Transfers recorded `.mcap` files over QUIC (`iroh`/LAN) with cryptographic hash verification to T7920 inbox and exports headless `.rrd` 3D/timeline recordings via `cli_runner`.
