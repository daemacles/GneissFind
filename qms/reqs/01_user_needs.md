# GneissFind User Needs (`qms/reqs/01_user_needs.md`)

## `UN-001`: Real-Time Standoff Rockhounding Detection
- **Status:** Active
- **Validation-Method:** Golden-MCAP-And-Field-Walk
- **Summary:** Highlight and sonify target PNW semi-precious stones (`>= 20 mm` diameter: chalcedony/agate, red/yellow jasper, green Dallasite/nephrite, petrified wood) from waist height (`~1.06 m`, `~45 deg` pitch, `~1.50 m` slant range) at strolling pace (`~0.5 m/s`).

## `UN-002`: Eyes-Up Low-Fatigue Field Operation
- **Status:** Active
- **Validation-Method:** HIL-And-UX-Inspection
- **Summary:** Provide eyes-up acoustic metal-detector feedback scaled to stone size and multi-frame confidence, suppress transient 1-frame confusers, warn on excessive sweep speed (`> 1.10 m/s`), and keep the live camera viewport 100% free of modal UI occlusion.

## `UN-003`: Zero-Friction Workstation <-> Phone V&V Iteration
- **Status:** Active
- **Validation-Method:** Desktop-CLI-Replay-Benchmark
- **Summary:** Record deterministic hardware-timestamped field sensor bags on the Pixel 10 Pro, sync wirelessly to the Dell T7920 without USB cable dependency, and replay the exact production Rust perception/tracking/voting code at `>= 50x` real-time speed on T7920 CUDA GPUs.
