# GneissFind Polyglot Monorepo Task Runner (`justfile`)
set shell := ["bash", "-euo", "pipefail", "-c"]

default: qms-check test-rust test-android-unit

# Run all pure-Rust unit and property tests natively on Ubuntu (< 3s)
test-rust:
    cargo test --workspace

# Build arm64-v8a Rust bridge + UniFFI Kotlin bindings & run Android JVM unit tests
test-android-unit:
    ./gradlew -p android assembleDebug testDebugUnitTest

# Verify Trace-as-Code QMS bidirectional coverage & regenerate docs/TRACEABILITY_MATRIX.md
qms-check:
    python3 scripts/qms_verify.py --write-matrix docs/TRACEABILITY_MATRIX.md

# Run desktop T7920 ORT <-> LiteRT quantization parity + golden synthetic MCAP replay + Rerun .rrd export
vv-desktop:
    cargo run --release -p cli_runner -- verify-parity
    cargo run --release -p cli_runner -- synth-mcap --out /tmp/gneissfind_synth_walk.mcap
    cargo run --release -p cli_runner -- replay --mcap /tmp/gneissfind_synth_walk.mcap --save-rrd /tmp/gneissfind_synth_walk.rrd

# Run dark-room / stationary USB-connected Pixel 10 Pro HIL verification suite
hil-darkroom:
    adb devices
    ./gradlew -p android connectedDebugAndroidTest
