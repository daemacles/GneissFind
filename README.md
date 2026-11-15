# GneissFind — Planetary Geological Push-Broom Scanner (`GEO-47`)

> *"Dammit Jim, I'm a rockhound, not a quarryman!"*
>
> **GneissFind** is a real-time, eyes-free mobile geological line scanner for Pacific Northwest gravel bars and cobble beaches. Because life is too short to take a good stone for *granite*—and when you finally spot a translucent carnelian nodule in wet river cobble, it's *gneiss* to have a 24th-century tactical sensor array backing you up.

---

## What Is This Thing? (The Away-Team Briefing)

Staring straight down at wet, sun-dappled river gravel while walking at `0.8 m/s` is a notoriously difficult visual search task for human eyes—and an even harder one for standard full-frame mobile object detectors, which waste 70% of their compute budget re-evaluating the same rocks 30 times or hallucinating on specular water glare.

**GneissFind** turns an Android flagship (targeted and benchmarked on the **Google Pixel 10 Pro XL**) into a **push-broom line-scan geological instrument** with a split-screen, 2360s-inspired retro-futuristic **LCARS** tactical interface and an eyes-free **metal-detector audio sonifier**:

1. **Optical Tripwire Geometry (`5 x 384x384` Push-Broom Swath):**
   - Locks the camera to `2.0x` lossless in-sensor center crop (`1920x1080` landscape `YUV_420_888`) and extracts **5 horizontally overlapping `384x384` tiles** (`1792x384 px` total width with `32 px` overlap seams).
   - At a comfortable $45^\circ$ waist-high scan angle (`~1.50 m` slant range), this projects a **`~1.05 m` wide $\times$ `~0.25 m` deep** ground tripwire at **`0.59 mm/px` Ground Sample Distance (GSD)**—putting ~34 pixels across a `20 mm` pebble.
2. **Defeating Motion Blur (`<= 1/500 s` Shutter Clamp + Closed-Loop Auto-ISO):**
   - Standard auto-exposure under Pacific Northwest tree canopy drops shutter speeds to `1/60 s`, smearing walking-speed gravel into unrecognizable streaks. GneissFind clamps `SENSOR_EXPOSURE_TIME <= 2,000,000 ns` (`1/500 s`) via `Camera2CameraControl` while dynamically servoing `SENSOR_SENSITIVITY` (ISO) and Canopy `EV` bias.
3. **120 Hz Inertial Pitch Stabilization (Software EIS):**
   - Wrist pitch wobble of just $\pm 4^\circ$ would sweep a narrow strip wildy across the ground. GneissFind fuses the 120 Hz `TYPE_GAME_ROTATION_VECTOR` IMU to dynamically shift `cropY` within `[0..696]` and translates the live viewport via GPU `RenderNode` at 120 Hz—locking the scan strip onto the target ground patch with zero Android `View.requestLayout()` jank.
4. **Two-Graph `RF-DETR-Nano` LiteRT 2.x GPU (FP16) Inference:**
   - Runs a two-subgraph split of `RF-DETR-Nano` (`384x384`, DINOv2-Small backbone + 2-layer deformable cross-attention decoder) compiled **100% onto Android's `LITERT_CL` OpenCL GPU delegate** (`909/909` nodes in `Part 1`, `350/350` nodes in `Part 2`), connected by a zero-allocation Kotlin Host `Top-300` proposal bridge and cross-tile `IoU + IoM` seam union merging.
5. **3-of-6 Temporal Voter & Eyes-Free Acoustic Sonification:**
   - Filters out transient wet-cobble specular glints by requiring **3 detections out of a sliding 6-sweep window** (`TemporalTargetTracker`) before locking a target reticle or firing the phase-continuous `AudioTrack` (`USAGE_MEDIA`) synthesizer—where tone frequency encodes mineral class (`320 / 520 / 780 / 1040 Hz`) and pulse cadence encodes physical stone diameter (`>= 20 mm`).

---

## Target Mineral Matrix (and Milestone 2 Indoor Stand-Ins)

| Tactical Slot | Target PNW Mineral Class | Reticle Color | Audio Ping | Milestone 2 MS-COCO Stand-In |
| :---: | :--- | :--- | :---: | :--- |
| **`01`** | **Chalcedony / Agate / Carnelian** | Cyan (`#00E5FF`) | `780 Hz` | `cup` (`class_id = 47`) |
| **`02`** | **Red & Yellow Jasper** | Gold (`#FFD740`) | `520 Hz` | `apple` (`class_id = 53`) |
| **`03`** | **Green Gem Rocks (Dallasite / Nephrite Jade)** | Green (`#00E676`) | `1040 Hz` | `toothbrush` (`class_id = 90`) |
| **`04`** | **Petrified Wood** | Orange (`#FF9100`) | `320 Hz` | `remote` (`class_id = 75`) |
| **`—`** | **Hard Confusers** *(Wet basalt, quartz veins, foam, glare)* | *Muted (Suppressed)* | *Silent* | All other confident classes |

---

## Loading the Project in Android Studio

> [!IMPORTANT]
> The Android Gradle project lives in the **`android/`** subdirectory, not the repository root.

1. Launch **Android Studio** (Ladybug / Meerkat or newer with Android SDK 36 installed).
2. Select **File → Open...** and choose the **`android/`** directory (`/path/to/gneissfind/android`), **not** the top-level `gneissfind/` folder.
3. Let Gradle sync complete. Android Studio will configure the `:app` module automatically.
4. Connect an ARCore-capable Android device (e.g., Pixel 10 Pro XL) with USB Debugging enabled, select the **`app`** run configuration, and click **Run (`Shift+F10`)**.

---

## Building & Running from the Command Line

### 1. Android App (`android/`)
You can run all Gradle tasks from the repository root using `-p android`:

```bash
# Assemble the debug APK (outputs to android/app/build/outputs/apk/debug/app-debug.apk)
./android/gradlew -p android assembleDebug

# Run the JVM unit test suite (cropper, cross-tile IoM seam merge, Top-K bridge, voter, sonifier)
./android/gradlew -p android testDebugUnitTest

# Build and install directly onto a connected Android device via ADB
./android/gradlew -p android installDebug
```

### 2. Python ML Export & Verification Pipeline (`ml-pipeline/`)
The model export and PyTorch-vs-LiteRT parity suite uses [`uv`](https://docs.astral.sh/uv/) for deterministic Python dependency management (`Python 3.11`, `torch`, `rfdetr`, `litert-torch`, `ai-edge-litert`):

```bash
cd ml-pipeline

# Run the PyTorch-vs-LiteRT numerical parity and COCO class-index oracle test suite
CUDA_VISIBLE_DEVICES=0 uv run pytest -v

# Re-stage the active 2-Graph B=5 .tflite assets and regenerate rf_detr_coco_metadata.json
CUDA_VISIBLE_DEVICES=0 uv run python scripts/export_rfdetr_litert.py \
  --active-topology two_graph \
  --active-batch 5 \
  --skip-export
```

### 3. On-Device LiteRT GPU Benchmark Harness (`scripts/`)
With a physical device connected via `adb`:

```bash
./scripts/benchmark_pixel10.sh --warmup 2 --sweeps 12 --cooldown 8
```

---

## So You'd Like to Start Working on This?

Start here:
- **[`docs/INTERNALS.md`](docs/INTERNALS.md)** — **The Developer Onboarding & Architecture Guide.** Covers the full sensor-to-audio dataflow, how to switch detector/graph configurations (build-time, in-app UI, and `adb` hot-swap), how to run the test suites, critical Camera2/OpenCL invariants, and an *"If you want to do X, look in file Y at class Z"* navigation map.
- **[`docs/ENGINEERING_LEDGER.md`](docs/ENGINEERING_LEDGER.md)** — **The Engineering & Validation Ledger.** Append-only record of hardware postmortems, `LITERT_CL` OpenCL compiler discoveries, and Pixel 10 Pro XL benchmark tables.
- **[`docs/ROADMAP.md`](docs/ROADMAP.md)** — **The Execution Roadmap.** Milestone definitions and acceptance criteria from Milestone 1 through Milestone 6.
- **[`docs/COMMENTING.md`](docs/COMMENTING.md)** — **Code Commenting Standard.** Purpose-first documentation conventions used across all Kotlin and Python files.

---

## License

### Source Code — MIT License
All original **GneissFind** source code, scripts, and documentation in this repository (`android/`, `ml-pipeline/`, `scripts/`, `docs/`) are licensed under the **MIT License** (see [`LICENSE`](LICENSE)).

### Third-Party Models & Assets
The detection model weights and bundled typography carry their own permissive open-source licenses:

| Component | Upstream Source | License | Notes |
| :--- | :--- | :--- | :--- |
| **GneissFind Codebase** | This repository | **MIT License** | All Kotlin, Python, and shell code. |
| **`RF-DETR-Nano` Architecture & Weights** | [`roboflow/rf-detr`](https://github.com/roboflow/rf-detr) | **Apache License 2.0** | Nano/Small/Medium variants are Apache-2.0 (`RF-DETR-Large` is PML-1.0 and is *not* distributed in the app). |
| **Windowed `DINOv2` Backbone** | [`facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2) | **Apache License 2.0** | Vision Transformer backbone inside `RF-DETR-Nano`. |
| **`Antonio` Font Family** | Vernon Adams / Google Fonts (`res/font/antonio_*.ttf`) | **SIL Open Font License 1.1 (OFL-1.1)** | Bundled for offline field operation without network font providers. |

> *Disclaimer: GneissFind is an independent open-source geological tool. Its retro-futuristic interface is an affectionate homage to classic 1990s television graphic design; this project is not affiliated with, endorsed by, or sponsored by Paramount Global or CBS Studios.*
