# GneissFind ML Pipeline (`ml-pipeline/`)

Python `uv` workspace for Milestone 2 (`RF-DETR-Nano` `384x384` MS-COCO LiteRT GPU FP16 export,
operator optimization, and empirical household stand-in verification) and Milestones 3–4 (SAM 2
auto-labeling and PNW mineral fine-tuning).

## Commands

Export all 4 benchmark `.tflite` graphs (`1-Graph` & `2-Graph` x `B=1` & `B=5`), stage the active
model into `android/app/src/main/assets/`, and write `rf_detr_coco_metadata.json`:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/export_rfdetr_litert.py
```

Re-stage a different topology/batch into `android/app/src/main/assets/` without re-tracing:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/export_rfdetr_litert.py \
  --skip-export --active-topology single_graph --active-batch 5
```

Run standalone COCO stand-in parity & operator verification:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/verify_coco_export.py
```

Run the offline `pytest` verification suite:

```bash
CUDA_VISIBLE_DEVICES=0 uv run pytest -v
```
