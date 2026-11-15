"""Model export, verification, and COCO stand-in metadata tooling for GneissFind.

Consumes pretrained or fine-tuned ``rfdetr`` checkpoints on the workstation and produces
static-shape ``.tflite`` graphs plus ``rf_detr_coco_metadata.json`` consumed by the Android
LiteRT GPU detector (`RfDetrTwoGraphGpuDetector`).
"""

from gneissfind_ml_pipeline.coco_oracle import (
    STAND_IN_SPECS,
    StandInClassSpec,
    build_coco_metadata,
    verify_stand_in_fixtures,
)
from gneissfind_ml_pipeline.wrappers import (
    FusedRfDetrSingleGraph,
    RfDetrPart1BackboneEncoder,
    RfDetrPart2DecoderHead,
    bilinear_matmul_grid_sample,
    host_select_topk_queries,
    load_prepared_lwdetr,
    pin_cuda_device_zero,
)

__all__ = [
    "FusedRfDetrSingleGraph",
    "RfDetrPart1BackboneEncoder",
    "RfDetrPart2DecoderHead",
    "STAND_IN_SPECS",
    "StandInClassSpec",
    "bilinear_matmul_grid_sample",
    "build_coco_metadata",
    "host_select_topk_queries",
    "load_prepared_lwdetr",
    "pin_cuda_device_zero",
    "verify_stand_in_fixtures",
]
