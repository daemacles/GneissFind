"""Pytest dual-scale (`120-200 px` medium + `28-45 px` small pebble) `1792x384` strip recall oracle.

Verifies that the deterministic evaluation strips (`eval_strip_medium_1792x384.png` and
`eval_strip_small_1792x384.png`) are generated reproducibly and evaluates baseline `RF-DETR-Nano`
(`5x 384x384`) on both medium and small pebble-scale targets across the full `1792 px` swath.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from gneissfind_ml_pipeline.coco_oracle import cxcywh_norm_to_xyxy_px, decode_tile_predictions
from gneissfind_ml_pipeline.dual_oracle import (
    FIVE_TILE_STRIP_X_OFFSETS,
    SOURCE_OBJECT_SPECS,
    StripBoxPrediction,
    build_or_load_dual_oracle_strips,
    evaluate_strip_detector,
    slice_strip_into_5x384_tiles,
)
from gneissfind_ml_pipeline.wrappers import FusedRfDetrSingleGraph, load_prepared_lwdetr

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
CLASS_ID_TO_STONE_ID = {
    spec.coco_class_id: spec.stone_class_id for spec in SOURCE_OBJECT_SPECS.values()
}


def make_5x384_pytorch_strip_predictor(
    wrapper: torch.nn.Module,
    score_threshold: float = 0.10,
):
    """Wraps a `(B, 384, 384, 3)` PyTorch detector into a `1792x384` 5-tile strip predictor."""

    def _predict_strip(strip_hwc_01: NDArray[np.float32]) -> list[StripBoxPrediction]:
        batch = slice_strip_into_5x384_tiles(strip_hwc_01)
        with torch.no_grad():
            boxes_t, scores_t = wrapper(torch.from_numpy(batch))
        boxes_np = boxes_t.numpy()
        scores_np = scores_t.numpy()
        out: list[StripBoxPrediction] = []
        for tile_idx, x_offset in enumerate(FIVE_TILE_STRIP_X_OFFSETS):
            dets = decode_tile_predictions(
                boxes_cxcywh=boxes_np[tile_idx],
                scores=scores_np[tile_idx],
                class_id_to_stone_id=CLASS_ID_TO_STONE_ID,
                score_threshold=score_threshold,
                resolution=384,
            )
            for d in dets:
                if d.stone_class_id == 4:
                    continue
                l, t, r, b = d.xyxy_px
                out.append(
                    StripBoxPrediction(
                        stone_class_id=d.stone_class_id,
                        score=d.score,
                        strip_xyxy_px=(l + x_offset, t, r + x_offset, b),
                    )
                )
        return out

    return _predict_strip


def test_dual_oracle_strip_generation_and_baseline_rfdetr_nano_recall() -> None:
    """Verifies that baseline RF-DETR-Nano (5x 384x384) achieves 100% recall on medium and 38-45px targets."""
    med_strip, med_targets, small_strip, small_targets = build_or_load_dual_oracle_strips(
        FIXTURES_DIR, force_rebuild=True
    )
    assert med_strip.shape == (384, 1792, 3)
    assert small_strip.shape == (384, 1792, 3)
    assert len(med_targets) == 4
    assert len(small_targets) == 12

    lwdetr, _ = load_prepared_lwdetr(resolution=384)
    wrapper = FusedRfDetrSingleGraph(lwdetr).eval()
    summary = evaluate_strip_detector(
        candidate_name="RF-DETR-Nano 5x384x384 (Baseline)",
        predict_strip_fn=make_5x384_pytorch_predictor_fn(wrapper),
        medium_strip_01=med_strip,
        medium_targets=med_targets,
        small_strip_01=small_strip,
        small_targets=small_targets,
    )
    print(f"\n[DUAL ORACLE BASELINE] {summary.candidate_name}:")
    print(
        f"  Medium (120-200px): recall@0.35={summary.medium_recall_035:.2%}, "
        f"mean_score={summary.medium_mean_score:.3f}"
    )
    print(
        f"  Small (38-45px)   : recall@0.35={summary.small_38_45_recall_035:.2%}, "
        f"recall@0.25={summary.small_38_45_recall_025:.2%}, "
        f"mean_score={summary.small_38_45_mean_score:.3f}"
    )
    print(
        f"  Small (28-45px)   : recall@0.25={summary.small_28_45_recall_025:.2%}, "
        f"mean_score={summary.small_28_45_mean_score:.3f}"
    )
    for tid, (sc, iou) in summary.per_target_scores.items():
        print(f"    {tid:22s}: score={sc:.3f}, IoU={iou:.3f}")

    assert summary.medium_recall_035 == 1.0
    assert summary.small_38_45_recall_035 >= 0.875


def make_5x384_pytorch_predictor_fn(wrapper: torch.nn.Module):
    """Alias helper returning the 5x384 predictor closure."""
    return make_5x384_pytorch_strip_predictor(wrapper, score_threshold=0.10)
