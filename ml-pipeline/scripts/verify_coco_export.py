#!/usr/bin/env python3
"""Standalone COCO stand-in verification and operator audit CLI for ``RF-DETR-Nano``.

Loads the exported 1-graph and 2-graph ``.tflite`` models in ``ml-pipeline/artifacts/``, inspects
their TFLite operator histograms, runs PyTorch-vs-LiteRT numerical and bounding-box IoU parity
checks across ``cup``, ``apple``, ``toothbrush``, ``remote``, and ``mixed_confusers`` fixtures,
and writes/verifies ``android/app/src/main/assets/rf_detr_coco_metadata.json``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ai_edge_litert.interpreter import Interpreter

from gneissfind_ml_pipeline.coco_oracle import (
    HARD_CONFUSER_STONE_CLASS_ID,
    build_coco_metadata,
    decode_tile_predictions,
    load_nhwc_rgb_01,
    run_single_graph_tflite,
    run_two_graph_tflite,
    verify_stand_in_fixtures,
    write_coco_metadata,
)
from gneissfind_ml_pipeline.wrappers import (
    DEFAULT_INPUT_RESOLUTION,
    FusedRfDetrSingleGraph,
    load_prepared_lwdetr,
    pin_cuda_device_zero,
)

ML_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_PIPELINE_ROOT.parent
DEFAULT_ARTIFACTS_DIR = ML_PIPELINE_ROOT / "artifacts"
DEFAULT_FIXTURES_DIR = ML_PIPELINE_ROOT / "tests" / "fixtures"
DEFAULT_METADATA_PATH = (
    REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "rf_detr_coco_metadata.json"
)


def summarize_tflite_ops(model_path: Path) -> Counter[str]:
    """Counts TFLite built-in and delegate operators inside ``model_path``.

    Args:
        model_path: Path to an exported ``.tflite`` flatbuffer.

    Returns:
        ``Counter`` mapping operator name (for example ``"CONV_2D"``) to node count.
    """
    interp = Interpreter(model_path=str(model_path))
    return Counter(op["op_name"] for op in interp._get_ops_details())


def parse_args() -> argparse.Namespace:
    """Parses CLI options for ``verify_coco_export.py``.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES_DIR)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--active-topology", choices=("single_graph", "two_graph"), default="two_graph")
    parser.add_argument("--active-batch", type=int, choices=(1, 5), default=5)
    return parser.parse_args()


def main() -> None:
    """Runs 1-graph and 2-graph parity checks, confuser check, and metadata emission."""
    args = parse_args()
    pin_cuda_device_zero()

    lwdetr, eager_detector = load_prepared_lwdetr(resolution=DEFAULT_INPUT_RESOLUTION)
    pt_wrapper = FusedRfDetrSingleGraph(lwdetr).eval()

    p_1g = args.artifacts_dir / "rf_detr_nano_1g_b1.tflite"
    p_p1 = args.artifacts_dir / "rf_detr_nano_part1_b1.tflite"
    p_p2 = args.artifacts_dir / "rf_detr_nano_part2_b1.tflite"

    for label, path in (("1-Graph B=1", p_1g), ("2-Graph Part1 B=1", p_p1), ("2-Graph Part2 B=1", p_p2)):
        ops = summarize_tflite_ops(path)
        print(
            f"[ops] {label:18s}: total={sum(ops.values()):4d}, "
            f"TOPK_V2={ops['TOPK_V2']}, GATHER_ND={ops['GATHER_ND']}"
        )

    interp_1g = Interpreter(model_path=str(p_1g), num_threads=4)
    interp_1g.allocate_tensors()
    reports_1g = verify_stand_in_fixtures(
        pytorch_wrapper=pt_wrapper,
        eager_detector=eager_detector,
        tflite_runner=lambda b: run_single_graph_tflite(interp_1g, b),
        fixtures_dir=args.fixtures_dir,
    )

    interp_p1 = Interpreter(model_path=str(p_p1), num_threads=4)
    interp_p2 = Interpreter(model_path=str(p_p2), num_threads=4)
    interp_p1.allocate_tensors()
    interp_p2.allocate_tensors()
    reports_2g = verify_stand_in_fixtures(
        pytorch_wrapper=pt_wrapper,
        eager_detector=eager_detector,
        tflite_runner=lambda b: run_two_graph_tflite(interp_p1, interp_p2, b),
        fixtures_dir=args.fixtures_dir,
    )

    reports = reports_2g if args.active_topology == "two_graph" else reports_1g
    metadata = build_coco_metadata(
        reports,
        batch_size=args.active_batch,
        topology=args.active_topology,
    )

    # Verify HARD_CONFUSER mapping on mixed household scene (cats + remotes + couch).
    class_map = {
        entry["class_id"]: entry["stone_class_id"]
        for entry in metadata["stand_in_map"].values()
    }
    mixed_nhwc = load_nhwc_rgb_01(args.fixtures_dir / "sample_mixed_confusers.jpg")
    boxes_m, scores_m = run_two_graph_tflite(interp_p1, interp_p2, mixed_nhwc)
    decoded = decode_tile_predictions(boxes_m[0], scores_m[0], class_id_to_stone_id=class_map)
    confusers = [d for d in decoded if d.stone_class_id == HARD_CONFUSER_STONE_CLASS_ID]
    remotes = [d for d in decoded if d.coco_label == "remote"]
    assert confusers and remotes, "Expected both remote target and muted HARD_CONFUSER in mixed scene"
    print(
        f"[confuser_check] Mixed scene decoded {len(remotes)} PETRIFIED_WOOD (remote) "
        f"and {len(confusers)} muted HARD_CONFUSER ({[c.coco_label for c in confusers]})"
    )

    write_coco_metadata(metadata, args.metadata_out)
    print(f"[verify_coco_export] Verified and wrote {args.metadata_out}")


if __name__ == "__main__":
    main()
