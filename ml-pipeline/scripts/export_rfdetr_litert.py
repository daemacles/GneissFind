#!/usr/bin/env python3
"""Exports ``RF-DETR-Nano`` (384x384 MS-COCO) to 1-graph and 2-graph LiteRT ``.tflite`` artifacts.

Produces the four candidate ``.tflite`` binaries required by the Milestone 2 benchmark matrix
(`.agents/rules/litert-inference.md` §3):

- ``rf_detr_nano_1g_b1.tflite`` and ``rf_detr_nano_1g_b5.tflite`` (single-graph)
- ``rf_detr_nano_part1_b1.tflite`` + ``rf_detr_nano_part2_b1.tflite`` (two-graph, ``B=1``)
- ``rf_detr_nano_part1_b5.tflite`` + ``rf_detr_nano_part2_b5.tflite`` (two-graph, ``B=5``)

Stages the selected active configuration into ``android/app/src/main/assets/`` and runs the
empirical MS-COCO stand-in oracle to generate ``rf_detr_coco_metadata.json``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import litert_torch
import torch
from ai_edge_litert.interpreter import Interpreter

from gneissfind_ml_pipeline.coco_oracle import (
    build_coco_metadata,
    run_single_graph_tflite,
    run_two_graph_tflite,
    verify_stand_in_fixtures,
    write_coco_metadata,
)
from gneissfind_ml_pipeline.wrappers import (
    DEFAULT_INPUT_RESOLUTION,
    ENCODER_GRID_HW,
    FusedRfDetrSingleGraph,
    RfDetrPart1BackboneEncoder,
    RfDetrPart2DecoderHead,
    load_prepared_lwdetr,
    pin_cuda_device_zero,
)

ML_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_PIPELINE_ROOT.parent
DEFAULT_ARTIFACTS_DIR = ML_PIPELINE_ROOT / "artifacts"
DEFAULT_FIXTURES_DIR = ML_PIPELINE_ROOT / "tests" / "fixtures"
DEFAULT_ANDROID_ASSETS_DIR = REPO_ROOT / "android" / "app" / "src" / "main" / "assets"


def export_single_graph_variant(
    lwdetr: torch.nn.Module,
    batch_size: int,
    output_path: Path,
) -> Path:
    """Traces and exports :class:`FusedRfDetrSingleGraph` for ``batch_size`` to ``output_path``.

    Args:
        lwdetr: Prepared export-mode ``LWDETR`` module from :func:`load_prepared_lwdetr`.
        batch_size: Static batch dimension ``B`` baked into the ``.tflite`` signature.
        output_path: Destination ``.tflite`` path.

    Returns:
        Written ``output_path``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = FusedRfDetrSingleGraph(lwdetr).eval()
    sample_nhwc = torch.rand(
        batch_size, DEFAULT_INPUT_RESOLUTION, DEFAULT_INPUT_RESOLUTION, 3, dtype=torch.float32
    )
    with torch.no_grad():
        edge_model = litert_torch.convert(wrapper, (sample_nhwc,))
        edge_model.export(str(output_path))
    return output_path


def export_two_graph_variant(
    lwdetr: torch.nn.Module,
    batch_size: int,
    part1_path: Path,
    part2_path: Path,
) -> tuple[Path, Path]:
    """Traces and exports ``Part 1`` and ``Part 2`` ``.tflite`` files for ``batch_size``.

    Args:
        lwdetr: Prepared export-mode ``LWDETR`` module from :func:`load_prepared_lwdetr`.
        batch_size: Static batch dimension ``B`` baked into both ``.tflite`` signatures.
        part1_path: Destination ``.tflite`` path for :class:`RfDetrPart1BackboneEncoder`.
        part2_path: Destination ``.tflite`` path for :class:`RfDetrPart2DecoderHead`.

    Returns:
        Tuple ``(part1_path, part2_path)``.
    """
    part1_path.parent.mkdir(parents=True, exist_ok=True)
    part2_path.parent.mkdir(parents=True, exist_ok=True)

    part1 = RfDetrPart1BackboneEncoder(lwdetr).eval()
    part2 = RfDetrPart2DecoderHead(
        lwdetr, resolution=DEFAULT_INPUT_RESOLUTION, grid_hw=ENCODER_GRID_HW
    ).eval()

    num_queries = int(lwdetr.num_queries)
    sample_nhwc = torch.rand(
        batch_size, DEFAULT_INPUT_RESOLUTION, DEFAULT_INPUT_RESOLUTION, 3, dtype=torch.float32
    )
    with torch.no_grad():
        sample_memory, sample_prenorm, _ = part1(sample_nhwc)
        sample_selected_prenorm = sample_prenorm[:, :num_queries, :].contiguous()
        sample_selected_proposals = torch.full(
            (batch_size, num_queries, 4), 0.25, dtype=torch.float32
        )

        edge_p1 = litert_torch.convert(part1, (sample_nhwc,))
        edge_p1.export(str(part1_path))

        edge_p2 = litert_torch.convert(
            part2, (sample_memory, sample_selected_prenorm, sample_selected_proposals)
        )
        edge_p2.export(str(part2_path))

    return part1_path, part2_path


def stage_active_assets(
    artifacts_dir: Path,
    assets_dir: Path,
    active_topology: str,
    active_batch: int,
) -> list[Path]:
    """Copies the chosen ``.tflite`` artifact(s) into ``android/app/src/main/assets/``.

    Removes stale ``.tflite`` files from the other topology so debug APKs bundle only one ~105 MB
    weight set at a time.

    Args:
        artifacts_dir: Directory containing the exported benchmark ``.tflite`` candidates.
        assets_dir: Android ``src/main/assets/`` directory.
        active_topology: ``"single_graph"`` or ``"two_graph"``.
        active_batch: ``1`` or ``5``.

    Returns:
        List of destination ``.tflite`` paths staged inside ``assets_dir``.
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
    single_dst = assets_dir / "rf_detr_nano_coco.tflite"
    part1_dst = assets_dir / "rf_detr_nano_part1_backbone_encoder.tflite"
    part2_dst = assets_dir / "rf_detr_nano_part2_decoder_head.tflite"

    staged: list[Path] = []
    if active_topology == "single_graph":
        src = artifacts_dir / f"rf_detr_nano_1g_b{active_batch}.tflite"
        shutil.copy2(src, single_dst)
        for obsolete in (part1_dst, part2_dst):
            obsolete.unlink(missing_ok=True)
        staged.append(single_dst)
    elif active_topology == "two_graph":
        src_p1 = artifacts_dir / f"rf_detr_nano_part1_b{active_batch}.tflite"
        src_p2 = artifacts_dir / f"rf_detr_nano_part2_b{active_batch}.tflite"
        shutil.copy2(src_p1, part1_dst)
        shutil.copy2(src_p2, part2_dst)
        single_dst.unlink(missing_ok=True)
        staged.extend([part1_dst, part2_dst])
    else:
        raise ValueError(f"Unsupported active_topology={active_topology!r}")
    return staged


def parse_args() -> argparse.Namespace:
    """Parses CLI options for exporting and staging RF-DETR-Nano LiteRT artifacts.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Output directory for the 4 benchmark .tflite candidates.",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ANDROID_ASSETS_DIR,
        help="Android assets directory for the active model and rf_detr_coco_metadata.json.",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_FIXTURES_DIR,
        help="Directory holding 384x384 stand-in verification images.",
    )
    parser.add_argument(
        "--active-topology",
        choices=("single_graph", "two_graph"),
        default="two_graph",
        help="Graph topology staged into Android assets/.",
    )
    parser.add_argument(
        "--active-batch",
        type=int,
        choices=(1, 5),
        default=5,
        help="Static batch size staged into Android assets/.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse existing .tflite files in --artifacts-dir and only re-stage + verify.",
    )
    return parser.parse_args()


def main() -> None:
    """Exports the 4-way LiteRT matrix, verifies COCO parity, and writes Android metadata."""
    args = parse_args()
    selected_device = pin_cuda_device_zero()
    print(f"[export_rfdetr_litert] CUDA_VISIBLE_DEVICES=0 pinned (probe device={selected_device})")

    lwdetr, eager_detector = load_prepared_lwdetr(resolution=DEFAULT_INPUT_RESOLUTION)

    if not args.skip_export:
        for batch in (1, 5):
            out_1g = args.artifacts_dir / f"rf_detr_nano_1g_b{batch}.tflite"
            print(f"[export_rfdetr_litert] Exporting 1-Graph B={batch} -> {out_1g}")
            export_single_graph_variant(lwdetr, batch_size=batch, output_path=out_1g)

            out_p1 = args.artifacts_dir / f"rf_detr_nano_part1_b{batch}.tflite"
            out_p2 = args.artifacts_dir / f"rf_detr_nano_part2_b{batch}.tflite"
            print(f"[export_rfdetr_litert] Exporting 2-Graph B={batch} -> {out_p1.name}, {out_p2.name}")
            export_two_graph_variant(lwdetr, batch_size=batch, part1_path=out_p1, part2_path=out_p2)

    staged = stage_active_assets(
        artifacts_dir=args.artifacts_dir,
        assets_dir=args.assets_dir,
        active_topology=args.active_topology,
        active_batch=args.active_batch,
    )
    print(f"[export_rfdetr_litert] Staged Android assets: {[p.name for p in staged]}")

    pt_wrapper = FusedRfDetrSingleGraph(lwdetr).eval()
    if args.active_topology == "single_graph":
        interp_1g = Interpreter(
            model_path=str(args.artifacts_dir / "rf_detr_nano_1g_b1.tflite"), num_threads=4
        )
        interp_1g.allocate_tensors()
        runner = lambda batch: run_single_graph_tflite(interp_1g, batch)
    else:
        interp_p1 = Interpreter(
            model_path=str(args.artifacts_dir / "rf_detr_nano_part1_b1.tflite"), num_threads=4
        )
        interp_p2 = Interpreter(
            model_path=str(args.artifacts_dir / "rf_detr_nano_part2_b1.tflite"), num_threads=4
        )
        interp_p1.allocate_tensors()
        interp_p2.allocate_tensors()
        runner = lambda batch: run_two_graph_tflite(interp_p1, interp_p2, batch)

    reports = verify_stand_in_fixtures(
        pytorch_wrapper=pt_wrapper,
        eager_detector=eager_detector,
        tflite_runner=runner,
        fixtures_dir=args.fixtures_dir,
    )
    for r in reports:
        print(
            f"  [VERIFIED] {r.stone_class_name:18s} <- {r.coco_label:10s} "
            f"(class_id={r.empirical_class_id:2d}, pt={r.pytorch_top_score:.3f}, "
            f"tflite={r.tflite_top_score:.3f}, IoU={r.top_box_iou:.4f}, "
            f"max_diff={r.max_score_abs_diff:.6f})"
        )

    metadata = build_coco_metadata(
        reports,
        batch_size=args.active_batch,
        topology=args.active_topology,
    )
    meta_path = write_coco_metadata(metadata, args.assets_dir / "rf_detr_coco_metadata.json")
    print(f"[export_rfdetr_litert] Wrote verified metadata -> {meta_path}")


if __name__ == "__main__":
    main()
