"""Empirical MS-COCO stand-in class oracle and PyTorch-vs-LiteRT parity verification.

``rfdetr`` checkpoints vary between 80 contiguous and 91 sparse MS-COCO category indices (for
example, index 47 is ``apple`` in 80-class contiguous layout but ``cup`` in 91-class sparse layout;
see `.agents/rules/litert-inference.md` §4). This module runs both PyTorch and exported ``.tflite``
graphs on real ``384x384`` fixture images of ``cup``, ``apple``, ``toothbrush``, and ``remote``,
verifies numerical and bounding-box IoU parity, empirically confirms the integer class IDs, and
writes ``rf_detr_coco_metadata.json`` for Android consumption.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter
from numpy.typing import NDArray
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES

from gneissfind_ml_pipeline.wrappers import (
    DEFAULT_INPUT_RESOLUTION,
    DEFAULT_PROPOSAL_WH,
    ENCODER_GRID_HW,
    IMAGENET_MEAN_RGB,
    IMAGENET_STD_RGB,
    FusedRfDetrSingleGraph,
    host_select_topk_queries,
)

HARD_CONFUSER_STONE_CLASS_ID: int = 4
DEFAULT_SCORE_THRESHOLD: float = 0.35


@dataclass(frozen=True, slots=True)
class StandInClassSpec:
    """Mapping specification between a GneissFind ``StoneClass`` and its Milestone 2 COCO proxy.

    Attributes:
        stone_class_name: Enum constant name in Kotlin ``StoneClass``.
        stone_class_id: Ordinal ``id`` in Kotlin ``StoneClass`` (``0..3``).
        coco_label: Expected human-readable MS-COCO label string in ``COCO_CLASSES``.
        fixture_filename: Filename under ``tests/fixtures/`` containing a clean sample of the item.
    """

    stone_class_name: str
    stone_class_id: int
    coco_label: str
    fixture_filename: str


STAND_IN_SPECS: tuple[StandInClassSpec, ...] = (
    StandInClassSpec(
        stone_class_name="CHALCEDONY_AGATE",
        stone_class_id=0,
        coco_label="cup",
        fixture_filename="sample_cup.jpg",
    ),
    StandInClassSpec(
        stone_class_name="JASPER_RED_YELLOW",
        stone_class_id=1,
        coco_label="apple",
        fixture_filename="sample_apple.jpg",
    ),
    StandInClassSpec(
        stone_class_name="GREEN_GEM_ROCK",
        stone_class_id=2,
        coco_label="toothbrush",
        fixture_filename="sample_toothbrush.jpg",
    ),
    StandInClassSpec(
        stone_class_name="PETRIFIED_WOOD",
        stone_class_id=3,
        coco_label="remote",
        fixture_filename="sample_remote.jpg",
    ),
)


@dataclass(frozen=True, slots=True)
class DecodedDetection:
    """Single decoded tile prediction in ``[0, resolution]`` pixel ``xyxy`` coordinates.

    Attributes:
        coco_class_id: Raw integer index in the model's ``num_classes`` output axis.
        coco_label: Resolved COCO class name, or ``"unmapped"`` when absent from ``COCO_CLASSES``.
        stone_class_id: Mapped GneissFind ``StoneClass.id`` (``0..3`` for targets, ``4`` confuser).
        score: Sigmoid probability in ``[0.0, 1.0]``.
        xyxy_px: ``(left, top, right, bottom)`` bounding box in tile pixel coordinates.
    """

    coco_class_id: int
    coco_label: str
    stone_class_id: int
    score: float
    xyxy_px: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FixtureParityReport:
    """Empirical verification result for one ``384x384`` household stand-in fixture image.

    Attributes:
        stone_class_name: Target Kotlin ``StoneClass`` enum name.
        coco_label: Verified COCO label string.
        empirical_class_id: Integer class index activated on the fixture by both PyTorch and LiteRT.
        pytorch_top_score: Highest confidence score for ``empirical_class_id`` from PyTorch.
        tflite_top_score: Highest confidence score for ``empirical_class_id`` from LiteRT.
        max_score_abs_diff: Worst-case ``|pt_scores - tflite_scores|`` across all ``300 x 91`` entries.
        max_box_abs_diff: Worst-case ``|pt_boxes - tflite_boxes|`` across all ``300 x 4`` entries.
        top_box_iou: Pixel-space ``xyxy`` IoU between PyTorch and LiteRT top-1 boxes for the class.
    """

    stone_class_name: str
    coco_label: str
    empirical_class_id: int
    pytorch_top_score: float
    tflite_top_score: float
    max_score_abs_diff: float
    max_box_abs_diff: float
    top_box_iou: float


def load_nhwc_rgb_01(image_path: Path, resolution: int = DEFAULT_INPUT_RESOLUTION) -> NDArray[np.float32]:
    """Loads an image file into ``(1, resolution, resolution, 3)`` ``NHWC`` float32 ``[0.0, 1.0]``.

    Matches ``PushBroomCropper``'s output layout and value range on Android.

    Args:
        image_path: Path to an RGB-convertible image file on disk.
        resolution: Target square edge size in pixels.

    Returns:
        Contiguous float32 array of shape ``(1, resolution, resolution, 3)`` with values in
        ``[0.0, 1.0]``.
    """
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        if rgb.size != (resolution, resolution):
            rgb = rgb.resize((resolution, resolution), Image.Resampling.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
    return np.ascontiguousarray(arr[None, ...])


def cxcywh_norm_to_xyxy_px(
    box_cxcywh: NDArray[np.floating[Any]],
    resolution: int = DEFAULT_INPUT_RESOLUTION,
) -> tuple[float, float, float, float]:
    """Converts one normalized ``(cx, cy, w, h)`` box into clamped ``(left, top, right, bottom)`` px.

    Args:
        box_cxcywh: 1D array of 4 normalized floats ``(cx, cy, w, h)``.
        resolution: Tile width and height in pixels.

    Returns:
        Clamped ``(left, top, right, bottom)`` tuple in ``[0.0, float(resolution)]`` pixels.
    """
    scale = float(resolution)
    cx = float(box_cxcywh[0]) * scale
    cy = float(box_cxcywh[1]) * scale
    w = float(box_cxcywh[2]) * scale
    h = float(box_cxcywh[3]) * scale
    left = max(0.0, min(scale, cx - 0.5 * w))
    top = max(0.0, min(scale, cy - 0.5 * h))
    right = max(0.0, min(scale, cx + 0.5 * w))
    bottom = max(0.0, min(scale, cy + 0.5 * h))
    return left, top, right, bottom


def compute_xyxy_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Computes intersection-over-union between two ``(left, top, right, bottom)`` boxes.

    Args:
        box_a: First bounding box ``(left, top, right, bottom)`` in pixels.
        box_b: Second bounding box ``(left, top, right, bottom)`` in pixels.

    Returns:
        IoU ratio in ``[0.0, 1.0]``; returns ``0.0`` when union area is non-positive.
    """
    inter_left = max(box_a[0], box_b[0])
    inter_top = max(box_a[1], box_b[1])
    inter_right = min(box_a[2], box_b[2])
    inter_bottom = min(box_a[3], box_b[3])
    inter_w = max(0.0, inter_right - inter_left)
    inter_h = max(0.0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


def decode_tile_predictions(
    boxes_cxcywh: NDArray[np.float32],
    scores: NDArray[np.float32],
    class_id_to_stone_id: dict[int, int],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    resolution: int = DEFAULT_INPUT_RESOLUTION,
) -> list[DecodedDetection]:
    """Decodes ``(Q, 4)`` normalized boxes and ``(Q, C)`` probabilities into GneissFind detections.

    Mirrors ``RfDetrTwoGraphGpuDetector``'s Android decoding rule: any query/class pair above
    ``score_threshold`` maps to ``class_id_to_stone_id[class_id]`` if present, or to
    ``HARD_CONFUSER_STONE_CLASS_ID`` (4) otherwise.

    Args:
        boxes_cxcywh: Normalized ``(cx, cy, w, h)`` array for one tile, shape ``(Q, 4)``.
        scores: Sigmoid class probability array for one tile, shape ``(Q, C)``.
        class_id_to_stone_id: Mapping from COCO ``class_id`` to ``StoneClass.id`` (``0..3``).
        score_threshold: Minimum probability required to emit a detection.
        resolution: Tile side length in pixels.

    Returns:
        Detections sorted by descending confidence score.
    """
    num_queries, num_classes = scores.shape
    flat_scores = scores.reshape(-1)
    passing = np.flatnonzero(flat_scores >= score_threshold)
    if passing.size == 0:
        return []

    order = passing[np.argsort(-flat_scores[passing], kind="stable")]
    decoded: list[DecodedDetection] = []
    for flat_idx in order:
        q_idx = int(flat_idx // num_classes)
        c_idx = int(flat_idx % num_classes)
        prob = float(flat_scores[flat_idx])
        stone_id = class_id_to_stone_id.get(c_idx, HARD_CONFUSER_STONE_CLASS_ID)
        decoded.append(
            DecodedDetection(
                coco_class_id=c_idx,
                coco_label=COCO_CLASSES.get(c_idx, "unmapped"),
                stone_class_id=stone_id,
                score=prob,
                xyxy_px=cxcywh_norm_to_xyxy_px(boxes_cxcywh[q_idx], resolution=resolution),
            )
        )
    return decoded


def run_single_graph_tflite(
    interpreter: Interpreter,
    nhwc_batch: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Executes a static-batch single-graph ``RF-DETR-Nano`` ``.tflite`` interpreter.

    Args:
        interpreter: Allocated ``ai_edge_litert.interpreter.Interpreter`` for the single-graph model.
        nhwc_batch: Float32 RGB input batch matching the graph's ``(B, 384, 384, 3)`` shape.

    Returns:
        Tuple ``(boxes_cxcywh, scores)`` with shapes ``(B, 300, 4)`` and ``(B, 300, 91)``.
    """
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()
    interpreter.set_tensor(in_details[0]["index"], np.ascontiguousarray(nhwc_batch, dtype=np.float32))
    interpreter.invoke()
    boxes = interpreter.get_tensor(out_details[0]["index"]).copy()
    scores = interpreter.get_tensor(out_details[1]["index"]).copy()
    return boxes, scores


def run_two_graph_tflite(
    part1_interpreter: Interpreter,
    part2_interpreter: Interpreter,
    nhwc_batch: NDArray[np.float32],
    top_k: int = 300,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Executes the two-graph ``Part 1 -> Host Top-K -> Part 2`` ``.tflite`` pipeline.

    Args:
        part1_interpreter: Allocated interpreter for ``RfDetrPart1BackboneEncoder``.
        part2_interpreter: Allocated interpreter for ``RfDetrPart2DecoderHead``.
        nhwc_batch: Float32 RGB input batch matching ``Part 1``'s ``(B, 384, 384, 3)`` shape.
        top_k: Number of top-scoring queries bridged from Stage 1 to Stage 2.

    Returns:
        Tuple ``(boxes_cxcywh, scores)`` with shapes ``(B, 300, 4)`` and ``(B, 300, 91)``.
    """
    p1_in = part1_interpreter.get_input_details()
    p1_out = part1_interpreter.get_output_details()
    part1_interpreter.set_tensor(p1_in[0]["index"], np.ascontiguousarray(nhwc_batch, dtype=np.float32))
    part1_interpreter.invoke()
    memory = part1_interpreter.get_tensor(p1_out[0]["index"])
    prenorm = part1_interpreter.get_tensor(p1_out[1]["index"])
    enc_max_scores = part1_interpreter.get_tensor(p1_out[2]["index"])

    selected_prenorm, selected_proposals, _ = host_select_topk_queries(
        output_memory_prenorm=prenorm,
        enc_max_scores=enc_max_scores,
        top_k=top_k,
    )

    p2_in = part2_interpreter.get_input_details()
    p2_out = part2_interpreter.get_output_details()
    part2_interpreter.set_tensor(p2_in[0]["index"], np.ascontiguousarray(memory, dtype=np.float32))
    part2_interpreter.set_tensor(p2_in[1]["index"], selected_prenorm)
    part2_interpreter.set_tensor(p2_in[2]["index"], selected_proposals)
    part2_interpreter.invoke()
    boxes = part2_interpreter.get_tensor(p2_out[0]["index"]).copy()
    scores = part2_interpreter.get_tensor(p2_out[1]["index"]).copy()
    return boxes, scores


def verify_stand_in_fixtures(
    pytorch_wrapper: FusedRfDetrSingleGraph,
    eager_detector: RFDETRNano,
    tflite_runner: Any,
    fixtures_dir: Path,
    min_box_iou: float = 0.85,
    max_score_tol: float = 1e-2,
    resolution: int = DEFAULT_INPUT_RESOLUTION,
    compare_class_max_scores: bool = False,
) -> list[FixtureParityReport]:
    """Empirically verifies COCO class IDs and PyTorch-vs-LiteRT parity on the 4 stand-in fixtures.

    Asserts on every fixture that:
    1. Both PyTorch and LiteRT activate the same top-1 integer ``class_id``.
    2. ``COCO_CLASSES[class_id]`` equals the expected household label (``cup``, ``apple``,
       ``toothbrush``, ``remote``), cross-checked against ``eager_detector.predict()``.
    3. ``|pt_scores - tflite_scores| <= max_score_tol`` across all query/class entries (or across
       permutation-invariant per-class maxima when ``compare_class_max_scores=True`` for quantized
       two-stage models where ``TopK`` rank ties can permute query indices).
    4. Pixel-space ``xyxy`` IoU between PyTorch and LiteRT top-1 boxes is ``>= min_box_iou``.

    Args:
        pytorch_wrapper: Eval-mode :class:`FusedRfDetrSingleGraph` instance.
        eager_detector: Unmodified ``RFDETRNano`` detector for ``predict()`` cross-checking.
        tflite_runner: Callable ``(nhwc_1xHxWx3) -> (boxes_cxcywh, scores)`` backed by LiteRT.
        fixtures_dir: Directory containing ``sample_cup.jpg``, ``sample_apple.jpg``,
            ``sample_toothbrush.jpg``, and ``sample_remote.jpg``.
        min_box_iou: Minimum required top-1 bounding-box IoU between PyTorch and LiteRT.
        max_score_tol: Maximum tolerated absolute probability difference.
        resolution: Square input resolution in pixels (``384`` default, ``320`` for Lock-On).
        compare_class_max_scores: When ``True``, measures max score delta over per-class max
            probabilities ``(91,)`` so ``W8-Dyn`` Stage-1 ``TopK`` rank tie swaps do not trigger
            spurious index-aligned diffs.

    Returns:
        Ordered list of :class:`FixtureParityReport`, one per entry in ``STAND_IN_SPECS``.

    Raises:
        AssertionError: If any parity bound or COCO class-label cross-check fails.
    """
    reports: list[FixtureParityReport] = []

    for spec in STAND_IN_SPECS:
        fixture_path = fixtures_dir / spec.fixture_filename
        if not fixture_path.is_file():
            raise FileNotFoundError(f"Missing stand-in verification fixture: {fixture_path}")

        nhwc = load_nhwc_rgb_01(fixture_path, resolution=resolution)
        with torch.no_grad():
            pt_boxes_t, pt_scores_t = pytorch_wrapper(torch.from_numpy(nhwc))
        pt_boxes = pt_boxes_t.numpy()[0]
        pt_scores = pt_scores_t.numpy()[0]

        tf_boxes_b, tf_scores_b = tflite_runner(nhwc)
        tf_boxes = tf_boxes_b[0]
        tf_scores = tf_scores_b[0]

        if compare_class_max_scores:
            max_score_diff = float(
                np.max(np.abs(np.max(pt_scores, axis=0) - np.max(tf_scores, axis=0)))
            )
        else:
            max_score_diff = float(np.max(np.abs(pt_scores - tf_scores)))
        confident_q = np.max(pt_scores, axis=-1) >= 0.05
        # Measure box delta on confident queries so rank-300 background TopK ties (~1e-3 score)
        # do not masquerade as coordinate drift.
        box_pool_pt = pt_boxes[confident_q] if np.any(confident_q) else pt_boxes
        box_pool_tf = tf_boxes[confident_q] if np.any(confident_q) else tf_boxes
        max_box_diff = float(np.max(np.abs(box_pool_pt - box_pool_tf)))
        assert max_score_diff <= max_score_tol, (
            f"{spec.coco_label}: max score diff {max_score_diff:.6f} > {max_score_tol}"
        )

        # Empirically discover the top-scoring (query, class_id) pair without assuming any ID.
        pt_flat_best = int(np.argmax(pt_scores))
        tf_flat_best = int(np.argmax(tf_scores))
        num_classes = pt_scores.shape[1]
        pt_q, pt_cid = divmod(pt_flat_best, num_classes)
        tf_q, tf_cid = divmod(tf_flat_best, num_classes)

        assert pt_cid == tf_cid, (
            f"{spec.coco_label}: PyTorch top class {pt_cid} != LiteRT top class {tf_cid}"
        )
        resolved_label = COCO_CLASSES.get(pt_cid)
        assert resolved_label == spec.coco_label, (
            f"Fixture {spec.fixture_filename} activated class_id={pt_cid} ({resolved_label!r}), "
            f"expected {spec.coco_label!r}"
        )

        # Cross-check against rfdetr's own high-level predict() pipeline on the PIL image.
        with Image.open(fixture_path) as pil_img:
            eager_preds = eager_detector.predict(pil_img.convert("RGB"), threshold=DEFAULT_SCORE_THRESHOLD)
        assert len(eager_preds.class_id) > 0 and int(eager_preds.class_id[0]) == pt_cid, (
            f"{spec.coco_label}: RFDETRNano.predict top class {eager_preds.class_id[:1]} != {pt_cid}"
        )

        pt_xyxy = cxcywh_norm_to_xyxy_px(pt_boxes[pt_q], resolution=resolution)
        tf_xyxy = cxcywh_norm_to_xyxy_px(tf_boxes[tf_q], resolution=resolution)
        iou = compute_xyxy_iou(pt_xyxy, tf_xyxy)
        assert iou >= min_box_iou, (
            f"{spec.coco_label}: top box IoU {iou:.4f} < {min_box_iou}"
        )

        reports.append(
            FixtureParityReport(
                stone_class_name=spec.stone_class_name,
                coco_label=spec.coco_label,
                empirical_class_id=pt_cid,
                pytorch_top_score=float(pt_scores[pt_q, pt_cid]),
                tflite_top_score=float(tf_scores[tf_q, tf_cid]),
                max_score_abs_diff=max_score_diff,
                max_box_abs_diff=max_box_diff,
                top_box_iou=iou,
            )
        )

    return reports


def build_coco_metadata(
    reports: list[FixtureParityReport],
    *,
    model_name: str = "rf_detr_nano_coco",
    input_resolution: int = DEFAULT_INPUT_RESOLUTION,
    encoder_grid_hw: tuple[int, int] = ENCODER_GRID_HW,
    quantization: str | None = None,
    batch_size: int = 5,
    topology: str = "two_graph",
    num_queries: int = 300,
    num_classes: int = 91,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    single_graph_asset: str = "rf_detr_nano_coco.tflite",
    part1_asset: str = "rf_detr_nano_part1_backbone_encoder.tflite",
    part2_asset: str = "rf_detr_nano_part2_decoder_head.tflite",
) -> dict[str, Any]:
    """Constructs the JSON-serializable COCO metadata payload from verified reports.

    Follows the schema specified in `.agents/rules/litert-inference.md` while adding explicit
    tensor layout, proposal grid, and asset filename metadata for the Kotlin GPU runner.

    Args:
        reports: Empirically verified :class:`FixtureParityReport` entries for all 4 target classes.
        model_name: Logical model identifier written into ``model_name``.
        input_resolution: Square input resolution in pixels (``384`` or ``320``).
        encoder_grid_hw: ``(height, width)`` of the encoder feature token grid.
        quantization: Optional weight quantization recipe string (e.g., ``"dynamic_wi8_afp32"``).
        batch_size: Default static tile batch size bundled into ``assets/`` (``1`` or ``5``).
        topology: Default bundled graph topology (``"single_graph"`` or ``"two_graph"``).
        num_queries: Decoder object query count per tile.
        num_classes: Classification output dimension per query.
        score_threshold: Default confidence floor in ``[0.0, 1.0]``.
        single_graph_asset: Asset filename when ``topology == "single_graph"``.
        part1_asset: Stage 1 asset filename when ``topology == "two_graph"``.
        part2_asset: Stage 2 asset filename when ``topology == "two_graph"``.

    Returns:
        Dictionary ready for ``json.dump(..., indent=2)``.
    """
    stand_in_map: dict[str, dict[str, Any]] = {}
    spec_by_name = {s.stone_class_name: s for s in STAND_IN_SPECS}
    verification_summary: dict[str, dict[str, Any]] = {}

    for rep in reports:
        spec = spec_by_name[rep.stone_class_name]
        stand_in_map[rep.stone_class_name] = {
            "coco_label": rep.coco_label,
            "class_id": rep.empirical_class_id,
            "stone_class_id": spec.stone_class_id,
        }
        verification_summary[rep.coco_label] = {
            "class_id": rep.empirical_class_id,
            "pytorch_top_score": round(rep.pytorch_top_score, 6),
            "tflite_top_score": round(rep.tflite_top_score, 6),
            "max_score_abs_diff": round(rep.max_score_abs_diff, 6),
            "max_box_abs_diff": round(rep.max_box_abs_diff, 6),
            "top_box_iou": round(rep.top_box_iou, 6),
        }

    payload: dict[str, Any] = {
        "model_name": model_name,
        "rfdetr_version": pkg_version("rfdetr"),
        "input_resolution": input_resolution,
        "input_layout": "NHWC_RGB_FLOAT32_0_TO_1",
        "fused_normalization": {
            "mean": list(IMAGENET_MEAN_RGB),
            "std": list(IMAGENET_STD_RGB),
        },
        "batch_size": batch_size,
        "topology": topology,
        "asset_files": {
            "single_graph": single_graph_asset,
            "part1": part1_asset,
            "part2": part2_asset,
        },
        "encoder_grid_hw": list(encoder_grid_hw),
        "num_encoder_tokens": encoder_grid_hw[0] * encoder_grid_hw[1],
        "hidden_dim": 256,
        "proposal_default_wh": DEFAULT_PROPOSAL_WH,
        "num_queries": num_queries,
        "num_classes": num_classes,
        "class_id_indexing": "coco_91_sparse",
        "scores_are_probabilities": True,
        "boxes_format": "cxcywh_normalized_0_to_1",
        "score_threshold": score_threshold,
        "stand_in_map": stand_in_map,
        "hard_confuser_stone_class_id": HARD_CONFUSER_STONE_CLASS_ID,
        "unmapped_confident_to_hard_confuser": True,
        "empirical_verification": verification_summary,
    }
    if quantization is not None:
        payload["quantization"] = quantization
    return payload


def write_coco_metadata(metadata: dict[str, Any], output_path: Path) -> Path:
    """Writes ``rf_detr_coco_metadata.json`` atomically with deterministic indentation.

    Args:
        metadata: Dictionary produced by :func:`build_coco_metadata`.
        output_path: Destination JSON path (typically ``android/app/src/main/assets/rf_detr_coco_metadata.json``).

    Returns:
        Resolved path written on disk. Creates parent directories if missing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_path
