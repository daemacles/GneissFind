"""Parity and class-oracle test suite for the Milestone 2 RF-DETR-Nano LiteRT export pipeline.

Covers:
1. GPU-safe separable matmul bilinear grid sampling vs ``torch.nn.functional.grid_sample``.
2. Single-graph vs two-graph PyTorch numerical equivalence across a 5-tile ``B=5`` batch.
3. Operator cleanliness of the two-graph ``.tflite`` split (zero ``TOPK_V2`` and ``GATHER_ND``).
4. PyTorch-vs-LiteRT numerical parity (``<= 1e-2``) and top-box IoU (``>= 0.85``) on real
   ``384x384`` fixtures for ``cup``, ``apple``, ``toothbrush``, and ``remote``.
5. ``B=5`` batched vs ``5 x B=1`` sequential ``.tflite`` equivalence and ``HARD_CONFUSER`` routing.
6. Schema and class-ID integrity of ``android/app/src/main/assets/rf_detr_coco_metadata.json``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from ai_edge_litert.interpreter import Interpreter

from gneissfind_ml_pipeline.coco_oracle import (
    HARD_CONFUSER_STONE_CLASS_ID,
    STAND_IN_SPECS,
    decode_tile_predictions,
    load_nhwc_rgb_01,
    run_single_graph_tflite,
    run_two_graph_tflite,
    verify_stand_in_fixtures,
)
from gneissfind_ml_pipeline.wrappers import (
    DEFAULT_INPUT_RESOLUTION,
    ENCODER_GRID_HW,
    FusedRfDetrSingleGraph,
    RfDetrPart1BackboneEncoder,
    RfDetrPart2DecoderHead,
    bilinear_matmul_grid_sample,
    host_select_topk_queries,
    load_prepared_lwdetr,
    pin_cuda_device_zero,
)

ML_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_PIPELINE_ROOT.parent
ARTIFACTS_DIR = ML_PIPELINE_ROOT / "artifacts"
FIXTURES_DIR = ML_PIPELINE_ROOT / "tests" / "fixtures"
METADATA_PATH = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "rf_detr_coco_metadata.json"


@pytest.fixture(scope="module")
def prepared_models() -> tuple[torch.nn.Module, object]:
    """Loads ``RFDETRNano`` once per test module with GPU 0 pinning configured."""
    pin_cuda_device_zero()
    return load_prepared_lwdetr(resolution=DEFAULT_INPUT_RESOLUTION)


@pytest.fixture(scope="module")
def five_tile_fixture_batch() -> np.ndarray:
    """Builds a realistic ``(5, 384, 384, 3)`` ``NHWC`` ``[0, 1]`` sweep from the 5 test fixtures."""
    names = [
        "sample_cup.jpg",
        "sample_apple.jpg",
        "sample_toothbrush.jpg",
        "sample_remote.jpg",
        "sample_mixed_confusers.jpg",
    ]
    tiles = [load_nhwc_rgb_01(FIXTURES_DIR / name) for name in names]
    return np.concatenate(tiles, axis=0)


def test_bilinear_matmul_grid_sample_matches_pytorch_grid_sample() -> None:
    torch.manual_seed(17)
    value = torch.randn(8, 32, 24, 24, dtype=torch.float32)
    # Span interior [-1, 1], half-pixel zero border, and fully out-of-bounds coordinates.
    grid = torch.empty(8, 300, 4, 2, dtype=torch.float32).uniform_(-1.35, 1.35)

    expected = F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    actual = bilinear_matmul_grid_sample(value, grid, padding_mode="zeros", align_corners=False)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)


def test_single_graph_and_two_graph_pytorch_equivalence(
    prepared_models: tuple[torch.nn.Module, object],
    five_tile_fixture_batch: np.ndarray,
) -> None:
    lwdetr, _ = prepared_models
    single = FusedRfDetrSingleGraph(lwdetr).eval()
    part1 = RfDetrPart1BackboneEncoder(lwdetr).eval()
    part2 = RfDetrPart2DecoderHead(
        lwdetr, resolution=DEFAULT_INPUT_RESOLUTION, grid_hw=ENCODER_GRID_HW
    ).eval()

    batch_t = torch.from_numpy(five_tile_fixture_batch)
    with torch.no_grad():
        boxes_1g, scores_1g = single(batch_t)
        mem, prenorm, enc_scores = part1(batch_t)
        sel_pre, sel_prop, _ = host_select_topk_queries(
            prenorm.numpy(), enc_scores.numpy(), top_k=300
        )
        boxes_2g, scores_2g = part2(
            mem, torch.from_numpy(sel_pre), torch.from_numpy(sel_prop)
        )

    assert float((boxes_1g - boxes_2g).abs().max()) < 1e-4
    assert float((scores_1g - scores_2g).abs().max()) < 1e-4


def test_two_graph_tflite_eliminates_topk_and_gather_nd() -> None:
    forbidden_gpu_ops = ("TOPK_V2", "GATHER_ND", "BROADCAST_TO", "FLOOR_DIV", "RELU_0_TO_1")
    for filename in (
        "rf_detr_nano_part1_b1.tflite",
        "rf_detr_nano_part2_b1.tflite",
        "rf_detr_nano_part1_b5.tflite",
        "rf_detr_nano_part2_b5.tflite",
    ):
        model_path = ARTIFACTS_DIR / filename
        assert model_path.is_file(), f"Missing exported artifact {model_path}; run export_rfdetr_litert.py"
        interp = Interpreter(model_path=str(model_path))
        ops = Counter(op["op_name"] for op in interp._get_ops_details())
        for op_name in forbidden_gpu_ops:
            assert ops[op_name] == 0, f"{filename} unexpectedly contains {op_name}"
        max_rank = max(len(t["shape"]) for t in interp.get_tensor_details())
        assert max_rank <= 4, f"{filename} has tensor rank {max_rank} > 4 (unsupported by LITERT_CL)"


def test_stand_in_coco_class_oracle_and_tflite_parity(
    prepared_models: tuple[torch.nn.Module, object],
) -> None:
    lwdetr, eager_detector = prepared_models
    pt_wrapper = FusedRfDetrSingleGraph(lwdetr).eval()

    interp_1g = Interpreter(model_path=str(ARTIFACTS_DIR / "rf_detr_nano_1g_b1.tflite"), num_threads=4)
    interp_1g.allocate_tensors()
    reports_1g = verify_stand_in_fixtures(
        pytorch_wrapper=pt_wrapper,
        eager_detector=eager_detector,
        tflite_runner=lambda b: run_single_graph_tflite(interp_1g, b),
        fixtures_dir=FIXTURES_DIR,
    )

    interp_p1 = Interpreter(model_path=str(ARTIFACTS_DIR / "rf_detr_nano_part1_b1.tflite"), num_threads=4)
    interp_p2 = Interpreter(model_path=str(ARTIFACTS_DIR / "rf_detr_nano_part2_b1.tflite"), num_threads=4)
    interp_p1.allocate_tensors()
    interp_p2.allocate_tensors()
    reports_2g = verify_stand_in_fixtures(
        pytorch_wrapper=pt_wrapper,
        eager_detector=eager_detector,
        tflite_runner=lambda b: run_two_graph_tflite(interp_p1, interp_p2, b),
        fixtures_dir=FIXTURES_DIR,
    )

    ids_1g = {r.coco_label: r.empirical_class_id for r in reports_1g}
    ids_2g = {r.coco_label: r.empirical_class_id for r in reports_2g}
    assert ids_1g == ids_2g == {"cup": 47, "apple": 53, "toothbrush": 90, "remote": 75}


def test_b5_batched_matches_sequential_b1_and_routes_hard_confusers(
    five_tile_fixture_batch: np.ndarray,
) -> None:
    interp_1g_b1 = Interpreter(model_path=str(ARTIFACTS_DIR / "rf_detr_nano_1g_b1.tflite"), num_threads=4)
    interp_1g_b5 = Interpreter(model_path=str(ARTIFACTS_DIR / "rf_detr_nano_1g_b5.tflite"), num_threads=4)
    interp_p1_b5 = Interpreter(model_path=str(ARTIFACTS_DIR / "rf_detr_nano_part1_b5.tflite"), num_threads=4)
    interp_p2_b5 = Interpreter(model_path=str(ARTIFACTS_DIR / "rf_detr_nano_part2_b5.tflite"), num_threads=4)
    for it in (interp_1g_b1, interp_1g_b5, interp_p1_b5, interp_p2_b5):
        it.allocate_tensors()

    seq_boxes, seq_scores = [], []
    for i in range(5):
        b_i, s_i = run_single_graph_tflite(interp_1g_b1, five_tile_fixture_batch[i : i + 1])
        seq_boxes.append(b_i)
        seq_scores.append(s_i)
    seq_boxes_arr = np.concatenate(seq_boxes, axis=0)
    seq_scores_arr = np.concatenate(seq_scores, axis=0)

    b5_1g_boxes, b5_1g_scores = run_single_graph_tflite(interp_1g_b5, five_tile_fixture_batch)
    b5_2g_boxes, b5_2g_scores = run_two_graph_tflite(interp_p1_b5, interp_p2_b5, five_tile_fixture_batch)

    assert float(np.max(np.abs(seq_scores_arr - b5_1g_scores))) < 1e-4
    assert float(np.max(np.abs(seq_scores_arr - b5_2g_scores))) < 1e-4
    assert float(np.max(np.abs(seq_boxes_arr - b5_2g_boxes))) < 1e-3

    # Tile 4 is sample_mixed_confusers.jpg (cats + remotes + couch).
    class_map = {47: 0, 53: 1, 90: 2, 75: 3}
    mixed_dets = decode_tile_predictions(b5_2g_boxes[4], b5_2g_scores[4], class_id_to_stone_id=class_map)
    stone_ids = {d.stone_class_id for d in mixed_dets}
    assert 3 in stone_ids, "Expected PETRIFIED_WOOD (remote=75) in mixed fixture"
    assert HARD_CONFUSER_STONE_CLASS_ID in stone_ids, "Expected muted HARD_CONFUSER in mixed fixture"


def test_generated_android_coco_metadata_json() -> None:
    assert METADATA_PATH.is_file(), f"Missing {METADATA_PATH}"
    payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    assert payload["model_name"] == "rf_detr_nano_coco"
    assert payload["input_resolution"] == 384
    assert payload["num_queries"] == 300
    assert payload["num_classes"] == 91
    assert payload["scores_are_probabilities"] is True
    assert payload["unmapped_confident_to_hard_confuser"] is True

    stand_in_map = payload["stand_in_map"]
    expected = {spec.stone_class_name: spec.coco_label for spec in STAND_IN_SPECS}
    assert set(stand_in_map.keys()) == set(expected.keys())
    for stone_name, coco_label in expected.items():
        assert stand_in_map[stone_name]["coco_label"] == coco_label
    assert stand_in_map["CHALCEDONY_AGATE"]["class_id"] == 47
    assert stand_in_map["JASPER_RED_YELLOW"]["class_id"] == 53
    assert stand_in_map["GREEN_GEM_ROCK"]["class_id"] == 90
    assert stand_in_map["PETRIFIED_WOOD"]["class_id"] == 75


def test_lockon_320_b1_w8dyn_gpu_cleanliness_and_stand_in_parity() -> None:
    """Verifies the Milestone 2C B=1 320x320 K=100 W8-Dyn Lock-On split graphs and metadata."""
    forbidden_gpu_ops = ("TOPK_V2", "GATHER_ND", "BROADCAST_TO", "FLOOR_DIV", "RELU_0_TO_1")
    p1_name = "rf_detr_nano_part1_b1_320_w8dyn.tflite"
    p2_name = "rf_detr_nano_part2_b1_320_k100_w8dyn.tflite"

    assets_dir = METADATA_PATH.parent
    for filename in (p1_name, p2_name):
        artifact_path = ARTIFACTS_DIR / filename
        asset_path = assets_dir / filename
        assert artifact_path.is_file(), f"Missing Lock-On artifact {artifact_path}"
        assert asset_path.is_file(), f"Missing staged Lock-On asset {asset_path}"
        interp = Interpreter(model_path=str(artifact_path))
        ops = Counter(op["op_name"] for op in interp._get_ops_details())
        for op_name in forbidden_gpu_ops:
            assert ops[op_name] == 0, f"{filename} unexpectedly contains {op_name}"
        max_rank = max(len(t["shape"]) for t in interp.get_tensor_details())
        assert max_rank <= 4, f"{filename} has tensor rank {max_rank} > 4"

    lwdetr_320, eager_detector = load_prepared_lwdetr(resolution=320, num_queries=100)
    pt_wrapper_320 = FusedRfDetrSingleGraph(lwdetr_320).eval()
    interp_p1 = Interpreter(model_path=str(ARTIFACTS_DIR / p1_name), num_threads=4)
    interp_p2 = Interpreter(model_path=str(ARTIFACTS_DIR / p2_name), num_threads=4)
    interp_p1.allocate_tensors()
    interp_p2.allocate_tensors()

    assert interp_p1.get_input_details()[0]["shape"].tolist() == [1, 320, 320, 3]
    assert interp_p1.get_output_details()[0]["shape"].tolist() == [1, 400, 256]
    assert interp_p2.get_output_details()[0]["shape"].tolist() == [1, 100, 4]
    assert interp_p2.get_output_details()[1]["shape"].tolist() == [1, 100, 91]

    reports = verify_stand_in_fixtures(
        pytorch_wrapper=pt_wrapper_320,
        eager_detector=eager_detector,
        tflite_runner=lambda b: run_two_graph_tflite(interp_p1, interp_p2, b, top_k=100),
        fixtures_dir=FIXTURES_DIR,
        min_box_iou=0.90,
        max_score_tol=0.08,
        resolution=320,
        compare_class_max_scores=True,
    )
    assert {r.coco_label: r.empirical_class_id for r in reports} == {
        "cup": 47,
        "apple": 53,
        "toothbrush": 90,
        "remote": 75,
    }

    lockon_meta_path = assets_dir / "rf_detr_320_b1_metadata.json"
    assert lockon_meta_path.is_file(), f"Missing {lockon_meta_path}"
    payload = json.loads(lockon_meta_path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "rf_detr_nano_320_b1_w8dyn"
    assert payload["input_resolution"] == 320
    assert payload["batch_size"] == 1
    assert payload["num_queries"] == 100
    assert payload["encoder_grid_hw"] == [20, 20]
    assert payload["num_encoder_tokens"] == 400
    assert payload["quantization"] == "dynamic_wi8_afp32"

