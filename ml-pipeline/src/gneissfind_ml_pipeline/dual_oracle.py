"""Deterministic 1792x384 dual-scale (medium 120-200px + small pebble 28-45px) strip recall oracle.

Synthesizes reproducible ``1792x384`` push-broom evaluation strips from the verified MS-COCO
household stand-in fixtures (`cup=47`, `apple=53`, `toothbrush=90`, `remote=75`) embedded into a
textured gravel-like background at both:
- **Medium scale (`120-180 px` object bbox):** 4 targets across the 5 tiles (including a seam-
  straddling target across `x = 352..384`).
- **Small pebble scale (`28-45 px` minor-axis width, `20-26 mm` at `0.59 mm/px` GSD):** 8 targets
  (`45 px` and `38 px` for all 4 classes, plus `32 px` and `28 px` sub-20mm boundary probes) placed
  across the `1792x384` strip.

Provides a geometry-agnostic strip evaluation harness (`evaluate_strip_detector`) that slices a
`1792x384` strip into whatever tile/panoramic layout a candidate model uses (`5x384x384`,
`5x256x256`, `6x256x256`, `1x256x1152` panoramic, `YOLO26n/s`, `RT-DETR`, etc.), fuses cross-tile
boxes in `1792x384` strip coordinates using the `PushBroomCropper` `IoU >= 0.45 OR IoM >= 0.65`
union rule, and measures both Medium-Target Recall and `28-45 px` Small-Pebble Recall.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter

from gneissfind_ml_pipeline.coco_oracle import compute_xyxy_iou

STRIP_WIDTH_PX: int = 1792
STRIP_HEIGHT_PX: int = 384
FIVE_TILE_STRIP_X_OFFSETS: tuple[int, ...] = (0, 352, 704, 1056, 1408)


@dataclass(frozen=True, slots=True)
class SourceObjectCropSpec:
    """Tight bounding box of each household stand-in inside its 384x384 fixture image.

    Attributes:
        coco_label: MS-COCO class label (`cup`, `apple`, `toothbrush`, `remote`).
        coco_class_id: 91-logit sparse COCO class index (`47, 53, 90, 75`).
        stone_class_id: Mapped GneissFind `StoneClass.id` (`0..3`).
        fixture_filename: Source image filename under `tests/fixtures/`.
        obj_bbox_384: `(left, top, right, bottom)` of the target object inside the 384x384 fixture.
    """

    coco_label: str
    coco_class_id: int
    stone_class_id: int
    fixture_filename: str
    obj_bbox_384: tuple[int, int, int, int]


SOURCE_OBJECT_SPECS: dict[str, SourceObjectCropSpec] = {
    "cup": SourceObjectCropSpec(
        coco_label="cup",
        coco_class_id=47,
        stone_class_id=0,
        fixture_filename="sample_cup.jpg",
        obj_bbox_384=(146, 52, 342, 318),
    ),
    "apple": SourceObjectCropSpec(
        coco_label="apple",
        coco_class_id=53,
        stone_class_id=1,
        fixture_filename="sample_apple.jpg",
        obj_bbox_384=(27, 27, 358, 364),
    ),
    "toothbrush": SourceObjectCropSpec(
        coco_label="toothbrush",
        coco_class_id=90,
        stone_class_id=2,
        fixture_filename="sample_toothbrush.jpg",
        obj_bbox_384=(231, 147, 339, 381),
    ),
    "remote": SourceObjectCropSpec(
        coco_label="remote",
        coco_class_id=75,
        stone_class_id=3,
        fixture_filename="sample_remote.jpg",
        obj_bbox_384=(84, 11, 198, 323),
    ),
}


@dataclass(frozen=True, slots=True)
class GroundTruthStripTarget:
    """Ground-truth object instance placed inside a `1792x384` evaluation strip.

    Attributes:
        target_id: Unique string identifier (e.g., `"med_cup"`, `"small45_apple"`).
        scale_band: `"medium_120_200px"`, `"small_38_45px"`, or `"small_28_32px"`.
        coco_label: Human-readable COCO label.
        coco_class_id: 91-logit COCO class index.
        stone_class_id: Mapped GneissFind `StoneClass.id` (`0..3`).
        minor_axis_px: Target minor-axis (width) in strip pixels.
        major_axis_px: Target major-axis (height) in strip pixels.
        strip_xyxy_px: `(left, top, right, bottom)` bounding box in `1792x384` strip coordinates.
    """

    target_id: str
    scale_band: str
    coco_label: str
    coco_class_id: int
    stone_class_id: int
    minor_axis_px: int
    major_axis_px: int
    strip_xyxy_px: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class StripBoxPrediction:
    """Unified detection in `1792x384` strip coordinates after cross-tile seam merging.

    Attributes:
        stone_class_id: Mapped GneissFind `StoneClass.id` (`0..3`, or `4` for `HARD_CONFUSER`).
        score: Confidence probability in `[0.0, 1.0]`.
        strip_xyxy_px: `(left, top, right, bottom)` in `1792x384` strip coordinates.
    """

    stone_class_id: int
    score: float
    strip_xyxy_px: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class DualOracleEvaluationSummary:
    """Aggregated recall and confidence metrics on the medium and small-pebble 1792x384 strips.

    Attributes:
        candidate_name: Identifier of the detector configuration evaluated.
        medium_recall_035: Recall (`0.0..1.0`) on `120-200 px` targets at `score >= 0.35`, `IoU >= 0.35`.
        medium_mean_score: Mean top matched confidence across the 4 medium targets.
        small_38_45_recall_035: Recall (`0.0..1.0`) on `38-45 px` (`22-26 mm` pebble) targets at
            `score >= 0.35`, `IoU >= 0.25`.
        small_38_45_recall_025: Recall (`0.0..1.0`) on `38-45 px` targets at `score >= 0.25`.
        small_38_45_mean_score: Mean top matched confidence across the 8 `38-45 px` targets.
        small_28_45_recall_025: Recall (`0.0..1.0`) across all 12 `28-45 px` small targets at
            `score >= 0.25`.
        small_28_45_mean_score: Mean top matched confidence across all 12 `28-45 px` targets.
        per_target_scores: Mapping from `target_id` to `(matched_score, matched_iou)`.
    """

    candidate_name: str
    medium_recall_035: float
    medium_mean_score: float
    small_38_45_recall_035: float
    small_38_45_recall_025: float
    small_38_45_mean_score: float
    small_28_45_recall_025: float
    small_28_45_mean_score: float
    per_target_scores: dict[str, tuple[float, float]]


def _build_textured_gravel_strip_canvas(seed: int = 20261116) -> Image.Image:
    """Generates a deterministic multi-scale neutral gravel-textured `1792x384` RGB background."""
    rng = np.random.default_rng(seed)
    # Base sandy-slate luminance with multi-scale pebble grain so background isn't flat solid RGB.
    coarse = rng.integers(108, 138, size=(STRIP_HEIGHT_PX // 4, STRIP_WIDTH_PX // 4, 3), dtype=np.uint8)
    coarse_img = Image.fromarray(coarse).resize(
        (STRIP_WIDTH_PX, STRIP_HEIGHT_PX), Image.Resampling.BILINEAR
    )
    fine = rng.integers(-8, 9, size=(STRIP_HEIGHT_PX, STRIP_WIDTH_PX, 3), dtype=np.int16)
    blended = np.clip(np.asarray(coarse_img, dtype=np.int16) + fine, 0, 255).astype(np.uint8)
    return Image.fromarray(blended).filter(ImageFilter.GaussianBlur(radius=1.8))


def _paste_scaled_fixture_object(
    strip_img: Image.Image,
    fixtures_dir: Path,
    spec: SourceObjectCropSpec,
    target_id: str,
    scale_band: str,
    center_x_px: int,
    center_y_px: int,
    minor_axis_px: int,
) -> GroundTruthStripTarget:
    """Pastes one stand-in object scaled to `minor_axis_px` onto `strip_img` with feathered edges."""
    full_img = Image.open(fixtures_dir / spec.fixture_filename).convert("RGB")
    x1, y1, x2, y2 = spec.obj_bbox_384
    obj_w = float(x2 - x1)
    obj_h = float(y2 - y1)
    scale = float(minor_axis_px) / min(obj_w, obj_h)

    patch_w = min(STRIP_HEIGHT_PX - 4, max(16, int(round(384.0 * scale))))
    patch_h = min(STRIP_HEIGHT_PX - 4, max(16, int(round(384.0 * scale))))
    actual_scale_x = patch_w / 384.0
    actual_scale_y = patch_h / 384.0

    resized = full_img.resize((patch_w, patch_h), Image.Resampling.LANCZOS)

    # Create a locally color-matched backdrop halo around the patch so the object's natural immediate
    # lighting transitions smoothly into the strip canvas without a sharp rectangular crop edge.
    edge_rgb = np.median(np.asarray(full_img)[:14, :, :], axis=(0, 1)).astype(np.uint8)
    halo_w = min(STRIP_HEIGHT_PX, patch_w + 28)
    halo_h = min(STRIP_HEIGHT_PX, patch_h + 28)
    halo = Image.new("RGB", (halo_w, halo_h), tuple(int(c) for c in edge_rgb))
    inner_mask = Image.new("L", (patch_w, patch_h), 0)
    draw_inner = ImageDraw.Draw(inner_mask)
    draw_inner.rectangle((2, 2, patch_w - 3, patch_h - 3), fill=255)
    inner_mask = inner_mask.filter(ImageFilter.GaussianBlur(radius=1.5))
    halo.paste(resized, ((halo_w - patch_w) // 2, (halo_h - patch_h) // 2), inner_mask)

    outer_mask = Image.new("L", (halo_w, halo_h), 0)
    draw_outer = ImageDraw.Draw(outer_mask)
    draw_outer.rounded_rectangle((4, 4, halo_w - 5, halo_h - 5), radius=10, fill=255)
    outer_mask = outer_mask.filter(ImageFilter.GaussianBlur(radius=4.0))

    # Align so the object center lands at (center_x_px, center_y_px) in the 1792x384 strip.
    obj_cx_in_halo = (halo_w - patch_w) * 0.5 + 0.5 * (x1 + x2) * actual_scale_x
    obj_cy_in_halo = (halo_h - patch_h) * 0.5 + 0.5 * (y1 + y2) * actual_scale_y
    paste_x = int(round(center_x_px - obj_cx_in_halo))
    paste_y = int(round(center_y_px - obj_cy_in_halo))
    paste_x = max(0, min(STRIP_WIDTH_PX - halo_w, paste_x))
    paste_y = max(0, min(STRIP_HEIGHT_PX - halo_h, paste_y))

    strip_img.paste(halo, (paste_x, paste_y), outer_mask)

    patch_origin_x = paste_x + (halo_w - patch_w) // 2
    patch_origin_y = paste_y + (halo_h - patch_h) // 2
    gt_left = float(patch_origin_x) + x1 * actual_scale_x
    gt_top = float(patch_origin_y) + y1 * actual_scale_y
    gt_right = float(patch_origin_x) + x2 * actual_scale_x
    gt_bottom = float(patch_origin_y) + y2 * actual_scale_y

    scaled_w = int(round(gt_right - gt_left))
    scaled_h = int(round(gt_bottom - gt_top))
    return GroundTruthStripTarget(
        target_id=target_id,
        scale_band=scale_band,
        coco_label=spec.coco_label,
        coco_class_id=spec.coco_class_id,
        stone_class_id=spec.stone_class_id,
        minor_axis_px=min(scaled_w, scaled_h),
        major_axis_px=max(scaled_w, scaled_h),
        strip_xyxy_px=(
            round(gt_left, 2),
            round(gt_top, 2),
            round(gt_right, 2),
            round(gt_bottom, 2),
        ),
    )


def build_or_load_dual_oracle_strips(
    fixtures_dir: Path,
    force_rebuild: bool = False,
) -> tuple[NDArray[np.float32], list[GroundTruthStripTarget], NDArray[np.float32], list[GroundTruthStripTarget]]:
    """Builds or loads the deterministic `1792x384` Medium and Small-Pebble evaluation strips.

    Writes:
    - `fixtures_dir / "eval_strip_medium_1792x384.png"`
    - `fixtures_dir / "eval_strip_small_1792x384.png"`
    - `fixtures_dir / "eval_strip_ground_truth.json"`

    Args:
        fixtures_dir: Directory containing `sample_cup.jpg`, `sample_apple.jpg`,
            `sample_toothbrush.jpg`, and `sample_remote.jpg`.
        force_rebuild: When `True`, regenerates the PNGs and JSON manifest unconditionally.

    Returns:
        Tuple `(medium_strip_nhwc_01, medium_targets, small_strip_nhwc_01, small_targets)` where
        strip arrays have shape `(384, 1792, 3)` in `float32 [0.0, 1.0]`.
    """
    med_png = fixtures_dir / "eval_strip_medium_1792x384.png"
    small_png = fixtures_dir / "eval_strip_small_1792x384.png"
    gt_json = fixtures_dir / "eval_strip_ground_truth.json"

    if not force_rebuild and med_png.is_file() and small_png.is_file() and gt_json.is_file():
        payload = json.loads(gt_json.read_text(encoding="utf-8"))
        med_targets = [
            GroundTruthStripTarget(
                target_id=t["target_id"],
                scale_band=t["scale_band"],
                coco_label=t["coco_label"],
                coco_class_id=int(t["coco_class_id"]),
                stone_class_id=int(t["stone_class_id"]),
                minor_axis_px=int(t["minor_axis_px"]),
                major_axis_px=int(t["major_axis_px"]),
                strip_xyxy_px=tuple(float(v) for v in t["strip_xyxy_px"]),  # type: ignore[arg-type]
            )
            for t in payload["medium_targets"]
        ]
        small_targets = [
            GroundTruthStripTarget(
                target_id=t["target_id"],
                scale_band=t["scale_band"],
                coco_label=t["coco_label"],
                coco_class_id=int(t["coco_class_id"]),
                stone_class_id=int(t["stone_class_id"]),
                minor_axis_px=int(t["minor_axis_px"]),
                major_axis_px=int(t["major_axis_px"]),
                strip_xyxy_px=tuple(float(v) for v in t["strip_xyxy_px"]),  # type: ignore[arg-type]
            )
            for t in payload["small_targets"]
        ]
        med_arr = np.asarray(Image.open(med_png).convert("RGB"), dtype=np.float32) / np.float32(255.0)
        small_arr = np.asarray(Image.open(small_png).convert("RGB"), dtype=np.float32) / np.float32(255.0)
        return med_arr, med_targets, small_arr, small_targets

    # 1. Medium-scale strip (120-200 px object bounding boxes across the 5 tiles, including seam x=368).
    med_canvas = _build_textured_gravel_strip_canvas(seed=20261116)
    med_placements = [
        ("med_cup_t0", "cup", 175, 192, 132),
        ("med_apple_seam01", "apple", 368, 192, 148),
        ("med_toothbrush_t2", "toothbrush", 896, 192, 92),
        ("med_remote_t4", "remote", 1590, 192, 96),
    ]
    med_targets = [
        _paste_scaled_fixture_object(
            strip_img=med_canvas,
            fixtures_dir=fixtures_dir,
            spec=SOURCE_OBJECT_SPECS[label],
            target_id=tid,
            scale_band="medium_120_200px",
            center_x_px=cx,
            center_y_px=cy,
            minor_axis_px=minor_px,
        )
        for tid, label, cx, cy, minor_px in med_placements
    ]
    med_canvas.save(med_png, format="PNG")

    # 2. Small pebble-scale strip (28-45 px minor-axis width across all 5 tiles).
    #    Places 12 well-separated targets (45 px, 38 px, and 30-32 px minor-axis probes) across the
    #    1792x384 strip so every tile [0..4] contains small targets.
    small_canvas = _build_textured_gravel_strip_canvas(seed=20261117)
    small_placements = [
        # 45 px minor-axis (~26.5 mm pebble at 0.59 mm/px GSD)
        ("small45_cup", "cup", "small_38_45px", 110, 115, 45),
        ("small45_apple", "apple", "small_38_45px", 480, 115, 45),
        ("small45_toothbrush", "toothbrush", "small_38_45px", 830, 130, 45),
        ("small45_remote", "remote", "small_38_45px", 1190, 140, 45),
        # 38 px minor-axis (~22.4 mm pebble at 0.59 mm/px GSD)
        ("small38_cup", "cup", "small_38_45px", 250, 275, 38),
        ("small38_apple", "apple", "small_38_45px", 615, 275, 38),
        ("small38_toothbrush", "toothbrush", "small_38_45px", 980, 265, 38),
        ("small38_remote", "remote", "small_38_45px", 1340, 255, 38),
        # 28-32 px minor-axis sub-20mm boundary probes (~16.5-18.9 mm) in Tile 4
        ("small32_cup", "cup", "small_28_32px", 1500, 110, 32),
        ("small32_apple", "apple", "small_28_32px", 1680, 110, 30),
        ("small32_toothbrush", "toothbrush", "small_28_32px", 1510, 275, 32),
        ("small32_remote", "remote", "small_28_32px", 1685, 275, 32),
    ]
    small_targets = [
        _paste_scaled_fixture_object(
            strip_img=small_canvas,
            fixtures_dir=fixtures_dir,
            spec=SOURCE_OBJECT_SPECS[label],
            target_id=tid,
            scale_band=band,
            center_x_px=cx,
            center_y_px=cy,
            minor_axis_px=minor_px,
        )
        for tid, label, band, cx, cy, minor_px in small_placements
    ]
    small_canvas.save(small_png, format="PNG")

    gt_payload = {
        "strip_width_px": STRIP_WIDTH_PX,
        "strip_height_px": STRIP_HEIGHT_PX,
        "medium_targets": [asdict(t) for t in med_targets],
        "small_targets": [asdict(t) for t in small_targets],
    }
    gt_json.write_text(json.dumps(gt_payload, indent=2) + "\n", encoding="utf-8")

    med_arr = np.asarray(med_canvas, dtype=np.float32) / np.float32(255.0)
    small_arr = np.asarray(small_canvas, dtype=np.float32) / np.float32(255.0)
    return med_arr, med_targets, small_arr, small_targets


def merge_strip_predictions_iou_iom(
    raw_predictions: list[StripBoxPrediction],
    iou_threshold: float = 0.45,
    iom_threshold: float = 0.65,
) -> list[StripBoxPrediction]:
    """Fuses same-class strip predictions using `IoU >= 0.45 OR IoM >= 0.65` union merging.

    Mirrors `PushBroomCropper.mergeAndFilterTileDetections` on Android.

    Args:
        raw_predictions: Unmerged detections projected into `1792x384` strip coordinates.
        iou_threshold: Intersection-over-Union threshold for union box merging.
        iom_threshold: Intersection-over-Minimum-area threshold for union box merging.

    Returns:
        Deduplicated list of :class:`StripBoxPrediction` sorted by descending score.
    """
    ordered = sorted(raw_predictions, key=lambda d: -d.score)
    consumed = [False] * len(ordered)
    merged: list[StripBoxPrediction] = []

    for i, seed in enumerate(ordered):
        if consumed[i] or seed.stone_class_id == 4:
            continue
        consumed[i] = True
        left, top, right, bottom = seed.strip_xyxy_px
        best_score = seed.score

        changed = True
        while changed:
            changed = False
            for j in range(i + 1, len(ordered)):
                if consumed[j]:
                    continue
                cand = ordered[j]
                if cand.stone_class_id != seed.stone_class_id:
                    continue
                c_left, c_top, c_right, c_bottom = cand.strip_xyxy_px
                inter_w = max(0.0, min(right, c_right) - max(left, c_left))
                inter_h = max(0.0, min(bottom, c_bottom) - max(top, c_top))
                inter_area = inter_w * inter_h
                area_a = max(1e-6, (right - left) * (bottom - top))
                area_b = max(1e-6, (c_right - c_left) * (c_bottom - c_top))
                iou = inter_area / (area_a + area_b - inter_area)
                iom = inter_area / min(area_a, area_b)
                if iou >= iou_threshold or iom >= iom_threshold:
                    left = min(left, c_left)
                    top = min(top, c_top)
                    right = max(right, c_right)
                    bottom = max(bottom, c_bottom)
                    best_score = max(best_score, cand.score)
                    consumed[j] = True
                    changed = True

        merged.append(
            StripBoxPrediction(
                stone_class_id=seed.stone_class_id,
                score=best_score,
                strip_xyxy_px=(left, top, right, bottom),
            )
        )
    return merged


def slice_strip_into_5x384_tiles(strip_hwc_01: NDArray[np.float32]) -> NDArray[np.float32]:
    """Slices a `(384, 1792, 3)` strip into `(5, 384, 384, 3)` tiles at `[0, 352, 704, 1056, 1408]`."""
    tiles = [
        strip_hwc_01[:, x0 : x0 + 384, :]
        for x0 in FIVE_TILE_STRIP_X_OFFSETS
    ]
    return np.ascontiguousarray(np.stack(tiles, axis=0), dtype=np.float32)


def evaluate_strip_detector(
    candidate_name: str,
    predict_strip_fn: Callable[[NDArray[np.float32]], list[StripBoxPrediction]],
    medium_strip_01: NDArray[np.float32],
    medium_targets: list[GroundTruthStripTarget],
    small_strip_01: NDArray[np.float32],
    small_targets: list[GroundTruthStripTarget],
) -> DualOracleEvaluationSummary:
    """Evaluates any `1792x384` strip detector on both Medium and Small-Pebble strips.

    Args:
        candidate_name: Human-readable configuration name.
        predict_strip_fn: Callable taking a `(384, 1792, 3)` float32 `[0..1]` strip and returning
            raw or merged :class:`StripBoxPrediction` instances in `1792x384` coordinates.
        medium_strip_01: `(384, 1792, 3)` float32 medium-target strip.
        medium_targets: Ground-truth targets on the medium strip.
        small_strip_01: `(384, 1792, 3)` float32 small-pebble strip.
        small_targets: Ground-truth targets on the small-pebble strip.

    Returns:
        :class:`DualOracleEvaluationSummary` with medium and small-pebble recall and mean scores.
    """
    med_preds = merge_strip_predictions_iou_iom(predict_strip_fn(medium_strip_01))
    small_preds = merge_strip_predictions_iou_iom(predict_strip_fn(small_strip_01))

    per_target: dict[str, tuple[float, float]] = {}

    def _match_targets(
        targets: list[GroundTruthStripTarget],
        preds: list[StripBoxPrediction],
        min_iou: float,
    ) -> list[tuple[GroundTruthStripTarget, float, float]]:
        matched: list[tuple[GroundTruthStripTarget, float, float]] = []
        for gt in targets:
            best_score = 0.0
            best_iou = 0.0
            for p in preds:
                if p.stone_class_id != gt.stone_class_id:
                    continue
                iou = compute_xyxy_iou(gt.strip_xyxy_px, p.strip_xyxy_px)
                if iou >= min_iou and p.score > best_score:
                    best_score = p.score
                    best_iou = iou
            per_target[gt.target_id] = (round(best_score, 4), round(best_iou, 4))
            matched.append((gt, best_score, best_iou))
        return matched

    med_matches = _match_targets(medium_targets, med_preds, min_iou=0.35)
    small_matches = _match_targets(small_targets, small_preds, min_iou=0.20)

    med_scores = [s for _, s, _ in med_matches]
    med_recall_035 = sum(1 for s in med_scores if s >= 0.35) / max(1, len(med_scores))

    small_38_45 = [(gt, s) for gt, s, _ in small_matches if gt.scale_band == "small_38_45px"]
    s38_45_scores = [s for _, s in small_38_45]
    small_38_45_recall_035 = sum(1 for s in s38_45_scores if s >= 0.35) / max(1, len(s38_45_scores))
    small_38_45_recall_025 = sum(1 for s in s38_45_scores if s >= 0.25) / max(1, len(s38_45_scores))

    all_small_scores = [s for _, s, _ in small_matches]
    small_28_45_recall_025 = sum(1 for s in all_small_scores if s >= 0.25) / max(1, len(all_small_scores))

    return DualOracleEvaluationSummary(
        candidate_name=candidate_name,
        medium_recall_035=round(float(med_recall_035), 4),
        medium_mean_score=round(float(np.mean(med_scores)), 4),
        small_38_45_recall_035=round(float(small_38_45_recall_035), 4),
        small_38_45_recall_025=round(float(small_38_45_recall_025), 4),
        small_38_45_mean_score=round(float(np.mean(s38_45_scores)), 4),
        small_28_45_recall_025=round(float(small_28_45_recall_025), 4),
        small_28_45_mean_score=round(float(np.mean(all_small_scores)), 4),
        per_target_scores=per_target,
    )
