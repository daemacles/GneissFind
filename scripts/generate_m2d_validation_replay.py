#!/usr/bin/env python3
"""Generates a deterministic 1920x1080@30fps H.264 clip.mp4 + 120Hz IMU trace.jsonl
validation bundle (m2d_validation_v1) exercising Milestone 2D end-to-end:
  1. PATROL multi-object scene acquisition (cup target + apple/remote/oversized-cup distractors)
  2. LOCK_ON_NEURAL multi-view stereo parallax -> PINPOINT_CONSTELLATION automatic handover
  3. 7-frame (233 ms <= 300 ms) specular glint coasting hysteresis
  4. 24-frame (800 ms) full target occlusion + sideways camera translation + nearby distractors
  5. Full off-screen-right pan across x = 1920..2380 + translation + viewport return
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np


WIDTH = 1920
HEIGHT = 1080
FPS = 30
IMU_HZ = 120
TOTAL_FRAMES = 240  # 8.0 seconds @ 30 FPS
FOCAL_PX = 2000.0
ZOOM_RATIO = 2.0
REF_PITCH_DEG = 45.0

# Ground plane texture map: East in [-1.20, +1.20] m, North in [0.10, 2.50] m at 1000 px/m (1 mm/px)
TEX_SIZE = 2400
PX_PER_METER = 1000.0
EAST_ORIGIN_PX = 1200.0
NORTH_TOP_M = 2.50


def build_landscape_rotation_matrix(pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    """Exact match to GroundPlaneRaycaster.buildLandscapeRotationMatrix (R_world_from_scr)."""
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    y = math.radians(yaw_deg)

    sin_p, cos_p = math.sin(p), math.cos(p)
    sin_r, cos_r = math.sin(r), math.cos(r)
    sin_y, cos_y = math.sin(y), math.cos(y)

    x0_e, x0_n, x0_u = cos_r, sin_r * sin_p, sin_r * cos_p
    y0_e, y0_n, y0_u = -sin_r, cos_r * sin_p, cos_r * cos_p
    z0_e, z0_n, z0_u = 0.0, -cos_p, sin_p

    return np.array(
        [
            [cos_y * x0_e + sin_y * x0_n, cos_y * y0_e + sin_y * y0_n, cos_y * z0_e + sin_y * z0_n],
            [-sin_y * x0_e + cos_y * x0_n, -sin_y * y0_e + cos_y * y0_n, -sin_y * z0_e + cos_y * z0_n],
            [x0_u, y0_u, z0_u],
        ],
        dtype=np.float64,
    )


def project_enu_to_frame(
    pts_enu: np.ndarray,
    cam_pos_enu: np.ndarray,
    r_world_from_scr: np.ndarray,
    focal_px: float = FOCAL_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Projects Nx3 ENU points into 1920x1080 pinhole coordinates (u, v) and z_cv depth."""
    rel = pts_enu - cam_pos_enu[None, :]
    scr = rel @ r_world_from_scr  # columns of R_world_from_scr are x_scr, y_scr, z_scr
    x_cv = scr[:, 0]
    y_cv = -scr[:, 1]
    z_cv = -scr[:, 2]
    z_safe = np.maximum(z_cv, 1e-4)
    u = WIDTH * 0.5 + focal_px * (x_cv / z_safe)
    v = HEIGHT * 0.5 + focal_px * (y_cv / z_safe)
    return np.stack([u, v], axis=-1), z_cv


def enu_to_tex_px(east_m: float, north_m: float) -> tuple[float, float]:
    tx = EAST_ORIGIN_PX + east_m * PX_PER_METER
    ty = (NORTH_TOP_M - north_m) * PX_PER_METER
    return tx, ty


def smoothstep(t: float) -> float:
    c = max(0.0, min(1.0, t))
    return c * c * (3.0 - 2.0 * c)


def pose_at_time(t_sec: float) -> tuple[np.ndarray, float, float, float]:
    """Continuous C^1 camera trajectory (cam_enu_m, pitch_deg, roll_deg, yaw_deg) at time t_sec in [0..8]."""
    # Phase 1 (0.0..1.0s): Steady PATROL acquisition centered on primary cup at (0, 1.06, 0)
    if t_sec < 1.0:
        wobble = 0.15 * math.sin(t_sec * 4.0)
        return np.array([0.0, 0.0, 1.06]), 45.0 + wobble, 0.0, 0.0

    # Phase 2 (1.0..3.1s): Smooth stereo baseline arc (+0.25m East, +0.18m North, crouch 1.06 -> 0.80m)
    if t_sec < 3.1:
        s = smoothstep((t_sec - 1.0) / 2.1)
        east = 0.25 * s
        north = 0.18 * s
        up = 1.06 - 0.26 * s
        yaw = -14.5 * s
        pitch = 45.0 - 1.5 * s
        return np.array([east, north, up]), pitch, 0.4 * math.sin(s * math.pi), yaw

    # Phase 3 (3.1..3.8s): Pinpoint hold (includes 7-frame specular glint at frames 95..101)
    if t_sec < 3.8:
        s = smoothstep((t_sec - 3.1) / 0.7)
        east = 0.25 - 0.01 * s
        north = 0.18 + 0.02 * s
        up = 0.80
        yaw = -14.5 + 0.5 * s
        return np.array([east, north, up]), 43.5, 0.0, yaw

    # Phase 4 (3.8..4.65s): Full foreground occlusion (frames 115..138) WHILE translating West (-0.13m)
    if t_sec < 4.65:
        s = smoothstep((t_sec - 3.8) / 0.85)
        east = 0.24 - 0.13 * s
        north = 0.20 - 0.02 * s
        up = 0.80 + 0.03 * s
        yaw = -14.0 + 7.2 * s
        return np.array([east, north, up]), 44.0, 0.0, yaw

    # Phase 5 (4.65..5.35s): Post-occlusion recovery hold at translated camera position
    if t_sec < 5.35:
        return np.array([0.11, 0.18, 0.83]), 44.2, 0.0, -6.8

    # Phase 6 (5.35..7.05s): Full off-screen-right pan across x = 1920..2380 + West step + return!
    if t_sec < 7.05:
        u = (t_sec - 5.35) / 1.70
        # Bell-shaped left yaw excursion (-6.8 -> -41.0 -> -1.2 deg) sweeps stone off right edge x > 2250
        pan_bell = math.sin(math.pi * min(1.0, max(0.0, u))) ** 2
        s = smoothstep(u)
        east = 0.11 - 0.09 * s
        north = 0.18 - 0.02 * s
        up = 0.83 + 0.03 * s
        base_yaw = -6.8 + 5.6 * s
        yaw = base_yaw - 52.0 * pan_bell
        return np.array([east, north, up]), 44.5, 0.0, yaw

    # Phase 7 (7.05..8.00s): Final terminal Pinpoint lock hold
    return np.array([0.02, 0.16, 0.86]), 44.5, 0.0, -1.2


def paste_feathered_decal(
    canvas: np.ndarray,
    patch_bgr: np.ndarray,
    east_m: float,
    north_m: float,
    width_m: float,
     foreshorten_comp: float = 1.38,
) -> None:
    """Pastes a photographic COCO object decal onto the ground texture map with soft elliptical feathering."""
    h_src, w_src = patch_bgr.shape[:2]
    aspect = h_src / max(1.0, float(w_src))
    w_px = max(24, int(round(width_m * PX_PER_METER)))
    h_px = max(24, int(round(w_px * aspect * foreshorten_comp)))

    resized = cv2.resize(patch_bgr, (w_px, h_px), interpolation=cv2.INTER_AREA)
    cx, cy = enu_to_tex_px(east_m, north_m)
    x0 = int(round(cx - w_px * 0.5))
    y0 = int(round(cy - h_px * 0.5))
    x1 = x0 + w_px
    y1 = y0 + h_px
    if x0 < 0 or y0 < 0 or x1 > TEX_SIZE or y1 > TEX_SIZE:
        return

    yy, xx = np.ogrid[:h_px, :w_px]
    nx = (xx - w_px * 0.5) / (w_px * 0.5)
    ny = (yy - h_px * 0.5) / (h_px * 0.5)
    rad = np.sqrt(nx * nx + ny * ny)
    alpha = np.clip((1.02 - rad) / 0.22, 0.0, 1.0)[..., None].astype(np.float32)

    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    blended = roi * (1.0 - alpha) + resized.astype(np.float32) * alpha
    canvas[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)


def build_ground_texture(repo_root: Path) -> np.ndarray:
    """Synthesizes a rich multi-scale river-gravel ground plane with real COCO stand-in targets."""
    rng = np.random.default_rng(20260204)
    # Warm river-bed base + multi-scale noise so 640x360 PyrLK and 192x192 / 480x270 AKAZE have crisp corners
    tex = np.full((TEX_SIZE, TEX_SIZE, 3), (92, 102, 112), dtype=np.uint8)
    fine_noise = rng.integers(-18, 19, size=(TEX_SIZE, TEX_SIZE, 1), dtype=np.int16)
    tex = np.clip(tex.astype(np.int16) + fine_noise, 20, 235).astype(np.uint8)

    # 5,500 multi-scale textured gravel clasts across the 2.4m x 2.4m bed
    for _ in range(5500):
        px = int(rng.integers(16, TEX_SIZE - 16))
        py = int(rng.integers(16, TEX_SIZE - 16))
        rx = int(rng.integers(4, 22))
        ry = int(rng.integers(3, 17))
        ang = float(rng.uniform(0.0, 180.0))
        base_v = int(rng.integers(42, 196))
        color = (
            max(20, base_v - int(rng.integers(0, 18))),
            base_v,
            min(240, base_v + int(rng.integers(0, 22))),
        )
        cv2.ellipse(tex, (px, py), (rx, ry), ang, 0, 360, color, -1, cv2.LINE_AA)
        cv2.circle(
            tex,
            (px + 2, py - 2),
            max(1, rx // 5),
            (min(245, color[0] + 32), min(245, color[1] + 32), min(245, color[2] + 32)),
            -1,
        )

    fixtures = repo_root / "ml-pipeline" / "tests" / "fixtures"
    strip_img = cv2.imread(str(fixtures / "eval_strip_medium_1792x384.png"))
    cup_img = cv2.imread(str(fixtures / "sample_cup.jpg"))
    apple_img = cv2.imread(str(fixtures / "sample_apple.jpg"))
    remote_img = cv2.imread(str(fixtures / "sample_remote.jpg"))

    # Crop tight, high-confidence COCO patches from verified eval_strip_medium_1792x384.png
    if strip_img is not None:
        cup_patch = strip_img[92:292, 102:256]
        apple_patch = strip_img[106:276, 284:452]
        remote_patch = strip_img[48:326, 1516:1632]
    else:
        cup_patch = cup_img
        apple_patch = apple_img
        remote_patch = remote_img

    # 1. Primary Target: cup (CHALCEDONY_AGATE, class 0) at (East=0.00m, North=1.06m) -> ~95x120 px @ t=0
    paste_feathered_decal(tex, cup_patch, east_m=0.00, north_m=1.06, width_m=0.095)

    # 2. Distractor A: apple (JASPER_RED_YELLOW, class 1) at (+0.15m, 1.09m) -> inside 640x640 wide crop!
    paste_feathered_decal(tex, apple_patch, east_m=0.15, north_m=1.09, width_m=0.092)

    # 3. Distractor B: oversized cup (CHALCEDONY_AGATE, class 0, 2.5x scale) at (-0.18m, 1.11m) -> inside 640x640 wide crop!
    paste_feathered_decal(tex, cup_patch, east_m=-0.18, north_m=1.11, width_m=0.240)

    # 4. Distractor C: remote (PETRIFIED_WOOD, class 3) at (+0.34m, 0.96m)
    paste_feathered_decal(tex, remote_patch, east_m=0.34, north_m=0.96, width_m=0.072)

    return tex


def render_frame_bgr(
    ground_tex: np.ndarray,
    cam_enu: np.ndarray,
    r_world_from_scr: np.ndarray,
    frame_idx: int,
) -> np.ndarray:
    """Warps the 3D z=0 ground plane into the 1920x1080 pinhole camera via exact planar homography."""
    quad_enu = np.array(
        [
            [-0.85, 1.95, 0.0],
            [0.85, 1.95, 0.0],
            [0.85, 0.35, 0.0],
            [-0.85, 0.35, 0.0],
        ],
        dtype=np.float64,
    )
    src_tex = np.array([enu_to_tex_px(p[0], p[1]) for p in quad_enu], dtype=np.float32)
    dst_img_64, _ = project_enu_to_frame(quad_enu, cam_enu, r_world_from_scr, FOCAL_PX)
    h_mat = cv2.getPerspectiveTransform(src_tex, dst_img_64.astype(np.float32))

    frame = cv2.warpPerspective(
        ground_tex,
        h_mat,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Projected primary stone center for glint / foreground occlusion injection
    stone_uv, stone_z = project_enu_to_frame(
        np.array([[0.00, 1.06, 0.00]], dtype=np.float64),
        cam_enu,
        r_world_from_scr,
        FOCAL_PX,
    )
    sx, sy = float(stone_uv[0, 0]), float(stone_uv[0, 1])

    # Frames 95..101 (7 frames = 233 ms <= 300 ms): brief specular wet-film collar washout
    if 95 <= frame_idx <= 101 and stone_z[0] > 0.1:
        cx, cy = int(round(sx)), int(round(sy))
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), 195, (248, 250, 252), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0.0, dst=frame)

    # Frames 115..138 (24 frames = 800 ms): full opaque foreground hand/trowel occlusion over target ONLY
    if 115 <= frame_idx <= 138 and stone_z[0] > 0.1:
        cx, cy = int(round(sx)), int(round(sy))
        cv2.ellipse(frame, (cx, cy + 18), (118, 128), -12.0, 0, 360, (68, 72, 78), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (cx - 48, cy + 90), (cx + 48, HEIGHT), (62, 66, 72), -1)

    return frame


def generate_replay_bundle(out_dir: Path, repo_root: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_path = out_dir / "clip.mp4"
    trace_path = out_dir / "trace.jsonl"

    ground_tex = build_ground_texture(repo_root)

    base_t_ns = 10_000_000_000
    imu_step_ns = round(1_000_000_000 / IMU_HZ)
    frame_step_ns = round(1_000_000_000 / FPS)

    trace_lines: list[str] = []
    trace_lines.append(
        '{"type":"header","schema_version":1,'
        f'"width":{WIDTH},"height":{HEIGHT},"fps":{FPS},'
        f'"zoom_ratio":{ZOOM_RATIO:.3f},"focal_length_px":{FOCAL_PX:.2f},'
        f'"ref_pitch_deg":{REF_PITCH_DEG:.3f},"display_rotation":1,'
        '"git_hash":"m2d_validation_v1"}'
    )

    total_imu = TOTAL_FRAMES * (IMU_HZ // FPS)
    for imu_idx in range(total_imu):
        t_ns = base_t_ns + imu_idx * imu_step_ns
        t_sec = (t_ns - base_t_ns) * 1e-9
        _, pitch_deg, roll_deg, yaw_deg = pose_at_time(t_sec)
        rot = build_landscape_rotation_matrix(pitch_deg, roll_deg, yaw_deg).reshape(-1)
        rot_str = ",".join(f"{v:.6f}" for v in rot)
        trace_lines.append(
            f'{{"type":"imu","t_ns":{t_ns},"rot_mat":[{rot_str}],"pitch_deg":{pitch_deg:.3f}}}'
        )

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-b:v",
        "14M",
        "-g",
        "15",
        "-movflags",
        "+faststart",
        str(clip_path),
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    max_right_u = -1e9
    for frame_idx in range(TOTAL_FRAMES):
        t_ns = base_t_ns + frame_idx * frame_step_ns
        pts_us = round((frame_idx * 1_000_000) / FPS)
        t_sec = (t_ns - base_t_ns) * 1e-9
        cam_enu, pitch_deg, roll_deg, yaw_deg = pose_at_time(t_sec)
        rot = build_landscape_rotation_matrix(pitch_deg, roll_deg, yaw_deg)

        uv, _ = project_enu_to_frame(np.array([[0.00, 1.06, 0.00]]), cam_enu, rot, FOCAL_PX)
        max_right_u = max(max_right_u, float(uv[0, 0]))

        frame_bgr = render_frame_bgr(ground_tex, cam_enu, rot, frame_idx)
        proc.stdin.write(frame_bgr.tobytes())

        trace_lines.append(
            f'{{"type":"frame","frame_idx":{frame_idx},"pts_us":{pts_us},"t_ns":{t_ns},'
            f'"exposure_ns":4000000,"iso":180,"zoom_ratio":{ZOOM_RATIO:.3f},'
            '"shutter_clamp":true,"ev_bias":0.00}'
        )

    proc.stdin.close()
    stderr_bytes = proc.stderr.read() if proc.stderr else b""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed ({rc}): {stderr_bytes.decode('utf-8', errors='replace')}")

    trace_path.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")
    print(
        f"Generated {clip_path} ({clip_path.stat().st_size / (1024 * 1024):.2f} MB) + "
        f"{trace_path} ({len(trace_lines)} records), max_offscreen_right_u={max_right_u:.1f}px"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/m2d_validation_v1"),
        help="Destination replay session directory",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    generate_replay_bundle(args.out_dir, repo_root)


if __name__ == "__main__":
    main()
