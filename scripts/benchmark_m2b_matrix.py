#!/usr/bin/env python3
"""Milestone 2B On-Device Pixel 10 Pro XL Benchmark Matrix Runner.

Enforces thermal hygiene (battery <= 36.0°C before every run, screen_brightness 1,
screen asleep during cooldowns) and benchmarks Tracks 1, 2, 3, 4, and 5 across
live CameraX + 120 Hz LCARS UI execution.
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SERIAL = "58110DLCQ00003"
REMOTE_DIR = "/sdcard/Android/data/com.gneissfind.app/files"
BENCH_RE = re.compile(
    r"RFDETR_BENCH sweep=(\d+) topology=(\S+) res=(\d+) batch=(\d+) tiles=(\d+) "
    r"intlv=(\S+) accel=(\S+) total_ms=([\d.]+) p1_ms=([\d.]+) topk_ms=([\d.]+) "
    r"p2_ms=([\d.]+) rolling_ms=([\d.]+) dets=(\d+)"
)
INIT_RE = re.compile(r"Initialized .* compile_ms=([\d.]+)")


def adb(*args: str, timeout: float = 30.0) -> str:
    cmd = ["adb", "-s", SERIAL, *args]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    return res.stdout + "\n" + res.stderr


def get_battery_temp_c() -> float:
    out = adb("shell", "dumpsys", "battery")
    for line in out.splitlines():
        if "temperature:" in line:
            val = int(line.split(":")[1].strip())
            return val / 10.0
    return 0.0


def get_thermal_status() -> str:
    out = adb("shell", "dumpsys", "thermalservice")
    for line in out.splitlines():
        if "Thermal Status:" in line:
            return line.strip()
    return "Thermal Status: unknown"


def ensure_cool_device(max_temp_c: float = 35.8) -> float:
    temp = get_battery_temp_c()
    while temp > max_temp_c:
        print(f"  [Thermal Guard] Battery at {temp:.1f}°C > {max_temp_c:.1f}°C; sleeping screen for 15s...")
        adb("shell", "am", "force-stop", "com.gneissfind.app")
        adb("shell", "input", "keyevent", "KEYCODE_SLEEP")
        time.sleep(15.0)
        temp = get_battery_temp_c()
    return temp


def wake_and_launch_app() -> None:
    adb("shell", "settings", "put", "system", "screen_brightness", "1")
    adb("shell", "am", "force-stop", "com.gneissfind.app")
    adb("shell", "input", "keyevent", "KEYCODE_WAKEUP")
    time.sleep(0.4)
    adb("shell", "am", "start", "-W", "-n", "com.gneissfind.app/.MainActivity")


def push_override_json(cfg: dict[str, Any]) -> None:
    tmp_path = Path("/tmp/benchmark_override.json")
    tmp_path.write_text(json.dumps(cfg, indent=2))
    adb("push", str(tmp_path), f"{REMOTE_DIR}/benchmark_override.json")
    adb("shell", f"touch {REMOTE_DIR}/benchmark_override.json")


def run_single_benchmark(
    name: str,
    track: str,
    override_cfg: dict[str, Any],
    duration_sec: float = 22.0,
    min_sweeps: int = 25,
    max_wait_sec: float = 45.0,
) -> dict[str, Any]:
    start_temp_c = ensure_cool_device()
    adb("logcat", "-c")
    push_override_json(override_cfg)
    wake_and_launch_app()

    t_start = time.monotonic()
    samples: list[dict[str, Any]] = []
    compile_ms: float | None = None
    delegate_notes: list[str] = []

    while (time.monotonic() - t_start) < max_wait_sec:
        time.sleep(2.0)
        log_out = adb("logcat", "-d", "-v", "time")
        samples.clear()
        delegate_notes.clear()
        for line in log_out.splitlines():
            if "RfDetrTwoGraphGpu" in line:
                m_init = INIT_RE.search(line)
                if m_init:
                    compile_ms = float(m_init.group(1))
                if any(k in line for k in ("Zero-copy", "failed", "unavailable", "Initialized")):
                    delegate_notes.append(line.split("RfDetrTwoGraphGpu")[-1].strip(": "))
                m_bench = BENCH_RE.search(line)
                if m_bench:
                    samples.append(
                        {
                            "sweep": int(m_bench.group(1)),
                            "topology": m_bench.group(2),
                            "res": int(m_bench.group(3)),
                            "batch": int(m_bench.group(4)),
                            "tiles": int(m_bench.group(5)),
                            "intlv": m_bench.group(6) == "true",
                            "accel": m_bench.group(7),
                            "total_ms": float(m_bench.group(8)),
                            "p1_ms": float(m_bench.group(9)),
                            "topk_ms": float(m_bench.group(10)),
                            "p2_ms": float(m_bench.group(11)),
                            "rolling_ms": float(m_bench.group(12)),
                            "dets": int(m_bench.group(13)),
                        }
                    )
            elif any(k in line.lower() for k in ("google_tensor", "npu", "darwinn", "edgetpu")):
                if len(delegate_notes) < 12:
                    delegate_notes.append(line.strip()[-140:])

        elapsed = time.monotonic() - t_start
        if len(samples) >= min_sweeps and elapsed >= duration_sec:
            break

    end_temp_c = get_battery_temp_c()
    thermal_str = get_thermal_status()

    # Skip first 4 warmup sweeps if enough samples exist
    steady = samples[4:] if len(samples) > 6 else samples
    if not steady:
        print(f"  [WARN] {name}: No steady samples collected! Notes: {delegate_notes[:4]}")
        return {
            "name": name,
            "track": track,
            "status": "FAILED",
            "compile_ms": compile_ms,
            "start_temp_c": start_temp_c,
            "end_temp_c": end_temp_c,
            "notes": delegate_notes[:8],
        }

    totals = [s["total_ms"] for s in steady]
    p1s = [s["p1_ms"] for s in steady]
    topks = [s["topk_ms"] for s in steady]
    p2s = [s["p2_ms"] for s in steady]

    totals_sorted = sorted(totals)
    p50 = statistics.median(totals_sorted)
    mean_ms = statistics.mean(totals_sorted)
    p95_idx = min(len(totals_sorted) - 1, int(round(0.95 * (len(totals_sorted) - 1))))
    p95 = totals_sorted[p95_idx]

    # Compute true wall-clock FPS from steady sweeps + ~3 ms async handoff overhead
    batch_fps_p50 = 1000.0 / p50 if p50 > 0 else 0.0
    batch_fps_mean = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    is_intlv = steady[-1]["intlv"]
    full_swath_fps_p50 = batch_fps_p50 * 0.5 if is_intlv else batch_fps_p50

    res = {
        "name": name,
        "track": track,
        "status": "OK",
        "accel": steady[-1]["accel"],
        "topology": steady[-1]["topology"],
        "res": steady[-1]["res"],
        "batch": steady[-1]["batch"],
        "interleaved": is_intlv,
        "compile_ms": round(compile_ms or 0.0, 1),
        "sweeps": len(steady),
        "p50_ms": round(p50, 2),
        "mean_ms": round(mean_ms, 2),
        "p95_ms": round(p95, 2),
        "min_ms": round(min(totals_sorted), 2),
        "max_ms": round(max(totals_sorted), 2),
        "p1_mean_ms": round(statistics.mean(p1s), 2),
        "topk_mean_ms": round(statistics.mean(topks), 2),
        "p2_mean_ms": round(statistics.mean(p2s), 2),
        "batch_fps_p50": round(batch_fps_p50, 2),
        "batch_fps_mean": round(batch_fps_mean, 2),
        "full_swath_fps_p50": round(full_swath_fps_p50, 2),
        "start_temp_c": start_temp_c,
        "end_temp_c": end_temp_c,
        "thermal": thermal_str,
        "notes": list(dict.fromkeys(delegate_notes))[:6],
    }
    print(
        f"  [{track}] {name:42s} | accel={res['accel']:12s} | "
        f"p50={res['p50_ms']:6.1f}ms (P1={res['p1_mean_ms']:5.1f} TopK={res['topk_mean_ms']:4.1f} P2={res['p2_mean_ms']:5.1f}) | "
        f"FPS={res['batch_fps_p50']:5.2f} (swath={res['full_swath_fps_p50']:5.2f}) | "
        f"temp={start_temp_c:.1f}->{end_temp_c:.1f}°C"
    )
    return res


def main() -> None:
    candidates = [
        # Track 1 & 4: Baseline 384 (K=300) vs Zero-Copy EncMemory + Async Pipeline
        (
            "RF-DETR 2G B=5 384 (K=300, Legacy Copy)",
            "Track1/4",
            {
                "topology": "two_graph",
                "input_resolution": 384,
                "batch_size": 5,
                "num_queries": 300,
                "part1_path": f"{REMOTE_DIR}/rf_detr_nano_part1_b5.tflite",
                "part2_path": f"{REMOTE_DIR}/rf_detr_nano_part2_b5.tflite",
                "zero_copy_enc_memory": False,
                "interleave_even_odd": False,
            },
        ),
        (
            "RF-DETR 2G B=5 384 (K=300, ZeroCopy+Async)",
            "Track1/4",
            {
                "topology": "two_graph",
                "input_resolution": 384,
                "batch_size": 5,
                "num_queries": 300,
                "part1_path": f"{REMOTE_DIR}/rf_detr_nano_part1_b5.tflite",
                "part2_path": f"{REMOTE_DIR}/rf_detr_nano_part2_b5.tflite",
                "zero_copy_enc_memory": True,
                "interleave_even_odd": False,
            },
        ),
        # Track 1A: K=100 Top-K Query Reduction at 384
        (
            "RF-DETR 2G B=5 384 (K=100, ZeroCopy)",
            "Track1A",
            {
                "topology": "two_graph",
                "input_resolution": 384,
                "batch_size": 5,
                "num_queries": 100,
                "part1_path": f"{REMOTE_DIR}/rf_detr_nano_part1_b5.tflite",
                "part2_path": f"{REMOTE_DIR}/rf_detr_nano_part2_b5_384_k100.tflite",
                "zero_copy_enc_memory": True,
                "interleave_even_odd": False,
            },
        ),
        # Track 5: ToMe-2:1 Token Merging in Blocks 6..11 at 384 (K=100)
        (
            "RF-DETR 2G B=5 384 ToMe-2:1 (K=100)",
            "Track5",
            {
                "topology": "two_graph",
                "input_resolution": 384,
                "batch_size": 5,
                "num_queries": 100,
                "part1_path": f"{REMOTE_DIR}/rf_detr_nano_part1_b5_384_tome.tflite",
                "part2_path": f"{REMOTE_DIR}/rf_detr_nano_part2_b5_384_k100.tflite",
                "zero_copy_enc_memory": True,
                "interleave_even_odd": False,
            },
        ),
        # Track 1B: 5x320x320 (K=100)
        (
            "RF-DETR 2G B=5 320 (K=100)",
            "Track1B",
            {
                "topology": "two_graph",
                "input_resolution": 320,
                "batch_size": 5,
                "num_queries": 100,
                "part1_path": f"{REMOTE_DIR}/rf_detr_nano_part1_b5_320.tflite",
                "part2_path": f"{REMOTE_DIR}/rf_detr_nano_part2_b5_320_k100.tflite",
                "zero_copy_enc_memory": True,
                "interleave_even_odd": False,
            },
        ),
        # Track 1B: 6x256x256 (K=100)
        (
            "RF-DETR 2G B=6 256 (K=100)",
            "Track1B",
            {
                "topology": "two_graph",
                "input_resolution": 256,
                "batch_size": 6,
                "num_queries": 100,
                "part1_path": f"{REMOTE_DIR}/rf_detr_nano_part1_b6_256.tflite",
                "part2_path": f"{REMOTE_DIR}/rf_detr_nano_part2_b6_256_k100.tflite",
                "zero_copy_enc_memory": True,
                "interleave_even_odd": False,
            },
        ),
        # Track 1B: 5x256x256 (K=100)
        (
            "RF-DETR 2G B=5 256 (K=100)",
            "Track1B",
            {
                "topology": "two_graph",
                "input_resolution": 256,
                "batch_size": 5,
                "num_queries": 100,
                "part1_path": f"{REMOTE_DIR}/rf_detr_nano_part1_b5_256.tflite",
                "part2_path": f"{REMOTE_DIR}/rf_detr_nano_part2_b5_256_k100.tflite",
                "zero_copy_enc_memory": True,
                "interleave_even_odd": False,
            },
        ),
        # Track 4: B=3 320 Even/Odd Priority Sub-Swath Interleaving (K=100)
        (
            "RF-DETR 2G B=3 320 Interleaved (K=100)",
            "Track4",
            {
                "topology": "two_graph",
                "input_resolution": 320,
                "batch_size": 3,
                "num_queries": 100,
                "part1_path": f"{REMOTE_DIR}/rf_detr_nano_part1_b3_320.tflite",
                "part2_path": f"{REMOTE_DIR}/rf_detr_nano_part2_b3_320_k100.tflite",
                "zero_copy_enc_memory": True,
                "interleave_even_odd": True,
            },
        ),
        # Track 4: B=3 256 Even/Odd Priority Sub-Swath Interleaving (K=100)
        (
            "RF-DETR 2G B=3 256 Interleaved (K=100)",
            "Track4",
            {
                "topology": "two_graph",
                "input_resolution": 256,
                "batch_size": 3,
                "num_queries": 100,
                "part1_path": f"{REMOTE_DIR}/rf_detr_nano_part1_b3_256.tflite",
                "part2_path": f"{REMOTE_DIR}/rf_detr_nano_part2_b3_256_k100.tflite",
                "zero_copy_enc_memory": True,
                "interleave_even_odd": True,
            },
        ),
        # Track 2: W8AFP32 Dynamic INT8 Weights on GPU (allowSrcQuantizedFcConvOps=true)
        (
            "RF-DETR 2G B=5 320 W8-Dyn GPU (K=100)",
            "Track2",
            {
                "topology": "two_graph",
                "input_resolution": 320,
                "batch_size": 5,
                "num_queries": 100,
                "part1_path": f"{REMOTE_DIR}/rf_detr_nano_part1_b5_320_w8dyn.tflite",
                "part2_path": f"{REMOTE_DIR}/rf_detr_nano_part2_b5_320_k100.tflite",
                "accelerator": "gpu",
                "allow_quantized_fc_conv": True,
                "zero_copy_enc_memory": True,
            },
        ),
        (
            "RF-DETR 2G B=5 256 W8-Dyn GPU (K=100)",
            "Track2",
            {
                "topology": "two_graph",
                "input_resolution": 256,
                "batch_size": 5,
                "num_queries": 100,
                "part1_path": f"{REMOTE_DIR}/rf_detr_nano_part1_b5_256_w8dyn.tflite",
                "part2_path": f"{REMOTE_DIR}/rf_detr_nano_part2_b5_256_k100.tflite",
                "accelerator": "gpu",
                "allow_quantized_fc_conv": True,
                "zero_copy_enc_memory": True,
            },
        ),
        # Track 2: Tensor G5 NPU (Accelerator.NPU) Tests (W8A8 & FP16)
        (
            "RF-DETR 2G B=5 256 W8A8 NPU_P1+GPU_P2",
            "Track2-NPU",
            {
                "topology": "two_graph",
                "input_resolution": 256,
                "batch_size": 5,
                "num_queries": 100,
                "part1_path": f"{REMOTE_DIR}/rf_detr_nano_part1_b5_256_w8a8.tflite",
                "part2_path": f"{REMOTE_DIR}/rf_detr_nano_part2_b5_256_k100.tflite",
                "accelerator": "npu_p1_gpu_p2",
                "zero_copy_enc_memory": False,
            },
        ),
        (
            "RF-DETR 2G 5xB=1 256 W8A8 NPU_P1+GPU_P2",
            "Track2-NPU",
            {
                "topology": "two_graph",
                "input_resolution": 256,
                "batch_size": 1,
                "num_queries": 300,
                "part1_path": f"{REMOTE_DIR}/rf_detr_nano_part1_b1_256_w8a8.tflite",
                "part2_path": f"{REMOTE_DIR}/rf_detr_nano_part2_b1.tflite",
                "accelerator": "npu_p1_gpu_p2",
                "zero_copy_enc_memory": False,
            },
        ),
        (
            "YOLO26n 1G B=5 384 W8A8 Tensor G5 NPU",
            "Track2-NPU",
            {
                "topology": "single_graph",
                "input_resolution": 384,
                "batch_size": 5,
                "num_queries": 3024,
                "num_classes": 80,
                "single_graph_path": f"{REMOTE_DIR}/yolo26n_b5_384_w8a8.tflite",
                "accelerator": "npu_strict",
            },
        ),
        # Track 3: YOLO26s & YOLO26n on GPU FP16 (384, 320, 256)
        (
            "YOLO26s 1G B=5 384 GPU FP16",
            "Track3",
            {
                "topology": "single_graph",
                "input_resolution": 384,
                "batch_size": 5,
                "num_queries": 3024,
                "num_classes": 80,
                "single_graph_path": f"{REMOTE_DIR}/yolo26s_b5_384.tflite",
                "accelerator": "gpu",
            },
        ),
        (
            "YOLO26s 1G B=5 320 GPU FP16",
            "Track3",
            {
                "topology": "single_graph",
                "input_resolution": 320,
                "batch_size": 5,
                "num_queries": 2100,
                "num_classes": 80,
                "single_graph_path": f"{REMOTE_DIR}/yolo26s_b5_320.tflite",
                "accelerator": "gpu",
            },
        ),
        (
            "YOLO26s 1G B=5 256 GPU FP16",
            "Track3",
            {
                "topology": "single_graph",
                "input_resolution": 256,
                "batch_size": 5,
                "num_queries": 1344,
                "num_classes": 80,
                "single_graph_path": f"{REMOTE_DIR}/yolo26s_b5_256.tflite",
                "accelerator": "gpu",
            },
        ),
        (
            "YOLO26n 1G B=5 384 GPU FP16",
            "Track3",
            {
                "topology": "single_graph",
                "input_resolution": 384,
                "batch_size": 5,
                "num_queries": 3024,
                "num_classes": 80,
                "single_graph_path": f"{REMOTE_DIR}/yolo26n_b5_384.tflite",
                "accelerator": "gpu",
            },
        ),
        (
            "YOLO26n 1G B=5 320 GPU FP16",
            "Track3",
            {
                "topology": "single_graph",
                "input_resolution": 320,
                "batch_size": 5,
                "num_queries": 2100,
                "num_classes": 80,
                "single_graph_path": f"{REMOTE_DIR}/yolo26n_b5_320.tflite",
                "accelerator": "gpu",
            },
        ),
        (
            "YOLO26n 1G B=5 256 GPU FP16",
            "Track3",
            {
                "topology": "single_graph",
                "input_resolution": 256,
                "batch_size": 5,
                "num_queries": 1344,
                "num_classes": 80,
                "single_graph_path": f"{REMOTE_DIR}/yolo26n_b5_256.tflite",
                "accelerator": "gpu",
            },
        ),
    ]

    results: list[dict[str, Any]] = []
    out_file = Path("/tmp/m2b_benchmark_matrix_results.json")
    print(f"Starting Milestone 2B Pixel 10 Pro XL Benchmark Matrix ({len(candidates)} configs)...")
    for name, track, cfg in candidates:
        res = run_single_benchmark(name, track, cfg)
        results.append(res)
        out_file.write_text(json.dumps(results, indent=2))

    # Clean up override file and put phone screen to sleep
    adb("shell", "rm", "-f", f"{REMOTE_DIR}/benchmark_override.json")
    adb("shell", "am", "force-stop", "com.gneissfind.app")
    adb("shell", "input", "keyevent", "KEYCODE_SLEEP")
    print(f"\nSaved {len(results)} benchmark results to {out_file}")


if __name__ == "__main__":
    main()
