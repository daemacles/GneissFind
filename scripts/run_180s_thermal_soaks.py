#!/usr/bin/env python3
"""180-second Live-App Thermal Soak verifier for Milestone 2B finalists on Pixel 10 Pro XL.

Runs each finalist inside `com.gneissfind.app` with live CameraX preview + `ImageAnalysis` + 120 Hz
LCARS UI active for a continuous 180-second window after ensuring battery temperature <= 35.8°C.
Records latency and FPS distributions across:
- `t = 5..35s` (cold steady-state)
- `t = 75..105s` (mid soak)
- `t = 150..180s` (final 30-second thermal-soaked steady-state required by the Milestone 2B gate)
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from benchmark_m2b_matrix import (
    BENCH_RE,
    INIT_RE,
    REMOTE_DIR,
    adb,
    ensure_cool_device,
    get_battery_temp_c,
    get_thermal_status,
    push_override_json,
    wake_and_launch_app,
)


def summarize_window(samples: list[dict[str, Any]], interleaved: bool) -> dict[str, Any]:
    """Computes p50, mean, p95 latency and FPS metrics for a slice of sweep samples."""
    if not samples:
        return {"count": 0}
    totals = sorted(s["total_ms"] for s in samples)
    p1s = [s["p1_ms"] for s in samples]
    topks = [s["topk_ms"] for s in samples]
    p2s = [s["p2_ms"] for s in samples]
    p50 = statistics.median(totals)
    mean_v = statistics.mean(totals)
    p95 = totals[min(len(totals) - 1, int(len(totals) * 0.95))]
    batch_fps_p50 = 1000.0 / p50 if p50 > 0 else 0.0
    full_swath_fps_p50 = batch_fps_p50 / 2.0 if interleaved else batch_fps_p50
    return {
        "count": len(samples),
        "p50_ms": round(p50, 2),
        "mean_ms": round(mean_v, 2),
        "p95_ms": round(p95, 2),
        "min_ms": round(totals[0], 2),
        "max_ms": round(totals[-1], 2),
        "p1_mean_ms": round(statistics.mean(p1s), 2),
        "topk_mean_ms": round(statistics.mean(topks), 2),
        "p2_mean_ms": round(statistics.mean(p2s), 2),
        "batch_fps_p50": round(batch_fps_p50, 2),
        "full_swath_fps_p50": round(full_swath_fps_p50, 2),
        "priority_tile_fps_p50": round(batch_fps_p50, 2),
    }


def run_180s_soak(name: str, override_cfg: dict[str, Any], soak_sec: float = 182.0) -> dict[str, Any]:
    """Executes a 180-second continuous live-app thermal soak and returns windowed telemetry."""
    start_temp_c = ensure_cool_device(max_temp_c=35.8)
    adb("logcat", "-G", "32M")
    adb("logcat", "-c")
    push_override_json(override_cfg)
    wake_and_launch_app()

    t_launch = time.monotonic()
    seen_sweeps: dict[int, dict[str, Any]] = {}
    compile_ms: float | None = None
    temp_trace: list[tuple[float, float]] = [(0.0, start_temp_c)]

    while True:
        elapsed = time.monotonic() - t_launch
        if elapsed >= soak_sec:
            break
        time.sleep(5.0)
        elapsed = time.monotonic() - t_launch
        cur_temp = get_battery_temp_c()
        temp_trace.append((round(elapsed, 1), cur_temp))

        log_out = adb("logcat", "-d")
        for line in log_out.splitlines():
            if "RfDetrTwoGraphGpu" not in line:
                continue
            if compile_ms is None:
                m_init = INIT_RE.search(line)
                if m_init:
                    compile_ms = float(m_init.group(1))
            m_bench = BENCH_RE.search(line)
            if m_bench:
                sw_id = int(m_bench.group(1))
                if sw_id not in seen_sweeps:
                    seen_sweeps[sw_id] = {
                        "sweep": sw_id,
                        "t_rel_sec": round(elapsed, 2),
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
                    }
        if int(elapsed) % 30 < 6:
            recent = [s for s in seen_sweeps.values() if s["t_rel_sec"] >= elapsed - 15.0]
            r_p50 = statistics.median([s["total_ms"] for s in recent]) if recent else 0.0
            print(
                f"  [{name}] t={elapsed:5.1f}s | sweeps={len(seen_sweeps):4d} | "
                f"recent_p50={r_p50:6.1f}ms ({1000.0/r_p50 if r_p50 > 0 else 0.0:4.2f} FPS) | "
                f"temp={cur_temp:.1f}°C",
                flush=True,
            )

    end_temp_c = get_battery_temp_c()
    thermal_str = get_thermal_status()
    adb("shell", "am", "force-stop", "com.gneissfind.app")
    adb("shell", "input", "keyevent", "KEYCODE_SLEEP")

    all_samples = [seen_sweeps[k] for k in sorted(seen_sweeps.keys()) if k > 4]
    interleaved = bool(override_cfg.get("interleave_even_odd", False))
    w_cold = [s for s in all_samples if 5.0 <= s["t_rel_sec"] <= 35.0]
    w_mid = [s for s in all_samples if 75.0 <= s["t_rel_sec"] <= 105.0]
    w_final = [s for s in all_samples if 150.0 <= s["t_rel_sec"] <= 185.0]

    res = {
        "name": name,
        "compile_ms": compile_ms,
        "total_sweeps": len(all_samples),
        "interleaved": interleaved,
        "start_temp_c": start_temp_c,
        "end_temp_c": end_temp_c,
        "thermal_status": thermal_str,
        "window_5_35s": summarize_window(w_cold, interleaved),
        "window_75_105s": summarize_window(w_mid, interleaved),
        "window_150_180s": summarize_window(w_final, interleaved),
        "overall_steady": summarize_window(all_samples, interleaved),
    }
    f_win = res["window_150_180s"]
    print(
        f"==> SOAK COMPLETE [{name}]: t=150..180s p50={f_win.get('p50_ms')}ms "
        f"| batch_FPS={f_win.get('batch_fps_p50')} | swath_FPS={f_win.get('full_swath_fps_p50')} "
        f"| temp={start_temp_c:.1f}->{end_temp_c:.1f}°C ({thermal_str})",
        flush=True,
    )
    return res


def main() -> None:
    soak_candidates = [
        (
            "RF-DETR 2G B=5 256 W8-Dyn P1+P2 (ZeroCopy, K=100)",
            {
                "topology": "two_graph",
                "part1_path": f"{REMOTE_DIR}/rf_detr_nano_part1_b5_256_w8dyn.tflite",
                "part2_path": f"{REMOTE_DIR}/rf_detr_nano_part2_b5_256_k100_w8dyn.tflite",
                "input_resolution": 256,
                "batch_size": 5,
                "num_queries": 100,
                "num_classes": 91,
                "accelerator": "gpu",
                "zero_copy_enc_memory": True,
                "interleave_even_odd": False,
            },
        ),
        (
            "RF-DETR 2G B=3 320 W8-Dyn P1+P2 Interleaved (ZeroCopy, K=100)",
            {
                "topology": "two_graph",
                "part1_path": f"{REMOTE_DIR}/rf_detr_nano_part1_b3_320_w8dyn.tflite",
                "part2_path": f"{REMOTE_DIR}/rf_detr_nano_part2_b3_320_k100_w8dyn.tflite",
                "input_resolution": 320,
                "batch_size": 3,
                "num_queries": 100,
                "num_classes": 91,
                "accelerator": "gpu",
                "zero_copy_enc_memory": True,
                "interleave_even_odd": True,
            },
        ),
        (
            "YOLO26s 1G B=5 384 GPU FP16",
            {
                "topology": "single_graph",
                "single_graph_path": f"{REMOTE_DIR}/yolo26s_b5_384.tflite",
                "input_resolution": 384,
                "batch_size": 5,
                "num_queries": 300,
                "num_classes": 80,
                "accelerator": "gpu",
                "interleave_even_odd": False,
            },
        ),
    ]

    out_path = Path("/tmp/m2b_thermal_soak_results.json")
    results: list[dict[str, Any]] = []
    for name, cfg in soak_candidates:
        r = run_180s_soak(name, cfg, soak_sec=182.0)
        results.append(r)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Saved {len(results)} soak results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
