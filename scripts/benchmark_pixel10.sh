#!/usr/bin/env bash
# Automated ADB on-device LiteRT GPU (FP16) benchmark harness for Milestone 2 (RF-DETR-Nano).
#
# Pushes the exported 1-Graph and 2-Graph (B=1 and B=5) .tflite artifacts from ml-pipeline/artifacts/
# to the connected physical Pixel 10 Pro XL, launches each configuration in an isolated process via
# benchmark_override.json (with cooldown between runs to avoid cross-config VRAM or thermal carry-over),
# parses LITERT_CL OpenCL delegate partition counts and RFDETR_BENCH sweep telemetry from logcat,
# and writes ml-pipeline/artifacts/pixel10_benchmark_report.json plus a Markdown summary table.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="${REPO_ROOT}/ml-pipeline/artifacts"
DEVICE_FILES_DIR="/sdcard/Android/data/com.gneissfind.app/files"
REPORT_JSON="${ARTIFACTS_DIR}/pixel10_benchmark_report.json"

SERIAL="${ADB_SERIAL:-}"
WARMUP_SWEEPS=2
MEASURE_SWEEPS=15
COOLDOWN_SEC=6
SKIP_PUSH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial)
      SERIAL="$2"
      shift 2
      ;;
    --warmup)
      WARMUP_SWEEPS="$2"
      shift 2
      ;;
    --sweeps)
      MEASURE_SWEEPS="$2"
      shift 2
      ;;
    --cooldown)
      COOLDOWN_SEC="$2"
      shift 2
      ;;
    --skip-push)
      SKIP_PUSH=1
      shift
      ;;
    *)
      echo "Usage: $0 [--serial SERIAL] [--warmup N] [--sweeps N] [--cooldown SEC] [--skip-push]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${SERIAL}" ]]; then
  SERIAL="$(adb devices | awk 'NR>1 && $2=="device" && $1 !~ /^emulator-/ {print $1; exit}')"
fi
if [[ -z "${SERIAL}" ]]; then
  echo "ERROR: No physical Android device connected via adb." >&2
  exit 1
fi

MODEL_NAME="$(adb -s "${SERIAL}" shell getprop ro.product.model | tr -d '\r')"
SOC_MODEL="$(adb -s "${SERIAL}" shell getprop ro.soc.model | tr -d '\r')"
BOARD_PLATFORM="$(adb -s "${SERIAL}" shell getprop ro.board.platform | tr -d '\r')"
BUILD_FINGERPRINT="$(adb -s "${SERIAL}" shell getprop ro.build.fingerprint | tr -d '\r')"

echo "================================================================================"
echo "GneissFind Milestone 2 LiteRT GPU FP16 Benchmark Harness"
echo "  Device   : ${MODEL_NAME} (serial=${SERIAL}, soc=${SOC_MODEL}, board=${BOARD_PLATFORM})"
echo "  Build    : ${BUILD_FINGERPRINT}"
echo "  Protocol : ${WARMUP_SWEEPS} warmup + ${MEASURE_SWEEPS} measured sweeps/config (${COOLDOWN_SEC}s cooldown)"
echo "================================================================================"

adb -s "${SERIAL}" shell svc power stayon true >/dev/null 2>&1 || true
adb -s "${SERIAL}" shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1 || true
adb -s "${SERIAL}" shell wm dismiss-keyguard >/dev/null 2>&1 || true

if [[ "${SKIP_PUSH}" -eq 0 ]]; then
  echo "[1/3] Pushing exported .tflite artifacts to ${DEVICE_FILES_DIR}..."
  adb -s "${SERIAL}" shell "mkdir -p ${DEVICE_FILES_DIR}"
  adb -s "${SERIAL}" push \
    "${ARTIFACTS_DIR}/rf_detr_nano_1g_b1.tflite" \
    "${ARTIFACTS_DIR}/rf_detr_nano_1g_b5.tflite" \
    "${ARTIFACTS_DIR}/rf_detr_nano_part1_b1.tflite" \
    "${ARTIFACTS_DIR}/rf_detr_nano_part2_b1.tflite" \
    "${ARTIFACTS_DIR}/rf_detr_nano_part1_b5.tflite" \
    "${ARTIFACTS_DIR}/rf_detr_nano_part2_b5.tflite" \
    "${DEVICE_FILES_DIR}/"
fi

LOG_DIR="$(mktemp -d)"
trap 'rm -rf "${LOG_DIR}"; adb -s "${SERIAL}" shell "rm -f ${DEVICE_FILES_DIR}/benchmark_override.json" >/dev/null 2>&1 || true' EXIT

run_config() {
  local key="$1"
  local label="$2"
  local override_payload="$3"
  local target_total=$((WARMUP_SWEEPS + MEASURE_SWEEPS))
  local logfile="${LOG_DIR}/${key}.log"

  adb -s "${SERIAL}" shell am force-stop com.gneissfind.app
  sleep "${COOLDOWN_SEC}"
  adb -s "${SERIAL}" shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1 || true

  local soc_temp
  soc_temp="$(adb -s "${SERIAL}" shell dumpsys thermalservice | grep "mName=soc_therm" | tail -n 1 | sed -E 's/.*mValue=([0-9.]+).*/\1/' | tr -d '\r' || echo "n/a")"
  echo "  -> Benchmarking ${label} [soc_therm=${soc_temp}C] (waiting for ${target_total} sweeps)..."

  adb -s "${SERIAL}" shell "echo '${override_payload}' > ${DEVICE_FILES_DIR}/benchmark_override.json"
  adb -s "${SERIAL}" logcat -c
  adb -s "${SERIAL}" shell am start -n com.gneissfind.app/.MainActivity >/dev/null

  local elapsed=0
  local max_wait=180
  while [[ "${elapsed}" -lt "${max_wait}" ]]; do
    sleep 2
    elapsed=$((elapsed + 2))
    adb -s "${SERIAL}" logcat -d > "${logfile}"
    local highest_sweep
    highest_sweep="$(grep -oE "RFDETR_BENCH sweep=[0-9]+" "${logfile}" | awk -F= '{print $2}' | sort -n | tail -n 1 || echo 0)"
    if [[ -n "${highest_sweep}" && "${highest_sweep}" -ge "${target_total}" ]]; then
      break
    fi
  done
  adb -s "${SERIAL}" shell am force-stop com.gneissfind.app
}

echo "[2/3] Running isolated on-device benchmark matrix..."
run_config "2g_b5_5t" "2-Graph B=5 Batched (5-Tile)" \
  "{\"topology\":\"two_graph\",\"batch_size\":5,\"active_tiles\":5,\"part1_path\":\"${DEVICE_FILES_DIR}/rf_detr_nano_part1_b5.tflite\",\"part2_path\":\"${DEVICE_FILES_DIR}/rf_detr_nano_part2_b5.tflite\"}"

run_config "2g_b1_5t" "2-Graph 5xB=1 Sequential (5-Tile)" \
  "{\"topology\":\"two_graph\",\"batch_size\":1,\"active_tiles\":5,\"part1_path\":\"${DEVICE_FILES_DIR}/rf_detr_nano_part1_b1.tflite\",\"part2_path\":\"${DEVICE_FILES_DIR}/rf_detr_nano_part2_b1.tflite\"}"

run_config "1g_b5_5t" "1-Graph B=5 Batched (5-Tile)" \
  "{\"topology\":\"single_graph\",\"batch_size\":5,\"active_tiles\":5,\"single_graph_path\":\"${DEVICE_FILES_DIR}/rf_detr_nano_1g_b5.tflite\"}"

run_config "1g_b1_5t" "1-Graph 5xB=1 Sequential (5-Tile)" \
  "{\"topology\":\"single_graph\",\"batch_size\":1,\"active_tiles\":5,\"single_graph_path\":\"${DEVICE_FILES_DIR}/rf_detr_nano_1g_b1.tflite\"}"

run_config "2g_b1_3t" "2-Graph 3xB=1 Sequential (3-Tile Narrow)" \
  "{\"topology\":\"two_graph\",\"batch_size\":1,\"active_tiles\":3,\"part1_path\":\"${DEVICE_FILES_DIR}/rf_detr_nano_part1_b1.tflite\",\"part2_path\":\"${DEVICE_FILES_DIR}/rf_detr_nano_part2_b1.tflite\"}"

# Restore default packaged 2-Graph B=5 assets and relaunch app
adb -s "${SERIAL}" shell "rm -f ${DEVICE_FILES_DIR}/benchmark_override.json"
adb -s "${SERIAL}" shell am start -n com.gneissfind.app/.MainActivity >/dev/null

echo "[3/3] Computing statistics and writing ${REPORT_JSON}..."
python3 - "${LOG_DIR}" "${WARMUP_SWEEPS}" "${MEASURE_SWEEPS}" "${REPORT_JSON}" "${SERIAL}" "${MODEL_NAME}" "${SOC_MODEL}" "${BUILD_FINGERPRINT}" <<'PY'
import json
import re
import sys
from pathlib import Path

import numpy as np

log_dir = Path(sys.argv[1])
warmup = int(sys.argv[2])
measure = int(sys.argv[3])
report_path = Path(sys.argv[4])
serial, model_name, soc_model, fingerprint = sys.argv[5:9]

configs = [
    ("2g_b5_5t", "2-Graph B=5 Batched (5-Tile)"),
    ("2g_b1_5t", "2-Graph 5xB=1 Sequential (5-Tile)"),
    ("1g_b5_5t", "1-Graph B=5 Batched (5-Tile)"),
    ("1g_b1_5t", "1-Graph 5xB=1 Sequential (5-Tile)"),
    ("2g_b1_3t", "2-Graph 3xB=1 Sequential (3-Tile)"),
]

bench_re = re.compile(
    r"RFDETR_BENCH sweep=(\d+) topology=(\S+) batch=(\d+) tiles=(\d+) accel=(\S+) "
    r"total_ms=([\d.]+) p1_ms=([\d.]+) topk_ms=([\d.]+) p2_ms=([\d.]+) rolling_ms=([\d.]+)"
)
part_re = re.compile(
    r"Replacing (\d+) out of (\d+) node\(s\) with delegate \(([^)]+)\) node, yielding (\d+) partitions"
)

results = {}
print()
print("| Configuration | LITERT_CL GPU Partitions | P1 Mean (ms) | TopK Mean (ms) | P2 Mean (ms) | Sweep Min (ms) | Sweep p50 (ms) | Sweep p95 (ms) | Throughput (Hz) |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")

for key, label in configs:
    text = (log_dir / f"{key}.log").read_text(encoding="utf-8", errors="replace")
    delegates = part_re.findall(text)
    gpu_parts = [f"{d}/{t} ({p}p)" for d, t, name, p in delegates if "LITERT_CL" in name]
    xnn_parts = [f"XNN:{d}/{t} ({p}p)" for d, t, name, p in delegates if "XNNPack" in name]
    part_summary = " + ".join(gpu_parts + xnn_parts) if (gpu_parts or xnn_parts) else "n/a"

    rows = []
    for m in bench_re.finditer(text):
        sweep_idx = int(m.group(1))
        if sweep_idx > warmup and len(rows) < measure:
            rows.append(
                {
                    "sweep": sweep_idx,
                    "total_ms": float(m.group(6)),
                    "p1_ms": float(m.group(7)),
                    "topk_ms": float(m.group(8)),
                    "p2_ms": float(m.group(9)),
                }
            )
    if not rows:
        continue
    totals = np.array([r["total_ms"] for r in rows], dtype=np.float64)
    p1s = np.array([r["p1_ms"] for r in rows], dtype=np.float64)
    topks = np.array([r["topk_ms"] for r in rows], dtype=np.float64)
    p2s = np.array([r["p2_ms"] for r in rows], dtype=np.float64)

    min_total = float(np.min(totals))
    p50 = float(np.percentile(totals, 50))
    p95 = float(np.percentile(totals, 95))
    mean_total = float(np.mean(totals))
    p1_mean = float(np.mean(p1s))
    topk_mean = float(np.mean(topks))
    p2_mean = float(np.mean(p2s))
    hz = 1000.0 / p50 if p50 > 0 else 0.0

    results[key] = {
        "label": label,
        "samples": len(rows),
        "partitions": part_summary,
        "p1_mean_ms": round(p1_mean, 2),
        "topk_mean_ms": round(topk_mean, 2),
        "p2_mean_ms": round(p2_mean, 2),
        "total_min_ms": round(min_total, 2),
        "total_mean_ms": round(mean_total, 2),
        "total_p50_ms": round(p50, 2),
        "total_p95_ms": round(p95, 2),
        "sustained_hz": round(hz, 2),
    }
    print(
        f"| {label} | `{part_summary}` | {p1_mean:.2f} | {topk_mean:.2f} | "
        f"{p2_mean:.2f} | {min_total:.2f} | **{p50:.2f}** | {p95:.2f} | **{hz:.2f} Hz** |"
    )

payload = {
    "device": {
        "serial": serial,
        "model": model_name,
        "soc": soc_model,
        "build_fingerprint": fingerprint,
    },
    "warmup_sweeps": warmup,
    "measured_sweeps": measure,
    "configurations": results,
}
report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print()
print(f"Saved JSON report -> {report_path}")
PY
