#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_FILE="${1:-/tmp/gneissfind-sync.md}"

{
  echo "# GneissFind Context Sync Snapshot"
  echo "Generated: $(date -Is)"
  echo ""
  echo "## Git Branch & Recent Commits"
  echo '```text'
  git -C "$REPO_ROOT" status -sb
  echo "---"
  git -C "$REPO_ROOT" log -n 12 --oneline
  echo '```'
  echo ""
  if [[ -f "$REPO_ROOT/android/app/src/main/assets/rf_detr_coco_metadata.json" ]]; then
    echo "## Exported LiteRT Model Metadata (rf_detr_coco_metadata.json)"
    echo '```json'
    cat "$REPO_ROOT/android/app/src/main/assets/rf_detr_coco_metadata.json"
    echo '```'
    echo ""
  fi
  echo "## Key Kotlin Domain & Detector Contracts"
  for rel in \
    "android/app/src/main/java/com/gneissfind/app/model/StoneModels.kt" \
    "android/app/src/main/java/com/gneissfind/app/inference/StoneDetector.kt"; do
    if [[ -f "$REPO_ROOT/$rel" ]]; then
      echo "### $rel"
      echo '```kotlin'
      cat "$REPO_ROOT/$rel"
      echo '```'
      echo ""
    fi
  done
  echo "## Engineering Ledger (docs/ENGINEERING_LEDGER.md)"
  cat "$REPO_ROOT/docs/ENGINEERING_LEDGER.md"
} > "$OUT_FILE"

echo "Wrote sync bundle to $OUT_FILE ($(wc -c < "$OUT_FILE") bytes)"
