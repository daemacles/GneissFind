#!/usr/bin/env bash
set -euo pipefail

# Packs active GneissFind contracts + hot Engineering Ledger into /tmp/gneissfind-sync.md.
# Cold archives under docs/archive/* are intentionally excluded to keep the sync bundle < 25 KB.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_FILE="${1:-/tmp/gneissfind-sync.md}"

compact_json_meta() {
  python3 -c '
import json, sys
def slim(obj):
    if isinstance(obj, list):
        return f"<{len(obj)} items>" if len(obj) > 10 else [slim(x) for x in obj]
    if isinstance(obj, dict):
        if len(obj) > 12:
            return {k: slim(v) for k, v in obj.items() if v not in (4, "HARD_CONFUSER")}
        return {k: slim(v) for k, v in obj.items()}
    return obj
print(json.dumps(slim(json.load(open(sys.argv[1], encoding="utf-8"))), separators=(",", ":")))
' "$1"
}

compact_kotlin_contracts() {
  python3 -c '
import re, sys
src = open(sys.argv[1], encoding="utf-8").read()
src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
lines = []
for ln in src.splitlines():
    ln = re.sub(r"\s*//.*$", "", ln).strip()
    if not ln or ln.startswith(("import ", "package ")):
        continue
    lines.append(ln)
text = "\n".join(lines)
text = re.sub(r"\(\n\s*", "(", text)
text = re.sub(r",\n\s*", ", ", text)
text = re.sub(r"\n\)", ")", text)
# Strip method bodies inside data/enum/object types while preserving type/property/interface declarations
out, depth, skip_body = [], 0, False
for ln in text.splitlines():
    opens, closes = ln.count("{"), ln.count("}")
    if skip_body:
        depth += opens - closes
        if depth <= 0:
            skip_body = False
        continue
    if re.match(r"^(?:override\s+|private\s+|internal\s+)?fun\b", ln):
        if "{" in ln and not ln.endswith("{}"):
            sig = ln.split("{", 1)[0].rstrip()
            if not ln.startswith(("override fun equals", "override fun hashCode")):
                out.append(sig)
            delta = opens - closes
            if delta > 0:
                depth = delta
                skip_body = True
            continue
        if ") =" in ln or "): " in ln and " = " in ln:
            out.append(ln.split(" = ", 1)[0].rstrip())
            continue
    out.append(ln)
res = "\n".join(out)
res = re.sub(r"(\bval\s+\w+:\s*[A-Za-z0-9_<>,?]+)\s*=\s*[^,)]+", r"\1", res)
res = re.sub(r"\b[a-zA-Z0-9_]+\s*=\s*(?=[\"0-9tfl-])", "", res)
print(res)
' "$1"
}

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
  for meta in \
    "android/app/src/main/assets/rf_detr_coco_metadata.json" \
    "android/app/src/main/assets/rf_detr_320_b1_metadata.json"; do
    if [[ -f "$REPO_ROOT/$meta" ]]; then
      echo "## Exported LiteRT Metadata ($meta)"
      echo '```json'
      compact_json_meta "$REPO_ROOT/$meta"
      echo '```'
      echo ""
    fi
  done
  echo "## Key Kotlin Domain & Detector Contracts"
  for rel in \
    "android/app/src/main/java/com/gneissfind/app/model/StoneModels.kt" \
    "android/app/src/main/java/com/gneissfind/app/inference/StoneDetector.kt"; do
    if [[ -f "$REPO_ROOT/$rel" ]]; then
      echo "### $rel"
      echo '```kotlin'
      compact_kotlin_contracts "$REPO_ROOT/$rel"
      echo '```'
      echo ""
    fi
  done
  echo "## Engineering Ledger (docs/ENGINEERING_LEDGER.md)"
  cat "$REPO_ROOT/docs/ENGINEERING_LEDGER.md"
} > "$OUT_FILE"

echo "Wrote sync bundle to $OUT_FILE ($(wc -c < "$OUT_FILE") bytes)"
