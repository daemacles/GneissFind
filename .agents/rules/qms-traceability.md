---
trigger: always_on
description: Lightweight Git-native Trace-as-Code QMS rules linking requirements (qms/reqs/) to Rust/Kotlin source symbols and automated Verification & Validation tests.
---

# Lightweight "Trace-as-Code" QMS Specification

## 1. Core Philosophy
Every functional requirement and hardware invariant must trace **downward to source code** and **forward to an automated Verification or Validation test**, verified in `< 2 seconds` by `python3 scripts/qms_verify.py` (`just qms-check`) with zero external SaaS or manual spreadsheets.

## 2. Requirement ID Taxonomy (`qms/reqs/*.md`)
- **`UN-xxx` (User Needs):** Field rockhounding outcomes (`qms/reqs/01_user_needs.md`).
- **`SYS-xxx` (System Requirements):** Optical geometry, seam merging, latency,HIL injection, and `.mcap` recording (`qms/reqs/02_system_reqs.md`).
- **`INV-xxx` (Sacred Hardware & UI Invariants):** Empirical Camera2, `LITERT_CL` 2-graph GPU, ring-buffer, and LCARS invariants from `ENGINEERING_LEDGER.md` (`qms/reqs/03_invariants_and_algos.md`).
- **`ALG-xxx` & `ALG-TUNE-xxx` (Algorithms & Open Tuning Constants):** Model quantization parity, N-of-M voting, and explicit tracking of the 6 `RATIONALE-UNKNOWN` constants (`ALG-TUNE-001..006`, allowed status `Open-M3-Sweep` until Milestone 3 field `.mcap` calibration locks them).

## 3. How to Annotate Code & Tests
`scripts/qms_verify.py` scans both Rust (`crates/`, `cli_runner/`) and Kotlin/Python (`android/`, `ml-pipeline/`) for canonical `@TracesReq` tags in doc-comments, attributes, or annotations:
- **In Rust Implementation & Tests (`*.rs`):**
  Place `/// @TracesReq(SYS-SEAM-001)` (or `#[doc = "@TracesReq(SYS-SEAM-001)"]`) directly above the implementing `fn`/`struct`/`impl` AND above the `#[test]` verifying it. Multiple IDs may be comma-separated: `/// @TracesReq(SYS-OPT-002, INV-CAM-002)`.
- **In Kotlin Implementation & Tests (`*.kt`):**
  Use `@TracesReq("INV-CAM-001")` (`package com.gneissfind.app.qms`, `RetentionPolicy.SOURCE` repeatable/vararg annotation) OR KDoc `/** @TracesReq(INV-CAM-001) */` above the class/function and unit/instrumented test.
- **In Python ML Scripts & Tests (`ml-pipeline/*.py`):**
  Use `# @TracesReq(ALG-DET-001)` above the exporter/parity function and `pytest` function.

## 4. Mandatory QMS Gate (`python3 scripts/qms_verify.py`)
Before closing any work batch or milestone:
1. Run `python3 scripts/qms_verify.py --write-matrix docs/TRACEABILITY_MATRIX.md`.
2. Every `Active` requirement (`SYS-*`, `INV-*`, `ALG-*` excluding `Status: Open-M3-Sweep`) MUST have:
   - `>= 1` production implementation link (`impl_refs`), AND
   - `>= 1` automated test or V&V oracle link (`test_refs`).
3. Zero unknown/orphan `@TracesReq(...)` IDs may exist in source code.
