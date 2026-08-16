#!/usr/bin/env bash
# Assemble the R24 quality-gate evidence from a finished full60 run.
# Usage: bash tools/r24_quality_gate.sh <full60-result.json>
set -Eeuo pipefail

RESULT="${1:?usage: r24_quality_gate.sh <full60-result.json>}"
ART=.codex/research-loop/artifacts/darvin-aligned-quality-20260816
OUT="$ART/quality-gate-r24-v3"
mkdir -p "$OUT"

source .venv/bin/activate

echo "== reaudit (evaluator 1.2 semantics on R24 rows)"
python validation_sets/public_resume_holdout/reaudit.py \
  --input "$RESULT" --output "$OUT/r24-full60-reaudit.json" >/dev/null 2>&1

echo "== r3 components (R24)"
python validation_sets/public_resume_holdout/darvin_components.py \
  --input "$OUT/r24-full60-reaudit.json" \
  --output "$OUT/r24-full60-darvin-r3.json" 2>/dev/null | grep -v weasyprint

echo "== r3 components (V2 immutable, refreshed with audited evaluator)"
python validation_sets/public_resume_holdout/darvin_components.py \
  --input "$ART/v2-full60-reaudit12.json" \
  --output "$OUT/v2-full60-darvin-r3-audited.json" 2>/dev/null | grep -v weasyprint

echo "== done -> $OUT"
