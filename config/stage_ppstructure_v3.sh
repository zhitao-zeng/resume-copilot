#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO_ROOT/config/ppstructure-v3-layout-ocr.sha256"
DEST_ROOT="$REPO_ROOT/models_slim/ppstructure-v3/official_models"

MODELS=(
  PP-DocLayout_plus-L
  PP-LCNet_x1_0_textline_ori
  PP-OCRv5_server_det
  PP-OCRv5_server_rec
)

verify_bundle() {
  local root="$1"
  local expected relative source_relative actual
  [ -d "$root" ] || return 1
  while read -r expected relative; do
    source_relative="${relative#ppstructure-v3/official_models/}"
    [ -f "$root/$source_relative" ] || return 1
    actual="$(sha256sum "$root/$source_relative" | awk '{print $1}')"
    [ "$actual" = "$expected" ] || return 1
  done < "$MANIFEST"
}

if verify_bundle "$DEST_ROOT"; then
  echo "PP-StructureV3 layout+OCR models already staged and verified: $DEST_ROOT"
  exit 0
fi

SOURCE="${PPSTRUCTURE_MODEL_SOURCE_DIR:-}"
if [ -z "$SOURCE" ]; then
  for candidate in \
    "/mnt/disk2/zengzhitao/data/resume-copilot/ppstructure-v3-cache-v33-ms/official_models" \
    "/models/ocr/ppstructure-v3/official_models"; do
    if verify_bundle "$candidate"; then
      SOURCE="$candidate"
      break
    fi
  done
fi

if [ -z "$SOURCE" ] || ! verify_bundle "$SOURCE"; then
  echo "PP-StructureV3 source is missing or failed checksum verification." >&2
  echo "Set PPSTRUCTURE_MODEL_SOURCE_DIR to the PaddleX official_models directory." >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  install -d -m 0755 "$DEST_ROOT/$model"
done
while read -r _expected relative; do
  source_relative="${relative#ppstructure-v3/official_models/}"
  install -m 0644 "$SOURCE/$source_relative" "$DEST_ROOT/$source_relative"
done < "$MANIFEST"

verify_bundle "$DEST_ROOT"
echo "Staged PP-StructureV3 layout+OCR models from $SOURCE to $DEST_ROOT"
