#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO_ROOT/config/ppocrv6-trt-a100-fp32.sha256"
DEST_PARENT="$REPO_ROOT/models_slim"
DEST="$DEST_PARENT/ppocrv6-trt-a100"

verify_bundle() {
  local bundle="$1"
  local expected relative filename actual
  [ -d "$bundle" ] || return 1
  while read -r expected relative; do
    filename="${relative##*/}"
    [ -f "$bundle/$filename" ] || return 1
    actual="$(sha256sum "$bundle/$filename" | awk '{print $1}')"
    [ "$actual" = "$expected" ] || return 1
  done < "$MANIFEST"
}

if verify_bundle "$DEST"; then
  echo "PP-OCRv6 A100 TensorRT FP32 bundle already staged and verified: $DEST"
  exit 0
fi

SOURCE="${PPOCRV6_TRT_SOURCE_DIR:-}"
if [ -z "$SOURCE" ]; then
  for candidate in \
    "$REPO_ROOT/../embodied-ai/models/ppocrv6-trt-a100-fp32" \
    "/models/ocr/ppocrv6-trt-a100-fp32"; do
    if verify_bundle "$candidate"; then
      SOURCE="$candidate"
      break
    fi
  done
fi

if [ -z "$SOURCE" ] || ! verify_bundle "$SOURCE"; then
  echo "PP-OCRv6 A100 TensorRT FP32 source is missing or failed checksum verification." >&2
  echo "Set PPOCRV6_TRT_SOURCE_DIR to a bundle built with TensorRT 10.13.3.9 on A100." >&2
  exit 1
fi

mkdir -p "$DEST"
for filename in primary-rec.engine secondary-rec.engine medium-rec.engine keys.txt; do
  install -m 0644 "$SOURCE/$filename" "$DEST/$filename"
done
verify_bundle "$DEST"
echo "Staged PP-OCRv6 A100 TensorRT FP32 bundle from $SOURCE to $DEST"
