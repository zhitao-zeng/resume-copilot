#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO_ROOT/config/ppocrv6-small-ort.sha256"
DEST_PARENT="$REPO_ROOT/models_slim"
DEST="$DEST_PARENT/ppocrv6-small-ort"

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
  echo "PP-OCRv6 Small already staged and verified: $DEST"
  exit 0
fi

SOURCE="${PPOCRV6_SOURCE_DIR:-}"
if [ -z "$SOURCE" ]; then
  for candidate in \
    "$REPO_ROOT/../embodied-ai/models/ppocrv6-small-ort" \
    "/models/ocr/ppocrv6-small-ort"; do
    if verify_bundle "$candidate"; then
      SOURCE="$candidate"
      break
    fi
  done
fi

if [ -z "$SOURCE" ] || ! verify_bundle "$SOURCE"; then
  echo "PP-OCRv6 Small source is missing or failed checksum verification." >&2
  echo "Set PPOCRV6_SOURCE_DIR to the embodied-ai ppocrv6-small-ort directory." >&2
  exit 1
fi

mkdir -p "$DEST"
for filename in det.onnx rec.onnx cls.onnx keys.txt; do
  install -m 0644 "$SOURCE/$filename" "$DEST/$filename"
done
verify_bundle "$DEST"
echo "Staged PP-OCRv6 Small from $SOURCE to $DEST"
