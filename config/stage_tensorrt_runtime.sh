#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="10.13.3.9"
MANIFEST="$REPO_ROOT/config/ppocrv6-tensorrt-runtime-${VERSION}.sha256"
DEST_PARENT="$REPO_ROOT/models_slim"
DEST="$DEST_PARENT/tensorrt-runtime-${VERSION}"

verify_bundle() {
  local bundle="$1"
  local expected relative local_path actual
  [ -d "$bundle" ] || return 1
  while read -r expected relative; do
    local_path="${relative#tensorrt-runtime-${VERSION}/}"
    [ -f "$bundle/$local_path" ] || return 1
    actual="$(sha256sum "$bundle/$local_path" | awk '{print $1}')"
    [ "$actual" = "$expected" ] || return 1
  done < "$MANIFEST"
}

if verify_bundle "$DEST"; then
  echo "TensorRT ${VERSION} inference runtime already staged and verified: $DEST"
  exit 0
fi

SOURCE="${TENSORRT_RUNTIME_SOURCE_DIR:-}"
if [ -z "$SOURCE" ]; then
  for candidate in \
    "$REPO_ROOT/../embodied-ai/models/tensorrt-runtime-${VERSION}" \
    "/models/ocr/tensorrt-runtime-${VERSION}"; do
    if verify_bundle "$candidate"; then
      SOURCE="$candidate"
      break
    fi
  done
fi

if [ -z "$SOURCE" ] || ! verify_bundle "$SOURCE"; then
  echo "TensorRT ${VERSION} inference runtime is missing or failed checksum verification." >&2
  echo "Set TENSORRT_RUNTIME_SOURCE_DIR to the pinned inference-only runtime bundle." >&2
  exit 1
fi

mkdir -p "$DEST"
cp -a "$SOURCE/tensorrt" "$DEST/"
cp -a "$SOURCE/tensorrt_bindings" "$DEST/"
cp -a "$SOURCE/tensorrt_libs" "$DEST/"
install -m 0644 "$SOURCE/LICENSE.txt" "$DEST/LICENSE.txt"
verify_bundle "$DEST"
echo "Staged TensorRT ${VERSION} inference runtime from $SOURCE to $DEST"
