#!/bin/bash
set -euo pipefail

export PORT="${PORT:-8001}"
export HOST="${HOST:-0.0.0.0}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/app/output}"
mkdir -p "$OUTPUT_DIR"

exec python3 main.py
