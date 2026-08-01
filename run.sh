
#!/bin/bash
set -euo pipefail

: "${MODELHUB_API_KEY:?Set MODELHUB_API_KEY in the environment or secret manager}"
: "${MODELHUB_BASE_URL:?Set MODELHUB_BASE_URL in the environment}"
: "${MODELHUB_MODEL_NAME:?Set MODELHUB_MODEL_NAME in the environment}"
export MODELHUB_API_KEY MODELHUB_BASE_URL MODELHUB_MODEL_NAME

ENABLE_HEURISTIC_AUDIT_FALLBACK=0 \
ENABLE_RESUME_SHRINK_GUARD=1 \
DEFAULT_OUTPUT_FORMAT="both" \
HOST="0.0.0.0" \
PORT="8001" \
python3 main.py
