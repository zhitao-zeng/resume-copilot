#!/usr/bin/env bash
# Python-interpreter-compatible adapter for local PP-Structure GPU evaluation.
#
# ``core.ppstructure_runtime`` invokes its configured interpreter as:
#   INTERPRETER SCRIPT --worker-input ... --worker-output ... --worker-format ...
# This adapter preserves that contract while running the worker inside the
# already-built Paddle GPU image.  It is evaluation-only; production runs the
# isolated interpreter directly inside its application container.

set -Eeuo pipefail

die() {
  printf '[ppstructure-docker] ERROR: %s\n' "$*" >&2
  exit 1
}

[[ $# -ge 1 ]] || die "missing worker script"
worker_script="$1"
shift

input_path=""
output_path=""
output_format="text"
while (($#)); do
  case "$1" in
    --worker-input)
      (($# >= 2)) || die "--worker-input requires a value"
      input_path="$2"
      shift 2
      ;;
    --worker-output)
      (($# >= 2)) || die "--worker-output requires a value"
      output_path="$2"
      shift 2
      ;;
    --worker-format)
      (($# >= 2)) || die "--worker-format requires a value"
      output_format="$2"
      shift 2
      ;;
    *)
      die "unsupported worker argument: $1"
      ;;
  esac
done

[[ -f "$worker_script" ]] || die "worker script not found: $worker_script"
[[ -f "$input_path" ]] || die "worker input not found: $input_path"
[[ -n "$output_path" && -d "$(dirname "$output_path")" ]] || die \
  "worker output directory is unavailable: $output_path"
[[ "$output_format" == text || "$output_format" == blocks ]] || die \
  "invalid worker format: $output_format"

gpu_id="${PPSTRUCTURE_DOCKER_GPU_ID:-}"
[[ "$gpu_id" =~ ^[0-9]+$ ]] || die "PPSTRUCTURE_DOCKER_GPU_ID must be explicit"
allowed_gpus=",${PPSTRUCTURE_DOCKER_ALLOWED_GPUS:-3,4,5,6},"
[[ "$allowed_gpus" == *",$gpu_id,"* ]] || die \
  "GPU $gpu_id is outside PPSTRUCTURE_DOCKER_ALLOWED_GPUS"

image="${PPSTRUCTURE_DOCKER_IMAGE:-resume-copilot:ppstructure-gpu-sharedcuda-local}"
source_root="${LOCAL_EVAL_SOURCE_ROOT:-$(cd "$(dirname "$worker_script")/.." && pwd)}"
[[ -f "$source_root/core/ppstructure_runtime.py" ]] || die \
  "source root has no PP-Structure worker: $source_root"

input_dir="$(dirname "$input_path")"
output_dir="$(dirname "$output_path")"
input_name="$(basename "$input_path")"
output_name="$(basename "$output_path")"

exec docker run --rm --gpus "device=$gpu_id" \
  --entrypoint /opt/ppstructure-venv/bin/python \
  -e PPSTRUCTURE_DEVICE=gpu:0 \
  -e PPSTRUCTURE_MODEL_DIR=/root/app/models/ppstructure-v3/official_models \
  -e DISABLE_MODEL_SOURCE_CHECK=True \
  -v "$source_root/core:/workspace/core:ro" \
  -v "$input_dir:/worker-input:ro" \
  -v "$output_dir:/worker-output" \
  "$image" \
  /workspace/core/ppstructure_runtime.py \
  --worker-input "/worker-input/$input_name" \
  --worker-output "/worker-output/$output_name" \
  --worker-format "$output_format"
