#!/usr/bin/env bash
# Reproducible local 27B/OCR evaluation cluster for the public 60-case holdout.
#
# The script deliberately refuses to reuse unowned API listeners or evaluate
# code that changed after the API processes started.  This prevents two common
# false comparisons: requests silently reaching an old process, and Composer
# falling back to its default 8K application-side context budget.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${LOCAL_EVAL_PYTHON:-$REPO_ROOT/.venv/bin/python}"
INSTANCE_COUNT="${LOCAL_EVAL_INSTANCE_COUNT:-4}"
API_PORT_BASE="${LOCAL_EVAL_API_PORT_BASE:-18085}"
MODEL_PORT_BASE="${LOCAL_EVAL_MODEL_PORT_BASE:-8007}"
GPU_IDS="${LOCAL_EVAL_GPU_IDS:-3,4,5,6}"
PIPELINE_PROFILE="${LOCAL_EVAL_PIPELINE_PROFILE:-current_control}"
FACT_COMPILER_MODE="${LOCAL_EVAL_FACT_COMPILER_MODE:-on}"

WORKSPACE_ROOT="${LOCAL_EVAL_WORKSPACE_ROOT:-/mnt/disk1/zengzhitao}"
TOKENIZER_DIR="${LOCAL_EVAL_TOKENIZER_DIR:-$WORKSPACE_ROOT/models/Qwen3.5-27B-AWQ}"
OCR_SITE="${LOCAL_EVAL_OCR_SITE:-$WORKSPACE_ROOT/embodied-ai/ocr_eval/.venv/lib/python3.13/site-packages}"
OCR_PRIMARY_DIR="${LOCAL_EVAL_OCR_PRIMARY_DIR:-$WORKSPACE_ROOT/embodied-ai/models/ppocrv6-small-finetune-ort}"
OCR_SECONDARY_REC="${LOCAL_EVAL_OCR_SECONDARY_REC:-$WORKSPACE_ROOT/embodied-ai/models/ppocrv6-small-ort/rec.onnx}"
OCR_NUMERIC_REC="${LOCAL_EVAL_OCR_NUMERIC_REC:-$WORKSPACE_ROOT/embodied-ai/models/ppocrv6-medium-ort/rec.onnx}"

VLLM_IMAGE="${LOCAL_EVAL_VLLM_IMAGE:-harbor-contest.4pd.io/zengzhitao/resume-copilot:f507820-post-guard-consistency}"
VLLM_CONTAINER_PREFIX="${LOCAL_EVAL_VLLM_CONTAINER_PREFIX:-resume-fcv1-vllm-gpu}"
VLLM_GPU_MEMORY_UTILIZATION="${LOCAL_EVAL_VLLM_GPU_MEMORY_UTILIZATION:-0.48}"
MODEL_WAIT_SECONDS="${LOCAL_EVAL_MODEL_WAIT_SECONDS:-1800}"

RUNTIME_DIR="${LOCAL_EVAL_RUNTIME_DIR:-$REPO_ROOT/.codex/research-loop/runtime/local-eval-cluster}"
ARTIFACT_ROOT="${LOCAL_EVAL_ARTIFACT_ROOT:-$REPO_ROOT/.codex/research-loop/artifacts/local-eval-cluster}"
CASES_PATH="${LOCAL_EVAL_CASES:-$REPO_ROOT/validation_sets/public_resume_holdout/holdout_v2/cases.jsonl}"
ANNOTATIONS_PATH="${LOCAL_EVAL_ANNOTATIONS:-$REPO_ROOT/validation_sets/public_resume_holdout/holdout_v2/annotations.jsonl}"
EVALUATOR="$REPO_ROOT/validation_sets/public_resume_holdout/evaluate.py"
MERGER="$REPO_ROOT/validation_sets/public_resume_holdout/merge_results.py"

log() {
  printf '[local-eval] %s\n' "$*"
}

die() {
  printf '[local-eval] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: bash tools/local_eval_cluster.sh COMMAND [ARGS]

Environment and lifecycle:
  preflight                 Validate paths, Python/OCR, GPUs, model APIs and ports
  status                    Show model/API ownership, health, context and code freshness
  models-up                 Start missing vLLM containers and wait for readiness
  models-down               Stop only the named local-eval vLLM containers
  api-up                    Start four managed business APIs; refuses occupied ports
  api-down                  Stop only APIs created by this script
  restart                   Restart managed APIs so they load the current source tree
  up                        Run models-up followed by api-up

Evaluation:
  eval-case CASE_ID [INDEX] Run one case on managed API INDEX (default: 0)
  eval-subset RUN_ID IDS    Run comma-separated case IDs across managed APIs
  eval-plan                 Print the deterministic four-way case assignment
  eval-full [RUN_ID]        Run the frozen 60 cases in four modulo shards and merge
  summary RESULT.json       Print promotion metrics from an evaluator result
  logs [INDEX]              Follow one managed API log (default: 0)

Common overrides:
  LOCAL_EVAL_GPU_IDS=3,4,5,6
  LOCAL_EVAL_PIPELINE_PROFILE=current_control|f507_compatible|ledger_shadow|local_repair|fact_compiler|candidate|quality_v2
  LOCAL_EVAL_FACT_COMPILER_MODE=legacy|shadow|on
  LOCAL_EVAL_VLLM_IMAGE=immutable-image-tag
  LOCAL_EVAL_ARTIFACT_ROOT=/durable/output/path

Safety:
  api-up never kills or adopts an existing listener. api-down only kills a
  process group recorded by this script whose command still matches this repo.
  eval-* requires all APIs to be managed, healthy, configured for 16K, and
  loaded from the current source digest.
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "missing directory: $1"
}

direct_curl() {
  curl --noproxy '*' "$@"
}

parse_gpu_ids() {
  IFS=',' read -r -a PARSED_GPU_IDS <<<"$GPU_IDS"
  [[ "${#PARSED_GPU_IDS[@]}" -eq "$INSTANCE_COUNT" ]] || die \
    "LOCAL_EVAL_GPU_IDS has ${#PARSED_GPU_IDS[@]} entries; expected $INSTANCE_COUNT"
  local gpu_id
  for gpu_id in "${PARSED_GPU_IDS[@]}"; do
    [[ "$gpu_id" =~ ^[0-9]+$ ]] || die "invalid GPU id: $gpu_id"
  done
}

code_digest() {
  (
    cd "$REPO_ROOT"
    {
      printf '%s\0' main.py
      find core -maxdepth 1 -type f -name '*.py' -print0
    } | sort -z | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

api_pid_file() {
  printf '%s/api-%s.pid' "$RUNTIME_DIR" "$1"
}

api_log_path_file() {
  printf '%s/api-%s.logpath' "$RUNTIME_DIR" "$1"
}

api_output_dir() {
  printf '%s/api-output-%s' "$RUNTIME_DIR" "$1"
}

manifest_path() {
  printf '%s/cluster.env' "$RUNTIME_DIR"
}

read_manifest_value() {
  local key="$1"
  local manifest
  manifest="$(manifest_path)"
  [[ -f "$manifest" ]] || return 1
  sed -n "s/^${key}=//p" "$manifest" | head -n 1
}

process_matches_repo_api() {
  local pid="$1"
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Fq "$REPO_ROOT/main.py"
}

managed_api_pid() {
  local port="$1"
  local pid_file pid
  pid_file="$(api_pid_file "$port")"
  [[ -f "$pid_file" ]] || return 1
  pid="$(sed -n '1p' "$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  process_matches_repo_api "$pid" || return 1
  printf '%s' "$pid"
}

listening_pid() {
  fuser -n tcp "$1" 2>/dev/null | awk '{print $1}' | head -n 1
}

process_env_value() {
  local pid="$1"
  local key="$2"
  [[ -r "/proc/$pid/environ" ]] || return 1
  tr '\0' '\n' <"/proc/$pid/environ" | sed -n "s/^${key}=//p" | head -n 1
}

api_is_ready() {
  direct_curl -fsS --max-time 2 "http://127.0.0.1:$1/ready" >/dev/null 2>&1
}

model_is_ready() {
  direct_curl -fsS --max-time 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1
}

model_id() {
  direct_curl -fsS --max-time 3 "http://127.0.0.1:$1/v1/models" 2>/dev/null \
    | jq -r '.data[0].id // "-"' 2>/dev/null || printf '-'
}

preflight_files() {
  local command_name
  for command_name in curl docker find flock fuser jq nvidia-smi setsid sha256sum ss tee xargs; do
    require_command "$command_name"
  done
  require_file "$PYTHON_BIN"
  require_dir "$TOKENIZER_DIR"
  require_file "$TOKENIZER_DIR/tokenizer_config.json"
  require_dir "$OCR_SITE"
  require_dir "$OCR_PRIMARY_DIR"
  require_file "$OCR_PRIMARY_DIR/det.onnx"
  require_file "$OCR_PRIMARY_DIR/cls.onnx"
  require_file "$OCR_PRIMARY_DIR/rec.onnx"
  require_file "$OCR_SECONDARY_REC"
  require_file "$OCR_NUMERIC_REC"
  require_file "$CASES_PATH"
  require_file "$ANNOTATIONS_PATH"
  require_file "$EVALUATOR"
  require_file "$MERGER"
  parse_gpu_ids
  [[ "$PIPELINE_PROFILE" =~ ^(current_control|f507_compatible|ledger_shadow|local_repair|fact_compiler|candidate|quality_v2)$ ]] || die \
    "invalid LOCAL_EVAL_PIPELINE_PROFILE: $PIPELINE_PROFILE"
  [[ "$FACT_COMPILER_MODE" =~ ^(legacy|shadow|on)$ ]] || die \
    "invalid LOCAL_EVAL_FACT_COMPILER_MODE: $FACT_COMPILER_MODE"

  local case_count annotation_count
  case_count="$(wc -l <"$CASES_PATH" | tr -d ' ')"
  annotation_count="$(wc -l <"$ANNOTATIONS_PATH" | tr -d ' ')"
  [[ "$case_count" -eq 60 ]] || die "expected frozen holdout_v2 to contain 60 cases; found $case_count"
  [[ "$annotation_count" -eq 60 ]] || die "expected 60 annotations; found $annotation_count"

  PYTHONPATH="$OCR_SITE:$REPO_ROOT/core:$REPO_ROOT" "$PYTHON_BIN" -c \
    'import rapidocr; from resume_io import extract_text_from_bytes' >/dev/null
}

status_models() {
  parse_gpu_ids
  printf '%-5s %-8s %-34s %-10s %-24s\n' GPU PORT CONTAINER STATE MODEL
  local index gpu_id port name running state health id
  for index in "${!PARSED_GPU_IDS[@]}"; do
    gpu_id="${PARSED_GPU_IDS[$index]}"
    port=$((MODEL_PORT_BASE + index))
    name="${VLLM_CONTAINER_PREFIX}${gpu_id}"
    running="$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || true)"
    if [[ "$running" == "true" ]]; then
      state=running
    elif docker inspect "$name" >/dev/null 2>&1; then
      state=stopped
    else
      state=missing
    fi
    health=down
    id=-
    if model_is_ready "$port"; then
      health=ready
      id="$(model_id "$port")"
    fi
    printf '%-5s %-8s %-34s %-10s %-24s\n' \
      "$gpu_id" "$port" "$name" "$state/$health" "$id"
  done
}

status_apis() {
  local current_digest loaded_digest
  current_digest="$(code_digest)"
  loaded_digest="$(read_manifest_value CODE_DIGEST || true)"
  printf '%-8s %-9s %-10s %-10s %-8s %-10s\n' API_PORT PID OWNER HEALTH CONTEXT CODE
  local index port pid managed owner health context freshness
  for ((index = 0; index < INSTANCE_COUNT; index++)); do
    port=$((API_PORT_BASE + index))
    pid="$(listening_pid "$port" || true)"
    managed="$(managed_api_pid "$port" || true)"
    owner=none
    [[ -n "$pid" ]] && owner=unmanaged
    if [[ -n "$managed" && "$managed" == "$pid" ]]; then
      owner=managed
    fi
    health=down
    api_is_ready "$port" && health=ready
    context=-
    [[ -n "$pid" ]] && context="$(process_env_value "$pid" LLM_CONTEXT_WINDOW || true)"
    [[ -n "$context" ]] || context=-
    freshness=unknown
    if [[ "$owner" == managed && -n "$loaded_digest" ]]; then
      freshness=current
      [[ "$loaded_digest" == "$current_digest" ]] || freshness=stale-code
    fi
    printf '%-8s %-9s %-10s %-10s %-8s %-10s\n' \
      "$port" "${pid:--}" "$owner" "$health" "$context" "$freshness"
  done
  printf 'current_code_digest=%s\n' "$current_digest"
  printf 'loaded_code_digest=%s\n' "${loaded_digest:--}"
}

command_preflight() {
  preflight_files
  log "paths, Python/OCR imports and frozen 60-case dataset are valid"
  status_models
  status_apis
}

models_all_ready() {
  local index port
  for ((index = 0; index < INSTANCE_COUNT; index++)); do
    port=$((MODEL_PORT_BASE + index))
    model_is_ready "$port" || return 1
  done
}

command_models_up() {
  preflight_files
  docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1 || die \
    "vLLM image is not local: $VLLM_IMAGE"
  local index gpu_id port name running mapped_port used_mib
  for index in "${!PARSED_GPU_IDS[@]}"; do
    gpu_id="${PARSED_GPU_IDS[$index]}"
    port=$((MODEL_PORT_BASE + index))
    name="${VLLM_CONTAINER_PREFIX}${gpu_id}"
    running="$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || true)"
    if [[ "$running" == "true" ]]; then
      mapped_port="$(docker inspect -f '{{(index (index .HostConfig.PortBindings "8000/tcp") 0).HostPort}}' "$name")"
      [[ "$mapped_port" == "$port" ]] || die \
        "$name maps model port $mapped_port, expected $port"
      continue
    fi
    if docker inspect "$name" >/dev/null 2>&1; then
      log "starting existing model container $name"
      docker start "$name" >/dev/null
      continue
    fi
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$port$"; then
      die "model port $port is occupied by an unowned service"
    fi
    used_mib="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    [[ "$used_mib" =~ ^[0-9]+$ ]] || die "cannot inspect GPU $gpu_id memory"
    [[ "$used_mib" -lt 1024 ]] || die \
      "GPU $gpu_id already uses ${used_mib} MiB; refusing to place another model"
    log "creating $name on GPU $gpu_id, host port $port"
    docker run -d \
      --name "$name" \
      --gpus "device=$gpu_id" \
      --mount "type=bind,src=$TOKENIZER_DIR,dst=/model,readonly" \
      -p "127.0.0.1:$port:8000" \
      --entrypoint vllm \
      "$VLLM_IMAGE" \
      serve /model \
      --host 0.0.0.0 --port 8000 \
      --trust-remote-code --dtype float16 \
      --max-model-len 16384 --quantization awq --enforce-eager \
      --served-model-name Qwen3.5-27B-AWQ \
      --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
      --limit-mm-per-prompt '{"image":0,"video":0}' \
      --max-num-batched-tokens 8192 --max-num-seqs 2 >/dev/null
  done

  local started_at now waited
  started_at="$(date +%s)"
  while ! models_all_ready; do
    now="$(date +%s)"
    waited=$((now - started_at))
    if ((waited >= MODEL_WAIT_SECONDS)); then
      status_models
      die "model cluster did not become ready within ${MODEL_WAIT_SECONDS}s"
    fi
    if ((waited > 0 && waited % 30 == 0)); then
      log "waiting for model shards (${waited}s)"
    fi
    sleep 2
  done
  log "all $INSTANCE_COUNT model endpoints are ready"
}

command_models_down() {
  parse_gpu_ids
  local gpu_id name
  for gpu_id in "${PARSED_GPU_IDS[@]}"; do
    name="${VLLM_CONTAINER_PREFIX}${gpu_id}"
    if [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || true)" == true ]]; then
      log "stopping $name"
      docker stop --time 30 "$name" >/dev/null
    fi
  done
}

managed_cluster_is_current() {
  local expected_digest loaded_digest index port pid
  expected_digest="$(code_digest)"
  loaded_digest="$(read_manifest_value CODE_DIGEST || true)"
  [[ -n "$loaded_digest" && "$loaded_digest" == "$expected_digest" ]] || return 1
  for ((index = 0; index < INSTANCE_COUNT; index++)); do
    port=$((API_PORT_BASE + index))
    pid="$(managed_api_pid "$port" || true)"
    [[ -n "$pid" ]] || return 1
    [[ "$(listening_pid "$port" || true)" == "$pid" ]] || return 1
    api_is_ready "$port" || return 1
    [[ "$(process_env_value "$pid" LLM_CONTEXT_WINDOW || true)" == 16384 ]] || return 1
    [[ "$(process_env_value "$pid" PIPELINE_PROFILE || true)" == "$PIPELINE_PROFILE" ]] || return 1
    [[ "$(process_env_value "$pid" FACT_COMPILER_MODE || true)" == "$FACT_COMPILER_MODE" ]] || return 1
  done
}

write_manifest() {
  local digest="$1"
  local temporary
  mkdir -p "$RUNTIME_DIR"
  temporary="$(manifest_path).$$"
  {
    printf 'CODE_DIGEST=%s\n' "$digest"
    printf 'STARTED_AT=%s\n' "$(date -Iseconds)"
    printf 'INSTANCE_COUNT=%s\n' "$INSTANCE_COUNT"
    printf 'API_PORT_BASE=%s\n' "$API_PORT_BASE"
    printf 'MODEL_PORT_BASE=%s\n' "$MODEL_PORT_BASE"
    printf 'LLM_CONTEXT_WINDOW=16384\n'
    printf 'PIPELINE_PROFILE=%s\n' "$PIPELINE_PROFILE"
    printf 'FACT_COMPILER_MODE=%s\n' "$FACT_COMPILER_MODE"
    printf 'OCR_PRIMARY_DIR=%s\n' "$OCR_PRIMARY_DIR"
  } >"$temporary"
  mv "$temporary" "$(manifest_path)"
}

stop_managed_apis() {
  local index port pid pid_file
  local -a owned_pids=()
  for ((index = 0; index < INSTANCE_COUNT; index++)); do
    port=$((API_PORT_BASE + index))
    pid="$(managed_api_pid "$port" || true)"
    if [[ -n "$pid" ]]; then
      owned_pids+=("$pid")
      log "stopping managed API port $port (process group $pid)"
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done

  local attempt alive owned_pid
  for attempt in $(seq 1 20); do
    alive=0
    for owned_pid in "${owned_pids[@]:-}"; do
      kill -0 "$owned_pid" 2>/dev/null && alive=$((alive + 1))
    done
    ((alive == 0)) && break
    sleep 1
  done
  for owned_pid in "${owned_pids[@]:-}"; do
    if kill -0 "$owned_pid" 2>/dev/null && process_matches_repo_api "$owned_pid"; then
      log "forcing stopped process group $owned_pid after grace period"
      kill -KILL -- "-$owned_pid" 2>/dev/null || kill -KILL "$owned_pid" 2>/dev/null || true
    fi
  done
  for ((index = 0; index < INSTANCE_COUNT; index++)); do
    port=$((API_PORT_BASE + index))
    pid_file="$(api_pid_file "$port")"
    [[ -e "$pid_file" ]] && unlink "$pid_file"
    [[ -e "$(api_log_path_file "$port")" ]] && unlink "$(api_log_path_file "$port")"
  done
  # A C-style arithmetic loop leaves the status of its final false condition
  # as 1.  Under ``set -e`` that made an already-stopped cluster abort
  # ``restart`` before API startup.  Stopping zero managed processes is a
  # successful idempotent operation.
  return 0
}

command_api_up() {
  preflight_files
  models_all_ready || die "model endpoints are not all ready; run models-up first"
  if managed_cluster_is_current; then
    log "managed API cluster is already healthy and current"
    return 0
  fi

  local index port occupied
  for ((index = 0; index < INSTANCE_COUNT; index++)); do
    port=$((API_PORT_BASE + index))
    occupied="$(listening_pid "$port" || true)"
    if [[ -n "$occupied" ]]; then
      die "API port $port is occupied by PID $occupied; refuse to adopt/kill it. Stop the old cluster explicitly."
    fi
  done

  mkdir -p "$RUNTIME_DIR/logs"
  local digest started_at
  digest="$(code_digest)"
  started_at="$(date +%Y%m%d-%H%M%S)"
  write_manifest "$digest"

  local model_port output_dir log_path pid
  local -a started_pids=()
  for ((index = 0; index < INSTANCE_COUNT; index++)); do
    port=$((API_PORT_BASE + index))
    model_port=$((MODEL_PORT_BASE + index))
    output_dir="$(api_output_dir "$port")"
    log_path="$RUNTIME_DIR/logs/api-${port}-${started_at}.log"
    mkdir -p "$output_dir"
    nohup setsid env \
      PYTHONPATH="$OCR_SITE:$REPO_ROOT/core:$REPO_ROOT" \
      HOST=127.0.0.1 PORT="$port" \
      MODELHUB_BASE_URL="http://127.0.0.1:$model_port/v1" \
      MODELHUB_API_KEY=not-needed MODELHUB_MODEL_NAME=Qwen3.5-27B-AWQ \
      MAX_MODEL_LEN=16384 LLM_CONTEXT_WINDOW=16384 LLM_TOKENIZER_PATH="$TOKENIZER_DIR" \
      OUTPUT_DIR="$output_dir" DEFAULT_OUTPUT_FORMAT=docx MAX_REQUEST_SIZE_BYTES=67108864 \
      REQUEST_TIMEOUT_SECONDS=480 TASK_DEADLINE_SECONDS=475 TASK_FINALIZATION_RESERVE_SECONDS=30 \
      LLM_TIMEOUT_SECONDS=300 REQUEST_CONCURRENCY=1 REQUEST_QUEUE_LIMIT=2 LLM_INFLIGHT_LIMIT=2 \
      LLM_COMPOSER_CONCURRENCY=2 LLM_COMPOSER_MAX_TOKENS=6144 LLM_COMPOSER_MAX_FACT_BLOCKS=50 \
      ENABLE_HEURISTIC_AUDIT_FALLBACK=0 \
      PIPELINE_PROFILE="$PIPELINE_PROFILE" FACT_COMPILER_MODE="$FACT_COMPILER_MODE" \
      PPOCRV6_MODEL_DIR="$OCR_PRIMARY_DIR" PPOCR_NUMERIC_CONSENSUS=1 \
      PPOCRV6_SECONDARY_REC_MODEL_PATH="$OCR_SECONDARY_REC" \
      PPOCRV6_NUMERIC_REC_MODEL_PATH="$OCR_NUMERIC_REC" \
      RAPID_OCR_MODEL=small RAPID_OCR_CPU_THREADS=2 RAPID_OCR_MAX_LONG_EDGE=3000 \
      OCR_HARD_TIMEOUT_SECONDS=60 LAYOUT_ORDER_ENGINE=bbox \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
      "$PYTHON_BIN" "$REPO_ROOT/main.py" >"$log_path" 2>&1 </dev/null &
    pid="$!"
    started_pids+=("$pid")
    printf '%s\n' "$pid" >"$(api_pid_file "$port")"
    printf '%s\n' "$log_path" >"$(api_log_path_file "$port")"
  done

  local attempt ready_count
  for attempt in $(seq 1 60); do
    ready_count=0
    for ((index = 0; index < INSTANCE_COUNT; index++)); do
      port=$((API_PORT_BASE + index))
      api_is_ready "$port" && ready_count=$((ready_count + 1))
    done
    if ((ready_count == INSTANCE_COUNT)); then
      break
    fi
    sleep 1
  done
  if ((ready_count != INSTANCE_COUNT)); then
    status_apis
    stop_managed_apis
    die "only $ready_count/$INSTANCE_COUNT APIs became ready"
  fi
  managed_cluster_is_current || {
    status_apis
    stop_managed_apis
    die "API environment verification failed after startup"
  }
  log "all $INSTANCE_COUNT APIs are managed, healthy, 16K, and loaded from $digest"
}

command_api_down() {
  stop_managed_apis
}

require_current_cluster() {
  managed_cluster_is_current || {
    status_apis
    die "evaluation requires a managed current cluster; run api-down then api-up"
  }
}

new_run_dir() {
  local requested="$1"
  local run_id
  run_id="${requested:-$(date +%Y%m%d-%H%M%S)}"
  [[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid run id: $run_id"
  local run_dir="$ARTIFACT_ROOT/$run_id"
  [[ ! -e "$run_dir" ]] || die "run directory already exists: $run_dir"
  mkdir -p "$run_dir"
  printf '%s' "$run_dir"
}

eval_identity() {
  local digest
  digest="$(code_digest)"
  EVAL_VERSION="${LOCAL_EVAL_VERSION:-local-${digest:0:12}-${PIPELINE_PROFILE}}"
  EVAL_IMAGE_DIGEST="${LOCAL_EVAL_IMAGE_DIGEST:-local27b-$digest}"
}

validate_case_ids() {
  local csv="$1"
  local case_id
  IFS=',' read -r -a requested_ids <<<"$csv"
  [[ "${#requested_ids[@]}" -gt 0 ]] || die "at least one case ID is required"
  for case_id in "${requested_ids[@]}"; do
    jq -s -e --arg id "$case_id" 'any(.[]; .id == $id)' "$CASES_PATH" >/dev/null || die \
      "unknown holdout case: $case_id"
  done
}

command_eval_case() {
  local case_id="${1:-}"
  local index="${2:-0}"
  preflight_files
  [[ -n "$case_id" ]] || die "eval-case requires CASE_ID"
  [[ "$index" =~ ^[0-9]+$ && "$index" -lt "$INSTANCE_COUNT" ]] || die \
    "API index must be between 0 and $((INSTANCE_COUNT - 1))"
  # ``jq -e select(...)`` on a JSONL stream returns status 4 when the match is
  # not the final row, even though it emitted a valid earlier object. Slurp the
  # frozen 60-row manifest and evaluate one explicit boolean instead.
  jq -s -e --arg id "$case_id" 'any(.[]; .id == $id)' "$CASES_PATH" >/dev/null || die \
    "unknown holdout case: $case_id"
  require_current_cluster
  eval_identity
  local run_dir port out
  run_dir="$(new_run_dir "case-${case_id}-$(date +%Y%m%d-%H%M%S)")"
  port=$((API_PORT_BASE + index))
  out="$run_dir/result.json"
  (
    cd "$REPO_ROOT"
    PYTHONPATH=core:. "$PYTHON_BIN" "$EVALUATOR" \
      --base-url "http://127.0.0.1:$port" \
      --version "$EVAL_VERSION" --image-digest "$EVAL_IMAGE_DIGEST" \
      --cases "$CASES_PATH" --annotations "$ANNOTATIONS_PATH" \
      --out "$out" --timeout 480 --max-attempts 1 --case-id "$case_id"
  ) | tee "$run_dir/evaluator.log"
  command_summary "$out"
  log "case artifact: $out"
}

command_eval_plan() {
  require_file "$CASES_PATH"
  require_command jq
  mapfile -t all_case_ids < <(jq -r '.id' "$CASES_PATH")
  [[ "${#all_case_ids[@]}" -eq 60 ]] || die "expected 60 case IDs"
  local index position port csv count
  for ((index = 0; index < INSTANCE_COUNT; index++)); do
    csv=""
    count=0
    for position in "${!all_case_ids[@]}"; do
      if ((position % INSTANCE_COUNT == index)); then
        csv+="${csv:+,}${all_case_ids[$position]}"
        count=$((count + 1))
      fi
    done
    port=$((API_PORT_BASE + index))
    printf 'shard=%s api=http://127.0.0.1:%s cases=%s ids=%s\n' \
      "$index" "$port" "$count" "$csv"
  done
}

command_eval_subset() {
  local requested_run_id="${1:-}"
  local requested_csv="${2:-}"
  [[ -n "$requested_run_id" ]] || die "eval-subset requires RUN_ID"
  [[ -n "$requested_csv" ]] || die "eval-subset requires comma-separated case IDs"
  preflight_files
  require_current_cluster
  validate_case_ids "$requested_csv"
  eval_identity
  mkdir -p "$RUNTIME_DIR"
  exec 9>"$RUNTIME_DIR/eval.lock"
  flock -n 9 || die "another managed evaluation holds $RUNTIME_DIR/eval.lock"

  local run_dir
  run_dir="$(new_run_dir "$requested_run_id")"
  IFS=',' read -r -a requested_ids <<<"$requested_csv"

  local index position port csv count shard log_path
  local -a shard_paths=() evaluator_pids=()
  for ((index = 0; index < INSTANCE_COUNT; index++)); do
    csv=""
    count=0
    for position in "${!requested_ids[@]}"; do
      if ((position % INSTANCE_COUNT == index)); then
        csv+="${csv:+,}${requested_ids[$position]}"
        count=$((count + 1))
      fi
    done
    ((count > 0)) || continue
    port=$((API_PORT_BASE + index))
    shard="$run_dir/shard-$index.json"
    log_path="$run_dir/shard-$index.log"
    shard_paths+=("$shard")
    (
      cd "$REPO_ROOT"
      PYTHONPATH=core:. "$PYTHON_BIN" "$EVALUATOR" \
        --base-url "http://127.0.0.1:$port" \
        --version "$EVAL_VERSION" --image-digest "$EVAL_IMAGE_DIGEST" \
        --cases "$CASES_PATH" --annotations "$ANNOTATIONS_PATH" \
        --out "$shard" --timeout 480 --max-attempts 1 --case-id "$csv"
    ) >"$log_path" 2>&1 &
    evaluator_pids+=("$!")
    log "subset shard $index -> API $port, PID ${evaluator_pids[-1]}, $count cases"
  done

  local failed=0 evaluator_pid
  for evaluator_pid in "${evaluator_pids[@]}"; do
    wait "$evaluator_pid" || failed=1
  done
  ((failed == 0)) || die "one or more subset shards failed; inspect $run_dir/shard-*.log"

  local merged="$run_dir/merged.json"
  (
    cd "$REPO_ROOT"
    PYTHONPATH=core:. "$PYTHON_BIN" "$MERGER" \
      --cases "$CASES_PATH" --out "$merged" --allow-partial "${shard_paths[@]}"
  ) | tee "$run_dir/merge.log"
  command_summary "$merged"
  log "merged subset artifact: $merged"
}

command_eval_full() {
  local requested_run_id="${1:-}"
  preflight_files
  require_current_cluster
  eval_identity
  mkdir -p "$RUNTIME_DIR"
  exec 9>"$RUNTIME_DIR/eval.lock"
  flock -n 9 || die "another managed full evaluation holds $RUNTIME_DIR/eval.lock"

  local run_dir
  run_dir="$(new_run_dir "${requested_run_id:-full60-$(date +%Y%m%d-%H%M%S)}")"
  mapfile -t all_case_ids < <(jq -r '.id' "$CASES_PATH")
  [[ "${#all_case_ids[@]}" -eq 60 ]] || die "expected 60 case IDs"

  local index position port csv count shard log_path
  local -a shard_paths=() evaluator_pids=()
  for ((index = 0; index < INSTANCE_COUNT; index++)); do
    csv=""
    count=0
    for position in "${!all_case_ids[@]}"; do
      if ((position % INSTANCE_COUNT == index)); then
        csv+="${csv:+,}${all_case_ids[$position]}"
        count=$((count + 1))
      fi
    done
    port=$((API_PORT_BASE + index))
    shard="$run_dir/shard-$index.json"
    log_path="$run_dir/shard-$index.log"
    shard_paths+=("$shard")
    (
      cd "$REPO_ROOT"
      PYTHONPATH=core:. "$PYTHON_BIN" "$EVALUATOR" \
        --base-url "http://127.0.0.1:$port" \
        --version "$EVAL_VERSION" --image-digest "$EVAL_IMAGE_DIGEST" \
        --cases "$CASES_PATH" --annotations "$ANNOTATIONS_PATH" \
        --out "$shard" --timeout 480 --max-attempts 1 --case-id "$csv"
    ) >"$log_path" 2>&1 &
    evaluator_pids+=("$!")
    log "shard $index -> API $port, PID ${evaluator_pids[$index]}, $count cases"
  done

  local failed=0 evaluator_pid
  for evaluator_pid in "${evaluator_pids[@]}"; do
    wait "$evaluator_pid" || failed=1
  done
  ((failed == 0)) || die "one or more evaluator shards failed; inspect $run_dir/shard-*.log"

  local merged="$run_dir/merged.json"
  (
    cd "$REPO_ROOT"
    PYTHONPATH=core:. "$PYTHON_BIN" "$MERGER" \
      --cases "$CASES_PATH" --out "$merged" "${shard_paths[@]}"
  ) | tee "$run_dir/merge.log"
  command_summary "$merged"
  log "merged artifact: $merged"
}

command_summary() {
  local result="${1:-}"
  [[ -n "$result" ]] || die "summary requires RESULT.json"
  require_file "$result"
  jq '{
    cases: .summary.case_count,
    request_failures: .summary.request_failure_count,
    mean_seconds: .summary.latency_seconds.mean,
    p95_seconds: .summary.latency_seconds.p95,
    max_seconds: .summary.latency_seconds.max,
    atomic_precision: .summary.atomic_factuality.micro_precision,
    atomic_recall: .summary.atomic_factuality.micro_recall,
    ownership_integrity: .summary.ownership_integrity.integrity_rate,
    critical_additions: .summary.critical_additions
  }' "$result"
}

command_logs() {
  local index="${1:-0}"
  [[ "$index" =~ ^[0-9]+$ && "$index" -lt "$INSTANCE_COUNT" ]] || die \
    "API index must be between 0 and $((INSTANCE_COUNT - 1))"
  local port log_path_file log_path
  port=$((API_PORT_BASE + index))
  log_path_file="$(api_log_path_file "$port")"
  require_file "$log_path_file"
  log_path="$(sed -n '1p' "$log_path_file")"
  require_file "$log_path"
  tail -n 80 -f "$log_path"
}

main() {
  local command="${1:-help}"
  shift || true
  case "$command" in
    help|-h|--help) usage ;;
    preflight) command_preflight "$@" ;;
    status) status_models; status_apis ;;
    models-up) command_models_up "$@" ;;
    models-down) command_models_down "$@" ;;
    api-up) command_api_up "$@" ;;
    api-down) command_api_down "$@" ;;
    restart) command_api_down; command_api_up ;;
    up) command_models_up; command_api_up ;;
    eval-case) command_eval_case "$@" ;;
    eval-subset) command_eval_subset "$@" ;;
    eval-plan) command_eval_plan "$@" ;;
    eval-full) command_eval_full "$@" ;;
    summary) command_summary "$@" ;;
    logs) command_logs "$@" ;;
    *) usage >&2; die "unknown command: $command" ;;
  esac
}

main "$@"
