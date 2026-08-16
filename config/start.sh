#!/bin/bash
set -e

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# === 版本信息 ===
VERSION_COMMIT="${RESUME_BUILD_COMMIT:-unknown}"
VERSION_DATE="${RESUME_BUILD_DATE:-unknown}"
log "============================================"
log "  resume-copilot 版本: ${VERSION_COMMIT} (${VERSION_DATE})"
log "============================================"

# ═══ 等模型就绪 (NFS 可能延迟, 最多等 120s) ═══
# subPath 直接挂载模型目录到 /model, config.json 在 /model/config.json
find_model() {
    for c in "/model" "/model/Qwen3-14B-GPTQ-Int4" "/model/Qwen3-8B-AWQ" "/model/Qwen3.5-9B-AWQ-4bit" "/model/Qwen3.5-27B-AWQ" "/mounted_model" "${MODEL_PATH:-}"; do
        [ -n "$c" ] && [ -f "$c/config.json" ] && echo "$c" && return 0
    done
    return 1
}

log "等待模型文件就绪..."
MODEL_FOUND=""
for i in $(seq 1 120); do
    if MODEL_FOUND=$(find_model); then
        log "模型文件就绪: $MODEL_FOUND (耗时 ${i}s)"
        break
    fi
    sleep 1
done

if [ -z "$MODEL_FOUND" ]; then
    log "错误: 模型未找到!"
    ls /model/ 2>/dev/null || echo "  /model 不存在"
    exit 1
fi

# ═══ vLLM 启动并等待模型加载完成 ═══
export MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.74}"
export MAX_MODEL_LEN
export LLM_CONTEXT_WINDOW="$MAX_MODEL_LEN"
export LLM_TOKENIZER_PATH="${LLM_TOKENIZER_PATH:-$MODEL_FOUND}"
export LLM_INFLIGHT_LIMIT="${LLM_INFLIGHT_LIMIT:-$MAX_NUM_SEQS}"
# Dense, OCR-reordered CVs need one extraction request to see company names,
# dates and duties together.  The 16k context/40 GiB profile can safely reserve
# a larger completion than the portable 8k default used by unit tests.
export LLM_COMPOSER_MAX_TOKENS="${LLM_COMPOSER_MAX_TOKENS:-6144}"
export LLM_COMPOSER_MAX_FACT_BLOCKS="${LLM_COMPOSER_MAX_FACT_BLOCKS:-36}"
export LLM_COMPOSER_CALL_TIMEOUT_SECONDS="${LLM_COMPOSER_CALL_TIMEOUT_SECONDS:-120}"
log "GPU memory split: vLLM=${VLLM_GPU_MEMORY_UTILIZATION}, PP-StructureV3 reserve enabled"

vllm serve "$MODEL_FOUND" \
    --host 0.0.0.0 --port 8000 \
    --quantization awq --dtype float16 \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --trust-remote-code \
    --enforce-eager \
    > /tmp/vllm_stdout.log 2>&1 &
VLLM_LOG="/tmp/vllm_stdout.log"
VLLM_PID=$!
log "vLLM 启动 (pid=$VLLM_PID), 等待模型加载就绪..."

# 打印模型文件大小，确认 NFS 上权重完整
log "模型文件大小:"
du -sh "$MODEL_FOUND" 2>/dev/null || true
ls -lh "$MODEL_FOUND"/*.safetensors 2>/dev/null | awk '{print $5, $9}' | head -10

# 轮询 vLLM /health, 一直等到启动为止
WAITED=0
while true; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        log "错误: vLLM 进程已退出 (pid=$VLLM_PID)!"
        log "vLLM 最后输出:"
        tail -100 $VLLM_LOG 2>/dev/null || true
        exit 1
    fi
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        log "vLLM 模型加载完成! (耗时 ${WAITED}s)"
        # 打印内存 profiling 明细
        log "================== vLLM 内存明细 =================="
        grep -E "Model loading took|Loading weights took|Available KV cache|GPU KV cache size|Maximum concurrency|model weights|gpu_memory|memory" $VLLM_LOG 2>/dev/null | tail -10
        log "===================================================="
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $((WAITED % 30)) -eq 0 ]; then
        # 显示 vLLM 实际加载进度（shard 百分比/总耗时）
        PROGRESS=$(grep -oP 'Loading safetensors.*?(\d+%)\|' $VLLM_LOG 2>/dev/null | tail -1 | grep -oP '\d+%' || echo "?")
        WEIGHT_TIME=$(grep -oP 'Loading weights took \K[\d.]+' $VLLM_LOG 2>/dev/null | tail -1)
        MODEL_TIME=$(grep -oP 'Model loading took \K[\d.]+ GiB memory' $VLLM_LOG 2>/dev/null | tail -1)
        [ -n "$WEIGHT_TIME" ] && WT=", 权重耗时=${WEIGHT_TIME}s" || WT=""
        log "vLLM 加载中... (已等待 ${WAITED}s, shard进度: ${PROGRESS:-?}${WT})"
    fi
done

if ! kill -0 $VLLM_PID 2>/dev/null; then
    log "错误: vLLM 进程已退出 (pid=$VLLM_PID)!"
    log "=============== vLLM 最后 100 行 ==============="
    tail -100 $VLLM_LOG 2>/dev/null || true
    log "=============== 内存 profiling (如果有) ==============="
    grep -E "Model loading took|Loading weights took|Available KV cache|GPU KV cache size|orig free|limit|total_gpu|Free memory on device|ValueError|gpu_memory" $VLLM_LOG 2>/dev/null | tail -15
    log "======================================================="
    exit 1
fi

# ═══ 模型就绪后启动 API ═══
export MODELHUB_BASE_URL="http://localhost:8000/v1"
export MODELHUB_API_KEY=not-needed
export MODELHUB_MODEL_NAME="$MODEL_FOUND"
export PORT=80
export PYTHONPATH="/root/app/core:$PYTHONPATH"
export ENABLE_HEURISTIC_AUDIT_FALLBACK="${ENABLE_HEURISTIC_AUDIT_FALLBACK:-0}"
export REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-480}"
export LLM_TIMEOUT_SECONDS="${LLM_TIMEOUT_SECONDS:-300}"
export DEFAULT_OUTPUT_FORMAT="${DEFAULT_OUTPUT_FORMAT:-docx}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/app/output}"
# `v3` enables the evidence compiler end to end. Keep `v2` as the deployment
# default until held-out/Darvin gates select V3 explicitly.
export RESUME_PIPELINE_VERSION="${RESUME_PIPELINE_VERSION:-v2}"
export V3_TRAINING_TRACE_ENABLED="${V3_TRAINING_TRACE_ENABLED:-0}"
export V3_SEMANTIC_CONCURRENCY="${V3_SEMANTIC_CONCURRENCY:-2}"
export V3_SEMANTIC_BATCH_FACTS="${V3_SEMANTIC_BATCH_FACTS:-28}"
export V3_SEMANTIC_BATCH_CHARS="${V3_SEMANTIC_BATCH_CHARS:-9000}"
export V3_REALIZER_MIN_REMAINING_SECONDS="${V3_REALIZER_MIN_REMAINING_SECONDS:-240}"
# Production defaults to the locally validated reference content path. New
# recovery/composer modules are opt-in profiles until their grouped ownership
# and platform gates pass; current OCR, deadlines and rendering stay shared.
export PIPELINE_PROFILE="${PIPELINE_PROFILE:-f507_compatible}"
# This immutable diagnostic build records the complete fictional benchmark
# payload and each factual transformation under one task ID. Credential-like
# mapping keys remain suppressed by diagnostic_trace.py.
export RESUME_DIAGNOSTIC_TRACE="${RESUME_DIAGNOSTIC_TRACE:-1}"
mkdir -p "$OUTPUT_DIR"
cd /root/app

log "vLLM 就绪, 启动 resume-copilot (port=$PORT, pipeline=$RESUME_PIPELINE_VERSION, diagnostic_trace=$RESUME_DIAGNOSTIC_TRACE)"
exec python3 main.py
