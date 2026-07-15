#!/bin/bash
set -e

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# ═══ 模型路径 ═══
MODEL_PATH_ENV="${MODEL_PATH:-}"
if [ -d "/model" ] && [ -f "/model/Qwen3.5-9B-AWQ-4bit/config.json" ]; then
    MODEL_PATH="/model/Qwen3.5-9B-AWQ-4bit"
    log "模型路径: $MODEL_PATH (/model)"
elif find /model -maxdepth 2 -name config.json -path "*Qwen*" 2>/dev/null | head -1 | read; then
    MODEL_PATH=$(dirname "$(find /model -maxdepth 2 -name config.json -path '*Qwen*' 2>/dev/null | head -1)")
    log "模型路径: $MODEL_PATH (/model auto-detect)"
elif [ -n "$MODEL_PATH_ENV" ] && [ -f "$MODEL_PATH_ENV/config.json" ]; then
    MODEL_PATH="$MODEL_PATH_ENV"
    log "模型路径(环境变量): $MODEL_PATH"
else
    log "错误: 模型未找到!"
    ls /model/ 2>/dev/null || echo "  /model 不存在"
    exit 1
fi

# ═══ transformers 直接加载模型 ═══
export MODEL_PATH="$MODEL_PATH"
export LLM_PORT="${LLM_PORT:-8000}"
log "启动 transformers LLM server (model=$MODEL_PATH, port=$LLM_PORT)"
python3 /root/app/llm_server.py &
LLM_PID=$!

# 轮询 /health, 最多等 1200s (20min)
WAITED=0
while [ $WAITED -lt 1200 ]; do
    if ! kill -0 $LLM_PID 2>/dev/null; then
        log "错误: LLM 进程已退出 (pid=$LLM_PID)!"
        exit 1
    fi
    if curl -sf http://localhost:$LLM_PORT/health >/dev/null 2>&1; then
        log "LLM server 就绪! (耗时 ${WAITED}s)"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $((WAITED % 30)) -eq 0 ]; then
        log "等待 LLM 加载... (已等待 ${WAITED}s)"
    fi
done

# ═══ 启动 Flask ═══
export MODELHUB_BASE_URL="http://localhost:8000/v1"
export MODELHUB_API_KEY=not-needed
export MODELHUB_MODEL_NAME="$MODEL_PATH"
export PORT="${PORT:-80}"
export ENABLE_HEURISTIC_AUDIT_FALLBACK="${ENABLE_HEURISTIC_AUDIT_FALLBACK:-0}"
export REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-480}"
export LLM_TIMEOUT_SECONDS="${LLM_TIMEOUT_SECONDS:-180}"
export DEFAULT_OUTPUT_FORMAT=docx
export OUTPUT_DIR="${OUTPUT_DIR:-/root/app/output}"
mkdir -p "$OUTPUT_DIR"
cd /root/app

log "LLM 就绪, 启动 resume-copilot (port=$PORT)"
exec python3 main.py
