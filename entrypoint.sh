#!/bin/bash
set -e

# Start vLLM with OpenAI-compatible API
# In v0.19.1, `vllm serve` is the new entrypoint that exposes /v1/ endpoints
vllm serve /data/Qwen3.6-27B-AWQ-INT4 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 &
VLLM_PID=$!

# Wait for vLLM to be ready
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM ready on :8000"
        break
    fi
    sleep 2
done

# Start OpenAI proxy on port 8001
python3 /opt/vllm_openai_proxy.py --host 0.0.0.0 --port 8001 &

wait $VLLM_PID
