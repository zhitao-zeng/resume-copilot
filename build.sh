#!/bin/bash
set -e
# 构建并推送生产一体化镜像（容器内启动 vLLM + API）
# 用法: bash build.sh [tag]
TAG="${1:-latest}"
IMAGE="harbor-contest.4pd.io/zengzhitao/resume-copilot:${TAG}"
docker build -f config/Dockerfile -t "${IMAGE}" .
docker push "${IMAGE}"
echo "Pushed: ${IMAGE}"
