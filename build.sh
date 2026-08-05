#!/bin/bash
set -e
# 构建并推送生产一体化镜像（容器内启动 vLLM + API）
# 用法: bash build.sh [tag]
TAG="${1:-latest}"
IMAGE="harbor-contest.4pd.io/zengzhitao/resume-copilot:${TAG}"
bash config/stage_ppocrv6_small.sh
BUILD_COMMIT="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BUILD_DATE="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
docker build \
  --build-arg "BUILD_COMMIT=${BUILD_COMMIT}" \
  --build-arg "BUILD_DATE=${BUILD_DATE}" \
  -f config/Dockerfile -t "${IMAGE}" .
docker push "${IMAGE}"
echo "Pushed: ${IMAGE}"
