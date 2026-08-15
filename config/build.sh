#!/bin/bash
set -e
# 构建并推送镜像
# 用法: bash build.sh [tag]
TAG="${1:-latest}"
IMAGE="harbor-contest.4pd.io/zengzhitao/resume-copilot:${TAG}"
bash config/stage_ppocrv6_small.sh
bash config/stage_ppocrv6_trt_a100.sh
bash config/stage_tensorrt_runtime.sh
bash config/stage_ppstructure_v3.sh
BUILD_COMMIT="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BUILD_DATE="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
export DOCKER_BUILDKIT=1
docker build \
  --build-arg "BUILD_COMMIT=${BUILD_COMMIT}" \
  --build-arg "BUILD_DATE=${BUILD_DATE}" \
  -f config/Dockerfile -t "${IMAGE}" .
docker push "${IMAGE}"
echo "Pushed: ${IMAGE}"
