#!/bin/bash
set -e
# 构建并推送镜像
# 用法: bash build.sh [tag]
TAG="${1:-latest}"
IMAGE="harbor-contest.4pd.io/zengzhitao/resume-copilot:${TAG}"
docker build -f config/Dockerfile -t "${IMAGE}" .
docker push "${IMAGE}"
echo "Pushed: ${IMAGE}"
