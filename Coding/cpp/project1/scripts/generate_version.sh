#!/bin/bash
# scripts/generate_version.sh
# 自动生成版本信息头文件

OUTPUT_FILE=$1

GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_TIME=$(date '+%Y-%m-%d %H:%M:%S')
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")

cat > $OUTPUT_FILE << EOF
#pragma once
#define GIT_HASH "${GIT_HASH}"
#define BUILD_TIME "${BUILD_TIME}"
#define GIT_BRANCH "${BRANCH}"
#define PROJECT_VERSION "${PROJECT_VERSION}"
EOF

echo "Generated version info: ${GIT_HASH} @ ${BUILD_TIME}"