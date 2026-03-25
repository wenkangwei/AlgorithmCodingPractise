#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_TYPE=${1:-Release}
BUILD_DIR="${PROJECT_ROOT}/build/${BUILD_TYPE,,}"
INSTALL_PREFIX=${2:-/usr/local}

echo "=========================================="
echo "Build Type: ${BUILD_TYPE}"
echo "Build Dir:  ${BUILD_DIR}"
echo "Install:    ${INSTALL_PREFIX}"
echo "=========================================="

# 1. 运行代码生成脚本
echo "[1/4] Running code generation..."
bash "${PROJECT_ROOT}/scripts/generate_version.sh" \
    "${BUILD_DIR}/generated/version.h"

# 2. 配置 CMake
echo "[2/4] Configuring CMake..."
cmake -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DENABLE_TESTING=ON \
    -S "${PROJECT_ROOT}"

# 3. 构建
echo "[3/4] Building..."
cmake --build "${BUILD_DIR}" --parallel $(nproc)

# 4. 运行测试（可选）
if [ "$3" == "--test" ]; then
    echo "[4/4] Running tests..."
    ctest --test-dir "${BUILD_DIR}" --output-on-failure
else
    echo "[4/4] Skipping tests (use --test to enable)"
fi

echo "=========================================="
echo "Build complete! Binaries at: ${BUILD_DIR}/bin"
echo "=========================================="