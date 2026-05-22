#!/bin/bash

# 构建脚本
# 用法: ./scripts/build.sh [clean]

set -e  # 遇到错误立即退出

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

echo "=== Recommendation System Build Script ==="
echo "Project directory: $PROJECT_DIR"
echo "Build directory: $BUILD_DIR"
echo ""

# 如果传入了clean参数，先清理
if [ "$1" = "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
    echo "Build directory cleaned."
    echo ""
fi

# 创建构建目录
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

# 进入构建目录
cd "$BUILD_DIR"

# 运行CMake配置
echo "Running CMake configuration..."
cmake -DCMAKE_BUILD_TYPE=Release ..

# 编译项目
echo ""
echo "Building project..."
make -j$(nproc)

echo ""
echo "=== Build completed successfully ==="
echo "Executable files are located in: $BUILD_DIR"
echo ""
echo "To run the server:"
echo "  cd $BUILD_DIR"
echo "  ./recommendation_server"
echo ""
echo "To run the client (in another terminal):"
echo "  cd $BUILD_DIR"
echo "  ./recommendation_client"