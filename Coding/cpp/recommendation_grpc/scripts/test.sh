#!/bin/bash

# 测试脚本 - 启动服务器和客户端进行测试

set -e

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

echo "=== Recommendation System Test Script ==="
echo ""

# 检查可执行文件是否存在
if [ ! -f "$BUILD_DIR/recommendation_server" ]; then
    echo "Error: recommendation_server not found in $BUILD_DIR"
    echo "Please run ./scripts/build.sh first"
    exit 1
fi

if [ ! -f "$BUILD_DIR/recommendation_client" ]; then
    echo "Error: recommendation_client not found in $BUILD_DIR"
    echo "Please run ./scripts/build.sh first"
    exit 1
fi

echo "1. Starting recommendation server..."
cd "$BUILD_DIR"
./recommendation_server > server.log 2>&1 &
SERVER_PID=$!
echo "   Server PID: $SERVER_PID"
echo "   Waiting for server to start..."
sleep 3

echo ""
echo "2. Running recommendation client..."
./recommendation_client
CLIENT_EXIT_CODE=$?

echo ""
echo "3. Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "4. Server output:"
cat server.log

# 清理日志文件
rm -f server.log

echo ""
if [ $CLIENT_EXIT_CODE -eq 0 ]; then
    echo "=== Test completed successfully ==="
else
    echo "=== Test failed with exit code $CLIENT_EXIT_CODE ==="
    exit $CLIENT_EXIT_CODE
fi