# gRPC Recommendation System

这是一个基于C++和gRPC的推荐系统示例项目，实现了完整的客户端-服务器通信协议。

## 项目结构

```
recommendation_grpc/
├── proto/                          # Protocol Buffers定义文件
│   └── recommendation.proto        # gRPC服务定义
├── include/                        # 头文件
│   └── recommendation_service_impl.h
├── src/                            # 源代码
│   ├── recommendation_service_impl.cpp  # 服务实现
│   ├── recommendation_server.cpp        # 服务器主程序
│   └── recommendation_client.cpp        # 客户端程序
├── build/                          # 构建输出目录
├── scripts/                        # 脚本文件
├── CMakeLists.txt                  # CMake构建配置
└── README.md                       # 本文档
```

## 功能特性

### 服务端
- 实现gRPC推荐服务接口
- 接收客户端推荐请求，包含以下字段：
  - `user_id`: 用户ID
  - `cookie_id`: Cookie ID
  - `trace_id`: 追踪ID
  - `timestamp`: 时间戳
  - `browser`: 浏览器信息
  - `location`: 地理位置
  - `application`: 应用名称
  - `headers`: HTTP头部信息（键值对）
  - `num_results`: 请求返回结果数量
- 返回模拟推荐数据，包含：
  - `item_id`: 项目ID
  - `score`: 推荐分数
  - `category`: 分类
  - `url`: URL地址
  - `title`: 标题
  - `description`: 描述
  - `metadata`: 额外元数据

### 客户端
- 发送推荐请求到服务器
- 解析并显示响应结果
- 支持命令行参数配置

## 依赖项

- **CMake**: >= 3.5
- **C++**: 支持 C++14 或更高版本
- **gRPC**: 最新稳定版本
- **Protobuf**: 与gRPC兼容的版本

## 安装依赖

### Ubuntu/Debian

```bash
# 安装构建工具
sudo apt-get update
sudo apt-get install -y build-essential cmake

# 安装gRPC和Protobuf
sudo apt-get install -y libgrpc++-dev libprotobuf-dev protobuf-compiler
```

### macOS

```bash
# 使用Homebrew安装
brew install cmake grpc protobuf
```

### 从源码构建gRPC

如果系统的包管理器没有提供所需版本的gRPC，可以从源码构建：

```bash
git clone -b v1.60.0 https://github.com/grpc/grpc
cd grpc
git submodule update --init
mkdir build && cd build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      ..
make -j$(nproc)
sudo make install
```

## 构建项目

1. 克隆或进入项目目录：
```bash
cd ai_project/AlgorithmCodingPractice/Coding/cpp/recommendation_grpc
```

2. 创建构建目录：
```bash
mkdir -p build
cd build
```

3. 运行CMake配置：
```bash
cmake ..
```

4. 编译项目：
```bash
make -j$(nproc)
```

编译成功后，在`build`目录下会生成两个可执行文件：
- `recommendation_server`: 推荐服务服务器
- `recommendation_client`: 推荐服务客户端

## 运行

### 启动服务器

在第一个终端中运行：

```bash
cd build
./recommendation_server [address]
```

默认地址为 `0.0.0.0:50051`，也可以指定其他地址：

```bash
./recommendation_server 0.0.0.0:8080
```

### 运行客户端

在第二个终端中运行：

```bash
cd build
./recommendation_client [server_address] [user_id] [num_results]
```

参数说明：
- `server_address`: 服务器地址（默认: `localhost:50051`）
- `user_id`: 用户ID（默认: `user_12345`）
- `num_results`: 请求的推荐结果数量（默认: 5）

示例：

```bash
# 使用默认参数
./recommendation_client

# 指定服务器地址
./recommendation_client localhost:8080

# 指定用户ID和结果数量
./recommendation_client localhost:50051 user_67890 10
```

## 示例输出

### 服务器端输出
```
=== Recommendation Server Started ===
Server listening on 0.0.0.0:50051
Press Ctrl+C to stop the server...
=== Received Recommendation Request ===
User ID: user_12345
Cookie ID: cookie_user_12345
Trace ID: trace_1234567890
Timestamp: 1234567890
Browser: Chrome/120.0.0.0
Location: Beijing, China
Application: WebApp
Number of results requested: 5
Headers:
  Content-Type: application/grpc
  Accept: application/json
  User-Agent: RecommendationClient/1.0
  X-Request-ID: trace_1234567890
=== Generated 5 recommendations ===
```

### 客户端输出
```
=== Recommendation Client ===
Server: localhost:50051
User ID: user_12345
Number of results: 5

=== Sending Recommendation Request ===
User ID: user_12345
Trace ID: trace_1234567890
Number of results: 5

=== Received Response ===
Status Code: 200
Message: Success
Trace ID: trace_1234567890
Number of items: 5

--- Item 1 ---
  ID: item_1
  Score: 1
  Category: Electronics
  Title: Product 1 - Electronics
  URL: https://example.com/products/item_1
  Description: This is a high-quality Electronics product...
  Metadata:
    brand: Brand1
    price: $19.99
    rating: 4.0
    in_stock: true

--- Item 2 ---
  ID: item_2
  Score: 0.95
  Category: Clothing
  Title: Product 2 - Clothing
  URL: https://example.com/products/item_2
  ...
```

## 开发说明

### 添加新的RPC方法

1. 在 `proto/recommendation.proto` 中定义新的service方法
2. 重新编译项目（CMake会自动生成新的代码）
3. 在 `include/recommendation_service_impl.h` 中声明新方法
4. 在 `src/recommendation_service_impl.cpp` 中实现新方法

### 修改数据结构

1. 编辑 `proto/recommendation.proto` 中的message定义
2. 重新编译项目
3. 在服务实现和客户端代码中使用新的字段

## 架构说明

本项目参考了 `project1` 的架构设计，采用了以下组织方式：

- **分离的目录结构**: proto、include、src、build分别存放不同类型的文件
- **CMake构建系统**: 使用CMake管理依赖和构建过程
- **模块化设计**: 服务实现、服务器、客户端分别在不同的文件中
- **清晰的接口**: 通过proto文件定义服务接口，实现前后端分离

## 注意事项

- 本项目使用不安全的连接（`InsecureServerCredentials`），仅用于开发和测试
- 在生产环境中应该使用SSL/TLS加密连接
- Mock数据的生成逻辑可以替换为实际的推荐算法
- 项目结构可以进一步扩展，例如添加日志、配置管理等模块

## 许可证

本项目仅用于学习和演示目的。

## 联系方式

如有问题或建议，欢迎提出Issue。