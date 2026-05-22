#include <iostream>
#include <memory>
#include <string>
#include <grpcpp/grpcpp.h>
#include "recommendation_service_impl.h"

void RunServer(const std::string& server_address) {
    // 创建服务实例
    RecommendationServiceImpl service;

    // 创建服务器构建器
    grpc::ServerBuilder builder;

    // 监听指定地址，不使用认证（仅用于开发测试）
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

    // 注册服务
    builder.RegisterService(&service);

    // 构建并启动服务器
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "=== Recommendation Server Started ===" << std::endl;
    std::cout << "Server listening on " << server_address << std::endl;
    std::cout << "Press Ctrl+C to stop the server..." << std::endl;

    // 等待服务器关闭
    server->Wait();
}

int main(int argc, char** argv) {
    // 默认服务器地址
    std::string server_address = "0.0.0.0:50051";
    
    // 如果提供了命令行参数，使用自定义地址
    if (argc > 1) {
        server_address = argv[1];
    }

    try {
        RunServer(server_address);
    } catch (const std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}