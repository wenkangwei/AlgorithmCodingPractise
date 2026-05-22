#include <iostream>
#include <memory>
#include <string>
#include <grpcpp/grpcpp.h>
#include <chrono>
#include <sys/time.h>
#include "recommendation.grpc.pb.h"

class RecommendationClient {
public:
    RecommendationClient(std::shared_ptr<grpc::Channel> channel)
        : stub_(recommendation::RecommendationService::NewStub(channel)) {}

    // 获取推荐
    void GetRecommendations(const std::string& user_id, int num_results) {
        // 创建请求
        recommendation::RecommendationRequest request;
        
        // 设置基本请求信息
        request.set_user_id(user_id);
        request.set_cookie_id("cookie_" + user_id);
        
        // 生成trace_id
        std::string trace_id = "trace_" + std::to_string(GetCurrentTimestamp());
        request.set_trace_id(trace_id);
        
        // 设置时间戳
        request.set_timestamp(GetCurrentTimestamp());
        
        // 设置浏览器信息
        request.set_browser("Chrome/120.0.0.0");
        
        // 设置位置信息
        request.set_location("Beijing, China");
        
        // 设置应用名称
        request.set_application("WebApp");
        
        // 设置请求结果数量
        request.set_num_results(num_results);
        
        // 添加headers
        auto& headers = *request.mutable_headers();
        headers["Content-Type"] = "application/grpc";
        headers["Accept"] = "application/json";
        headers["User-Agent"] = "RecommendationClient/1.0";
        headers["X-Request-ID"] = trace_id;
        
        std::cout << "=== Sending Recommendation Request ===" << std::endl;
        std::cout << "User ID: " << request.user_id() << std::endl;
        std::cout << "Trace ID: " << request.trace_id() << std::endl;
        std::cout << "Number of results: " << request.num_results() << std::endl;
        std::cout << std::endl;
        
        // 创建响应容器
        recommendation::RecommendationResponse response;
        
        // 创建RPC上下文
        grpc::ClientContext context;
        
        // 设置超时时间（5秒）
        std::chrono::system_clock::time_point deadline =
            std::chrono::system_clock::now() + std::chrono::seconds(5);
        context.set_deadline(deadline);
        
        // 发送RPC请求
        grpc::Status status = stub_->GetRecommendations(&context, request, &response);
        
        // 处理响应
        if (status.ok()) {
            std::cout << "=== Received Response ===" << std::endl;
            std::cout << "Status Code: " << response.status_code() << std::endl;
            std::cout << "Message: " << response.message() << std::endl;
            std::cout << "Trace ID: " << response.trace_id() << std::endl;
            std::cout << "Number of items: " << response.items_size() << std::endl;
            std::cout << std::endl;
            
            // 打印推荐项目
            for (int i = 0; i < response.items_size(); ++i) {
                const auto& item = response.items(i);
                std::cout << "--- Item " << (i + 1) << " ---" << std::endl;
                std::cout << "  ID: " << item.item_id() << std::endl;
                std::cout << "  Score: " << item.score() << std::endl;
                std::cout << "  Category: " << item.category() << std::endl;
                std::cout << "  Title: " << item.title() << std::endl;
                std::cout << "  URL: " << item.url() << std::endl;
                std::cout << "  Description: " << item.description() << std::endl;
                
                // 打印元数据
                if (item.metadata_size() > 0) {
                    std::cout << "  Metadata:" << std::endl;
                    for (const auto& meta : item.metadata()) {
                        std::cout << "    " << meta.first << ": " << meta.second << std::endl;
                    }
                }
                std::cout << std::endl;
            }
        } else {
            std::cerr << "RPC failed: " << status.error_code() << ": "
                      << status.error_message() << std::endl;
        }
    }

private:
    std::unique_ptr<recommendation::RecommendationService::Stub> stub_;
    
    // 获取当前时间戳（毫秒）
    int64_t GetCurrentTimestamp() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (int64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
    }
};

int main(int argc, char** argv) {
    // 默认服务器地址
    std::string server_address = "localhost:50051";
    
    // 默认参数
    std::string user_id = "user_12345";
    int num_results = 5;
    
    // 解析命令行参数
    if (argc > 1) {
        server_address = argv[1];
    }
    if (argc > 2) {
        user_id = argv[2];
    }
    if (argc > 3) {
        num_results = std::atoi(argv[3]);
    }
    
    std::cout << "=== Recommendation Client ===" << std::endl;
    std::cout << "Server: " << server_address << std::endl;
    std::cout << "User ID: " << user_id << std::endl;
    std::cout << "Number of results: " << num_results << std::endl;
    std::cout << std::endl;
    
    try {
        // 创建客户端
        RecommendationClient client(grpc::CreateChannel(
            server_address, grpc::InsecureChannelCredentials()));
        
        // 发送推荐请求
        client.GetRecommendations(user_id, num_results);
        
    } catch (const std::exception& e) {
        std::cerr << "Client error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}