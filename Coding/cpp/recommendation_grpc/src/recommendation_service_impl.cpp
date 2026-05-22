#include "recommendation_service_impl.h"
#include <iostream>
#include <sstream>

RecommendationServiceImpl::RecommendationServiceImpl()
    : rd_(), gen_(rd_()), score_dist_(0.0, 1.0) {
}

grpc::Status RecommendationServiceImpl::GetRecommendations(
    grpc::ServerContext* context,
    const recommendation::RecommendationRequest* request,
    recommendation::RecommendationResponse* response) {
    
    // 记录请求信息
    std::cout << "=== Received Recommendation Request ===" << std::endl;
    std::cout << "User ID: " << request->user_id() << std::endl;
    std::cout << "Cookie ID: " << request->cookie_id() << std::endl;
    std::cout << "Trace ID: " << request->trace_id() << std::endl;
    std::cout << "Timestamp: " << request->timestamp() << std::endl;
    std::cout << "Browser: " << request->browser() << std::endl;
    std::cout << "Location: " << request->location() << std::endl;
    std::cout << "Application: " << request->application() << std::endl;
    std::cout << "Number of results requested: " << request->num_results() << std::endl;
    
    // 打印headers信息
    if (request->headers_size() > 0) {
        std::cout << "Headers:" << std::endl;
        for (const auto& header : request->headers()) {
            std::cout << "  " << header.first << ": " << header.second << std::endl;
        }
    }
    
    // 设置响应基本信息
    response->set_trace_id(request->trace_id());
    response->set_status_code(200);
    response->set_message("Success");
    
    // 生成mock推荐数据
    GenerateMockRecommendations(*request, *response);
    
    std::cout << "=== Generated " << response->items_size() << " recommendations ===" << std::endl;
    
    return grpc::Status::OK;
}

void RecommendationServiceImpl::GenerateMockRecommendations(
    const recommendation::RecommendationRequest& request,
    recommendation::RecommendationResponse& response) {
    
    // 确定返回数量
    int num_results = request.num_results();
    if (num_results <= 0 || num_results > 100) {
        num_results = 10; // 默认返回10个结果
    }
    
    // 生成推荐项目
    for (int i = 0; i < num_results; ++i) {
        auto* item = response.add_items();
        GenerateMockItem(i, *item);
    }
}

double RecommendationServiceImpl::GenerateRandomScore() {
    return score_dist_(gen_);
}

std::string RecommendationServiceImpl::GenerateRandomCategory() {
    static const std::vector<std::string> categories = {
        "Electronics",
        "Clothing",
        "Books",
        "Home & Garden",
        "Sports",
        "Toys",
        "Automotive",
        "Health",
        "Beauty",
        "Food"
    };
    std::uniform_int_distribution<> dist(0, categories.size() - 1);
    return categories[dist(gen_)];
}

void RecommendationServiceImpl::GenerateMockItem(
    int index,
    recommendation::RecommendationItem& item) {
    
    // 生成item_id
    std::stringstream item_id_ss;
    item_id_ss << "item_" << (index + 1);
    item.set_item_id(item_id_ss.str());
    
    // 生成分数（按索引递减，模拟排序）
    double base_score = 1.0 - (index * 0.05);
    if (base_score < 0.0) base_score = 0.0;
    item.set_score(base_score);
    
    // 设置类别
    item.set_category(GenerateRandomCategory());
    
    // 生成URL
    std::stringstream url_ss;
    url_ss << "https://example.com/products/" << item.item_id();
    item.set_url(url_ss.str());
    
    // 生成标题
    std::stringstream title_ss;
    title_ss << "Product " << (index + 1) << " - " << item.category();
    item.set_title(title_ss.str());
    
    // 生成描述
    std::stringstream desc_ss;
    desc_ss << "This is a high-quality " << item.category() 
            << " product with excellent features and great value. "
            << "Recommended score: " << item.score();
    item.set_description(desc_ss.str());
    
    // 添加一些元数据
    auto& metadata = *item.mutable_metadata();
    metadata["brand"] = "Brand" + std::to_string((index % 5) + 1);
    metadata["price"] = "$" + std::to_string((index + 1) * 10 + 9) + ".99";
    metadata["rating"] = std::to_string(4.0 + (index % 6) * 0.2);
    metadata["in_stock"] = (index % 3 == 0) ? "true" : "false";
}