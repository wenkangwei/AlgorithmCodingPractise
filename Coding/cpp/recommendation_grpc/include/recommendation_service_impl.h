#ifndef RECOMMENDATION_SERVICE_IMPL_H
#define RECOMMENDATION_SERVICE_IMPL_H

#include <grpcpp/grpcpp.h>
#include "recommendation.grpc.pb.h"

#include <string>
#include <vector>
#include <random>
#include <chrono>

// 推荐服务实现类
class RecommendationServiceImpl final : public recommendation::RecommendationService::Service {
public:
    RecommendationServiceImpl();
    
    // 实现GetRecommendations RPC方法
    grpc::Status GetRecommendations(
        grpc::ServerContext* context,
        const recommendation::RecommendationRequest* request,
        recommendation::RecommendationResponse* response) override;

private:
    // 生成mock推荐数据
    void GenerateMockRecommendations(
        const recommendation::RecommendationRequest& request,
        recommendation::RecommendationResponse& response);
    
    // 生成随机分数
    double GenerateRandomScore();
    
    // 生成随机类别
    std::string GenerateRandomCategory();
    
    // 生成模拟项目数据
    void GenerateMockItem(
        int index,
        recommendation::RecommendationItem& item);
    
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_real_distribution<double> score_dist_;
};

#endif // RECOMMENDATION_SERVICE_IMPL_H