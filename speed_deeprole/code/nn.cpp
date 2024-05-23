#include "./nn.h"

#include <map>
#include <sstream>
#include <iostream>

#include "eigen_types.h"
#include "lookup_tables.h"
#include "game_constants.h"

std::map<std::tuple<int, int, int>, std::shared_ptr<Model>> model_cache;//一个静态的model_cache映射，用来缓存已加载的模型

//生成给定参数的模型文件名
static std::string get_model_filename(const std::string& search_dir, const int num_succeeds, const int num_fails, const int propose_count) {
    std::stringstream sstream;
    sstream << ((search_dir.empty()) ? "" : (search_dir + "/")) << num_succeeds << "_" << num_fails << "_" << propose_count << ".json";
    return sstream.str();
}

//首先检查模型是否已经在缓存中。如果是，直接返回它。
//如果不在缓存中，函数将加载一个模型，将其放入缓存，并返回对它的共享指针
std::shared_ptr<Model> load_model(const std::string& search_dir, const int num_succeeds, const int num_fails, const int propose_count) {
    auto cache_key = std::make_tuple(num_succeeds, num_fails, propose_count);
    if (model_cache.count(cache_key) != 0) {
        return model_cache[cache_key];
    }

    auto model_filename = get_model_filename(search_dir, num_succeeds, num_fails, propose_count);
    std::cerr << "Model's name: " << model_filename << std::endl;
    auto model = jdeep::model::load_model(model_filename);
    auto model_ptr = std::make_shared<Model>(num_succeeds, num_fails, propose_count, std::move(model));

    model_cache[cache_key] = model_ptr;

    return model_ptr;
}

//将提议者的索引和一组输入概率作为参数，然后执行神经网络的前向传播，以生成一个输出向量。
void Model::predict(const int proposer, const AssignmentProbs& input_probs, ViewpointVector* output_values) {
    //创建一个大小为65的Eigen向量input_tensor，并将其初始化为零
    EigenVector input_tensor(65);
    input_tensor.setZero();

    //将input_tensor的索引proposer处的元素设置为1.0，用于表示当前的proposer
    input_tensor(proposer) = 1.0;

    //将input_probs转换为浮点数类型，并将其赋值给input_tensor的后60个元素
    input_tensor.bottomRows<60>() = input_probs.cast<float>();

    //得到预测结果，将结果存储在result中
    EigenVector result = this->model.predict(input_tensor);

    //对于每个玩家，使用block函数从result中提取一个大小为NUM_VIEWPOINTS行、1列的数据块
    //并将其转换为double类型
    //然后将结果赋值给output_values对应的玩家位置
    for (int player = 0; player < NUM_PLAYERS; player++) {
        output_values[player] = result.block<NUM_VIEWPOINTS, 1>(NUM_VIEWPOINTS*player, 0).cast<double>();
    }
}

//
void print_loaded_models(const std::string& search_dir) {
    if (model_cache.size() == 0) {
        std::cerr << "No models loaded." << std::endl;
        return;
    }
    //std::cerr << "model_cache.size() ==" << std::endl;
    //std::cerr << model_cache.size() << std::endl;
    for (const auto& pair : model_cache) {
        const auto& tuple = pair.first;
        std::cerr << get_model_filename(search_dir, std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple)) << std::endl;
    }
}
