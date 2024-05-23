#include "./fdeep_replace.h"

#include "json.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wreturn-std-move"
#include <fdeep/import_model.hpp>
#pragma GCC diagnostic pop

#include <cassert>
#include <fstream>
#include <iostream>

#include "./lookup_tables.h"

namespace jdeep {

//计算全连接层的输出。它接受输入向量、权重矩阵、偏置向量和激活函数类型。
static EigenVector calculate_dense_layer(
    const EigenVector& input,
    const DenseWeights& weights,
    const DenseBiases& biases,
    const activation_type activation
) { 

    // std::cout << "dense input: [";
    // for (int i = 0; i < input.size(); ++i) {
    //     std::cout << input(i);
    //     if (i != input.size() - 1) std::cout << ", ";
    // }
    // std::cout << "]" << std::endl;

    // std::cout << "weights: [";
    // for (int i = 0; i < weights.size(); ++i) {
    //     std::cout << weights(i);
    //     if (i != weights.size() - 1) std::cout << ", ";
    // }
    // std::cout << "]" << std::endl;

    // std::cout << "biases: [";
    // for (int i = 0; i < biases.size(); ++i) {
    //     std::cout << biases(i);
    //     if (i != biases.size() - 1) std::cout << ", ";
    // }
    // std::cout << "]" << std::endl;
    //将输入向量 input 转置后与权重矩阵 weights 相乘，然后加上偏置向量 biases，最后再次进行转置
    EigenVector result = (input.matrix().transpose() * weights + biases.transpose()).transpose().array();

    switch (activation) {
    case RELU://将结果中小于零的元素设置为零
        result = result.max(0.0);
        break;
    case SIGMOID://将结果应用于 sigmoid 函数，将结果映射到范围 [0, 1]
        result = (1.0 / (1.0 + (-result).exp()));
        break;
    default: break;
    }
    return result;
}

//给定输入向量 inp (belief)和 CFV,计算最终结果
static EigenVector calculate_cfv_mask_and_adjust_layer(const EigenVector& inp, const EigenVector& cfvs) {
    EigenVector result(NUM_PLAYERS * NUM_VIEWPOINTS);
    result.setZero();

    //创建一个名为 mask 的布尔数组，其大小为 NUM_PLAYERS * NUM_VIEWPOINTS，并初始化为全零（false）
    bool mask[NUM_PLAYERS * NUM_VIEWPOINTS] = {0};

    //如果第i个角色分配概率大于0，则把mask[NUM_VIEWPOINTS * player + ASSIGNMENT_TO_VIEWPOINT[i][player]]设置为true
    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        if (inp(i) > 0.0) {
            for (int player = 0; player < NUM_PLAYERS; player++) {
                mask[NUM_VIEWPOINTS * player + ASSIGNMENT_TO_VIEWPOINT[i][player]] = true;
            }
        }   
    }

    //计算 mask 中为 true 的元素个数，存储在变量 num_left 中
    int num_left = 0;
    for (int i = 0; i < NUM_PLAYERS * NUM_VIEWPOINTS; i++) {
        num_left += (int) (mask[i]);
    }

    //对于 mask 中为 true 的位置，将对应的 cfvs 元素加到 masked_sum 中
    float masked_sum = 0.0;
    for (int i = 0; i < NUM_PLAYERS * NUM_VIEWPOINTS; i++) {
        if (mask[i]) {
            masked_sum += cfvs(i);
        }
    }

    //计算每个位置需要减去的值 subtract_amount，即将 masked_sum 除以 num_left
    float subtract_amount = masked_sum / num_left;

    //对于 mask 中为 true 的位置，将对应的 cfvs 元素减去 subtract_amount，并将结果存储在 result 中
    for (int i = 0; i < NUM_PLAYERS * NUM_VIEWPOINTS; i++) {
        if (mask[i]) {
            result(i) = cfvs(i) - subtract_amount;
        }
    }

    return result;
};

//计算从胜利概率到 CFV 的转换
static EigenVector calculate_cfv_from_win_layer(const EigenVector& input_probs, const EigenVector& win_probs) {

    EigenVector result(NUM_PLAYERS * NUM_VIEWPOINTS);
    result.setZero();

    // std::cout << "input_probs: [";
    // for (int i = 0; i < input_probs.size(); ++i) {
    //     std::cout << input_probs(i);
    //     if (i != input_probs.size() - 1) std::cout << ", ";
    // }
    // std::cout << "]" << std::endl;

    // std::cout << "win_probs: [";
    // for (int i = 0; i < win_probs.size(); ++i) {
    //     std::cout << win_probs(i);
    //     if (i != win_probs.size() - 1) std::cout << ", ";
    // }
    // std::cout << "]" << std::endl;

    for (int assignment = 0; assignment < NUM_ASSIGNMENTS; assignment++) {
        //计算正方胜利的期望收益
        float good_expected_payoff = 2 * GOOD_WIN_PAYOFF * win_probs(assignment) - GOOD_WIN_PAYOFF;
        //计算反方胜利的期望收益
        float bad_expected_payoff = good_expected_payoff * EVIL_LOSE_PAYOFF;

        for (int player = 0; player < NUM_PLAYERS; player++) {
            int viewpoint = ASSIGNMENT_TO_VIEWPOINT[assignment][player];
            //根据视角的位置判断是正方视角还是反方视角
            float payoff = (viewpoint < NUM_GOOD_VIEWPOINTS) ? good_expected_payoff : bad_expected_payoff;
            //
            result(NUM_VIEWPOINTS*player + viewpoint) += input_probs(NUM_PLAYERS + assignment) * payoff;
        }
    }

    return result;
};

//用于确保层输出的总和为零
static EigenVector calculate_zero_sum_layer(const EigenVector& input) {
    EigenVector result = input;
    result -= input.sum() / (NUM_VIEWPOINTS * NUM_PLAYERS);
    return result;
}

//预测给定输入向量的输出。它按照定义的层顺序遍历每一层，根据每层的类型和前置层的输出来计算当前层的输出。
EigenVector model::predict(const EigenVector& input) const {
    //用于存储每个层级的计算结果
    std::map<layer_id, EigenVector> layer_results;

    //输入 input 存储在layer_results中，键为第一个层级
    layer_results[this->ordered_layers.front()] = input;

    //遍历已排序的层级列表 this->ordered_layers
    for (const layer_id id : this->ordered_layers) {
        //对于每个层级 id，判断其类型 type
        const layer_type type = this->layer_info.at(id).first;

        //如果类型为 INPUT，则跳过该层级
        if (type == INPUT) continue;

        //对于其他类型的层级，获取对应的激活函数类型 activation 和父层级列表 parents
        const activation_type activation = this->layer_info.at(id).second;
        const std::vector<layer_id>& parents = this->predecessors.at(id);

        //根据不同的层级类型，使用 switch 语句进行处理
        switch (type) {
        case INPUT: break;
        //
        case DENSE: {
            assert(parents.size() == 1);
            const EigenVector& input = layer_results[parents.front()];
            layer_results[id] = calculate_dense_layer(input, this->layer_weights.at(id), this->layer_biases.at(id), activation);
        } break;
        case CFV_MASK: {
            assert(parents.size() == 2);
            const EigenVector& input_1 = layer_results[parents.front()];
            const EigenVector& input_2 = layer_results[parents.back()];
            layer_results[id] = calculate_cfv_mask_and_adjust_layer(input_1, input_2);
        } break;
        case CFV_FROM_WIN: {
            assert(parents.size() == 2);
            const EigenVector& input_1 = layer_results[parents.front()];
            const EigenVector& input_2 = layer_results[parents.back()];
            layer_results[id] = calculate_cfv_from_win_layer(input_1, input_2);
        } break;
        case ZERO_SUM: {
            assert(parents.size() == 1);
            const EigenVector& input = layer_results[parents.front()];
            layer_results[id] = calculate_zero_sum_layer(input); 
        } break;
        }
    }

    return layer_results[this->output_layer];
};

// private:
//     std::map<layer_id, DenseWeights> layer_weights;
//     std::map<layer_id, DenseBiases> layer_biases;
//     std::vector<layer_id> ordered_layers;
//     std::map<layer_id, std::pair<layer_type, activation_type>> layer_info;
//     std::map<layer_id, std::vector<layer_id>> predecessors;
// };

//从 JSON 数据中加载权重
DenseWeights load_weights(const nlohmann::json& json_data, const int output_size) {
    //从 json_data 中解码出一个浮点数向量 floats
    const std::vector<float> floats = fdeep::internal::decode_floats(json_data);
    //确保浮点数向量的大小能够被 output_size 整除，以确保权重矩阵的维度正确
    assert(floats.size() % output_size == 0);
    const int input_size = floats.size() / output_size;

    //创建一个 DenseWeights 对象 result，其大小为 input_size 行、output_size 列
    DenseWeights result(input_size, output_size);

    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < output_size; j++) {
            result(i, j) = floats[i*output_size + j];//将浮点数向量中的值按序填充到 result 中的对应位置
        }
    }

    return result;
}

//从 JSON 数据中加载偏置
DenseBiases load_biases(const nlohmann::json& json_data, const int output_size) {
    ////从 json_data 中解码出一个浮点数向量 floats
    const std::vector<float> floats = fdeep::internal::decode_floats(json_data);

    //创建一个 DenseBiases 对象 result，其大小为 output_size
    DenseBiases result(output_size);

    for (int i = 0; i < output_size; i++) {
        result(i) = floats[i];//将浮点数向量中的值按序填充到 result 中的对应位置
    }

    return result;
}

//运行模型的测试用例并进行结果验证
void run_test(const nlohmann::json& test_case, const model& test_model) {
    const int input_size = test_case["inputs"][0]["shape"][4];
    //printf("Input size:%d\n",input_size);
    const int output_size = test_case["outputs"][0]["shape"][4];
    //printf("Output size:%d\n",output_size);

    EigenVector input_vec = load_biases(test_case["inputs"][0]["values"], input_size).array();
    // std::cout << "test input_vec: [";
    // for (int i = 0; i < input_vec.size(); ++i) {
    //     std::cout << input_vec(i);
    //     if (i != input_vec.size() - 1) std::cout << ", ";
    // }
    // std::cout << "]" << std::endl;

    EigenVector output_vec = load_biases(test_case["outputs"][0]["values"], output_size).array();
    // std::cout << "test output_vec: [";
    // for (int i = 0; i < output_vec.size(); ++i) {
    //     std::cout << output_vec(i);
    //     if (i != output_vec.size() - 1) std::cout << ", ";
    // }
    // std::cout << "]" << std::endl;

    EigenVector result = test_model.predict(input_vec);
    // std::cout << "result: [";
    // for (int i = 0; i < result.size(); ++i) {
    //     std::cout << result(i);
    //     if (i != result.size() - 1) std::cout << ", ";
    // }
    // std::cout << "]" << std::endl;

    if (result.size() != output_size) {
        std::cerr << "Result size: (" << result.rows() << ", " << result.cols() << "). Size: " << result.size() << std::endl;
        std::cerr << "Output size: " << output_size << std::endl;
        throw std::length_error("Test result is not the correct shape");
    }

    for (int i = 0; i < output_size; i++) {
        if (std::abs(result(i) - output_vec(i)) > 0.0001) {
            std::cerr << std::setprecision(16);
            std::cerr << "Result:" << std::endl;
            std::cerr << result.transpose().leftCols(6) << std::endl;
            std::cerr << "Expected:" << std::endl;
            std::cerr << output_vec.transpose().leftCols(6) << std::endl;
            throw std::domain_error("results don't match");
        }
    }
}

//从 JSON 文件加载模型的结构和参数。它读取层的配置，权重和偏置，并构建一个 model 对象，该对象可以用于预测。
model model::load_model(const std::string& filename) {
    std::ifstream in_stream(filename);
    //使用断言来确保文件流成功打开
    assert(in_stream.good());
    // result，用于存储加载后的模型信息
    model result;

    //通过读取 JSON 数据，解析模型的结构和参数
    std::cerr << "Loading model: " << filename << std::endl;
    nlohmann::json json_data;
    in_stream >> json_data;

    layer_id current_id = 0;
    //将层的名称映射到层的 ID
    std::map<std::string, layer_id> name_to_id;
    //对于所有层来说
    for (const nlohmann::json& layer : json_data["architecture"]["config"]["layers"]) {
        //获取层的类名 layer_class_name 
        const std::string layer_class_name = layer["class_name"];
        //获取层的名称 layer_name
        const std::string layer_name = layer["name"];
        //将层的名称和当前的 ID 关联起来
        name_to_id[layer_name] = current_id;

        //如果是输入层，确定层的类型和激活函数类型，并将其存储在 result.layer_info 中
        if (layer_class_name == "InputLayer") {
            result.layer_info[current_id] = std::make_pair(INPUT, NA);
        } else if (layer_class_name == "Dense") {//如果层是 Dense 层
            activation_type activation = NA;//确定层的激活函数类型
            if (layer["config"]["activation"] == "linear") {
                activation = LINEAR;
            } else if (layer["config"]["activation"] == "relu") {
                activation = RELU;
            } else if (layer["config"]["activation"] == "sigmoid") {
                activation = SIGMOID;
            } else {
                assert(false);
            }
            result.layer_info[current_id] = std::make_pair(DENSE, activation);

            const int output_size = layer["config"]["units"];
            //加载该层的权重和偏置
            result.layer_weights[current_id] = load_weights(json_data["trainable_params"][layer_name]["weights"], output_size);
            result.layer_biases[current_id] = load_biases(json_data["trainable_params"][layer_name]["bias"], output_size);
        } else if (layer_class_name == "CFVMaskAndAdjustLayer") {
            result.layer_info[current_id] = std::make_pair(CFV_MASK, NA);
        } else if (layer_class_name == "CFVFromWinProbsLayer") {
            result.layer_info[current_id] = std::make_pair(CFV_FROM_WIN, NA);
        } else if (layer_class_name == "ZeroSumLayer") {
            result.layer_info[current_id] = std::make_pair(ZERO_SUM, NA);
        }

        //遍历层的输入节点，将输入节点所对应的层的 ID 存储在 result.predecessors 中
        for (const nlohmann::json& inbound_nodes : layer["inbound_nodes"]) {
            for (const nlohmann::json& inbound_layer : inbound_nodes) {
                const std::string layer_name = inbound_layer[0];
                assert(name_to_id.count(layer_name) > 0);
                result.predecessors[current_id].push_back(name_to_id[layer_name]);
            }
        }

        result.ordered_layers.push_back(current_id);
        current_id++;
    }

    //获取输出层的 ID
    result.output_layer = name_to_id.at(json_data["architecture"]["config"]["output_layers"][0][0]);

    int test_num = 1;
    for (const nlohmann::json& test_case : json_data["tests"]) {
        std::cerr << "Running test " << test_num << "..." << std::endl;
        run_test(test_case, result);
    }

    return result;
}

} // namespace jdeep;
