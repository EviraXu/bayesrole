#ifndef FDEEP_REPLACE_H_
#define FDEEP_REPLACE_H_

#include <Eigen/Core>
#include <string>
#include <map>
#include <utility>
#include <vector>
//
typedef Eigen::Array<float, Eigen::Dynamic, 1> EigenVector;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> EigenMatrix;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> DenseWeights;//全连接层（Dense层）的权重
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> DenseBiases;//全连接层（Dense层）的偏置

namespace jdeep {

enum layer_type {
    INPUT,//输入层
    DENSE,//全连接层
    CFV_MASK,
    CFV_FROM_WIN,//胜率层
    CFV_FROM_WIN_v2,
    ZERO_SUM,
    RNN,//RNN层
};//定义了不同类型的层

enum activation_type {
    NA,
    LINEAR,
    RELU,
    SIGMOID
};//层使用的激活函数类型

typedef int layer_id;

class model {
public:
    EigenVector predict(const EigenVector& input) const;//用于预测的predict函数，输入是EigenVector类型，返回同样类型的输出

    static model load_model(const std::string& filename);//从文件加载模型

private:
    std::map<layer_id, DenseWeights> layer_weights;//权重
    std::map<layer_id, DenseBiases> layer_biases;//偏置
    std::map<layer_id, DenseWeights> layer_kernel;
    std::map<layer_id, DenseWeights> layer_recurrent_kernel;
    std::vector<layer_id> ordered_layers;//已排序的层
    std::map<layer_id, std::pair<layer_type, activation_type>> layer_info;//层信息（包含层类型和激活类型）
    std::map<layer_id, std::vector<layer_id>> predecessors;//每个层的前序层节点信息

    layer_id output_layer;//成员变量指示输出层的ID
};

} // namespace jdeep;

#endif // FDEEP_REPLACE_H_
