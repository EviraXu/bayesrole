#ifndef SERIALIZATION_H_
#define SERIALIZATION_H_

#include <iostream>

#include "./lookahead.h"
//序列化相关的函数

//反序列化函数
//从输入流中读取JSON格式的数据，并将其反序列化为starting_reach_probs对象。这个对象可能代表了游戏开始时的各种状态概率。
void json_deserialize_starting_reach_probs(std::istream& in_stream, AssignmentProbs* starting_reach_probs);

//序列化函数
//将LookaheadNode的根节点和与之关联的起始概率starting_reach_probs序列化为JSON格式，并将其写入输出流。
//这可以用于保存游戏的某个状态，以便将来可以恢复或分析。
void json_serialize_lookahead(const LookaheadNode* root, const AssignmentProbs& starting_reach_probs, std::ostream& out_stream);

void json_serialize_lookahead_file(const LookaheadNode* root, const AssignmentProbs& starting_reach_probs, const std::string& filename);

//Eigen转换为double向量的模板
template <typename Derived>
inline std::vector<std::vector<double>> eigen_to_double_vector(const Eigen::ArrayBase<Derived>& array) {
    std::vector<std::vector<double>> result;

    for (int i = 0; i < array.rows(); i++) {
        std::vector<double> row;
        for (int j = 0; j < array.cols(); j++) {
            row.push_back(array(i, j));
        }
        result.push_back(row);
    }

    return result;
}

//Eigen转换为单一向量的模板
template <typename Derived>
inline std::vector<double> eigen_to_single_vector(const Eigen::ArrayBase<Derived>& array) {
    std::vector<double> result;

    for (int i = 0; i < array.rows(); i++) {
        result.push_back(array(i));
    }

    return result;
}

#endif // SERIALIZATION_H_
