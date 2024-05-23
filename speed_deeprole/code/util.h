#ifndef UTIL_H_
#define UTIL_H_

#include <string>
#include <random>

#include "./lookahead.h"

extern std::mt19937 rng;

//包含了多个与游戏初始化和配置相关的参数
struct Initialization {
    int depth;
    int num_succeeds;
    int num_fails;
    int propose_count;

    int proposer;
    std::string generate_start_technique;//是一个字符串，表示用于生成开始状态的方法或算法
    AssignmentProbs starting_probs;//存储概率分配

    //用于配置模拟或算法处理中的迭代次数
    int iterations;
    int wait_iterations;
    ViewpointVector solution_values[NUM_PLAYERS];

    std::string Stringify() const;//用于将 Initialization 数据转换为字符串表示，以便显示或记录。

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

void seed_rng();//用于给随机数生成器（rng）种下种子，以确保随机性的初始化

//使用提供的值来准备一个 Initialization 对象，为游戏的初始化过程做好准备。
void prepare_initialization(
    const int depth,
    const int num_succeeds,
    const int num_fails,
    const int propose_count,
    Initialization* init
);

//运行了涉及到CFR的初始化
void run_initialization_with_cfr(
    const int iterations,
    const int wait_iterations,
    const std::string& model_search_dir,
    Initialization* init
);

#endif // UTIL_H_
