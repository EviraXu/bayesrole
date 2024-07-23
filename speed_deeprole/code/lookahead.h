//声明了一个与前瞻机制相关的结构和函数
#ifndef LOOKAHEAD_H_
#define LOOKAHEAD_H_

#include <memory>
#include <array>
#include <vector>

#include "lookup_tables.h"
#include "eigen_types.h"

#include "nn.h"

//定义了前瞻树中可能存在的不同类型的节点
enum NodeType {
    PROPOSE,//提议
    VOTE,//投票
    MISSION,//执行任务
    TERMINAL_MERLIN,//识别梅林终止状态
    TERMINAL_NO_CONSENSUS,//提议执行任务没有共识终止状态
    TERMINAL_TOO_MANY_FAILS,//任务失败太多终止状态
    TERMINAL_PROPOSE_NN//由神经网络生成提议终止状态
};

//代表了前瞻机制中的一个节点
struct LookaheadNode {
    NodeType type;//节点类型
    int num_succeeds;//胜利轮数
    int num_fails;//失败轮数
    int proposer;//提议者
    int propose_count;//当前提议数量
    uint32_t proposal;//提议的执行任务玩家（00110）
    int merlin_pick;//梅林是谁

    //reach_probs是一个二维数组，每行代表一个玩家（5个玩家），每列代表一个信息集（15个信息集）
    //存储每个玩家在不同信息集下的到达概率
    ViewpointVector reach_probs[NUM_PLAYERS];
    ViewpointVector counterfactual_values[NUM_PLAYERS];//来自不同玩家信息集的遗憾值
    
    //储存每个角色分配到达节点的概率
    std::unique_ptr<AssignmentProbs> full_reach_probs;//

    //改-新增
    std::unique_ptr<std::array<MerlinData, NUM_PLAYERS>> finding_merlin;
    std::unique_ptr<std::array<MerlinData, NUM_PLAYERS>> finding_evil;
    //

    //存储提议的概率、策略和累计数据
    std::unique_ptr<ProposeData> propose_regrets;
    std::unique_ptr<ProposeData> propose_strategy;
    std::unique_ptr<ProposeData> propose_cum;

    //存储投票的概率、策略和累计数据
    std::unique_ptr<std::array<VoteData, NUM_PLAYERS>> vote_regrets;
    std::unique_ptr<std::array<VoteData, NUM_PLAYERS>> vote_strategy;
    std::unique_ptr<std::array<VoteData, NUM_PLAYERS>> vote_cum;

    //存储任务的概率、策略和累计数据
    std::unique_ptr<std::array<MissionData, NUM_PLAYERS>> mission_regrets;

    //mission_strategy->at(player)(viewpoint, 1)表示从mission_strategy中获取第player个元素
    //然后在该元素的二维数组中，使用viewpoint作为行索引，1作为列索引，获取对应的值
    std::unique_ptr<std::array<MissionData, NUM_PLAYERS>> mission_strategy;
    std::unique_ptr<std::array<MissionData, NUM_PLAYERS>> mission_cum;

    //存储梅林的概率、策略和累计数据
    std::unique_ptr<std::array<MerlinData, NUM_PLAYERS>> merlin_regrets;
    std::unique_ptr<std::array<MerlinData, NUM_PLAYERS>> merlin_strategy;
    std::unique_ptr<std::array<MerlinData, NUM_PLAYERS>> merlin_cum;

    //fails 是一个 std::vector 对象，它存储了一组类型为 std::pair<uint32_t, int> 的元素
    //fails中的其中一个对象为fail，表示一个由两个值组成的有序对
    //fail.first表示当前执行任务玩家；fail.second表示该失败情况中需要evil角色参与的数量
    std::vector<std::pair<uint32_t, int>> fails;//失败
    std::vector<std::unique_ptr<LookaheadNode>> children;//子节点

    std::shared_ptr<Model> nn_model;//神经网络模型

    LookaheadNode() = default;

    int round() const;

    std::string typeAsString() const;

    static std::unique_ptr<LookaheadNode> RootProposal(int num_succeeds, int num_fails, int proposer, int propose_count);
    static std::unique_ptr<LookaheadNode> CopyParent(const LookaheadNode& parent);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

std::unique_ptr<LookaheadNode> create_avalon_lookahead(
    const int num_succeeds,
    const int num_fails,
    const int proposer,
    const int propose_count,
    const int depth,
    const std::string& model_search_dir);

int count_lookahead_type(LookaheadNode* node, const NodeType type);
int count_lookahead(LookaheadNode* node);

#endif // LOOKAHEAD_H_
