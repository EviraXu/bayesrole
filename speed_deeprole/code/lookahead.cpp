#include "lookahead.h"
#include "lookup_tables.h"

//返回当前节点已完成的游戏轮次数，通过将成功和失败的次数相加计算得出
int LookaheadNode::round() const {
    return this->num_succeeds + this->num_fails;
}

//创建并返回一个新的 LookaheadNode 实例，这个实例被初始化为一个根提议节点。它包括成功和失败的任务数量、提议者、提议次数等属性。
std::unique_ptr<LookaheadNode> LookaheadNode::RootProposal(int num_succeeds, int num_fails, int proposer, int propose_count) {
    auto result = std::make_unique<LookaheadNode>();
    result->type = PROPOSE;
    //printf("Root PROPOSE node.\n");
    result->num_succeeds = num_succeeds;
    result->num_fails = num_fails;
    result->proposer = proposer;
    result->propose_count = propose_count;
    result->proposal = 0;
    result->merlin_pick = -1;
    return result;
}

//创建一个新的 LookaheadNode 实例，它是父节点的复制，包括所有父节点的状态信息。
std::unique_ptr<LookaheadNode> LookaheadNode::CopyParent(const LookaheadNode& parent) {
    auto result = std::make_unique<LookaheadNode>();
    result->type = parent.type;
    result->num_succeeds = parent.num_succeeds;
    result->num_fails = parent.num_fails;
    result->proposer = parent.proposer;
    result->propose_count = parent.propose_count;
    result->proposal = parent.proposal;
    result->merlin_pick = parent.merlin_pick;
    result->fails = parent.fails;
    return result;
}

//返回节点类型的字符串表示。每个节点类型对应游戏中的一个动作或结果。
std::string LookaheadNode::typeAsString() const {
    switch(type) {
    case PROPOSE:
        return "PROPOSE";
    case VOTE:
        return "VOTE";
    case MISSION:
        return "MISSION";
    case TERMINAL_MERLIN:
        return "TERMINAL_MERLIN";
    case TERMINAL_PROPOSE_NN:
        return "TERMINAL_PROPOSE_NN";
    case TERMINAL_NO_CONSENSUS:
        return "TERMINAL_NO_CONSENSUS";
    case TERMINAL_TOO_MANY_FAILS:
        return "TERMINAL_TOO_MANY_FAILS";
    }
    return "?????";
}

//这个递归函数用于向决策树中添加子节点。根据节点的类型，函数将生成对应的子节点。
//例如，对于提议节点，它将为每个可能的提议添加一个投票节点。对于投票节点，它将根据投票结果添加一个新的提议节点或任务节点。
void add_lookahead_children(const int depth, LookaheadNode* node) {
    switch (node->type) {
    case PROPOSE: {
        const int* index_to_proposal = (ROUND_TO_PROPOSE_SIZE[node->round()] == 2) ? INDEX_TO_PROPOSAL_2 : INDEX_TO_PROPOSAL_3;
        for (int i = 0; i < NUM_PROPOSAL_OPTIONS; i++) {
            auto new_child = LookaheadNode::CopyParent(*node);
            new_child->type = VOTE;
            // printf("depth is %d\n",depth);
            // printf("Propose generate a Vote node.\n");            
            new_child->proposal = index_to_proposal[i];
            add_lookahead_children(depth - 1, new_child.get());
            node->children.push_back(std::move(new_child));
        }
    } break;
    case VOTE: {
        for (int i = 0; i < (1 << NUM_PLAYERS); i++) {
            auto new_child = LookaheadNode::CopyParent(*node);
            new_child->proposer = (new_child->proposer + 1) % NUM_PLAYERS;

            if (__builtin_popcount(i) <= NUM_PLAYERS/2 ) {//检查整数 i 的二进制表示中设置的位数是否小于或等于 NUM_PLAYERS/2,即投票不通过
                new_child->propose_count++;
                new_child->proposal = 0;

                // Vote fails
                if (new_child->propose_count == 5) {//所有人的提议都没有被通过
                    new_child->type = TERMINAL_NO_CONSENSUS;
                    // printf("depth is %d\n",depth);
                    // printf("VOTE generate a TERMINAL_NO_CONSENSUS node.\n");
                } else if (depth == 0) {
                    new_child->type = TERMINAL_PROPOSE_NN;
                    // printf("depth is %d\n",depth);
                    // printf("VOTE generate a TERMINAL_PROPOSE_NN node.\n");
                } else {
                    new_child->type = PROPOSE;//继续提议
                    // printf("depth is %d\n",depth);
                    // printf("VOTE generate a PROPOSE node.\n");
                }
            } else {
                // Vote passes
                new_child->propose_count = 0;
                new_child->type = MISSION;//提议通过，开始执行任务
                // printf("depth is %d\n",depth);
                // printf("VOTE generate a MISSION node.\n");
            }

            add_lookahead_children(depth, new_child.get());
            node->children.push_back(std::move(new_child));
        }
    } break;
    case MISSION: {
        for (int i = 0; i < NUM_EVIL + 1; i++) {
            auto new_child = LookaheadNode::CopyParent(*node);
            if (i == 0) {//任务成功
                new_child->num_succeeds++;
            } else {//任务失败
                new_child->num_fails++;
                new_child->fails.push_back(std::make_pair(new_child->proposal, i));
            }
            if (new_child->num_fails == 3) {//达成任务失败条件，游戏结束
                new_child->type = TERMINAL_TOO_MANY_FAILS;
                // printf("depth is %d\n",depth);
                // printf("MISSION generate a TERMINAL_TOO_MANY_FAILS node.\n");
            } else if (new_child->num_succeeds == 3) {//达成任务成功条件，游戏结束
                new_child->type = TERMINAL_MERLIN;
                // printf("depth is %d\n",depth);
                // printf("MISSION generate a TERMINAL_MERLIN node.\n");
            } else if (depth == 0) {
                new_child->type = TERMINAL_PROPOSE_NN;
                // printf("depth is %d\n",depth);
                // printf("MISSION generate a TERMINAL_PROPOSE_NN node.\n");
            } else {//游戏继续，从propose
                new_child->type = PROPOSE;
                // printf("depth is %d\n",depth);
                // printf("MISSION generate a PROPOSE node.\n");
            }
            add_lookahead_children(depth, new_child.get());
            node->children.push_back(std::move(new_child));
        }
    } break;
    default: break;
    }
}

//初始化 LookaheadNode 的概率和策略相关字段。
//这包括设置概率向量，以及根据节点类型初始化策略和累计遗憾数据结构。对于某些终端节点，它还会加载神经网络模型。
void populate_lookahead_fields(const std::string& model_search_dir, LookaheadNode* node) {
    //改-初始化
    node->finding_evil = std::make_unique<std::array<MerlinData, NUM_PLAYERS>>();
    node->finding_merlin = std::make_unique<std::array<MerlinData, NUM_PLAYERS>>();
    double probability = 1.0 / NUM_PLAYERS;
    for (int i = 0; i < NUM_PLAYERS; i++) {
        node->finding_evil->at(i).setConstant(probability);
        node->finding_merlin->at(i).setConstant(1.0);
    }
    //
    for (int i = 0; i < NUM_PLAYERS; i++) {
        //将 reach_probs 数组的第 i 个元素设置为 ViewpointVector 类型的常量值1.0
        node->reach_probs[i] = ViewpointVector::Constant(1.0);
        //将 counterfactual_values 数组的第 i 个元素设置为0
        node->counterfactual_values[i].setZero();
    }

    switch (node->type) {
    case PROPOSE: {
        // Initialize the node's memory
        //创建了 ProposeData 类型的智能指针 node->propose_regrets、node->propose_strategy 和 node->propose_cum
        node->propose_regrets = std::make_unique<ProposeData>();
        node->propose_regrets->setZero();
        node->propose_strategy = std::make_unique<ProposeData>();
        node->propose_strategy->setZero();
        node->propose_cum = std::make_unique<ProposeData>();
        node->propose_cum->setZero();
    } break;
    case VOTE: {
        //创建数组并初始化为0
        node->vote_regrets = std::make_unique<std::array<VoteData, NUM_PLAYERS>>();
        node->vote_strategy = std::make_unique<std::array<VoteData, NUM_PLAYERS>>();
        node->vote_cum = std::make_unique<std::array<VoteData, NUM_PLAYERS>>();
        for (int i = 0; i < NUM_PLAYERS; i++) {
            node->vote_regrets->at(i).setZero();
            node->vote_strategy->at(i).setZero();
            node->vote_cum->at(i).setZero();
        }
    } break;
    case MISSION: {
        //创建数组并初始化为0
        node->mission_regrets = std::make_unique<std::array<MissionData, NUM_PLAYERS>>();
        node->mission_strategy = std::make_unique<std::array<MissionData, NUM_PLAYERS>>();
        node->mission_cum = std::make_unique<std::array<MissionData, NUM_PLAYERS>>();
        for (int i = 0; i < NUM_PLAYERS; i++) {
            node->mission_regrets->at(i).setZero();
            node->mission_strategy->at(i).setZero();
            node->mission_cum->at(i).setZero();
        }
    } break;
    case TERMINAL_MERLIN: {
        node->merlin_regrets = std::make_unique<std::array<MerlinData, NUM_PLAYERS>>();
        node->merlin_strategy = std::make_unique<std::array<MerlinData, NUM_PLAYERS>>();
        node->merlin_cum = std::make_unique<std::array<MerlinData, NUM_PLAYERS>>();
        for (int i = 0; i < NUM_PLAYERS; i++) {
            node->merlin_regrets->at(i).setZero();
            node->merlin_strategy->at(i).setZero();
            node->merlin_cum->at(i).setZero();
        }
        // Intentional missing break.
    }
    case TERMINAL_NO_CONSENSUS:
    case TERMINAL_TOO_MANY_FAILS: {
        node->full_reach_probs = std::make_unique<AssignmentProbs>();
    } break;
    case TERMINAL_PROPOSE_NN: {
        node->full_reach_probs = std::make_unique<AssignmentProbs>();
        node->nn_model = load_model(model_search_dir, node->num_succeeds, node->num_fails, node->propose_count);
    } break;
    default: break;
    }

    for (auto& child : node->children) {
        populate_lookahead_fields(model_search_dir, child.get());
    }
}

//使用指定的游戏状态（成功次数、失败次数、提议者、提议次数）和深度创建一个前瞻树的根节点，并填充整个树。
std::unique_ptr<LookaheadNode> create_avalon_lookahead(
    const int num_succeeds,
    const int num_fails,
    const int proposer,
    const int propose_count,
    const int depth,
    const std::string& model_search_dir) {

    auto root_node = LookaheadNode::RootProposal(num_succeeds, num_fails, proposer, propose_count);
    add_lookahead_children(depth, root_node.get());
    populate_lookahead_fields(model_search_dir, root_node.get());
    return root_node;
}

//递归计算树中特定类型节点的数量。
int count_lookahead_type(LookaheadNode* node, NodeType type) {
    int total_count = (node->type == type) ? 1 : 0;
    for (auto& child : node->children) {
        total_count += count_lookahead_type(child.get(), type);
    }
    return total_count;
}

//递归计算树中所有节点的数量。
int count_lookahead(LookaheadNode* node) {
    int total_count = 1;
    for (auto& child : node->children) {
        total_count += count_lookahead(child.get());
    }
    return total_count;
}
