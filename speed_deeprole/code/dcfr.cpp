#define CFR_PLUS
#include <iostream>
#include <memory>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <array>
#include <utility>
#include <fstream> 

#include "dcfr.h"
#include "lookup_tables.h"

using namespace std;


#define ASSERT(left,operator,right) { if(!((left) operator (right))){ std::cerr << "ASSERT FAILED: " << #left << #operator << #right << " @ " << __FILE__ << " (" << __LINE__ << "). " << #left << "=" << (left) << "; " << #right << "=" << (right) << std::endl; } }

//计算在 MISSION 类型的 LookaheadNode 节点中，特定参与者对mission的单次responsibility
static double my_single_pass_responsibility(LookaheadNode* node, int me, int my_viewpoint, int my_partner) {
    //验证节点类型是否是MISSION
    assert(node->type == MISSION);
    //验证在me的视角中，my_partner是反方角色
    assert(VIEWPOINT_TO_BAD[me][my_viewpoint] == my_partner);
    assert(my_viewpoint >= NUM_GOOD_VIEWPOINTS);
    //
    int partner_viewpoint = VIEWPOINT_TO_PARTNER_VIEWPOINT[me][my_viewpoint];
    assert(VIEWPOINT_TO_PARTNER_VIEWPOINT[my_partner][partner_viewpoint] == my_viewpoint);
    //验证任务的执行者是不是me和my_partner
    assert(node->proposal & (1 << me));
    assert(node->proposal & (1 << my_partner));

    //我让任务通过的概率
    double my_pass_prob = node->mission_strategy->at(me)(my_viewpoint, 0);
    //队友让任务通过的概率
    double partner_pass_prob = node->mission_strategy->at(my_partner)(partner_viewpoint, 0);
    //计算我们俩其中之一让任务失败的概率
    double outcome_prob = my_pass_prob * (1.0 - partner_pass_prob) + (1.0 - my_pass_prob) * partner_pass_prob;
    //
    double my_responsibility_portion = my_pass_prob * my_pass_prob + (1.0 - my_pass_prob) * (1.0 - my_pass_prob);
    double partner_responsibility_portion = partner_pass_prob * partner_pass_prob + (1.0 - partner_pass_prob) * (1.0 - partner_pass_prob);
    double my_responsibility_exponent = my_responsibility_portion / (my_responsibility_portion + partner_responsibility_portion);
    return pow(outcome_prob, my_responsibility_exponent);
}

//向给定的 LookaheadNode 节点中添加中间的反事实值（counterfactual values）
static void add_middle_cfvs(LookaheadNode* node, int me, int my_viewpoint, int my_partner, double* pass_cfv, double* fail_cfv) {
    assert(node->type == MISSION);
    assert(VIEWPOINT_TO_BAD[me][my_viewpoint] == my_partner);
    assert(my_viewpoint >= NUM_GOOD_VIEWPOINTS);
    int partner_viewpoint = VIEWPOINT_TO_PARTNER_VIEWPOINT[me][my_viewpoint];
    assert(VIEWPOINT_TO_PARTNER_VIEWPOINT[my_partner][partner_viewpoint] == my_viewpoint);
    assert(node->proposal & (1 << me));
    assert(node->proposal & (1 << my_partner));

    //计算 my_partner 参与者在给定视角下对mission的负责度 partner_responsibility
    double partner_responsibility = my_single_pass_responsibility(node, my_partner, partner_viewpoint, me);
    //计算middle_cfv
    //获取 node 的第二个子节点的 counterfactual_values[me](my_viewpoint) 的值
    //并将其除以 partner_responsibility 得到
    double middle_cfv = node->children[1]->counterfactual_values[me](my_viewpoint);
    middle_cfv /= partner_responsibility;

    #ifndef NDEBUG
    double my_prob = node->mission_strategy->at(me)(my_viewpoint, 0);
    double partner_prob = node->mission_strategy->at(my_partner)(partner_viewpoint, 0);
    double my_resp = my_single_pass_responsibility(node, me, my_viewpoint, my_partner);
    double partner_resp = my_single_pass_responsibility(node, my_partner, partner_viewpoint, me);
    double combined_resp = my_resp * partner_resp;
    double expected_resp = (1.0 - my_prob) * partner_prob + my_prob * (1.0 - partner_prob);

    if (abs(combined_resp - expected_resp) > 1e-10) {
        std::cerr << "combined_resp: " << combined_resp << endl;
        std::cerr << "expected_resp: " << expected_resp << endl;
        std::cerr << "   difference: " << abs(combined_resp - expected_resp) << endl;
        assert(false);
    }

    //计算另一种方式计算的 middle_cfv_2
    //将 node 的第二个子节点的 counterfactual_values[me](my_viewpoint) 乘以 my_resp
    //然后除以 expected_resp 得到
    double middle_cfv_2 = node->children[1]->counterfactual_values[me](my_viewpoint) * my_resp;
    middle_cfv_2 /= expected_resp;
    if (abs(middle_cfv - middle_cfv_2) > 1e-10) {
        std::cerr << "middle_cfv: " << middle_cfv << endl;
        std::cerr << "middle_cfv_2: " << middle_cfv_2 << endl;
        std::cerr << "   difference: " << abs(middle_cfv - middle_cfv_2) << endl;
        assert(false);
    }
    #endif

    double partner_pass_prob = node->mission_strategy->at(my_partner)(partner_viewpoint, 0);
    *pass_cfv += middle_cfv * (1.0 - partner_pass_prob);
    *fail_cfv += middle_cfv * partner_pass_prob;
}

//填充与mission节点相关的到达概率
static void fill_reach_probabilities_for_mission_node(LookaheadNode* node) {
    assert(node->type == MISSION);

    for (int fails = 0; fails < NUM_EVIL + 1; fails++) {//按照
        //函数获取节点的 children[fails] 成员，并将其存储在 child 引用
        auto& child = node->children[fails];
        // First, pass-through all of the players not on the mission.
        // Second, pass-through all of the viewpoints where the player is good.
        //      - this step won't work on it's own, we need to fix things at the leaf node to ensure we remove possibilities where a good player was forced to fail.
        // This is optimized - we'll be multiplying "correctly" later down in this function.
        //先初始化
        for (int player = 0; player < NUM_PLAYERS; player++) {
            child->reach_probs[player] = node->reach_probs[player];
        }

        switch (fails) {
        case 0: {
            // No one failed.没有人选择失败
            // For evil players, their move probability gets multiplied by the pass probability.
            for (int player = 0; player < NUM_PLAYERS; player++) {
                // Skip players not on the mission.
                if (((1 << player) & node->proposal) == 0) continue;
                //因为没人选择失败，所以乘以所有人都选择成功的策略（也就是第二个参数为0）
                for (int viewpoint = NUM_GOOD_VIEWPOINTS; viewpoint < NUM_VIEWPOINTS; viewpoint++) {
                    child->reach_probs[player](viewpoint) *= node->mission_strategy->at(player)(viewpoint, 0);
                    //std::cout<< "Player  " <<player<< "viewpoint " <<viewpoint<<" child->reach_probs:\n"<<child->reach_probs[player] <<std::endl;

                }
            }
        } break;
        case 1: {
            // This is the hard case, since we're combining probabilities oddly.
            //执行任务中，有一位玩家选择失败
            for (int player = 0; player < NUM_PLAYERS; player++) {
                // Skip players not on the mission.
                if (((1 << player) & node->proposal) == 0) continue;
                for (int viewpoint = NUM_GOOD_VIEWPOINTS; viewpoint < NUM_VIEWPOINTS; viewpoint++) {
                    int player_partner = VIEWPOINT_TO_BAD[player][viewpoint];
                    if ((1 << player_partner) & node->proposal) {
                        // The player's partner is on the mission. Weird stuff!
                        //执行任务的玩家是反方队友，但是任务失败了
                        //reach_probs[player](viewpoint)就用player对mission的单次responsibility计算
                        child->reach_probs[player](viewpoint) *= my_single_pass_responsibility(node, player, viewpoint, player_partner);
                        //std::cout<< "Player  " <<player<< "viewpoint " <<viewpoint<<" child->reach_probs:\n"<<child->reach_probs[player] <<std::endl;
                    } else {
                        // The player's partner is not on the mission, fail normally.
                        //参与行动的不是我的反方队友，那么参加任务一定有个反方角色，那就是我，显然我选择了失败
                        child->reach_probs[player](viewpoint) *= node->mission_strategy->at(player)(viewpoint, 1);
                    }
                }
            }
        } break;
        case 2: {
            // Everyone failed.两个人都选择了失败
            // For evil players, their move probability gets multiplied by the fail probability.
            for (int player = 0; player < NUM_PLAYERS; player++) {
                // Skip players not on the mission.
                if (((1 << player) & node->proposal) == 0) continue;
                for (int viewpoint = NUM_GOOD_VIEWPOINTS; viewpoint < NUM_VIEWPOINTS; viewpoint++) {
                    child->reach_probs[player](viewpoint) *= node->mission_strategy->at(player)(viewpoint, 1);
                    //std::cout<< "Player  " <<player<< "viewpoint " <<viewpoint<<" child->reach_probs:\n"<<child->reach_probs[player] <<std::endl;
                }
            }
        } break;
        }
    }
}

//填充其他节点的到达概率（PROPOSE、VOTE、MISSION）
static void fill_reach_probabilities(LookaheadNode* node) {
    switch (node->type) {
    case PROPOSE: {
        int player = node->proposer;
        for (int proposal = 0; proposal < NUM_PROPOSAL_OPTIONS; proposal++) {
            auto& child = node->children[proposal];
            for (int i = 0; i < NUM_PLAYERS; i++) {
                child->reach_probs[i] = node->reach_probs[i];
            }
            //propose_strategy->col(proposal)获取propose_strategy指针指向的二维数组的第proposal列向量
            //改
            child->reach_probs[player] *= node->propose_strategy->col(proposal);
            //std::cout<< "Proposal " <<proposal<< "Player  " <<player<<" child->reach_probs:\n"<<child->reach_probs[player] <<std::endl;
        }
    } break;
    case VOTE: {
        for (uint32_t vote_pattern = 0; vote_pattern < (1 << NUM_PLAYERS); vote_pattern++) {//遍历所有可能的投票模式
            auto& child = node->children[vote_pattern];
            for (int player = 0; player < NUM_PLAYERS; player++) {
                //vote 将被赋值为 0 或 1，代表投票结果的二进制位值
                int vote = (vote_pattern >> player) & 1;
                //函数将每个玩家的到达概率乘以节点的投票策略中对应玩家的列向量（0表示反对，1表示赞成）
                //改
                child->reach_probs[player] = node->reach_probs[player] * node->vote_strategy->at(player).col(vote);
                //std::cout<< "Player  " <<player<<" node->reach_probs[player]:\n"<<node->reach_probs[player] <<std::endl;
                //std::cout<< "Player  " <<player<< " Vote "<<vote<<" node->vote_strategy->at(player).col(vote):\n"<<node->vote_strategy->at(player).col(vote) <<std::endl;
                //std::cout<< "Vote "<<vote_pattern<< "Player  " <<player<<" child->reach_probs:\n"<<child->reach_probs[player] <<std::endl;
            }
        }
    } break;
    case MISSION: {
        fill_reach_probabilities_for_mission_node(node);
    } break;
    default: break;
    }
}

//计算每个分配情况的完整到达概率
//TERMINAL_NO_CONSENSUS、TERMINAL_TOO_MANY_FAILS、TERMINAL_PROPOSE_NN
//不可能的情况置为0
//可能的话，概率等于所有玩家在各自信息集下到达当前节点的概率乘积
static void populate_full_reach_probs(LookaheadNode* node) {
    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        double probability = 1.0;
        int evil = ASSIGNMENT_TO_EVIL[i];
        //遍历节点的失败情况
        for (auto fail : node->fails) {
            //判断执行任务的玩家中evil的个数是否小于当前轮次任务失败需要的evil角色参与的数量
            //如果小于的话，直接置node->full_reach_probs(i)=0
            if (__builtin_popcount(fail.first & evil) < fail.second) {
                probability = 0.0;
                break;
            }
        }
        for (int player = 0; player < NUM_PLAYERS && probability != 0; player++) {
            int viewpoint = ASSIGNMENT_TO_VIEWPOINT[i][player];
            //std::cout << "Player "<< player << "viewpoint "<< viewpoint <<"populate_full_reach_probs array:" << node->reach_probs[player](viewpoint) << std::endl;
            probability *= node->reach_probs[player](viewpoint);
        }
        (*(node->full_reach_probs))(i) = probability;
    }
}

void calculate_strategy(LookaheadNode* node, const double alpha,const double beta,const double gamma) {
    switch (node->type) {
    case PROPOSE: {
        // Initialize the node's memory
        auto& player_regrets = *(node->propose_regrets);
        auto& player_strategy = *(node->propose_strategy);

        // std::cout << "propose_regrets array:" << std::endl;
        // std::cout << player_regrets << std::endl;

        // std::cout << "After discounted array:" << std::endl;
        // std::cout << player_regrets << std::endl;


        //根据预处理宏 CFR_PLUS ，将 player_strategy 赋值为 player_regrets 
        #ifdef CFR_PLUS
        // These are already maxed
        //player_strategy = player_regrets;
        ProposeData positive_regrets = player_regrets.unaryExpr([](double v) { return std::max(v, 0.0); });
        // std::cout << "After positive_regrets array:" << std::endl;
        // std::cout << positive_regrets << std::endl;

        Eigen::ArrayXd strategy_sums = positive_regrets.rowwise().sum();
        // std::cout << "strategy_sums:\n" << strategy_sums << std::endl;

        for (int i = 0; i < player_regrets.rows(); ++i) {
            if (strategy_sums(i) > 0) {
                player_strategy.row(i) = positive_regrets.row(i) / strategy_sums(i);
            } else {
                player_strategy.row(i) = Eigen::ArrayXd::Constant(player_regrets.cols(), 1.0 / NUM_PROPOSAL_OPTIONS);
            }
        }
        #else
        // These aren't so we have to max them.
        player_strategy = player_regrets.max(0.0);
        #endif

        // std::cout << "After DCFR array:" << std::endl;
        // std::cout << player_strategy << std::endl;

        
        //改-更新策略
        int proposer = node->proposer;
        double evil_threshold = 0.5;
        ProposeData tmp_strategy = player_strategy;
        const int* index_to_proposal = (ROUND_TO_PROPOSE_SIZE[node->round()] == 2) ? INDEX_TO_PROPOSAL_2 : INDEX_TO_PROPOSAL_3;
        for(int i = 0; i < NUM_VIEWPOINTS; i++){
            std::vector<int> evil_players;
            for (int idx = 0; idx < NUM_PLAYERS; ++idx) {
                if (node->finding_evil->at(proposer)(i,idx) > evil_threshold) {
                    evil_players.push_back(idx);
                }
            }
            for(int j = 0; j < NUM_PROPOSAL_OPTIONS; j++){
                int proposal = index_to_proposal[j];
                for(int m = 0; m < evil_players.size(); m++){
                    if((1 << evil_players[m]) & proposal){
                        tmp_strategy(i,j) *= 0.5;
                    }
                }
            }
        }

        //将 player_strategy 中的每个元素归一化，使得每列的元素之和等于1。这样可以将策略表示为概率分布

        // std::cout << "After bayes array:" << std::endl;
        // std::cout << tmp_strategy << std::endl;

        ProposeData tmp_holder = tmp_strategy.colwise() / tmp_strategy.rowwise().sum();
        //ProposeData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
        //处理可能出现的非有限值，确保 tmp_holder 中的每个元素都是有效的概率值（介于0和1之间）。
        //如果某个元素为非有限值，那么它将被替换为一个平均概率值，即 1.0/NUM_PROPOSAL_OPTIONS
        tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 1.0/NUM_PROPOSAL_OPTIONS; });

        // std::cout << "To one array:" << std::endl;
        // std::cout << tmp_holder << std::endl;
        //将player_strategy更新为一个加权平衡了原始策略和均匀分布概率的混合策略，其中颤抖值 TREMBLE_VALUE 控制了两者之间的权衡
        player_strategy = (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * ProposeData::Constant(1.0/NUM_PROPOSAL_OPTIONS);

        // std::cout << "Propose player_strategy:" << std::endl;
        // std::cout << player_strategy << std::endl;


        if (gamma != 0) {//如果累积策略权重 cum_strat_weight 不为零
            //根据当前节点的到达概率、玩家的策略选择以及累积策略权重，更新节点的累积策略
            *(node->propose_cum) += (player_strategy.colwise() * node->reach_probs[node->proposer]) * gamma;
        }
    } break;
    case VOTE: {//于propose节点类似
        for (int i = 0; i < NUM_PLAYERS; i++) {
            auto& player_regrets = node->vote_regrets->at(i);
            auto& player_strategy = node->vote_strategy->at(i);

            // std::cout <<"Player "<<i<< " vote_regrets array:" << std::endl;
            // std::cout << player_regrets << std::endl;

            // std::cout <<"Player "<<i<< " vote_strategy array:" << std::endl;
            // std::cout << player_strategy << std::endl;


            for (int i = 0; i < player_regrets.rows(); ++i) {
                for (int j = 0; j < player_regrets.cols(); ++j) {
                    if (player_regrets(i, j) > 0.0) {
                        player_regrets(i, j) *= alpha;
                    }else if(player_regrets(i, j) < 0.0){
                        player_regrets(i, j) *= beta;
                    }
                }
            }

            // std::cout <<"Player "<<i<< " after discounted array:" << std::endl;
            // std::cout << player_regrets << std::endl;

            //根据预处理宏 CFR_PLUS ，将 player_strategy 赋值为 player_regrets 
            #ifdef CFR_PLUS
            // These are already maxed
            //player_strategy = player_regrets;
            VoteData positive_regrets = player_regrets.unaryExpr([](double v) { return std::max(v, 0.0); });
            Eigen::ArrayXd strategy_sums = positive_regrets.rowwise().sum();
            for (int i = 0; i < player_regrets.rows(); ++i) {
                if (strategy_sums(i) > 0) {
                    player_strategy.row(i) = positive_regrets.row(i) / strategy_sums(i);
                } else {
                    player_strategy.row(i) = Eigen::ArrayXd::Constant(player_regrets.cols(), 0.5);
                }
            }
            #else
            // These aren't so we have to max them.
            player_strategy = player_regrets.max(0.0);
            #endif

            // std::cout <<"Player "<<i<< " after dcfr array:" << std::endl;
            // std::cout << player_strategy << std::endl;


            //改-更新策略
            //设置阈值
            double evil_threshold = 0.5;
            
            VoteData tmp_strategy = player_strategy;
            int proposal = node->proposal;
            for(int j = 0;j < NUM_VIEWPOINTS; j++){
                std::vector<int> evil_players;
                for (int idx = 0; idx < NUM_PLAYERS; ++idx) {
                    if (node->finding_evil->at(i)(j,idx) > evil_threshold) {
                        evil_players.push_back(idx);
                    }
                }
                for(int m = 0; m < evil_players.size(); m++){
                    if((1 << evil_players[m]) & proposal){
                        tmp_strategy(j,0) *= 0.5;
                    }
                }
            }
            VoteData tmp_holder = tmp_strategy.colwise() / tmp_strategy.rowwise().sum();

            //VoteData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
            
            //处理可能出现的非有限值，确保 tmp_holder 中的每个元素都是有效的概率值（介于0和1之间）。
            //如果某个元素为非有限值，那么它将被替换为0.5
            tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.5; });

            // std::cout <<"Player "<<i<< " after toone array:" << std::endl;
            // std::cout << tmp_holder << std::endl;

            player_strategy = (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * VoteData::Constant(0.5);

            // std::cout <<"Player "<< i << " Vote player_strategy:" << std::endl;
            // std::cout << player_strategy << std::endl;

            if (gamma != 0) {
                node->vote_cum->at(i) += (player_strategy.colwise() * node->reach_probs[i]) * gamma;
            }
        }
    } break;
    case MISSION: {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            if (((1 << i) & node->proposal) == 0) continue;//跳过不执行任务的玩家
            auto& player_regrets = node->mission_regrets->at(i);
            auto& player_strategy = node->mission_strategy->at(i);

            // std::cout <<"Player "<<i<< " mission_regrets array:" << std::endl;
            // std::cout << player_regrets << std::endl;

            // std::cout <<"Player "<<i<< " mission_strategy array:" << std::endl;
            // std::cout << player_strategy << std::endl;

            for (int i = 0; i < player_regrets.rows(); ++i) {
                for (int j = 0; j < player_regrets.cols(); ++j) {
                    if (player_regrets(i, j) > 0.0) {
                        player_regrets(i, j) *= alpha;
                    }else if(player_regrets(i, j) < 0.0){
                        player_regrets(i, j) *= beta;
                    }
                }
            }

            // std::cout <<"Player "<<i<< " after discounted array:" << std::endl;
            // std::cout << player_regrets << std::endl;

            #ifdef CFR_PLUS
            // These are already maxed
            //player_strategy = player_regrets;
            //printf("CFR_PLUS\n");
            MissionData positive_regrets = player_regrets.unaryExpr([](double v) { return std::max(v, 0.0); });
            Eigen::ArrayXd strategy_sums = positive_regrets.rowwise().sum();
            for (int i = 0; i < player_regrets.rows(); ++i) {
                if (strategy_sums(i) > 0) {
                    player_strategy.row(i) = positive_regrets.row(i) / strategy_sums(i);
                } else {
                    player_strategy.row(i) = Eigen::ArrayXd::Constant(player_regrets.cols(), 0.5);
                }
            }
            #else
            // These aren't so we have to max them.
            player_strategy = player_regrets.max(0.0);
            #endif

            // std::cout <<"Player "<<i<< " after cfr array:" << std::endl;
            // std::cout << player_strategy << std::endl;

            MissionData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
            //处理可能出现的非有限值，确保 tmp_holder 中的每个元素都是有效的概率值（介于0和1之间）。
            //如果某个元素为非有限值，那么它将被替换为0.5
            tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.5; });

            // std::cout <<"Player "<<i<< " after toone array:" << std::endl;
            // std::cout << tmp_holder << std::endl;

            player_strategy = (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * MissionData::Constant(0.5);

            // std::cout <<"Player "<< i << " mission player_strategy:" << std::endl;
            // std::cout << player_strategy << std::endl;

            if (gamma != 0) {
                node->mission_cum->at(i) += (player_strategy.colwise() * node->reach_probs[i]) * gamma;
            }
        }
    } break;
    case TERMINAL_MERLIN: {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            auto& player_regrets = node->merlin_regrets->at(i);
            auto& player_strategy = node->merlin_strategy->at(i);

            for (int i = 0; i < player_regrets.rows(); ++i) {
                for (int j = 0; j < player_regrets.cols(); ++j) {
                    if (player_regrets(i, j) > 0.0) {
                        player_regrets(i, j) *= alpha;
                    }else if(player_regrets(i, j) < 0.0){
                        player_regrets(i, j) *= beta;
                    }
                }
            }

            #ifdef CFR_PLUS
            // These are already maxed
            //player_strategy = player_regrets;
            MerlinData positive_regrets = player_regrets.unaryExpr([](double v) { return std::max(v, 0.0); });
            Eigen::ArrayXd strategy_sums = positive_regrets.rowwise().sum();
            for (int i = 0; i < player_regrets.rows(); ++i) {
                if (strategy_sums(i) > 0) {
                    player_strategy.row(i) = positive_regrets.row(i) / strategy_sums(i);
                } else {
                    player_strategy.row(i) = Eigen::ArrayXd::Constant(player_regrets.cols(), 1.0/NUM_PLAYERS);
                }
            }
            #else
            // These aren't so we have to max them.
            player_strategy = player_regrets.max(0.0);
            #endif
            
            //改-更新策略
            MerlinData tmp_strategy;
            MerlinData finding_merlin = node->finding_merlin->at(i);
            tmp_strategy = player_strategy;
            //tmp_strategy = 0.5*finding_merlin + 0.5*player_strategy;
            MerlinData tmp_holder = tmp_strategy.colwise() / tmp_strategy.rowwise().sum();
            
            //MerlinData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
            tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 1.0/NUM_PLAYERS; });
            player_strategy = (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * MerlinData::Constant(1.0/NUM_PLAYERS);

            // std::cout <<"Player "<< i << " merlin player_strategy:" << std::endl;
            // std::cout << player_strategy << std::endl;

            if (gamma != 0) {
                node->merlin_cum->at(i) += (player_strategy.colwise() * node->reach_probs[i]) * gamma;
            }
        }
        // Intentional missing break.
    }
    case TERMINAL_NO_CONSENSUS:
    case TERMINAL_TOO_MANY_FAILS:
    case TERMINAL_PROPOSE_NN: {
        populate_full_reach_probs(node);
    } break;
    default: break;
    }

    fill_reach_probabilities(node);

    for (auto& child : node->children) {
        calculate_strategy(child.get(), alpha, beta,gamma);
    }
}

//计算propose的cfv
static void calculate_propose_cfvs(LookaheadNode* node) {
    for (int player = 0; player < NUM_PLAYERS; player++) {
        for (int proposal = 0; proposal < NUM_PROPOSAL_OPTIONS; proposal++) {
            auto& child = node->children[proposal];
            //如果当前玩家是提议者,将当前子节点的cfv与提议策略矩阵的对应列相乘，并累加到提议者的cfv中
            if (player == node->proposer) {
                // std::cout << "Player " << player << "Proposal " << proposal <<" child->counterfactual_values:" << std::endl;
                // std::cout << child->counterfactual_values[player] << std::endl;
                // std::cout << "Player " << player << "Proposal " << proposal <<" propose_strategy->col(proposal):" << std::endl;
                // std::cout << node->propose_strategy->col(proposal) << std::endl;
                node->counterfactual_values[player] += child->counterfactual_values[player] * node->propose_strategy->col(proposal);
            //如果当前玩家不是提议者，则直接将当前子节点的cfv累加到相应玩家的cfv中
            } else {
                // std::cout << "Player " << player << "Proposal " << proposal <<" child->counterfactual_values:" << std::endl;
                // std::cout << child->counterfactual_values[player] << std::endl;
                node->counterfactual_values[player] += child->counterfactual_values[player];
            }
        }
        // std::cout << "Player " << player << " node->counterfactual_values:" << std::endl;
        // std::cout << node->counterfactual_values[player] << std::endl;

    }


    // Update regrets更新悔值
    for (int proposal = 0; proposal < NUM_PROPOSAL_OPTIONS; proposal++) {
        //计算当前提议选项对应子节点的提议者cfv与当前节点的提议者cfv之差，并将其累加到策略cfv的propose_regrets中
        node->propose_regrets->col(proposal) += node->children[proposal]->counterfactual_values[node->proposer] - node->counterfactual_values[node->proposer];

    }
    //DCFR算法，更新悔值
    // auto& player_regrets = *(node->propose_regrets);
    // for (int i = 0; i < player_regrets.rows(); ++i) {
    //     for (int j = 0; j < player_regrets.cols(); ++j) {
    //         if (player_regrets(i, j) > 0.0) {
    //             player_regrets(i, j) *= alpha;
    //         }else if(player_regrets(i, j) < 0.0){
    //             player_regrets(i, j) *= beta;
    //         }
    //     }
    // }
    // *(node->propose_regrets) = player_regrets;

    //CFR+算法，将regrets小于0的值设为0
    // #ifdef CFR_PLUS
    // *(node->propose_regrets) = node->propose_regrets->max(0.0);
    // #endif

    //改-更新finding_evil数组
    int proposer = node->proposer;
    for (int player = 0; player < NUM_PLAYERS; player++){
        for (int viewpoint = 0; viewpoint < NUM_VIEWPOINTS; viewpoint++){
            double rmax = node->children[0]->counterfactual_values[proposer](viewpoint);
            for (int proposal = 0; proposal < NUM_PROPOSAL_OPTIONS; proposal++){
                if(node->children[proposal]->counterfactual_values[proposer](viewpoint) >= rmax){
                    rmax = node->children[proposal]->counterfactual_values[proposer](viewpoint);
                }
            }
            double sum_dif = 0;
            for (int proposal = 0; proposal < NUM_PROPOSAL_OPTIONS; proposal++){
                sum_dif += rmax - node->children[proposal]->counterfactual_values[proposer](viewpoint);
            }
            for (int proposal = 0; proposal < NUM_PROPOSAL_OPTIONS; proposal++){
                double dif = rmax - node->children[proposal]->counterfactual_values[proposer](viewpoint);
                node ->children[proposal]->finding_evil->at(player)(viewpoint,proposer) = node->finding_evil->at(player)(viewpoint,proposer) * (dif / sum_dif);
            }
        }
    }

}

//计算vote的cfv
static void calculate_vote_cfvs(LookaheadNode* node) {
    for (int player = 0; player < NUM_PLAYERS; player++) {
        VoteData cfvs = VoteData::Constant(0.0);

        for (int vote_pattern = 0; vote_pattern < (1 << NUM_PLAYERS); vote_pattern++) {
            auto& child = node->children[vote_pattern];
            int vote = (vote_pattern >> player) & 1;
            //将当前子节点的cfv累加到对应投票结果的列向量中
            cfvs.col(vote) += child->counterfactual_values[player];
        }

        // std::cout << "Vote Player " << player <<" cfvs:" << std::endl;
        // std::cout << cfvs << std::endl;


        // Update regrets更新悔值
        node->counterfactual_values[player] = (cfvs * node->vote_strategy->at(player)).rowwise().sum();
        // std::cout << "Player " << player <<" node->counterfactual_values:" << std::endl;
        // std::cout << node->counterfactual_values[player] << std::endl;
        node->vote_regrets->at(player) += cfvs.colwise() - node->counterfactual_values[player];

        // auto& player_regrets = node->vote_regrets->at(player);
        // for (int i = 0; i < player_regrets.rows(); ++i) {
        //     for (int j = 0; j < player_regrets.cols(); ++j) {
        //         if (player_regrets(i, j) > 0.0) {
        //             player_regrets(i, j) *= alpha;
        //         }else if(player_regrets(i, j) < 0.0){
        //             player_regrets(i, j) *= beta;
        //         }
        //     }
        // }
        // node->vote_regrets->at(player) = player_regrets;
        // #ifdef CFR_PLUS
        // node->vote_regrets->at(player) = node->vote_regrets->at(player).max(0.0);
        // #endif
    }

    //改-更新finding_evil数组
    //player在viewpoint视角认为player是evil的概率
    for (int player = 0; player < NUM_PLAYERS; player++){
        for (int viewpoint = 0; viewpoint < NUM_VIEWPOINTS; viewpoint++){
            for (int i = 0; i < NUM_PLAYERS; i++){
                double rmax = 0;
                for (int vote_pattern = 0; vote_pattern < (1 << NUM_PLAYERS); vote_pattern++){
                    if(node->children[vote_pattern]->counterfactual_values[i](viewpoint) > rmax){
                        rmax = node->children[vote_pattern]->counterfactual_values[i](viewpoint);
                    }
                }
                double sum_dif = 0;
                for (int vote_pattern = 0; vote_pattern < (1 << NUM_PLAYERS); vote_pattern++){
                    sum_dif += node->children[vote_pattern]->counterfactual_values[i](viewpoint);
                }
                for (int vote_pattern = 0; vote_pattern < (1 << NUM_PLAYERS); vote_pattern++){
                    double dif = rmax - node->children[vote_pattern]->counterfactual_values[i](viewpoint);
                    node->children[vote_pattern]->finding_evil->at(player)(viewpoint,i) = node->finding_evil->at(player)(viewpoint,i) * (dif / sum_dif);
                }
            }
        }
    }

}

static void calculate_mission_cfvs(LookaheadNode* node) {
    // For players not on the mission, the CFVs are just the sum.
    //不执行任务的玩家，CFV就是子节点和
    for (int player = 0; player < NUM_PLAYERS; player++) {
        // Skip players on the mission
        if ((1 << player) & node->proposal) continue;

        for (int num_fails = 0; num_fails < NUM_EVIL + 1; num_fails++) {
            node->counterfactual_values[player] += node->children[num_fails]->counterfactual_values[player];
        }
    }

    // For players on the mission, the CFVs are a little more complicated.
    //执行任务的玩家
    for (int player = 0; player < NUM_PLAYERS; player++) {
        // Skip players not on the mission.
        if (((1 << player) & node->proposal) == 0) continue;

        // For good viewpoints, CFVs are just the sum of the number of possible fails
        for (int viewpoint = 0; viewpoint < NUM_GOOD_VIEWPOINTS; viewpoint++) {
            for (int num_fails = 0; num_fails < NUM_EVIL + 1; num_fails++) {
                node->counterfactual_values[player](viewpoint) += node->children[num_fails]->counterfactual_values[player](viewpoint);
            }   
        }

        // For bad viewpoints, CFVs are split.
        for (int viewpoint = NUM_GOOD_VIEWPOINTS; viewpoint < NUM_VIEWPOINTS; viewpoint++) {
            double pass_cfv = 0.0;
            double fail_cfv = 0.0;
            int partner = VIEWPOINT_TO_BAD[player][viewpoint];
            if ((1 << partner) & node->proposal) {
                // The partner is on the mission.
                pass_cfv = node->children[0]->counterfactual_values[player](viewpoint);
                fail_cfv = node->children[2]->counterfactual_values[player](viewpoint);
                add_middle_cfvs(node, player, viewpoint, partner, &pass_cfv, &fail_cfv);
            } else {
                // The partner is not on the mission. CFVs are "simple" - 0 or 1 fails possible.
                assert(node->children[2]->counterfactual_values[player](viewpoint) == 0.0);
                pass_cfv = node->children[0]->counterfactual_values[player](viewpoint);
                fail_cfv = node->children[1]->counterfactual_values[player](viewpoint);
            }
            //mission更新悔值
            double my_pass_prob = node->mission_strategy->at(player)(viewpoint, 0);
            double result_cfv = pass_cfv * my_pass_prob + fail_cfv * (1.0 - my_pass_prob);
            node->counterfactual_values[player](viewpoint) = result_cfv;
            node->mission_regrets->at(player)(viewpoint, 0) += pass_cfv - result_cfv;
            node->mission_regrets->at(player)(viewpoint, 1) += fail_cfv - result_cfv;
        }
        // std::cout << "Mission Player " << player <<" cfvs:" << std::endl;
        // std::cout << node->counterfactual_values[player] << std::endl;

        //auto& player_regrets = node->mission_regrets->at(player);

        // #ifdef CFR_PLUS
        // node->mission_regrets->at(player) = node->mission_regrets->at(player).max(0.0);
        // #endif
    }

    //改-更新finding_evil
    //player在viewpoint下认为参与mission的玩家是evil的概率
    for (int player = 0; player < NUM_PLAYERS; player++){
        for (int viewpoint = 0; viewpoint < NUM_VIEWPOINTS; viewpoint++){
            for (int i = 0; i < NUM_PLAYERS; i++){
                if (((1 << i) & node->proposal) == 0) continue;
                double rmax = 0;
                for (int num_fails = 0; num_fails < NUM_EVIL + 1; num_fails++){
                    if(node->children[num_fails]->counterfactual_values[i](viewpoint) > rmax){
                        rmax = node->children[num_fails]->counterfactual_values[i](viewpoint);
                    }
                }
                double sum_dif = 0;
                for (int num_fails = 0; num_fails < NUM_EVIL + 1; num_fails++){
                    sum_dif += node->children[num_fails]->counterfactual_values[i](viewpoint);
                }
                for (int num_fails = 0; num_fails < NUM_EVIL + 1; num_fails++){
                    double dif = rmax - node->children[num_fails]->counterfactual_values[i](viewpoint);
                    node->children[num_fails]->finding_evil->at(player)(viewpoint,i) = node->finding_evil->at(player)(viewpoint,i) * (dif / sum_dif);
                }
            }
        }
    }

}

//计算Merlin终端节点的cfv值
static void calculate_merlin_cfvs(LookaheadNode* node, const AssignmentProbs& starting_probs) {
    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        //假设当前分配i=0是梅林、刺客、爪牙、正方、正方
        
        //获取梅林角色的编号 merlin
        //merlin=0
        int merlin = ASSIGNMENT_TO_ROLES[i][0];

        //刺客角色的编号 assassin
        //assassin=1
        int assassin = ASSIGNMENT_TO_ROLES[i][1];

        //刺客角色的观点编号 assassin_viewpoint
        //assassin_viewpoint=8
        int assassin_viewpoint = ASSIGNMENT_TO_VIEWPOINT[i][assassin];

        //反方编号 evil
        //evil=6(00110)
        int evil = ASSIGNMENT_TO_EVIL[i];

        //到达该分配情况的概率 reach_prob，
        //该概率等于节点 node 的完整到达概率（full reach probabilities）乘以起始概率（starting probabilities）中对应分配情况的概率
        double reach_prob = (*(node->full_reach_probs))(i) * starting_probs(i);
        if (reach_prob == 0.0) continue;

        //刺客猜对梅林的概率
        double correct_prob = node->merlin_strategy->at(assassin)(assassin_viewpoint, merlin);

        //根据player是否是反方，计算终端结点的cfv
        for (int player = 0; player < NUM_PLAYERS; player++) {
            int viewpoint = ASSIGNMENT_TO_VIEWPOINT[i][player];
            //假设玩家player在视角viewpoint下已经到达了当前节点，然后计算这种情况下的概率。
            double counterfactual_reach_prob = reach_prob / node->reach_probs[player](viewpoint);
            //如果当前玩家player在当前角色分配i中是evil角色
            if ((1 << player) & evil) {
                node->counterfactual_values[player](viewpoint) += counterfactual_reach_prob * (
                    EVIL_WIN_PAYOFF * correct_prob +
                    EVIL_LOSE_PAYOFF * (1.0 - correct_prob)
                );
            } else {
                node->counterfactual_values[player](viewpoint) += counterfactual_reach_prob * (
                    GOOD_LOSE_PAYOFF * correct_prob +
                    GOOD_WIN_PAYOFF * (1.0 - correct_prob)
                );
            }
            // std::cout << "Merlin Player " << player <<" cfvs:" << std::endl;
            // std::cout << node->counterfactual_values[player] << std::endl;
        }
        
        //更新悔值
        double assassin_counterfactual_reach_prob = reach_prob / node->reach_probs[assassin](assassin_viewpoint);
        double expected_assassin_payoff = assassin_counterfactual_reach_prob * (
            EVIL_WIN_PAYOFF * correct_prob +
            EVIL_LOSE_PAYOFF * (1.0 - correct_prob)
        );
        for (int assassin_choice = 0; assassin_choice < NUM_PLAYERS; assassin_choice++) {
            // double choice_prob = node->merlin_strategy->at(assassin)(assassin_viewpoint, assassin_choice);
            double payoff = (
                (assassin_choice == merlin) ?
                (EVIL_WIN_PAYOFF * assassin_counterfactual_reach_prob) :
                (EVIL_LOSE_PAYOFF * assassin_counterfactual_reach_prob)
            );

            node->merlin_regrets->at(assassin)(assassin_viewpoint, assassin_choice) += payoff - expected_assassin_payoff;
        }


    }
    
    // #ifdef CFR_PLUS
    // for (int player = 0; player < NUM_PLAYERS; player++) {
    //     node->merlin_regrets->at(player) = node->merlin_regrets->at(player).max(0.0);
    // }
    // #endif
};

//计算正方失败终止节点（terminal nodes）的cfv
//针对TERMINAL_NO_CONSENSUS和TERMINAL_TOO_MANY_FAILS两种节点
//这两种节点，都是正方输，反方赢。
static void calculate_terminal_cfvs(LookaheadNode* node, const AssignmentProbs& starting_probs) {
    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        //获得该角色分配i中evil角色信息
        int evil = ASSIGNMENT_TO_EVIL[i];
        //计算在当前角色分配i中到达当前节点的概率
        double reach_prob = (*(node->full_reach_probs))(i) * starting_probs(i);
        // std::cout << "node->full_reach_probs: "<< std::endl;
        // std::cout << (*(node->full_reach_probs))(i) << std::endl;
        // std::cout << "starting_probs: "<< std::endl;
        // std::cout << starting_probs(i) << std::endl;
        // std::cout << "reach_prob: "<< std::endl;
        // std::cout << reach_prob << std::endl;
        //对于每个玩家
        for (int player = 0; player < NUM_PLAYERS; player++) {
            //获得在当前角色分配i中player的信息集视角viewpoint
            int viewpoint = ASSIGNMENT_TO_VIEWPOINT[i][player];
            //计算反事实到达概率
            double counterfactual_reach_prob = reach_prob / node->reach_probs[player](viewpoint);
            // std::cout << "counterfactual_reach_prob: "<< std::endl;
            // std::cout << counterfactual_reach_prob << std::endl;
            //如果当前玩家player在当前角色分配i中是evil角色
            if ((1 << player) & evil) {
                //将玩家player在信息集viewpoint下的cfv值增加  counterfactual_reach_prob*evil获胜回报
                node->counterfactual_values[player](viewpoint) += counterfactual_reach_prob * EVIL_WIN_PAYOFF; // In these terminal nodes, evil wins, good loses.
            } else {//如果是正方角色
                //将玩家player在信息集viewpoint下的cfv值增加  counterfactual_reach_prob*正方输的回报
                node->counterfactual_values[player](viewpoint) += counterfactual_reach_prob * GOOD_LOSE_PAYOFF;
            }
            // std::cout << "Terminal Player " << player <<" cfvs:" << std::endl;
            // std::cout << node->counterfactual_values[player] << std::endl;
        }
    }
}

//计算神经网络模型下的cfv
static void calculate_neural_net_cfvs(LookaheadNode* node, const AssignmentProbs& starting_probs) {
    AssignmentProbs real_probs = (*(node->full_reach_probs)) * starting_probs;
    double sum = real_probs.sum();
    if (sum == 0.0) {
        return;
    }
    //标准化概率
    AssignmentProbs normalized_probs = real_probs / sum;
    //使用神经网络模型 node->nn_model 对提议者（proposer）和标准化概率 normalized_probs 进行预测
    //得到cfv并存储在节点 node 的 counterfactual_values 数组中
    node->nn_model->predict(node->proposer, normalized_probs, node->counterfactual_values);

    //规范化cfv
    for (int i = 0; i < NUM_PLAYERS; i++) {
        // Re-normalize the values so they are counterfactual.
        node->counterfactual_values[i] /= node->reach_probs[i];
        node->counterfactual_values[i] *= sum;
        // std::cout << "net Player " << i <<" cfvs:" << std::endl;
        // std::cout << node->counterfactual_values[i] << std::endl;
    }
};

//计算各种节点的cfv
void calculate_counterfactual_values(LookaheadNode* node, const AssignmentProbs& starting_probs) {
    for (auto& child : node->children) {
        calculate_counterfactual_values(child.get(), starting_probs);
    }

    for (int player = 0; player < NUM_PLAYERS; player++) {
        node->counterfactual_values[player].setZero();
    }

    switch (node->type) {
    case PROPOSE:
        calculate_propose_cfvs(node); break;
    case VOTE:
        calculate_vote_cfvs(node); break;
    case MISSION:
        calculate_mission_cfvs(node); break;
    case TERMINAL_MERLIN:
        calculate_merlin_cfvs(node, starting_probs); break;
    case TERMINAL_NO_CONSENSUS:
    case TERMINAL_TOO_MANY_FAILS:
        calculate_terminal_cfvs(node, starting_probs); break;
    case TERMINAL_PROPOSE_NN:
        calculate_neural_net_cfvs(node, starting_probs); break;
    }

    #ifndef NDEBUG
    double check = 0.0;
    for (int i = 0; i < NUM_PLAYERS; i++) {
        check += (node->counterfactual_values[i] * node->reach_probs[i]).sum();
    }
    if (abs(check) > 1e-5) {
        std::cout << "NODE TYPE: " << node->typeAsString() << std::endl;
        std::cout << "MY SUM: " << check << std::endl;
        for (const auto& child : node->children) {
            double s = 0.0;
            for (int i = 0; i < NUM_PLAYERS; i++) s += (child->counterfactual_values[i] * child->reach_probs[i]).sum();
            std::cout << "CHILD SUM: " << s << std::endl;
        }
    }
    ASSERT(abs(check), <, 1e-5);
    assert(abs(check) < 1e-5);
    #endif
}

//执行计算对cfv的CFR算法
void cfr_get_values(
    LookaheadNode* root,
    const int iterations,
    const int wait_iterations,
    const AssignmentProbs& starting_probs,
    const bool save_strategy,
    ViewpointVector* values
) {

    const double alpha =3;
    const double beta=0;
    const double gamma=2;

    //初始化 values 和 last_values 数组为零向量
    ViewpointVector last_values[NUM_PLAYERS];
    for (int i = 0; i < NUM_PLAYERS; i++) {
        values[i].setZero();
        last_values[i].setZero();
    }

    //累计权重
    double total_weight = 0;
    int consecutive_low_change_count = 0;
    //long long total_size = 0;

    //使用循环迭代 iterations 次进行CFR算法的计算
    for (int iter = 0; iter < iterations; iter++) {
        double discount_strategy = pow((double)iter / (iter + 1), gamma);
        //int weight = (iter < wait_iterations) ? 0 : (iter - wait_iterations);

        double discount_alpha = pow(iter, alpha);
        discount_alpha = discount_alpha / (discount_alpha + 1);

        double discount_beta = pow(iter, beta);
        discount_beta = discount_beta / (discount_beta + 1);
        //discount_beta = 0;

        //调用 calculate_strategy 函数计算策略
        calculate_strategy(root, discount_alpha, discount_beta, save_strategy ? discount_strategy : 0.0);
        //calculate_strategy(root, discount_alpha, discount_beta, save_strategy ? weight : 0.0);
        //

        //调用 calculate_counterfactual_values 函数计算cfv
        calculate_counterfactual_values(root, starting_probs);

        //bool equals_previous_iteration = !save_strategy; // Save strategy disables checking for early finish
        bool equals_previous_iteration = true;
        
        //检查当前迭代的cfv是否与上一次迭代的cfv相等。
        //如果所有玩家的cfv都近似相等，则表示算法已收敛，将当前cfv存储在 values 数组中，并返回结果
        for (int i = 0; i < NUM_PLAYERS; i++) {
            if (!last_values[i].isApprox(root->counterfactual_values[i])) {
                equals_previous_iteration = false;
            }else{
                consecutive_low_change_count++;
            }
        }
        //std::cout << "Iteration " << iter <<" consecutive_low_change_count "<< consecutive_low_change_count << std::endl;
        if (equals_previous_iteration && consecutive_low_change_count > 1000) {
            for (int i = 0; i < NUM_PLAYERS; i++) {
                values[i] = root->counterfactual_values[i];
            }
            //std::cout << "Convergence achieved at iteration " << iter << std::endl;
            return;
        }


        // 使用折扣策略权重来更新total_weight
        total_weight += discount_strategy;
        //total_size += weight;

        //追踪差值
        double strategy_change = 0.0;
        for (int i = 0; i < NUM_PLAYERS; i++) {
            // std::cout << "Player:" << i <<" last values:"<< std::endl;
            // std::cout << last_values[i] << std::endl;
            // std::cout << "Player:" << i <<" values:"<< std::endl;
            // std::cout << root->counterfactual_values[i] << std::endl;
            // bool flag = last_values[i].isApprox(root->counterfactual_values[i]);
            // std::cout << "Player:" << i <<" isApprox:"<< std::endl;
            // std::cout << flag << std::endl;
            double norm = (last_values[i] - root->counterfactual_values[i]).matrix().squaredNorm() / NUM_VIEWPOINTS;
            strategy_change += norm;
        }
        //std::cout << "Iteration " << iter <<": Weight "<< total_weight << "; Strategy Change = " << strategy_change << std::endl;
        //std::cout << "Iteration " << iter <<": Weight "<< total_size << "; Strategy Change = " << strategy_change << std::endl;

        for (int player = 0; player < NUM_PLAYERS; player++) {
            last_values[player] = root->counterfactual_values[player];
            values[player] += root->counterfactual_values[player] * discount_strategy;
            //values[player] += root->counterfactual_values[player] * weight;
        }

    }
    //对 values 数组中的每个元素除以 total_size，以计算平均cfv
    for (int i = 0; i < NUM_PLAYERS; i++) {
        values[i] /= total_weight;
        //values[i] /= total_size;
    }
}

//计算累积策略
void calculate_cumulative_strategy(LookaheadNode* node) {
    switch (node->type) {
    case PROPOSE: {
        // Initialize the node's memory
        auto& player_cumulative = *(node->propose_cum);
        auto& player_strategy = *(node->propose_strategy);

        player_strategy = player_cumulative;

        ProposeData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
        tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 1.0/NUM_PROPOSAL_OPTIONS; });
        player_strategy = tmp_holder; // (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * ProposeData::Constant(1.0/NUM_PROPOSAL_OPTIONS);
    } break;
    case VOTE: {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            auto& player_cumulative = node->vote_cum->at(i);
            auto& player_strategy = node->vote_strategy->at(i);

            player_strategy = player_cumulative;

            VoteData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
            tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.5; });
            player_strategy = tmp_holder; // (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * VoteData::Constant(0.5);
        }
    } break;
    case MISSION: {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            if (((1 << i) & node->proposal) == 0) continue;
            auto& player_cumulative = node->mission_cum->at(i);
            auto& player_strategy = node->mission_strategy->at(i);

            player_strategy = player_cumulative;

            MissionData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
            tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.5; });
            player_strategy = tmp_holder; // (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * MissionData::Constant(0.5);
        }
    } break;
    case TERMINAL_MERLIN: {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            auto& player_cumulative = node->merlin_cum->at(i);
            auto& player_strategy = node->merlin_strategy->at(i);

            player_strategy = player_cumulative;

            MerlinData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
            tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 1.0/NUM_PLAYERS; });
            player_strategy = tmp_holder; // (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * MerlinData::Constant(1.0/NUM_PLAYERS);
        }
        // Intentional missing break.
    }
    case TERMINAL_NO_CONSENSUS:
    case TERMINAL_TOO_MANY_FAILS:
    case TERMINAL_PROPOSE_NN: {
        populate_full_reach_probs(node);
    } break;
    default: break;
    }

    fill_reach_probabilities(node);

    for (auto& child : node->children) {
        calculate_cumulative_strategy(child.get());
    }
}
