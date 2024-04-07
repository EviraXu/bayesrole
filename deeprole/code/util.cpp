#include <vector>
#include <algorithm>
#include <sstream>
#include <map>

#include "util.h"
#include "cfr_plus.h"

std::mt19937 rng;

//负责初始化 rng（随机数生成器），用于在整个程序中生成随机数
void seed_rng() {
    std::random_device rd;
    std::array<int, std::mt19937::state_size> seed_data;
    std::generate_n(seed_data.data(), seed_data.size(), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    rng = std::mt19937(seq);
}

//将结构体的各个成员变量转换成一个逗号分隔值（CSV）格式的字符串，用于输出或记录。
std::string Initialization::Stringify() const {
    //定义 CSV 格式(十位数的浮点精度、不对齐列、使用逗号作为分隔符)
    static Eigen::IOFormat CSVFmt(10, Eigen::DontAlignCols, ",", ",", "", "", "", "");
    std::stringstream stream;
    // Add starting data
    stream << depth << ",";
    stream << num_succeeds << ",";
    stream << num_fails << ",";
    stream << propose_count << ",";
    stream << proposer << ",";
    stream << iterations << ",";
    stream << wait_iterations << ",";
    stream << generate_start_technique << ",";
    stream << starting_probs.format(CSVFmt) << ",";
    for (int i = 0; i < NUM_PLAYERS; i++) {
        stream << solution_values[i].format(CSVFmt) << ((i < NUM_PLAYERS - 1) ? "," : "");
    }
    return stream.str();
}

//递归函数用于在 probs 数组中填充随机概率值，这些值加起来等于 amt_left（剩余的概率总和）
void fill_random_probability(const double amt_left, const int length, double* probs) {
    if (length == 1) {
        *probs = amt_left;
        return;
    }

    std::uniform_real_distribution<> dis(0.0, amt_left);
    double part_a = dis(rng);
    double part_b = amt_left - part_a;

    fill_random_probability(part_a, length/2, probs);
    fill_random_probability(part_b, length - (length/2), probs + (length/2));
}

//生成 Merlin的概率分布
void generate_merlin_probs(double* merlin_probs) {
    double tmp_merlin[NUM_PLAYERS];
    std::vector<int> player_to_index;
    for (int i = 0; i < NUM_PLAYERS; i++) {
        player_to_index.push_back(i);
    }
    //使用一个随机数生成器 rng 来随机打乱 player_to_index 向量中的元素。这样可以确保梅林角色的分配是随机的。
    std::shuffle(player_to_index.begin(), player_to_index.end(), rng);
    fill_random_probability(1.0, NUM_PLAYERS, tmp_merlin);

    for (int i = 0; i < NUM_PLAYERS; i++) {
        merlin_probs[player_to_index[i]] = tmp_merlin[i];
    }
}

//根据成功和失败的轮数，函数返回一组随机失败的轮次
std::vector<int> get_random_fail_rounds(const int num_succeeds, const int num_fails) {
    std::vector<int> result;
    for (int i = 0; i < (num_succeeds + num_fails); i++) {
        result.push_back(ROUND_TO_PROPOSE_SIZE[i]);
    }
    std::shuffle(result.begin(), result.end(), rng);
    result.resize(num_fails);
    return result;
}

//为游戏中的邪恶角色生成所有可能的分布组合。它使用轮数和失败的数量来决定哪些组合是可能的，并返回这些组合的列表。
//返回的数据evil_possibilities=(00011,00101,00110......)代表有可能是反方的玩家
std::vector<uint32_t> get_random_evil_possibilities(const int num_succeeds, const int num_fails) {
    std::vector<int> fail_rounds = get_random_fail_rounds(num_succeeds, num_fails);

    //随机选择选择任务失败的玩家数量（1或2）
    std::uniform_int_distribution<> get_num_fails(1, 2);
    //随机选择一个proposal
    std::uniform_int_distribution<> get_proposal(0, NUM_PROPOSAL_OPTIONS - 1);

    //获得之前失败轮数对应的随机数(proposal, num_fails)
    std::vector<std::pair<uint32_t, int>> fails;
    for (int round_size : fail_rounds) {
        int num_fails = get_num_fails(rng);
        const int* index_to_proposal = (round_size == 2) ? INDEX_TO_PROPOSAL_2 : INDEX_TO_PROPOSAL_3;
        uint32_t proposal = index_to_proposal[get_proposal(rng)];
        fails.push_back(std::make_pair(proposal, num_fails));
    }

    std::vector<uint32_t> evil_possibilities;

    for (int i = 0; i < NUM_PROPOSAL_OPTIONS; i++) {
        uint32_t evil = INDEX_TO_PROPOSAL_2[i];
        bool valid = true;

        for (auto fail : fails) {
            if (__builtin_popcount(fail.first & evil) < fail.second) {
                valid = false;
                break;
            }
        }

        if (valid) {
            evil_possibilities.push_back(evil);
        }
    }

    if (evil_possibilities.size() == 0) {
        // If it's impossible, try again
        return get_random_evil_possibilities(num_succeeds, num_fails);
    } else {
        return evil_possibilities;
    }
}

//生成并返回一个 std::map，这个映射表中的键为玩家对（两个玩家的索引构成的 std::pair），值为这对玩家被指定为恶人（evil）的概率
//它使用了 get_random_evil_possibilities 函数的输出，并对结果进行随机分配，产生一组邪恶角色的可能分布和对应的概率。
std::map<std::pair<int, int>, double> generate_evil_probs(const int num_succeeds, const int num_fails) {
    //evil_possibilities=(00011,00101,00110......)代表有可能是反方的玩家
    auto evil_possibilities = get_random_evil_possibilities(num_succeeds, num_fails);

    //使用随机数生成器（rng）随机打乱这些组合，并使用 fill_random_probability 函数为每个可能的反方组合分配一个概率
    std::shuffle(evil_possibilities.begin(), evil_possibilities.end(), rng);
    double evil_probs[evil_possibilities.size()];
    fill_random_probability(1.0, evil_possibilities.size(), evil_probs);

    //返回值，存储每对玩家成为邪恶角色的概率
    //<1,2>,0.5  代表玩家1,2是邪恶角色的概率为0.5
    std::map<std::pair<int, int>, double> result;

    std::uniform_real_distribution<> get_split(0.0, 1.0);

    for (size_t i = 0; i < evil_possibilities.size(); i++) {
        uint32_t evil = evil_possibilities[i];
        double prob = evil_probs[i];
        int player_1 = __builtin_ctz(evil);
        evil &= ~(1 << player_1);
        int player_2 = __builtin_ctz(evil);

        double split_prob = get_split(rng);

        result[std::make_pair(player_1, player_2)] = prob * split_prob;
        result[std::make_pair(player_2, player_1)] = prob * (1.0 - split_prob);
    }

    return result;
}

//生成特定状态下游戏开始时各个角色分配的概率分布
void generate_starting_probs_v1(AssignmentProbs& starting_probs, const int num_succeeds, const int num_fails) {
    //随机生成玩家是梅林的概率
    //举例merlin_probs[5]=(0.1,0.3,0.2,0.2,0.2)
    double merlin_probs[NUM_PLAYERS];
    generate_merlin_probs(merlin_probs);

    //随机生成在指定当前成功轮次和失败轮次下，可能的evil组合及其概率
    //例如evil_probs = (<1,2>,0.5)
    auto evil_probs = generate_evil_probs(num_succeeds, num_fails);

    starting_probs.setZero();
    starting_probs = AssignmentProbs::Constant(1.0/NUM_ASSIGNMENTS);

    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        //指出每个分配下merlin,assassin和minion的玩家序号
        int merlin = ASSIGNMENT_TO_ROLES[i][0];
        int assassin = ASSIGNMENT_TO_ROLES[i][1];
        int minion = ASSIGNMENT_TO_ROLES[i][2];
        //根据三个特殊角色，计算当前分配的概率
        starting_probs(i) = merlin_probs[merlin] * evil_probs[std::make_pair(assassin, minion)];
    }
    double sum = starting_probs.sum();

    if (sum == 0.0) {
        // Try again, something went wrong.
        generate_starting_probs_v1(starting_probs, num_succeeds, num_fails);
        return;
    } else {
        starting_probs /= sum;
    }
}

//准备初始化过程，给 Initialization 结构体的实例设置初始值。
void prepare_initialization(
    const int depth,
    const int num_succeeds,
    const int num_fails,
    const int propose_count,
    Initialization* init
) {
    std::uniform_int_distribution<> dis(0, NUM_PLAYERS - 1);

    init->depth = depth;
    init->num_succeeds = num_succeeds;
    init->num_fails = num_fails;
    init->proposer = dis(rng);//生成随机proposer
    init->propose_count = propose_count;
    init->generate_start_technique = "v1";
    generate_starting_probs_v1(init->starting_probs, num_succeeds, num_fails);
}

//使用反向遗憾最小化（Counterfactual Regret Minimization, CFR）算法运行初始化。
void run_initialization_with_cfr(
    const int iterations,
    const int wait_iterations,
    const std::string& model_search_dir,
    Initialization* init
) {
    init->iterations = iterations;
    init->wait_iterations = wait_iterations;

    auto lookahead = create_avalon_lookahead(
        init->num_succeeds,
        init->num_fails,
        init->proposer,
        init->propose_count,
        init->depth,
        model_search_dir
    );

    AssignmentProbs& starting_probs = init->starting_probs;
    ViewpointVector* out_values = init->solution_values;
    cfr_get_values(lookahead.get(), iterations, wait_iterations, starting_probs, true, out_values);
}
