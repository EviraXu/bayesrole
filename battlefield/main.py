from battlefield.bots import (
    RandomBot, RandomBotUV,
    SimpleBot,
    ObserveBot, ExamineAgreementBot,
    ISMCTSBot, MOISMCTSBot,
    HumanBot, HumanLikeBot,
    NNBot, NNBotWithObservePropose,
    SimpleStatsBot,
    SingleMCTSPlayoutBot, SingleMCTSHeuristicBot, SingleMCTSBaseOpponentBot,
    CFRBot,
    LearningBot,
    ObserveBeaterBot,
    DeepBayes,
    Deeprole,
    Deeprole_assignment,
    Deeprole_3_1,
    Deeprole_10_5,
    Deeprole_30_15,
    Deeprole_100_50,
    Deeprole_300_150,
    Deeprole_100_0,
    Deeprole_100_25,
    Deeprole_100_75,
    Deeprole_100_100,
    Deeprole_NoZeroing,
    Deeprole_OldNNs,
    Deeprole_NoZeroUnconstrained,
    Deeprole_ZeroingUnconstrained,
    Deeprole_ZeroingWinProbs,
)
from battlefield.tournament import (
    run_simple_tournament,
    run_large_tournament,
    run_all_combos_parallel,
    run_all_combos_simple,
    run_all_combos,
    print_tournament_statistics,
    check_config,
    run_learning_tournament,
    run_single_threaded_tournament,
    run_and_print_game,
    run_and_collect_information,
    prepare_data_for_lstm,
    run_game_and_create_bots
)
from battlefield.compare_to_human import compute_human_statistics, print_human_statistics
from battlefield.predict_roles import predict_evil_over_human_data, predict_evil_using_voting, grid_search
from battlefield.determine_reachable_states import determine_reachable
from battlefield.subgame import calculate_subgame_ll, test_calculate
from battlefield.predict_merlin import get_merlin_prediction
from battlefield.merlin import recover_all_deeprole_thoughts
from battlefield.avalon_types import ASSIGNMENT_TO_ROLES
import multiprocessing
import pandas as pd
import random
import sys
import gzip
import pickle
import itertools
import os
import datetime
from tqdm import tqdm
import logging
import time
from collections import defaultdict

TOURNAMENT_CONFIG = [
    {
        'bot': Deeprole_assignment,
        'role': 'merlin'
    },
    {
        'bot': Deeprole_assignment,
        'role': 'minion'
    },
    {
        'bot': Deeprole_assignment,
        'role': 'servant'
    },
    {
        'bot': Deeprole_assignment,
        'role': 'servant'
    },
    {
        'bot': Deeprole_assignment,
        'role': 'assassin'
    }
]

INFO_CONFIG = [
    {
        'bot': Deeprole,
        'role': 'merlin'
    },
    {
        'bot': Deeprole,
        'role': 'minion'
    },
    {
        'bot': Deeprole,
        'role': 'servant'
    },
    {
        'bot': Deeprole,
        'role': 'servant'
    },
    {
        'bot': Deeprole,
        'role': 'assassin'
    }
]

def tournament():
    check_config(TOURNAMENT_CONFIG)
    #print("hidden", tuple(x['role'] for x in TOURNAMENT_CONFIG))
    #tournament_results = run_single_threaded_tournament(TOURNAMENT_CONFIG, num_games=2000, granularity=200)
    run_single_threaded_tournament(TOURNAMENT_CONFIG, num_games=1, granularity=1)
    #print_tournament_statistics(tournament_results)

#将多个DataFrame结果合并，并写入压缩的.msg.gz文件中
def join_and_write(results):
    result_by_bot = defaultdict(lambda: [])
    for result in results:
        bot_name = result.loc[0].bot
        result_by_bot[bot_name].append(result)

    for bot, results in result_by_bot.items():
        print("Saving", bot)
        with gzip.open('human/{}.msg.gz'.format(bot), 'w') as f:
            data = pd.concat(results)
            data.reset_index(drop=True, inplace=True)
            data.to_msgpack(f)

#并行执行compute_human_statistics函数，可能用于比较机器人和人类玩家的表现
def parallel_human_compare():
    pool = multiprocessing.Pool()
    try:
        results = []
        for bot in [CFRBot(4000000)]:
            for tremble in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]:
                results.append(pool.apply_async(compute_human_statistics, (bot, tremble, False, 5)))

        print("Waiting for results")
        sys.stdout.flush()
        results = [ result.get() for result in results ]
        print("Concatenating and writing results")
        sys.stdout.flush()
        join_and_write(results)
    except KeyboardInterrupt:
        print('terminating early')
        pool.terminate()


STRING_TO_BOT = {
    "RandomBot": RandomBot,
    "RandomBotUV": RandomBotUV,
    "SimpleBot": SimpleBot,
    "ObserveBot": ObserveBot,
    "SimpleStatsBot": SimpleStatsBot,
    "CFRBot_3000000": CFRBot(3000000),
    "CFRBot_6000000": CFRBot(6000000),
    "CFRBot_10000000": CFRBot(10000000),
    "ISMCTSBot": ISMCTSBot,
    "MOISMCTSBot": MOISMCTSBot,
    "DeepBayes":DeepBayes,
    "Deeprole": Deeprole,
    "Deeprole_assignment":Deeprole_assignment,
    "Deeprole_3_1": Deeprole_3_1,
    "Deeprole_10_5": Deeprole_10_5,
    "Deeprole_30_15": Deeprole_30_15,
    "Deeprole_100_50": Deeprole_100_50,
    "Deeprole_300_150": Deeprole_300_150,
    "Deeprole_100_0": Deeprole_100_0,
    "Deeprole_100_25": Deeprole_100_25,
    "Deeprole_100_75": Deeprole_100_75,
    "Deeprole_100_100": Deeprole_100_100,
    "Deeprole_OldNNs": Deeprole_OldNNs,
    "Deeprole_NoZeroing": Deeprole_NoZeroing,
    "Deeprole_NoZeroUnconstrained": Deeprole_NoZeroUnconstrained,
    "Deeprole_ZeroingUnconstrained": Deeprole_ZeroingUnconstrained,
    "Deeprole_ZeroingWinProbs": Deeprole_ZeroingWinProbs,
}

#比较机器人和人类玩家的表现
def human_compare():
    assert len(sys.argv) == 4, "Need 3 arguments to collect human stats"
    bot = STRING_TO_BOT[sys.argv[1]]
    min_game_id = int(sys.argv[2])
    max_game_id = int(sys.argv[3])
    stats = compute_human_statistics(
        bot,
        verbose=True,
        num_players=5,
        max_num_players=None,
        min_game_id=min_game_id,
        max_game_id=max_game_id,
        roles=set(['servant', 'merlin', 'minion', 'assassin'])
    )
    with gzip.open('human/{}-{}-{}.msg.gz'.format(bot.__name__, min_game_id, max_game_id), 'w') as f:
        stats.to_msgpack(f)

#预测角色
def predict_roles(bot, tremble):
    bot = STRING_TO_BOT[bot]
    tremble = float(tremble)
    dataframe, particles = predict_evil_over_human_data(bot, tremble)
    print("Writing dataframe...")
    with gzip.open('predict_roles/{}_{}_df.msg.gz'.format(bot.__name__, tremble), 'w') as f:
        dataframe.to_msgpack(f)

    print("Writing particles...")
    with gzip.open('predict_roles/{}_{}_particles.pkl.gz'.format(bot.__name__, tremble), 'w') as f:
        pickle.dump(particles, f)

#预测梅林角色
def predict_merlin():
    assert len(sys.argv) == 4, "Need 3 arguments to collect human stats"
    bot = STRING_TO_BOT[sys.argv[1]]
    min_game_id = int(sys.argv[2])
    max_game_id = int(sys.argv[3])

    result = get_merlin_prediction(bot, num_players=5, min_game_id=min_game_id, max_game_id=max_game_id)

    with gzip.open('merlin/{}-{}-{}.msg.gz'.format(bot.__name__, min_game_id, max_game_id), 'w') as f:
        result.to_msgpack(f)

#用于运行比赛，评估不同机器人策略的表现
def benchmark_performance_simple():
    games_per_matching = int(sys.argv[1])
    bot_names = sys.argv[2:]
    bot_classes = [ STRING_TO_BOT[name] for name in bot_names ]
    roles = ['merlin', 'servant', 'assassin', 'minion', 'servant']
    run_all_combos_simple(bot_classes, roles, games_per_matching=games_per_matching)

#使用40个并行进程来加速比赛的运行
def benchmark_performance():
    print("Launching...")
    games_per_matching = int(sys.argv[1])
    bot_names = sys.argv[2:]
    bot_classes = [ STRING_TO_BOT[name] for name in bot_names ]
    roles = ['merlin', 'servant', 'assassin', 'minion', 'servant']
    run_all_combos(bot_classes, roles, games_per_matching=games_per_matching, parallelization=40)

#运行比赛对局收集对局信息（状态）
def collect_information():
    check_config(INFO_CONFIG)
    print("hidden", tuple(x['role'] for x in INFO_CONFIG))
    #tournament_results = run_single_threaded_tournament(TOURNAMENT_CONFIG, num_games=2000, granularity=200)
    tournament_results = run_single_threaded_tournament(INFO_CONFIG, num_games=1, granularity=1)
    print_tournament_statistics(tournament_results)

def setup_logging():
    # 创建日志器
    logger = logging.getLogger('AssignmentLogger')
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO

    # 创建文件处理器，指定日志文件名和写入模式
    file_handler = logging.FileHandler('assignment_processing.log', mode='a')  # 追加模式
    file_handler.setLevel(logging.INFO)

    # 创建日志格式器，并添加到文件处理器中
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将文件处理器添加到日志器中
    logger.addHandler(file_handler)
    return logger

def process_assignment(args):
    assignment, index, base_directory, num_games = args
    # 设置目录和文件名
    directory_name = f"{base_directory}/assignment_{index+1}"
    os.makedirs(directory_name, exist_ok=True)

    # 设置局部 Logger
    logger = logging.getLogger(f'Assignment_{index+1}')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f'{directory_name}/processing.log', mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    start_time = time.time()  # 开始计时

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{directory_name}/assignment_{index+1}_game_{num_games}_{timestamp}.csv"
    for game in range(num_games):
        config = [{'bot': Deeprole, 'role': role} for role in assignment]
        history_file = f"{directory_name}/game_history_{game+1}.json"
        
        # 运行游戏收集信息和处理数据
        run_and_collect_information(config, history_file)
        prepare_data_for_lstm(history_file, filename)
        os.remove(history_file)  # 删除临时的JSON文件

    elapsed_time = time.time() - start_time  # 计算经过时间
    logger.info(f"Processing for assignment {index+1} took {elapsed_time:.2f} seconds.")  # 记录时间到日志
    logger.removeHandler(fh)
    fh.close()

if __name__ == "__main__":
    #简单比较不同bot在相同环境下的胜率等
    #修改TOURNAMENT_CONFIG来修改不同角色对应的bot
    #python3 main.py
    #tournament()


    #打印出一个随机身份分配的游戏过程并打印胜率
    #python3 main.py
    
    # #运行对局收集训练信息
    # roles = ['merlin', 'assassin', 'minion', 'servant', 'servant']
    # # 生成所有唯一的角色分配
    # unique_assignments = list(itertools.permutations(roles))
    # unique_assignments = list(set(unique_assignments))  # 去除重复项，虽然permutations已保证唯一性

    # # 选取前60个作为我们的角色分配配置
    # assignments = unique_assignments[:60]
    # #每个配置运行1k次
    # num_games_per_assignment = 1000
    # base_directory = "/root/DeepRole-master/battlefield/collect_information"
    # os.makedirs(base_directory, exist_ok=True)  # 确保基目录存在
    # args = [(assignment, i, base_directory, num_games_per_assignment) for i, assignment in enumerate(assignments)]
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     pool.map(process_assignment, args)


    #运行一次对局查看belief信息
    #tournament()
        
    

    #使用并行进程加速游戏计算过程
    #python3 main.py 50 Deeprole RandomBot
    benchmark_performance()

    #收集对局信息
    #collect_information()
    
    
    # bots = [ ObserveBeaterBot, ObserveBot, ObserveBot, ObserveBot, ObserveBot ]
    # run_learning_tournament(bots, winrate_track=0)
    # grid_search()
    # predict_evil_using_voting()
    # import sys
    # modulenames = set(sys.modules) & set(globals())
    # print [sys.modules[name] for name in modulenames]
    
    # recover_all_deeprole_thoughts()
    # predict_merlin()
    # human_compare()
    # determine_reachable(RandomBot, set(['merlin', 'minion', 'assassin', 'servant']), 5)
    # test_calculate()
    # df, _ = predict_evil_over_human_data(HumanLikeBot, 0.01)
    # import IPython; IPython.embed()
