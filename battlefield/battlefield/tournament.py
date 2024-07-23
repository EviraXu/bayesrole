import itertools
import pandas as pd
import multiprocessing
import gzip
import random
import os
import json
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences

from battlefield.avalon_types import GOOD_ROLES, EVIL_ROLES, possible_hidden_states, starting_hidden_states, ASSIGNMENT_TO_ROLES
from battlefield.avalon import AvalonState
import pickle


def run_game(state, hidden_state, bots):
    while not state.is_terminal():
        #获取当前需要行动的玩家列表
        moving_players = state.moving_players()
        #获取当前状态和该玩家在当前状态下的合法行动列表。
        moves = [
            bots[player].get_action(state, state.legal_actions(player, hidden_state))
            for player in moving_players
        ]
        new_state, _, observation = state.transition(moves, hidden_state)
        for player, bot in enumerate(bots):
            if player in moving_players:
                move = moves[moving_players.index(player)]
            else:
                move = None
            bot.handle_transition(state, new_state, observation, move=move)
            #print("-----------transition finish----------------")
        state = new_state

    return state.terminal_value(hidden_state), state.game_end



def run_large_tournament(bots_classes, roles, games_per_matching=50):
    print("Running {}".format(' '.join(map(lambda c: c.__name__, bots_classes))))

    start_state = AvalonState.start_state(len(roles))
    result = []
    all_hidden_states = possible_hidden_states(set(roles), num_players=len(roles))

    seen_hidden_states = set([])
    for hidden_state in itertools.permutations(roles):
        if hidden_state in seen_hidden_states:
            continue
        seen_hidden_states.add(hidden_state)

        beliefs = [
            starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(hidden_state))
        ]
        seen_bot_orders = set([])
        for bot_order in itertools.permutations(bots_classes):
            bot_order_str = tuple([bot_cls.__name__ for bot_cls in bot_order])
            if bot_order_str in seen_bot_orders:
                continue
            seen_bot_orders.add(bot_order_str)

            for _ in range(games_per_matching):
                bots = [
                    bot_cls.create_and_reset(start_state, player, role, beliefs[player])
                    for player, (bot_cls, role) in enumerate(zip(bot_order, hidden_state))
                ]
                values, game_end =  (start_state, hidden_state, bots)
                game_stat = {
                    'winner': game_end[0],
                    'win_type': game_end[1],
                }
                for player, (bot_cls, role) in enumerate(zip(bot_order, hidden_state)):
                    game_stat['bot_{}'.format(player)] = bot_cls.__name__
                    game_stat['bot_{}_role'.format(player)] = role
                    game_stat['bot_{}_payoff'.format(player)] = values[player]
                result.append(game_stat)

    df = pd.DataFrame(result, columns=sorted(result[0].keys()))
    df['winner'] = df['winner'].astype('category')
    df['win_type'] = df['win_type'].astype('category')
    for player in range(len(roles)):
        df['bot_{}'.format(player)] = df['bot_{}'.format(player)].astype('category')
        df['bot_{}_role'.format(player)] = df['bot_{}_role'.format(player)].astype('category')

    return df


def large_tournament_parallel_helper(bot_order, hidden_state, beliefs, start_state):
    bots = [
        bot_cls.create_and_reset(start_state, player, role, beliefs[player])
        for player, (bot_cls, role) in enumerate(zip(bot_order, hidden_state))
    ]
    values, game_end = run_game(start_state, hidden_state, bots)
    game_stat = {
        'winner': game_end[0],
        'win_type': game_end[1],
    }
    for player, (bot_cls, role) in enumerate(zip(bot_order, hidden_state)):
        game_stat['bot_{}'.format(player)] = bot_cls.__name__
        game_stat['bot_{}_role'.format(player)] = role
        game_stat['bot_{}_payoff'.format(player)] = values[player]

    return game_stat


def run_large_tournament_parallel(pool, bots_classes, roles, games_per_matching=50):
    print("Running {}".format(' '.join(map(lambda c: c.__name__, bots_classes))))

    start_state = AvalonState.start_state(len(roles))
    async_results = []
    all_hidden_states = possible_hidden_states(set(roles), num_players=len(roles))

    seen_hidden_states = set([])
    for hidden_state in itertools.permutations(roles):
        if hidden_state in seen_hidden_states:
            continue
        seen_hidden_states.add(hidden_state)

        beliefs = [
            starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(hidden_state))
        ]
        seen_bot_orders = set([])
        for bot_order in itertools.permutations(bots_classes):
            bot_order_str = tuple([bot_cls.__name__ for bot_cls in bot_order])
            if bot_order_str in seen_bot_orders:
                continue
            seen_bot_orders.add(bot_order_str)

            for _ in range(games_per_matching):
                async_result = pool.apply_async(large_tournament_parallel_helper, (bot_order, hidden_state, beliefs, start_state))
                async_results.append(async_result)
                
    result = [ async_result.get() for async_result in async_results ]

    df = pd.DataFrame(result, columns=sorted(result[0].keys()))
    df['winner'] = df['winner'].astype('category')
    df['win_type'] = df['win_type'].astype('category')
    for player in range(len(roles)):
        df['bot_{}'.format(player)] = df['bot_{}'.format(player)].astype('category')
        df['bot_{}_role'.format(player)] = df['bot_{}_role'.format(player)].astype('category')

    return df


def run_game_and_create_bots(hidden_state, beliefs, config):
    start_state = AvalonState.start_state(len(hidden_state))
    bots = [ bot['bot']() for bot in config ]
    for player, (bot, config) in enumerate(zip(bots, config)):
        bot.reset(start_state, player, config['role'], beliefs[player])
    return run_game(start_state, hidden_state, bots)


def run_simple_tournament(config, num_games=1000, granularity=100):
    tournament_statistics = {
        'bots': [
            { 'bot': bot['bot'].__name__, 'role': bot['role'], 'wins': 0, 'total': 0, 'win_percent': 0, 'payoff': 0.0 }
            for bot in config
        ],
        'end_types': {}
    }


    hidden_state = tuple([bot['role'] for bot in config])
    all_hidden_states = possible_hidden_states(set(hidden_state), num_players=len(config))
    beliefs = [
        starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(config))
    ]

    pool = multiprocessing.Pool(4)
    results = []

    for i in range(num_games):
        results.append(pool.apply_async(run_game_and_create_bots, (hidden_state, beliefs, config)))

    for i, result in enumerate(results):
        if i % granularity == 0:
            print("Waiting for game {}".format(i))
        payoffs, end_type = result.get()
        tournament_statistics['end_types'][end_type] = 1 + tournament_statistics['end_types'].get(end_type, 0)

        for b, payoff in zip(tournament_statistics['bots'], payoffs):
            b['wins'] += 1 if payoff > 0.0 else 0
            b['payoff'] += payoff
            b['total'] += 1

    for b in tournament_statistics['bots']:
        b['win_percent'] = float(b['wins'])/float(b['total'])

    pool.close()
    pool.join()

    return tournament_statistics


def check_config(config):
    role_counts = defaultdict(lambda: 0)

    start_state = AvalonState.start_state(len(config))

    for bot in config:
        # Count roles
        role_counts[bot['role']] += 1

    assert 'merlin' in role_counts
    assert 'assassin' in role_counts
    for role, count in role_counts.items():
        assert role == 'servant' or role == 'minion' or count == 1
        assert role in GOOD_ROLES or role in EVIL_ROLES


def print_tournament_statistics(tournament_statistics):
    print("       Role |                      Bot | Evil |      Winrate |        Payoff ")
    print("-----------------------------------------------------------------------------")
    for bot in tournament_statistics['bots']:
        print("{: >11} | {: >24} | {: >4} | {: >11.02f}% | {: >13.02f} ".format(bot['role'], bot['bot'], 'Yes' if bot['role'] in EVIL_ROLES else '', 100*bot['win_percent'], bot['payoff']))

    for game_end, count in sorted(tournament_statistics['end_types'].items(), key=lambda x: x[1], reverse=True):
        print("{}: {} - {}".format(count, game_end[0], game_end[1]))



def run_all_combos_simple(bots, roles, games_per_matching=50):
    results = []
    for combination in itertools.combinations_with_replacement(bots, r=len(roles)):
        combo_name = '-'.join(map(lambda c: c.__name__, combination))
        results.append(
            (combo_name, run_large_tournament(combination, roles, games_per_matching=games_per_matching))
        )

    job_id = os.urandom(10).encode('hex')

    for combo_name, dataframe in results:
        filename = 'tournaments/{}_{}.msg.gz'.format(combo_name, job_id)
        print("Writing {}".format(filename))
        with gzip.open(filename, 'w') as f:
            dataframe.to_msgpack(f)


def run_all_combos(bots, roles, games_per_matching=50, parallelization=16):
    pool = multiprocessing.Pool(parallelization)
    #dataframe = []
    job_id = os.urandom(10).hex()
    bots_folder_name = '_'.join([bot.__name__ for bot in bots])  # 使用bots的名称构建文件夹名称
    base_directory = f'tournaments/{bots_folder_name}'  # 构建基于bots名称的路径
    for combination in itertools.combinations_with_replacement(bots, r=len(roles)):
        combo_name = '-'.join(map(lambda c: c.__name__, combination))
        dataframe = run_large_tournament_parallel(pool, combination, roles, games_per_matching=games_per_matching)
        #filename = 'tournaments/{}_{}.msg.gz'.format(combo_name, job_id)
        #filename = 'tournaments/{}_{}.csv'.format(combo_name, job_id)
        filename = f'{base_directory}/{combo_name}.csv'  # 修改文件路径以包含bots名称的文件夹
        print("Writing {}".format(filename))
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # with gzip.open(filename, 'w') as f:
        #     dataframe.to_msgpack(f)
        dataframe.to_csv(filename, index=False)

    pool.close()
    pool.join()



def run_all_combos_parallel(bots, roles):
    pool = multiprocessing.Pool()
    results = []

    for combination in itertools.combinations_with_replacement(bots, r=len(roles)):
        combo_name = '-'.join(map(lambda c: c.__name__, combination))
        results.append(
            (combo_name, pool.apply_async(run_large_tournament, (combination, roles)))
        )

    for combo_name, async_result in results:
        dataframe = async_result.get()
        filename = 'tournaments/{}.msg.gz'.format(combo_name)
        print("Writing {}".format(filename))
        with gzip.open(filename, 'w') as f:
            dataframe.to_msgpack(f)



#用于运行学习型比赛，适用于机器学习或强化学习场景，其中机器人可以在多次游戏中学习和改进其策略。
#bot_classes: 机器人类的列表，每个类代表一个不同的策略或玩家类型。
#winrate_track: 指定要追踪胜率的机器人在bot_classes列表中的索引。如果为None，则不追踪任何机器人的胜率。
#winrate_window: 胜率计算的窗口大小，表示用于计算胜率的最近游戏数。
def run_learning_tournament(bot_classes, winrate_track=None, winrate_window=10000000):
    bots = [
        ( bot_class(), bot_class.__name__, num == winrate_track )
        for num, bot_class in enumerate(bot_classes)
    ]

    hidden_state = ['merlin', 'servant', 'servant', 'assassin', 'minion']
    all_hidden_states = possible_hidden_states(set(hidden_state), num_players=len(hidden_state))

    beliefs_for_hidden_state = {}
    start_state = AvalonState.start_state(len(hidden_state))

    wins = []
    game_num = 0

    while True:
        game_num += 1
        random.shuffle(hidden_state)
        random.shuffle(bots)
        bot_ids = [ bot_name for _, bot_name, _ in bots ]

        if tuple(hidden_state) not in beliefs_for_hidden_state:
            beliefs_for_hidden_state[tuple(hidden_state)] = [
                starting_hidden_states(
                    player,
                    tuple(hidden_state),
                    all_hidden_states
                ) for player in range(len(hidden_state))
            ]

        beliefs = beliefs_for_hidden_state[tuple(hidden_state)]

        track_num = None

        bot_objs = []
        for i, (bot, bot_name, track) in enumerate(bots):
            if track:
                track_num = i
            bot.reset(start_state, i, hidden_state[i], beliefs[i])
            bot.set_bot_ids(bot_ids)
            bot_objs.append(bot)

        results, _ = run_game(start_state, tuple(hidden_state), bot_objs)

        for i, (bot, bot_name, track) in enumerate(bots):
            bot.show_roles(hidden_state, bot_ids)

        if track_num is not None:
            wins.append(int(results[track_num] > 0))
            if len(wins) > winrate_window:
                wins.pop(0)

        if game_num % 10 == 0 and winrate_track is not None:
            print("Winrate: {}%".format(100 * float(sum(wins)) / len(wins)))


def run_single_threaded_tournament(config, num_games=10, granularity=1):
    #用于收集比赛过程中的统计数据，包括每个机器人的胜场数、总场数、胜率和收益等信息。
    tournament_statistics = {
        'bots': [
            { 'bot': bot['bot'].__name__, 'role': bot['role'], 'wins': 0, 'total': 0, 'win_percent': 0, 'payoff': 0.0 }
            for bot in config
        ],
        'end_types': {}
    }

    #从config中提取所有机器人的角色，表示一种角色分配
    hidden_state = tuple([bot['role'] for bot in config])
    #计算所有可能的角色分配（60种）
    all_hidden_states = possible_hidden_states(set(hidden_state), num_players=len(config))
    #为每个玩家计算在这种角色分配下，认为可能的队伍角色分配
    beliefs = [
        starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(config))
    ]
    #创建游戏的初始状态
    start_state = AvalonState.start_state(len(hidden_state))
    #实例化每个配置中指定的机器人类
    bots = [ bot['bot']() for bot in config ]

    # pool = multiprocessing.Pool()
    results = []

    for i in range(num_games):
        if i % granularity == 0:
            print(i)
        for player, (bot, c) in enumerate(zip(bots, config)):
            bot.reset(start_state, player, c['role'], beliefs[player])

        payoffs, end_type = run_game(start_state, hidden_state, bots)
        tournament_statistics['end_types'][end_type] = 1 + tournament_statistics['end_types'].get(end_type, 0)

        for b, payoff in zip(tournament_statistics['bots'], payoffs):
            b['wins'] += 1 if payoff > 0.0 else 0
            b['payoff'] += payoff
            b['total'] += 1

    for b in tournament_statistics['bots']:
        b['win_percent'] = float(b['wins'])/float(b['total'])

    return tournament_statistics

#打印游戏进程
def run_and_print_game(config):
    bot_counts = {}
    bot_names = []
    for bot in config:
        base_name = bot['bot'].__name__
        bot_counts[base_name] = bot_counts.get(base_name, 0) + 1
        bot_names.append("{}_{}".format(base_name, bot_counts[base_name]))

    print("       Role |                      Bot | Evil ")
    print("----------------------------------------------")
    for name, bconf in zip(bot_names, config):
        print("{: >11} | {: >24} | {: >4}".format(bconf['role'], name, 'Yes' if bconf['role'] in EVIL_ROLES else ''))

    hidden_state = tuple([bot['role'] for bot in config])
    all_hidden_states = possible_hidden_states(set(hidden_state), num_players=len(config))
    beliefs = [
        starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(config))
    ]
    state = AvalonState.start_state(len(hidden_state))
    bots = [ bot['bot']() for bot in config ]
    for i, bot in enumerate(bots):
        bot.reset(state, i, hidden_state[i], beliefs[i])

    print("=============== Round 1 ================")
    while not state.is_terminal():

        moving_players = state.moving_players()
        moves = [
            bots[player].get_action(state, state.legal_actions(player, hidden_state))
            for player in moving_players
        ]
        if state.status == 'propose':
            player = moving_players[0]
            legal_actions = state.legal_actions(player, hidden_state)
            move_probs = bots[player].get_move_probabilities(state, legal_actions)
            move_prob = move_probs[legal_actions.index(moves[0])]
            print("Proposal #{}. {} proposes ({:0.2f}):".format(state.propose_count + 1, bot_names[moving_players[0]], move_prob))
            for player in moves[0].proposal:
                print(" - {}".format(bot_names[player]))
        elif state.status == 'vote':
            for player, move in zip(moving_players, moves):
                legal_actions = state.legal_actions(player, hidden_state)
                move_probs = bots[player].get_move_probabilities(state, legal_actions)
                move_prob = move_probs[legal_actions.index(move)]
                print("{: >24} votes {: <4} ({:0.2f})".format(bot_names[player], 'UP' if move.up else 'DOWN', move_prob))
        elif state.status == 'run':
            print("--- Mission results ---")
            for player, move in zip(moving_players, moves):
                legal_actions = state.legal_actions(player, hidden_state)
                move_probs = bots[player].get_move_probabilities(state, legal_actions)
                move_prob = move_probs[legal_actions.index(move)]
                print("{: >24}: {} ({:0.2f})".format(bot_names[player], 'FAIL' if move.fail else 'SUCCEED', move_prob))
        elif state.status == 'merlin':
            print("===== Final chance: pick merlin! =====")
            assassin = hidden_state.index('assassin')
            legal_actions = state.legal_actions(assassin, hidden_state)
            move_probs = bots[assassin].get_move_probabilities(state, legal_actions)
            move_prob = move_probs[legal_actions.index(moves[0])]
            assassin_pick = moves[assassin].merlin
            print('{} picked {} - {}! ({:0.2f})'.format(bot_names[assassin], bot_names[assassin_pick], 'CORRECT' if assassin_pick == hidden_state.index('merlin') else 'WRONG', move_prob))

        new_state, _, observation = state.transition(moves, hidden_state)
        for player, bot in enumerate(bots):
            if player in moving_players:
                move = moves[moving_players.index(player)]
            else:
                move = None
            bot.handle_transition(state, new_state, observation, move=move)

        if state.status == 'run' and new_state.status == 'propose':
            print("=============== Round {} ================".format(new_state.succeeds + new_state.fails + 1))

        state = new_state

    print(state.game_end)

#收集信息
def run_and_collect_information(config,file):
    #初始化
    bot_counts = {}
    bot_names = []
    role_positions = [None, None, None]
    history = []
    history_file = file
    role_assignment = {
        'role_assignment': None
    }

    for index,bot in enumerate(config):
        base_name = bot['bot'].__name__
        bot_counts[base_name] = bot_counts.get(base_name, 0) + 1
        bot_names.append("{}_{}".format(base_name, bot_counts[base_name]))
        if bot['role'] == 'merlin':
            role_positions[0] = index
        elif bot['role'] == 'assassin':
            role_positions[1] = index
        elif bot['role'] == 'minion':
            role_positions[2] = index
    
    #当前角色分配
    hidden_state = tuple([bot['role'] for bot in config])

    for i, assignment in enumerate(ASSIGNMENT_TO_ROLES):
        if assignment == role_positions:
            role_assignment['role_assignment'] = i
            break
    
    #记录当前角色分配
    history.append(role_assignment)


    #所有可能的角色分配
    all_hidden_states = possible_hidden_states(set(hidden_state), num_players=len(config))

    #返回对于每个玩家的角色来说，可能的角色分配组合
    beliefs = [
        starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(config))
    ]

    #当前阿瓦隆初始状态
    state = AvalonState.start_state(len(hidden_state))

    #为当前玩家分配的bot算法
    bots = [ bot['bot']() for bot in config ]
    #重置bot
    for i, bot in enumerate(bots):
        bot.reset(state, i, hidden_state[i], beliefs[i])
    
    current_round = 1
    round_data=[]
    while not state.is_terminal():
        moving_players = state.moving_players()
        moves = [
            bots[player].get_action(state, state.legal_actions(player, hidden_state))
            for player in moving_players
        ]
        all_votes_passed = False
        #propose_entry = {'round': current_round}
        if state.status == 'propose':
            player = moving_players[0]
            propose_entry = {
                'round' : current_round,
                'propose_count' : state.propose_count+1,
                'proposer_one_hot' : [int(i == player) for i in range(len(config))],
                'proposal_one_hot' : [int(i in moves[0].proposal) for i in range(len(config))]
            }
            
        elif state.status == 'vote':
            player_votes = [1 if move.up else 0 for move in moves]
            propose_entry['votes'] = player_votes
            all_votes_passed = sum(player_votes) > len(player_votes) / 2
            propose_entry['votes_passed'] = [1 if all_votes_passed is True else 0]
            task_status=[]
            if all_votes_passed == False:
                task_status = [-1]
                propose_entry['mission_status'] = task_status
                round_data.append(propose_entry)
            
        elif state.status == 'run':
            task_status = [0 if any(move.fail for move in moves) else 1]
            propose_entry['mission_status'] = task_status
            round_data.append(propose_entry)

        new_state, _, observation = state.transition(moves, hidden_state)
        for player, bot in enumerate(bots):
            if player in moving_players:
                move = moves[moving_players.index(player)]
            else:
                move = None
            bot.handle_transition(state, new_state, observation, move=move)

        if state.status == 'run' and new_state.status == 'propose':
            current_round += 1
        state = new_state

    round_datas = {
        'round_data': round_data
    }
    history.append(round_datas)
        
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)



def prepare_data_for_lstm(history_file, data_file, max_rounds=5, max_proposals_per_round=5):
    with open(history_file, 'r') as f:
        data = json.load(f)

    all_games_features = []
    role_assignment = data[0]['role_assignment']
    round_data = data[1]['round_data']
    
    game_features = []
    for round_index in range(1, max_rounds + 1):
        round_proposals = [x for x in round_data if x['round'] == round_index]
        #print(len(round_proposals))
        for proposal_index in range(1, max_proposals_per_round + 1):
            if proposal_index <= len(round_proposals):
                proposal = round_proposals[proposal_index - 1]
                features = [
                    proposal.get('round', -1),
                    proposal.get('propose_count', -1),
                    *proposal.get('proposer_one_hot', [-1]*5),
                    *proposal.get('proposal_one_hot', [-1]*5),
                    *proposal.get('votes', [-1]*5),
                    proposal.get('votes_passed', [-1])[0],
                    proposal.get('mission_status', [-1])[0]
                ]
            else:
                # 填充空数据
                features = [-1] * (19)
                
            game_features.extend(features)

    # 添加角色分配信息
    game_features.append(role_assignment)
    all_games_features.append(game_features)
    
    # 创建DataFrame并保存到CSV
    df = pd.DataFrame(all_games_features)
    df.to_csv(data_file, mode='a', header=False,index=False)


