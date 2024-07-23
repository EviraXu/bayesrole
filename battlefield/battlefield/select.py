import pandas as pd
from collections import defaultdict
import json
import numpy as np
from keras.preprocessing.sequence import pad_sequences

from battlefield.avalon_types import GOOD_ROLES, EVIL_ROLES, possible_hidden_states, starting_hidden_states
from battlefield.avalon import AvalonState
from battlefield.bots import Deeprole


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

def run_and_collect_information(config, num_games=10):
    for game_index in range(num_games):
        bot_counts = {}
        bot_names = []
        for bot in config:
            base_name = bot['bot'].__name__
            bot_counts[base_name] = bot_counts.get(base_name, 0) + 1
            bot_names.append("{}_{}".format(base_name, bot_counts[base_name]))


        history = []
        current_round = 1

        hidden_state = tuple([bot['role'] for bot in config])
        all_hidden_states = possible_hidden_states(set(hidden_state), num_players=len(config))
        beliefs = [
            starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(config))
        ]
        state = AvalonState.start_state(len(hidden_state))
        bots = [ bot['bot']() for bot in config ]
        for i, bot in enumerate(bots):
            bot.reset(state, i, hidden_state[i], beliefs[i])

        while not state.is_terminal():
            moving_players = state.moving_players()
            moves = [
                bots[player].get_action(state, state.legal_actions(player, hidden_state))
                for player in moving_players
            ]
            all_votes_passed = False
            if state.status == 'propose':
                player = moving_players[0]
                propose_entry = {
                    'round': current_round,
                    'propose_count': state.propose_count+1,
                    'proposer_one_hot': [int(i == player) for i in range(len(config))],
                    'proposal_one_hot': [int(i in moves[0].proposal) for i in range(len(config))],
                }
                legal_actions = state.legal_actions(player, hidden_state)
                move_probs = bots[player].get_move_probabilities(state, legal_actions)
                move_prob = move_probs[legal_actions.index(moves[0])]
            elif state.status == 'vote':
                for player, move in zip(moving_players, moves):
                    legal_actions = state.legal_actions(player, hidden_state)
                    move_probs = bots[player].get_move_probabilities(state, legal_actions)
                    move_prob = move_probs[legal_actions.index(move)]
                    #print("{: >24} votes {: <4} ({:0.2f})".format(bot_names[player], 'UP' if move.up else 'DOWN', move_prob))
            
                player_votes = [1 if move.up else 0 for move in moves]
                propose_entry['votes'] = player_votes
                all_votes_passed = sum(player_votes) > len(player_votes) / 2
                propose_entry['votes_passed'] = [1 if all_votes_passed is True else 0]
                task_status=[]
                if all_votes_passed == False:
                    task_status = [-1]
                    propose_entry['mission_status'] = task_status
                    history.append(propose_entry)
            elif state.status == 'run':
                task_status = []            
                for player, move in zip(moving_players, moves):
                    legal_actions = state.legal_actions(player, hidden_state)
                    move_probs = bots[player].get_move_probabilities(state, legal_actions)
                    move_prob = move_probs[legal_actions.index(move)]
                    #print("{: >24}: {} ({:0.2f})".format(bot_names[player], 'FAIL' if move.fail else 'SUCCEED', move_prob))
            
                task_status = [0 if any(move.fail for move in moves) else 1]
                propose_entry['mission_status'] = task_status
                history.append(propose_entry)
            elif state.status == 'merlin':
                assassin = hidden_state.index('assassin')
                legal_actions = state.legal_actions(assassin, hidden_state)
                move_probs = bots[assassin].get_move_probabilities(state, legal_actions)
                move_prob = move_probs[legal_actions.index(moves[0])]
                assassin_pick = moves[assassin].merlin
            
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

        with open(f'game_history_{game_index+1}.json', 'w') as f:
            json.dump(history, f, indent=2)

        #prepare_data_for_lstm(f'game_history_{game_index+1}.json')

# def prepare_data_for_lstm(history_file):
#     # Load JSON data
#     with open(history_file, 'r') as f:
#         events = json.load(f)

#     sequences = []
#     for event in events:
#         # Construct feature vector for each event
#         # Your logic here

#     # Pad sequences to the maximum length
#     game_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', dtype='float32')
#     print(f"Data from {history_file} prepared for LSTM.")

#     # Optionally save the prepared data
#     np.save(f'{history_file}_prepared.npy', game_sequences)

if __name__ == "__main__":
    run_and_collect_information(INFO_CONFIG, num_games=10)


