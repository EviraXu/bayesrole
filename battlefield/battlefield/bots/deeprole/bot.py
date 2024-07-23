import numpy as np

from battlefield.bots.bot import Bot
from battlefield.avalon_types import EVIL_ROLES, GOOD_ROLES, MissionAction, VoteAction, ProposeAction, PickMerlinAction
from battlefield.bots.deeprole.lookup_tables import get_deeprole_perspective, assignment_id_to_hidden_state, print_top_k_belief, print_top_k_viewpoint_belief
from battlefield.bots.deeprole.run_deeprole import run_deeprole_on_node,run_deeprole_assignment_on_node
from battlefield.bots.cfr_bot import proposal_to_bitstring, bitstring_to_proposal
from nn_train.train import predict


def get_start_node(proposer):
    return {
        "type": "TERMINAL_PROPOSE_NN",
        "succeeds": 0,
        "fails": 0,
        "propose_count": 0,
        "proposer": proposer,
        "new_belief": list(np.ones(60)/60.0),
        "nozero_belief": list(np.ones(60)/60.0)
    }

def assert_eq(a, b):
    assert a == b, "{} != {}".format(a, b)

def votes_to_bitstring(vote_moves):
    result = 0
    for i, vote in enumerate(vote_moves):
        if vote.up:
            result |= (1 << i)
    return result

def print_move_probs(probs, legal_actions, cutoff=0.95):
    zipped = zip(probs, legal_actions)
    zipped.sort(reverse=True)
    total = 0.0
    print("----- move probs -----")
    while total < cutoff and len(zipped) > 0:
        prob, move = zipped[0]
        zipped = zipped[1:]
        print(move, prob)
        total += prob

# Plays randomly, except always fails missions if bad.
class Deeprole(Bot):
    ITERATIONS = 100
    WAIT_ITERATIONS = 50
    NO_ZERO=False
    NN_FOLDER='deeprole_models'
    DEEPROLE_BINARY='deeprole'

    def __init__(self):
        self.belief = list(np.ones(60)/60.0)
        pass


    def reset(self, game, player, role, hidden_states):
        #print("------reset-----")
        self.node = run_deeprole_on_node(
            get_start_node(game.proposer),
            self.ITERATIONS,
            self.WAIT_ITERATIONS,
            no_zero=self.NO_ZERO,
            nn_folder=self.NN_FOLDER,
            binary=self.DEEPROLE_BINARY
        )
        self.player = player
        self.perspective = get_deeprole_perspective(player, hidden_states[0])
        self.belief = list(np.ones(60)/60.0)
        self.thought_log = []
        # print self.perspective


    def handle_transition(self, old_state, new_state, observation, move=None):
        # 将关于旧状态的信息和当前信念记录到日志中。这有助于调试或回顾AI的决策过程。
        self.thought_log.append((
            old_state.succeeds, old_state.fails, old_state.propose_count, self.belief
        ))
        #print("thought_log is :",self.thought_log)

        #print("self.node:")
        #print(self.node)
        #print("old state is :",old_state.status)
        #print("new_state is :",new_state.status)

        #通过判断 old_state.status，该方法根据游戏的当前阶段执行不同的逻辑
        if old_state.status == 'merlin':
            return
        
        #获取提议信息，并将其转换为比特串
        #根据比特串找到对应的子节点，并更新AI的决策树节点 (self.node) 到该子节点
        if old_state.status == 'propose':
            proposal = observation
            bitstring = proposal_to_bitstring(proposal)
            child_index = self.node['propose_options'].index(bitstring)
            self.node = self.node['children'][child_index]
        elif old_state.status == 'vote':
            child_index = votes_to_bitstring(observation)
            self.node = self.node['children'][child_index]
        elif old_state.status == 'run':
            num_fails = observation
            self.node = self.node['children'][num_fails]

        if self.node['type'] == 'TERMINAL_PROPOSE_NN':
            # print_top_k_belief(self.node['new_belief'])
            # print "Player {} perspective {}".format(self.player, self.perspective)
            # print_top_k_viewpoint_belief(self.node['new_belief'], self.player, self.perspective)
            # print self.node['new_belief']
            self.belief = self.node['new_belief']
            #print("bot self.belief:")
            #print(self.belief)
            self.node = run_deeprole_on_node(
                self.node,
                self.ITERATIONS,
                self.WAIT_ITERATIONS,
                no_zero=self.NO_ZERO,
                nn_folder=self.NN_FOLDER,
                binary=self.DEEPROLE_BINARY
            )

        if self.node['type'].startswith("TERMINAL_") and self.node['type'] != "TERMINAL_MERLIN":
            return

        assert_eq(new_state.succeeds, self.node["succeeds"])
        assert_eq(new_state.fails, self.node["fails"])

        if new_state.status == 'propose':
            assert_eq(new_state.proposer, self.node['proposer'])
            assert_eq(new_state.propose_count, self.node['propose_count'])
        elif new_state.status == 'vote':
            assert_eq(new_state.proposer, self.node['proposer'])
            assert_eq(new_state.propose_count, self.node['propose_count'])
            assert_eq(new_state.proposal, bitstring_to_proposal(self.node['proposal']))
        elif new_state.status == 'run':
            assert_eq(new_state.proposal, bitstring_to_proposal(self.node['proposal']))

    #根据给定的合法行动列表和这些行动的概率分布，随机选择并返回一个行动
    def get_action(self, state, legal_actions):
        probs = self.get_move_probabilities(state, legal_actions)
        index = np.random.choice(len(legal_actions), p=probs)
        return legal_actions[index]

    #用于计算在给定游戏状态下执行每个合法行动的概率。
    def get_move_probabilities(self, state, legal_actions):
        if len(legal_actions) == 1:
            return np.array([1.0])
        #从类的属性（可能是由外部传入的决策树或类似结构）中获取当前玩家的提议策略 propose_strategy 和可选的提议 propose_options
        if state.status == 'propose':
            probs = np.zeros(len(legal_actions))
            propose_strategy = self.node['propose_strat'][self.perspective]
            propose_options = self.node['propose_options']
            for strategy_prob, proposal_bitstring in zip(propose_strategy, propose_options):
                action = ProposeAction(proposal=bitstring_to_proposal(proposal_bitstring))
                probs[legal_actions.index(action)] = strategy_prob
            # print probs
            # print_move_probs(probs, legal_actions)
            return probs
        elif state.status == 'vote':
            probs = np.zeros(len(legal_actions))
            vote_strategy = self.node['vote_strat'][self.player][self.perspective]
            for strategy_prob, vote_up in zip(vote_strategy, [False, True]):
                action = VoteAction(up=vote_up)
                probs[legal_actions.index(action)] = strategy_prob
            return probs
        elif state.status == 'run':
            probs = np.zeros(len(legal_actions))
            mission_strategy = self.node['mission_strat'][self.player][self.perspective]
            for strategy_prob, fail in zip(mission_strategy, [False, True]):
                action = MissionAction(fail=fail)
                probs[legal_actions.index(action)] = strategy_prob
            return probs
        elif state.status == 'merlin':
            # print np.array(self.node['merlin_strat'][self.player][self.perspective])
            return np.array(self.node['merlin_strat'][self.player][self.perspective])

class Deeprole_assignment(Bot):
    ITERATIONS = 100
    WAIT_ITERATIONS = 50
    NO_ZERO=False
    NN_FOLDER='deeprole_models'
    DEEPROLE_BINARY='deeprole'

    def __init__(self):
        self.belief = list(np.ones(60)/60.0)
        self.history = [-1]*475
        pass


    def reset(self, game, player, role, hidden_states):
        #print("------reset-----")
        history = np.array(self.history).reshape((1,25, 19))
        self.node = run_deeprole_assignment_on_node(
            get_start_node(game.proposer),
            self.ITERATIONS,
            self.WAIT_ITERATIONS,
            history,
            no_zero=self.NO_ZERO,
            nn_folder=self.NN_FOLDER,
            binary=self.DEEPROLE_BINARY
        )
        self.player = player
        self.perspective = get_deeprole_perspective(player, hidden_states[0])
        self.belief = list(np.ones(60)/60.0)
        self.history = [-1]*475
        self.thought_log = []



    def handle_transition(self, old_state, new_state, observation, move=None):
        # 将关于旧状态的信息和当前信念记录到日志中。这有助于调试或回顾AI的决策过程。
        self.thought_log.append((
            old_state.succeeds, old_state.fails, old_state.propose_count, self.belief
        ))
        

        #通过判断 old_state.status，该方法根据游戏的当前阶段执行不同的逻辑
        if old_state.status == 'merlin':
            return
        
        #获取提议信息，并将其转换为比特串
        #根据比特串找到对应的子节点，并更新AI的决策树节点 (self.node) 到该子节点
        if old_state.status == 'propose':
            proposal = observation
            #print("proposal:",proposal)
            bitstring = proposal_to_bitstring(proposal)
            #print("bitstring:",bitstring)
            round = old_state.fails + old_state.succeeds + 1
            #print("round:",round)
            propose_count = old_state.propose_count + 1
            #print("propose_count:",propose_count)
            proposer = [1 if i == old_state.proposer else 0 for i in range(5)]
            #print("proposer:",proposer)
            bitproposal = [1 if i in proposal else 0 for i in range(5)]
            #print("bitproposal:",bitproposal)

            index = ((round - 1) * 5 + (propose_count - 1))*19
            #print("index:",index)
            self.history[index+0] = round
            self.history[index+1] = propose_count
            self.history[index+2:index+7] = proposer
            self.history[index+7:index+12] = bitproposal
            #print(self.history)
            child_index = self.node['propose_options'].index(bitstring)
            self.node = self.node['children'][child_index]
        elif old_state.status == 'vote':
            player_votes = [1 if move.up else 0 for move in observation]
            #print("player_votes:",player_votes)
            vote_result = 1 if sum(player_votes) > len(player_votes) / 2 else 0
            #print("vote_result:",vote_result)
            
            round = old_state.fails + old_state.succeeds + 1
            propose_count = old_state.propose_count + 1
            index = ((round - 1) * 5 + (propose_count - 1))*19
            #print("index:",index)
            self.history[index+12:index+17] = player_votes
            self.history[index+17] = vote_result
            #print(self.history)

            child_index = votes_to_bitstring(observation)
            self.node = self.node['children'][child_index]
        elif old_state.status == 'run':
            #print("observation:",observation)
            mission_status = 0 if observation != 0 else 1
            #print("mission_status:",mission_status)

            round = old_state.fails + old_state.succeeds + 1
            propose_count = old_state.propose_count + 1
            index = ((round - 1) * 5 + (propose_count - 1))*19
            #print("index:",index)
            self.history[index+18] = mission_status
            #print(self.history)

            num_fails = observation
            self.node = self.node['children'][num_fails]

        if self.node['type'] == 'TERMINAL_PROPOSE_NN':
            # print_top_k_belief(self.node['new_belief'])
            # print "Player {} perspective {}".format(self.player, self.perspective)
            # print_top_k_viewpoint_belief(self.node['new_belief'], self.player, self.perspective)
            # print self.node['new_belief']
            #self.belief = self.node['new_belief']
            history = np.array(self.history).reshape((1,25, 19))
            #print("history:",history)
            self.belief = predict(history)
            #print("belief:",self.belief)
            self.node = run_deeprole_assignment_on_node(
                self.node,
                self.ITERATIONS,
                self.WAIT_ITERATIONS,
                history,
                no_zero=self.NO_ZERO,
                nn_folder=self.NN_FOLDER,
                binary=self.DEEPROLE_BINARY
            )

        if self.node['type'].startswith("TERMINAL_") and self.node['type'] != "TERMINAL_MERLIN":
            return

        assert_eq(new_state.succeeds, self.node["succeeds"])
        assert_eq(new_state.fails, self.node["fails"])

        if new_state.status == 'propose':
            assert_eq(new_state.proposer, self.node['proposer'])
            assert_eq(new_state.propose_count, self.node['propose_count'])
        elif new_state.status == 'vote':
            assert_eq(new_state.proposer, self.node['proposer'])
            assert_eq(new_state.propose_count, self.node['propose_count'])
            assert_eq(new_state.proposal, bitstring_to_proposal(self.node['proposal']))
        elif new_state.status == 'run':
            assert_eq(new_state.proposal, bitstring_to_proposal(self.node['proposal']))

    #根据给定的合法行动列表和这些行动的概率分布，随机选择并返回一个行动
    def get_action(self, state, legal_actions):
        probs = self.get_move_probabilities(state, legal_actions)
        index = np.random.choice(len(legal_actions), p=probs)
        return legal_actions[index]

    #用于计算在给定游戏状态下执行每个合法行动的概率。
    def get_move_probabilities(self, state, legal_actions):
        if len(legal_actions) == 1:
            return np.array([1.0])
        #从类的属性（可能是由外部传入的决策树或类似结构）中获取当前玩家的提议策略 propose_strategy 和可选的提议 propose_options
        if state.status == 'propose':
            probs = np.zeros(len(legal_actions))
            propose_strategy = self.node['propose_strat'][self.perspective]
            propose_options = self.node['propose_options']
            for strategy_prob, proposal_bitstring in zip(propose_strategy, propose_options):
                action = ProposeAction(proposal=bitstring_to_proposal(proposal_bitstring))
                probs[legal_actions.index(action)] = strategy_prob
            # print probs
            # print_move_probs(probs, legal_actions)
            return probs
        elif state.status == 'vote':
            probs = np.zeros(len(legal_actions))
            vote_strategy = self.node['vote_strat'][self.player][self.perspective]
            for strategy_prob, vote_up in zip(vote_strategy, [False, True]):
                action = VoteAction(up=vote_up)
                probs[legal_actions.index(action)] = strategy_prob
            return probs
        elif state.status == 'run':
            probs = np.zeros(len(legal_actions))
            mission_strategy = self.node['mission_strat'][self.player][self.perspective]
            for strategy_prob, fail in zip(mission_strategy, [False, True]):
                action = MissionAction(fail=fail)
                probs[legal_actions.index(action)] = strategy_prob
            return probs
        elif state.status == 'merlin':
            # print np.array(self.node['merlin_strat'][self.player][self.perspective])
            return np.array(self.node['merlin_strat'][self.player][self.perspective])


# For examining the effect of more iterations
class Deeprole_3_1(Deeprole):
    ITERATIONS = 3
    WAIT_ITERATIONS = 1

class Deeprole_10_5(Deeprole):
    ITERATIONS = 10
    WAIT_ITERATIONS = 5

class Deeprole_30_15(Deeprole):
    ITERATIONS = 30
    WAIT_ITERATIONS = 15

class Deeprole_100_50(Deeprole):
    ITERATIONS = 100
    WAIT_ITERATIONS = 50

class Deeprole_300_150(Deeprole):
    ITERATIONS = 300
    WAIT_ITERATIONS = 150


# For examining the effect of "Wait iterations"
class Deeprole_100_0(Deeprole):
    ITERATIONS = 100
    WAIT_ITERATIONS = 0

class Deeprole_100_25(Deeprole):
    ITERATIONS = 100
    WAIT_ITERATIONS = 25

# class Deeprole_100_50(Deeprole):

class Deeprole_100_75(Deeprole):
    ITERATIONS = 100
    WAIT_ITERATIONS = 75


class Deeprole_100_100(Deeprole):
    ITERATIONS = 100
    WAIT_ITERATIONS = 99


class Deeprole_NoZeroing(Deeprole):
    NO_ZERO = True


class Deeprole_OldNNs(Deeprole):
    NN_FOLDER = 'deeprole_models_old'



class Deeprole_NoZeroUnconstrained(Deeprole):
    DEEPROLE_BINARY = 'deeprole_nozero'
    NN_FOLDER = 'deeprole_nozero_unconstrained'


class Deeprole_ZeroingUnconstrained(Deeprole):
    DEEPROLE_BINARY = 'deeprole_zeroing'
    NN_FOLDER = 'deeprole_zeroing_unconstrained'


class Deeprole_ZeroingWinProbs(Deeprole):
    DEEPROLE_BINARY = 'deeprole_zeroing'
    NN_FOLDER = 'deeprole_zeroing_winprobs'

class DeepBayes(Deeprole):
    DEEPROLE_BINARY = 'deepbayes'
    NN_FOLDER = 'deepbayes_models'

