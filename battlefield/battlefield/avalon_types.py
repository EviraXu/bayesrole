from collections import namedtuple
import itertools

AVALON_PROPOSE_SIZES = {
    5:  [(2, 1), (3, 1), (2, 1), (3, 1), (3, 1)],
    6:  [(2, 1), (3, 1), (4, 1), (3, 1), (4, 1)],
    7:  [(2, 1), (3, 1), (3, 1), (4, 2), (4, 1)],
    8:  [(3, 1), (4, 1), (4, 1), (5, 2), (5, 1)],
    9:  [(3, 1), (4, 1), (4, 1), (5, 2), (5, 1)],
    10: [(3, 1), (4, 1), (4, 1), (5, 2), (5, 1)],
}

AVALON_PLAYER_COUNT = {
    5:  (3, 2),
    6:  (4, 2),
    7:  (4, 3),
    8:  (5, 3),
    9:  (5, 4),
    10: (6, 4),
}

EVIL_ROLES = set(['minion', 'mordred', 'morgana', 'assassin', 'oberon'])
GOOD_ROLES = set(['servant', 'merlin', 'percival'])

ASSIGNMENT_TO_ROLES = [
    [0, 1, 2],
    [0, 1, 3],
    [0, 1, 4],
    [0, 2, 1],
    [0, 2, 3],
    [0, 2, 4],
    [0, 3, 1],
    [0, 3, 2],
    [0, 3, 4],
    [0, 4, 1],
    [0, 4, 2],
    [0, 4, 3],
    [1, 0, 2],
    [1, 0, 3],
    [1, 0, 4],
    [1, 2, 0],
    [1, 2, 3],
    [1, 2, 4],
    [1, 3, 0],
    [1, 3, 2],
    [1, 3, 4],
    [1, 4, 0],
    [1, 4, 2],
    [1, 4, 3],
    [2, 0, 1],
    [2, 0, 3],
    [2, 0, 4],
    [2, 1, 0],
    [2, 1, 3],
    [2, 1, 4],
    [2, 3, 0],
    [2, 3, 1],
    [2, 3, 4],
    [2, 4, 0],
    [2, 4, 1],
    [2, 4, 3],
    [3, 0, 1],
    [3, 0, 2],
    [3, 0, 4],
    [3, 1, 0],
    [3, 1, 2],
    [3, 1, 4],
    [3, 2, 0],
    [3, 2, 1],
    [3, 2, 4],
    [3, 4, 0],
    [3, 4, 1],
    [3, 4, 2],
    [4, 0, 1],
    [4, 0, 2],
    [4, 0, 3],
    [4, 1, 0],
    [4, 1, 2],
    [4, 1, 3],
    [4, 2, 0],
    [4, 2, 1],
    [4, 2, 3],
    [4, 3, 0],
    [4, 3, 1],
    [4, 3, 2]
]



ProposeAction = namedtuple('ProposeAction', ['proposal'])
VoteAction = namedtuple('VoteAction', ['up'])
MissionAction = namedtuple('MissionAction', ['fail'])
PickMerlinAction = namedtuple('PickMerlinAction', ['merlin'])


def filter_hidden_states(hidden_states, proposal, num_fails_observed):
    return [
        hidden_state
        for hidden_state in hidden_states
        if sum([1 for player in proposal if hidden_state[player] in EVIL_ROLES ]) >= num_fails_observed
    ]


def possible_hidden_states(roles, num_players):
    #print("---------possible_hidden_states-----------")
    roles = set(roles)
    assert 'assassin' in roles, "All games require an assassin: {}".format(roles)
    assert 'merlin' in roles, "All games require a merlin: {}".format(roles)
    output_roles = list(roles)
    num_good_needed, num_evil_needed = AVALON_PLAYER_COUNT[num_players]
    num_good_have = len(roles & GOOD_ROLES)
    num_evil_have = len(roles & EVIL_ROLES)
    output_roles.extend(['minion']*(num_evil_needed - num_evil_have))
    output_roles.extend(['servant']*(num_good_needed - num_good_have))
    assert len(output_roles) == num_players, "not sure what happened"
    #print(list(set(itertools.permutations(output_roles))))
    return list(set(itertools.permutations(output_roles)))


#
def starting_hidden_states(player, real_hidden_state, possible_hidden_states):
    #knowledge 列表初始化包括了玩家自己的角色，因为每个玩家都清楚自己的角色
    knowledge = [(player, [real_hidden_state[player]])] # We know what we are
    #如果玩家是除奥伯伦外的邪恶角色（例如刺客、莫甘娜等），他们知道除奥伯伦外其他所有邪恶角色的身份。
    if real_hidden_state[player] in EVIL_ROLES and real_hidden_state[player] != 'oberon': # If we're evil and not oberon
        for p, role in enumerate(real_hidden_state):
            if p != player and role in EVIL_ROLES and role != 'oberon': # We know who's who for evil, but not who's oberon
                knowledge.append((p, [role]))
    
    #如果玩家是梅林，他知道除了莫德雷德和奥伯伦以外的所有邪恶角色。梅林不知道莫德雷德是邪恶的，因为莫德雷德对梅林是隐形的。
    if real_hidden_state[player] == 'merlin': # If we're merlin
        for p, role in enumerate(real_hidden_state):
            if p != player and role in EVIL_ROLES and role not in ['oberon', 'mordred']: # We know who's evil, but not oberon or mordred
                knowledge.append((p, EVIL_ROLES - set(['oberon', 'mordred'])))

    #
    if real_hidden_state[player] == 'percival': # If we're percival
        for p, role in enumerate(real_hidden_state):
            if p != player and role in ['merlin', 'morgana']:
                knowledge.append((p, ['merlin', 'morgana']))

    return [
        hidden_state
        for hidden_state in possible_hidden_states
        if all(hidden_state[p] in possible_roles for p, possible_roles in knowledge)
    ]


if __name__ == "__main__":
    hidden_states = possible_hidden_states(['assassin', 'mordred', 'merlin', 'oberon'], 7)
    real_hidden_state = hidden_states[len(hidden_states)/2]
    print("REAL ROLES")
    print(real_hidden_state)

    for player, role in enumerate(real_hidden_state):
        print("====== {}'s perspective:".format(role))
        for h in starting_hidden_states(player, real_hidden_state, hidden_states):
            print(h)


