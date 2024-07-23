import json
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# 加载 JSON 数据
with open('game_history.json', 'r') as f:
    events = json.load(f)

# 初始化列表来存储游戏的数据
game_sequences = []

num_players = 5  # 玩家数量
default_proposer = [-1] * num_players  # 默认提议者向量
default_proposal = [-1] * num_players  # 默认提议向量
default_votes = [-1] * num_players  # 默认投票向量
default_mission_status = [-1]  # 默认任务状态

sequences = []
for event in events:
    # 创建特征向量，这里需要根据你的具体特征进行调整
    print(event)
    round = [event.get('round')]
    propose_count = [event.get('propose_count')]
    proposer = event.get('proposer_one_hot', default_proposer)
    proposal = event.get('proposal_one_hot', default_proposal)
    votes = event.get('votes', default_votes)
    votes_passed = event.get('votes_passed', [0])
    mission_status = event.get('mission_status', default_mission_status)
    # 合并所有特征为一个单一的特征向量
    features = round + propose_count + proposer + proposal + votes + votes_passed + mission_status
    sequences.append(features)

# 将列表转换为 NumPy 数组，并进行填充
max_len = max([len(sequence) for sequence in sequences])  # 计算所有事件中的最大长度
game_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', dtype='float32')


# 打印 game_sequences 的维度，检查是否符合预期（num_samples, max_len, num_features）
print("Shape of game_sequences:", game_sequences.shape)

# 打印第一局游戏的全部数据
print("Game sequence:")
print(game_sequences[0])

# 如果你的数组是用 pad_sequences 填充的，可能包含很多-1，可以检查非-默认值元素
non_default_entries = np.count_nonzero(game_sequences[0] != -1)
print("Number of non-default entries in the game:", non_default_entries)
