import pandas as pd

# 加载CSV文件
df = pd.read_csv('/root/DeepRole-master/battlefield/tournaments/DeepBayes_Deeprole/Deeprole-Deeprole-Deeprole-Deeprole-Deeprole_a17f6f0a9079806e39b9.csv')

# 确定正方和反方角色
positive_roles = ['servant', 'merlin']
negative_roles = ['minion', 'assassin']

# 筛选bot_4为Deeprole/RandomBot的行
df_bot4_deeprole = df[df['bot_4'] == 'Deeprole']
#df_bot4_deeprole = df[df['bot_4'] == 'DeepBayes']


# 计算处于正方角色下的胜率
df_bot4_positive = df_bot4_deeprole[df_bot4_deeprole['bot_4_role'].isin(positive_roles)]
positive_win_rate = (df_bot4_positive['winner'] == 'good').mean() 

# 计算处于反方角色下的胜率
df_bot4_negative = df_bot4_deeprole[df_bot4_deeprole['bot_4_role'].isin(negative_roles)]
negative_win_rate = (df_bot4_negative['winner'] == 'evil').mean()   

# 计算bot_4胜率
# 计算作为正方角色时赢的次数
good_wins = df_bot4_deeprole[(df_bot4_deeprole['bot_4_role'].isin(positive_roles)) & (df_bot4_deeprole['winner'] == 'good')].shape[0]

# 计算作为反方角色时赢的次数
evil_wins = df_bot4_deeprole[(df_bot4_deeprole['bot_4_role'].isin(negative_roles)) & (df_bot4_deeprole['winner'] == 'evil')].shape[0]

# 计算总游戏次数
total_games = df_bot4_deeprole.shape[0]

# 计算综合胜率
overall_win_rate = (good_wins + evil_wins) / total_games if total_games > 0 else 0

# 打印结果
print(f"P(win)={overall_win_rate:.2f}")
print(f"P(S win)={negative_win_rate:.2f}")
print(f"P(R win)={positive_win_rate:.2f}")

