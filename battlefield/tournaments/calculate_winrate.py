import pandas as pd
import os
import sys
import re

# 全局变量
X_positive_win_rates = [None] * 5
X_negative_win_rates = [None] * 5
X_overall_win_rates = [None] * 5
Y_positive_win_rates = [None] * 5
Y_negative_win_rates = [None] * 5
Y_overall_win_rates = [None] * 5

def calculate_win_rates(file_path,agent_x,x_count,agent_y,y_count):

    # 加载CSV文件
    df = pd.read_csv(file_path)
    # 确定正方和反方角色
    positive_roles = ['servant', 'merlin']
    negative_roles = ['minion', 'assassin']
    df_bot4_X = df[df['bot_4'] == agent_x]
    df_bot4_Y = df[df['bot_4'] == agent_y]
    
    if x_count>=1:
        #对于智能体X
        df_bot4_X_positive = df_bot4_X[df_bot4_X['bot_4_role'].isin(positive_roles)]
        X_positive_win_rate = (df_bot4_X_positive['winner'] == 'good').mean() 

        df_bot4_X_negative = df_bot4_X[df_bot4_X['bot_4_role'].isin(negative_roles)]
        X_negative_win_rate = (df_bot4_X_negative['winner'] == 'evil').mean()

        X_good_wins = df_bot4_X[(df_bot4_X['bot_4_role'].isin(positive_roles)) & (df_bot4_X['winner'] == 'good')].shape[0]
        X_evil_wins = df_bot4_X[(df_bot4_X['bot_4_role'].isin(negative_roles)) & (df_bot4_X['winner'] == 'evil')].shape[0]
        X_total_games = df_bot4_X.shape[0]
        X_overall_win_rate = (X_good_wins + X_evil_wins) / X_total_games if X_total_games > 0 else 0   

        X_positive_win_rates[x_count - 1] = X_positive_win_rate
        X_negative_win_rates[x_count - 1] = X_negative_win_rate
        X_overall_win_rates[x_count - 1] = X_overall_win_rate
    

    if y_count>=1:
        #对于智能体Y
        df_bot4_Y_positive = df_bot4_Y[df_bot4_Y['bot_4_role'].isin(positive_roles)]
        Y_positive_win_rate = (df_bot4_Y_positive['winner'] == 'good').mean() 

        df_bot4_Y_negative = df_bot4_Y[df_bot4_Y['bot_4_role'].isin(negative_roles)]
        Y_negative_win_rate = (df_bot4_Y_negative['winner'] == 'evil').mean()

        Y_good_wins = df_bot4_Y[(df_bot4_Y['bot_4_role'].isin(positive_roles)) & (df_bot4_Y['winner'] == 'good')].shape[0]
        Y_evil_wins = df_bot4_Y[(df_bot4_Y['bot_4_role'].isin(negative_roles)) & (df_bot4_Y['winner'] == 'evil')].shape[0]
        Y_total_games = df_bot4_Y.shape[0]
        Y_overall_win_rate = (Y_good_wins + Y_evil_wins) / Y_total_games if Y_total_games > 0 else 0 

        Y_positive_win_rates[x_count] = Y_positive_win_rate
        Y_negative_win_rates[x_count] = Y_negative_win_rate
        Y_overall_win_rates[x_count] = Y_overall_win_rate
        
def save_results_to_csv(agent_x, agent_y):
    # Create DataFrame
    data = {
        "X_Positive_Win_Rates": X_positive_win_rates,
        "X_Negative_Win_Rates": X_negative_win_rates,
        "X_Overall_Win_Rates": X_overall_win_rates,
        "Y_Positive_Win_Rates": Y_positive_win_rates,
        "Y_Negative_Win_Rates": Y_negative_win_rates,
        "Y_Overall_Win_Rates": Y_overall_win_rates
    }
    df_results = pd.DataFrame(data)
    df_results = df_results.round(2)

    directory = 'win_rates'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{agent_x}_{agent_y}.csv"
    full_path = os.path.join(directory, filename)
    df_results.to_csv(full_path, index=False)

def calculate_every_file(directory, agent_x, agent_y):
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            #print(file)
            file_name = os.path.splitext(file)[0]
            parts = file_name.split('-')
            #print(parts)
            x_count = parts.count(agent_x)
            y_count = parts.count(agent_y)
            # print(x_count)
            # print(y_count)
            calculate_win_rates(file_path,agent_x,x_count,agent_y,y_count)

    save_results_to_csv(agent_x, agent_y)


if __name__ == "__main__":
    #python3 calculate_winrate.py /deeprole-master/battlefield/tournaments/Bayesrole_Deeprole_v2 Bayesrole Deeprole_v2
    if len(sys.argv) < 4:
        print("Usage: python calculate_winrate.py <directory> <agent_x> <agent_y>")
    else:
        directory = sys.argv[1]
        agent_x = sys.argv[2]
        agent_y = sys.argv[3]
        calculate_every_file(directory, agent_x, agent_y)