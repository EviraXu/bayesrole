import matplotlib.pyplot as plt


# X轴
deeprole_counts = [0, 1, 2, 3, 4]
# bot_types
bot_types = ['Random', 'LogicBot', 'ISMCTS', 'MOISMCTS', 'CFR']
win_rates = {
    'Random': {
        'DeepRole': [0.75, 0.68, 0.65, 0.51, 0.48],
        'Other': [0.50, 0.45, 0.48, 0.46, 0.03]
    },
    'LogicBot': {
        'DeepRole': [0.75, 0.68, 0.65, 0.51, 0.48],
        'Other': [0.50, 0.45, 0.48, 0.46, 0.03]
    },
    'ISMCTS': {
        'DeepRole': [0.75, 0.68, 0.65, 0.51, 0.48],
        'Other': [0.50, 0.45, 0.48, 0.46, 0.03]
    },
    'MOISMCTS': {
        'DeepRole': [0.75, 0.68, 0.65, 0.51, 0.48],
        'Other': [0.50, 0.45, 0.48, 0.46, 0.03]
    },
    'CFR': {
        'DeepRole': [0.75, 0.68, 0.65, 0.51, 0.48],
        'Other': [0.50, 0.45, 0.48, 0.46, 0.03]
    },
}
s_win_rates = {
    'Random': {
        'DeepRole': [0.12,0.23,0.45,0.61,0.39], 
        'Other': [0.78,0.23,0.47,0.87,0.15]
    },
    'LogicBot': {
        'DeepRole': [0.12,0.23,0.45,0.61,0.39], 
        'Other': [0.78,0.23,0.47,0.87,0.15]
    },
    'ISMCTS': {
        'DeepRole': [0.12,0.23,0.45,0.61,0.39], 
        'Other': [0.78,0.23,0.47,0.87,0.15]
    },
    'MOISMCTS': {
        'DeepRole': [0.12,0.23,0.45,0.61,0.39], 
        'Other': [0.78,0.23,0.47,0.87,0.15]
    },
    'CFR': {
        'DeepRole': [0.12,0.23,0.45,0.61,0.39], 
        'Other': [0.78,0.23,0.47,0.87,0.15]
    }, 
}
r_win_rates = {
    'Random': {
        'DeepRole': [0.42,0.23,0.45,0.61,0.39], 
        'Other': [0.78,0.73,0.47,0.87,0.15]
    },
    'LogicBot': {
        'DeepRole': [0.52,0.23,0.45,0.61,0.39], 
        'Other': [0.78,0.23,0.47,0.27,0.15]
    },
    'ISMCTS': {
        'DeepRole': [0.12,0.23,0.35,0.61,0.39], 
        'Other': [0.78,0.23,0.47,0.67,0.15]
    },
    'MOISMCTS': {
        'DeepRole': [0.12,0.23,0.45,0.21,0.39], 
        'Other': [0.78,0.23,0.47,0.17,0.15]
    },
    'CFR': {
        'DeepRole': [0.12,0.23,0.45,0.61,0.99], 
        'Other': [0.78,0.23,0.47,0.87,0.05]
    }, 
}
label_font_size=16
# 创建绘图
fig, axs = plt.subplots(3, 5, figsize=(15, 9)) 

for i, bot_type in enumerate(bot_types):
    # 对每一行的第一列添加y轴标签
    if i == 0:
        axs[0, i].set_ylabel('P(Win)',fontsize=label_font_size)
        axs[1, i].set_ylabel('P(S Win)',fontsize=label_font_size)
        axs[2, i].set_ylabel('P(R Win)',fontsize=label_font_size)

    # 绘制Win曲线
    axs[0, i].plot(deeprole_counts, win_rates[bot_type]['DeepRole'], label='DeepRole', marker='o', color='blue')
    axs[0, i].plot(deeprole_counts, win_rates[bot_type]['Other'], label='Other', marker='o', color='orange')

    # 绘制S Win曲线
    axs[1, i].plot(deeprole_counts, s_win_rates[bot_type]['DeepRole'], marker='o', color='blue')
    axs[1, i].plot(deeprole_counts, s_win_rates[bot_type]['Other'], marker='o', color='orange')

    # 绘制R Win曲线
    axs[2, i].plot(deeprole_counts, r_win_rates[bot_type]['DeepRole'], marker='o', color='blue')
    axs[2, i].plot(deeprole_counts, r_win_rates[bot_type]['Other'], marker='o', color='orange')

    # 设置每个子图的标题
    axs[0, i].set_title(bot_type)

    # 设置图例仅在每行第一列显示
    if i == 0:
        axs[0, i].legend()

# 调整子图的布局
plt.tight_layout()

# 保存绘图
plt.savefig('battlefield/tournaments/multi_line_chart.png')

# Display the plot
plt.show()
