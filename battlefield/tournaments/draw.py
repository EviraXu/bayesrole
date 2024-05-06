import matplotlib.pyplot as plt


# X轴
deeprole_counts = [0, 1, 2, 3, 4]
# bot_types
bot_types = ['Random', 'LogicBot', 'ISMCTS', 'MOISMCTS', 'Deeprole']
#bot_types = ['Deeprole']
win_rates = {
    'Random': {
        'DeepBayes': [0,0,0,0,0],
        'Other': [0,0,0,0,0]
    },
    'LogicBot': {
        'DeepBayes': [0,0,0,0,0],
        'Other': [0,0,0,0,0]
    },
    'ISMCTS': {
        'DeepBayes': [0,0,0,0,0],
        'Other': [0,0,0,0,0]
    },
    'MOISMCTS': {
        'DeepBayes': [0,0,0,0,0],
        'Other': [0,0,0,0,0]
    },
    'Deeprole': {
        'DeepBayes': [0.52,0.55,0.50,0.49,0.41],
        'Other': [0.44,0.48,0.52,0.52,0.53]
    },
}
s_win_rates = {
    'Random': {
        'DeepBayes': [0,0,0,0,0],
        'Other': [0,0,0,0,0]
    },
    'LogicBot': {
        'DeepBayes': [0,0,0,0,0],
        'Other': [0,0,0,0,0]
    },
    'ISMCTS': {
        'DeepBayes': [0,0,0,0,0],
        'Other': [0,0,0,0,0]
    },
    'MOISMCTS': {
        'DeepBayes': [0,0,0,0,0],
        'Other': [0,0,0,0,0]
    },
    'Deeprole': {
        'DeepBayes': [0.47,0.60,0.61,0.63,0.50],
        'Other': [0.50,0.58,0.67,0.70,0.64]
    },
}
r_win_rates = {
    'Random': {
        'DeepBayes': [0,0,0,0,0],
        'Other': [0,0,0,0,0]
    },
    'LogicBot': {
        'DeepBayes': [0,0,0,0,0],
        'Other': [0,0,0,0,0]
    },
    'ISMCTS': {
        'DeepBayes': [0,0,0,0,0],
        'Other': [0,0,0,0,0]
    },
    'MOISMCTS': {
        'DeepBayes': [0,0,0,0,0],
        'Other': [0,0,0,0,0]
    },
    'Deeprole': {
        'DeepBayes': [0.56,0.51,0.42,0.39,0.35],
        'Other': [0.41,0.41,0.42,0.40,0.46]
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
    axs[0, i].plot(deeprole_counts, win_rates[bot_type]['DeepBayes'], label='DeepBayes', marker='o', color='blue')
    axs[0, i].plot(deeprole_counts, win_rates[bot_type]['Other'], label='Other', marker='o', color='orange')

    # 绘制S Win曲线
    axs[1, i].plot(deeprole_counts, s_win_rates[bot_type]['DeepBayes'], marker='o', color='blue')
    axs[1, i].plot(deeprole_counts, s_win_rates[bot_type]['Other'], marker='o', color='orange')

    # 绘制R Win曲线
    axs[2, i].plot(deeprole_counts, r_win_rates[bot_type]['DeepBayes'], marker='o', color='blue')
    axs[2, i].plot(deeprole_counts, r_win_rates[bot_type]['Other'], marker='o', color='orange')

    # 设置每个子图的标题
    axs[0, i].set_title(bot_type)

    # 设置图例仅在每行第一列显示
    if i == 0:
        axs[0, i].legend()


# 设置Y轴范围
y_axis_limits = (0, 1)  # 例如从0到1
for row in axs:
    for ax in row:
        ax.set_ylim(y_axis_limits)

        
# 调整子图的布局
plt.tight_layout()

# 保存绘图
plt.savefig('multi_line_chart.png')

# Display the plot
plt.show()
