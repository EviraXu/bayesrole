import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('/root/DeepRole-master/battlefield/tournaments/win_rates/Deeprole_Deeprole_assignment.csv')  # 确保这里的路径指向您的CSV文件

# 创建画布，设置大小
plt.figure(figsize=(18,4))

# 总体胜率图
plt.subplot(1, 3, 1)  # 3行1列的第一个
plt.plot(range(len(df)), df['Y_Overall_Win_Rates'], marker='o', color='orange', label='+Deeprole_assignment')
plt.plot(range(len(df)), df['X_Overall_Win_Rates'], marker='o', color='blue', label='+Deeprole')
plt.title('Overall Win Rates')
# plt.xlabel('# DeepRole')
plt.ylabel('P(Win)')
plt.ylim(0, 1)
plt.xticks(range(0, 5)) 
plt.legend()

# 正方角色胜率图
plt.subplot(1, 3, 2)  # 3行1列的第二个
plt.plot(range(len(df)), df['Y_Positive_Win_Rates'], marker='o', color='orange', label='+Deeprole_assignment (S)')
plt.plot(range(len(df)), df['X_Positive_Win_Rates'], marker='o', color='blue', label='+Deeprole (S)')
plt.title('Positive Role Win Rates')
# plt.xlabel('# DeepRole')
plt.ylabel('P(S Win)')
plt.ylim(0, 1)
plt.xticks(range(0, 5)) 
plt.legend()

# 反方角色胜率图
plt.subplot(1, 3, 3)  # 3行1列的第三个
plt.plot(range(len(df)), df['Y_Negative_Win_Rates'], marker='o', color='orange', label='+Deeprole_assignment (R)')
plt.plot(range(len(df)), df['X_Negative_Win_Rates'], marker='o', color='blue', label='+Deeprole (R)')
plt.title('Negative Role Win Rates')
# plt.xlabel('# DeepRole')
plt.ylabel('P(R Win)')
plt.ylim(0, 1)
plt.xticks(range(0, 5)) 
plt.legend()

plt.savefig('Deeprole_assignment_Deeprole.png')
# 调整布局并显示图表
plt.tight_layout()
plt.show()

#python3 draw_one_chart.py
