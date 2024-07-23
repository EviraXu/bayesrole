import pandas as pd
import os

def load_data(base_directory):
    all_data = pd.DataFrame()

    # 遍历每个文件夹，这里我们预设最多有33个文件夹
    for i in range(1, 60):
        folder_name = f"assignment_{i}"
        folder_path = os.path.join(base_directory, folder_name)

        # 遍历每个文件夹中的所有文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):  # 确保只处理CSV文件
                file_path = os.path.join(folder_path, file_name)
                # 读取CSV文件
                data = pd.read_csv(file_path)
                # 将数据追加到总的DataFrame中
                all_data = pd.concat([all_data, data], ignore_index=True)
    
    return all_data

# 使用示例
base_directory = "/root/DeepRole-master/battlefield/collect_information"
all_data = load_data(base_directory)
