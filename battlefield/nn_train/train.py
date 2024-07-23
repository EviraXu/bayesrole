import pandas as pd
import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense,Masking,Dropout
from keras.callbacks import TensorBoard
from tensorflow.keras.layers import Layer
import tensorflow as tf

#加载数据
def load_data(base_directory):
    combined_cache_filename = os.path.join(base_directory, 'all_combined.npy')

    # 如果已经存在合并后的文件，则直接加载并返回
    if os.path.isfile(combined_cache_filename):
        print(f"Loading combined data from {combined_cache_filename}")
        return np.load(combined_cache_filename)

    # 如果不存在合并后的文件，开始处理每个文件夹
    all_data_arrays = []

    # 处理 base_directory 文件夹中的所有 CSV 文件
    base_directory_arrays = []
    for file_name in glob.glob(os.path.join(base_directory, '*.csv')):
        data = np.loadtxt(fname=file_name, delimiter=',')
        base_directory_arrays.append(data)

    if base_directory_arrays:
        base_directory_concatenated = np.concatenate(base_directory_arrays)
        all_data_arrays.append(base_directory_concatenated)

    #收集60个文件夹的数据
    # for i in range(1, 61):  # 假设从1到60的文件夹
    #     folder_name = f"assignment_{i}"
    #     folder_path = os.path.join(base_directory, folder_name)
    #     cache_filename = os.path.join(folder_path, 'combined.npy')

    #     # 检查是否已经有缓存的numpy文件
    #     if os.path.isfile(cache_filename):
    #         concatenated = np.load(cache_filename)
    #     else:
    #         arrays = []
    #         for file_name in glob.glob(os.path.join(folder_path, '*.csv')):
    #             data = np.loadtxt(fname=file_name, delimiter=',')
    #             arrays.append(data)

    #         if arrays:
    #             concatenated = np.concatenate(arrays)
    #             np.save(cache_filename, concatenated)  # 保存每个文件夹的合并数据
    #         else:
    #             concatenated = np.array([])

    #     all_data_arrays.append(concatenated)
    
    for i, array in enumerate(all_data_arrays):
        print(f"Array {i} has shape: {array.shape}")

    # 合并所有文件夹的数据到一个数组
    if all_data_arrays:
        all_data_combined = np.concatenate(all_data_arrays)
        np.save(combined_cache_filename, all_data_combined)  # 保存总的合并数据
        print(f"Combined data saved to {combined_cache_filename}")
    else:
        all_data_combined = np.array([])

    return all_data_combined

#处理数据
def preprocess_data(data):
    x = data[:, :-1]  # 使用 NumPy 切片来选择除了最后一列的所有列作为特征
    y = data[:, -1]   # 最后一列作为目标标签

    # 将目标变量转换为one-hot编码
    y = to_categorical(y, num_classes=60)

    # 标准化特征
    # scaler = StandardScaler()  # 或 MinMaxScaler()
    # X_scaled = scaler.fit_transform(X)

    # 调整数据以适应LSTM输入格式 [samples, time steps, features]
    #print(X_scaled.shape[0])
    #print(X_scaled.shape[1])
    X_scaled = x.reshape(x.shape[0], 25, 19)

    return X_scaled, y

class LayerOutputCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, validation_data, log_file="layer_output_log.txt"):
        super(LayerOutputCallback, self).__init__()
        self.model = model
        self.log_file = log_file
        self.validation_data = validation_data
        
        # Clear the log file by opening it in write mode
        with open(self.log_file, 'w') as f:
            pass

    def on_epoch_end(self, epoch, logs=None):
        batch = self.validation_data[0]
        
        # Select a batch of input data
        with open(self.log_file, 'a') as f:
            f.write(f"\nlogs keys: {list(logs.keys())}\n")
            f.write(f"\nEpoch {epoch + 1}\n")
        
        # Log the initial input data
        with open(self.log_file, 'a') as f:
            f.write("Initial input data (first example):\n")
            f.write(f"{batch[0]}\n\n")
        
        # Iterate over layers and get their outputs
        for layer in self.model.layers:
            if 'input' not in layer.name.lower():
                intermediate_model = tf.keras.models.Model(inputs=self.model.input, outputs=layer.output)
                intermediate_output = intermediate_model.predict(batch)
                
                with open(self.log_file, 'a') as f:
                    f.write(f"Layer: {layer.name}\n")
                    f.write(f"Output shape: {intermediate_output.shape}\n")
                    f.write(f"Output data (first example):\n{intermediate_output[0]}\n\n")
        
        # Log the output of the input layer (if there's a layer named 'input')
        for layer in self.model.layers:
            if 'input' in layer.name.lower():
                input_layer_output = intermediate_model.predict(batch)
                with open(self.log_file, 'a') as f:
                    f.write(f"Input layer output:\n{input_layer_output[0]}\n\n")
                break

class StatisticsLayer(Layer):
    def __init__(self, **kwargs):
        super(StatisticsLayer, self).__init__(**kwargs)

    def call(self, inputs):
        input_data = inputs
        #print("input_data:", input_data.shape)
        #probs = inputs[1]
        #print("probs:", probs.shape)
        #batch_size = tf.shape(input_data)[0]
        return input_data

def get_possible_assignments(proposal, assignments):
    log_file = "file://output_log.txt"
    
    #possible_assignments = []
    possible_assignments = tf.constant([], dtype=tf.int64)

    
    # 识别proposal中参与活动的玩家的索引
    active_players = tf.where(proposal[0] == 1)[:, 0]
    tf.print("active_players:", active_players, output_stream=log_file, summarize=-1)

    for assignment in assignments:
        # 获取assignment的第一列和第二列的玩家
        first_col_players = tf.convert_to_tensor(assignment[-2:], dtype=tf.int64)
        tf.print("first_col_players:", first_col_players, output_stream=log_file, summarize=-1)
        # 检查 active_players 是否在 first_col_players 中
        active_in_first_col = tf.reduce_any(tf.equal(tf.expand_dims(active_players, 0), tf.expand_dims(first_col_players, 1)), axis=1)
        tf.print("active_in_first_col:", active_in_first_col, output_stream=log_file, summarize=-1)
        
        tf.print("assignment:", assignment, output_stream=log_file, summarize=-1)
        result = tf.reduce_all(tf.logical_not(active_in_first_col))
        tf.print("result:", result, output_stream=log_file, summarize=-1)

        # 如果没有活跃玩家在第一列和第二列中
        if result:
            tf.print("succeed before possible_assignments:", possible_assignments, output_stream=log_file, summarize=-1)
            tf.print("succeed assignment:", assignment, output_stream=log_file, summarize=-1)
            #possible_assignments.append(assignment)
            # 将 assignment 添加到 possible_assignments
            assignment_tensor = tf.convert_to_tensor([assignment], dtype=tf.int64)
            possible_assignments = tf.concat([possible_assignments, assignment_tensor], axis=0)
            #print("possible_assignments:",possible_assignments)
            tf.print("succeed possible_assignments:", possible_assignments, output_stream=log_file, summarize=-1)
        else:
            tf.print("fail beforepossible_assignments:", possible_assignments, output_stream=log_file, summarize=-1)
            tf.print("fail assignment:", assignment, output_stream=log_file, summarize=-1)

    return possible_assignments

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

class MaskImpossibleClassesLayer(Layer):
    def __init__(self, **kwargs):
        super(MaskImpossibleClassesLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # 打开文件以写入
        log_file = "file://output_log.txt"

        logits = inputs[0]
        #tf.print("logits:", logits, output_stream=log_file)


        input_data = inputs[1]
        tf.print("values of input_data:", input_data, output_stream=log_file, summarize=-1)

        #将input_data中每个batch_size中为第三列第18个数为0的这列数据的第7到第12个数据挑出来记为proposal_play，每行是被挑出来的五个数据
        last_column = input_data[:, :, -1]
        tf.print("last_column:", last_column, output_stream=log_file, summarize=-1)

        # 创建布尔掩码，标识每行最后一个元素是否为0
        mask = tf.equal(last_column, 0)

        # 使用布尔掩码选择符合条件的行
        filtered_rows = tf.boolean_mask(input_data, mask)
        tf.print("filtered_rows:", filtered_rows, output_stream=log_file, summarize=-1)

        # 提取每行的第8到第12个数据
        proposal = filtered_rows[:, 7:12]
        tf.print("proposal:", proposal, output_stream=log_file, summarize=-1)

        #proposal中的全是正方的可能的角色分配概率为0
        possible_assignments = get_possible_assignments(proposal, ASSIGNMENT_TO_ROLES)
        #tf.print("possible_assignments:", possible_assignments, output_stream=log_file, summarize=-1)

        mask=1
        masked_logits = logits + (mask * -1e9)  # 将不可能的类的logit值变得非常小
        return masked_logits

def build_model(input_shape):
    # 定义输入层
    inp = tf.keras.layers.Input(shape=input_shape)
    # 使用自定义层处理
    #inp = StatisticsLayer()(inp)
    # 原始LSTM层
    x = Masking(mask_value=-1.0, input_shape=input_shape)(inp)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = LSTM(128, return_sequences=False)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)

    #1不做处理
    #probs = Dense(60, activation='softmax')(x)

    #2自定义层处理
    logits = Dense(60, activation='softmax')(x)
    # 使用自定义层将不可能的类别的logit值设置为非常小
    masked_logits = MaskImpossibleClassesLayer()([logits, inp])
    # 最后应用softmax
    probs = tf.keras.layers.Activation('softmax')(masked_logits)

    # 定义和编译模型
    model = tf.keras.models.Model(inputs=inp, outputs=probs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
    #layer_output_callback = LayerOutputCallback(model, validation_data=(X_test, y_test), log_file='layer_output_log.txt')
    history = model.fit(
        X_train, 
        y_train, 
        epochs=10, 
        batch_size=128, 
        validation_data=(X_test, y_test), 
        verbose=1, 
        callbacks=[tensorboard_callback]
    )
    model.save('model_masking_128_d2.h5')  # 保存模型
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy*100:.2f}%")
    return history

def predict(input_data):
    # 确认输入数据的形状是否正确
    if input_data.shape[1:] != (25, 19):
        raise ValueError("输入数据的形状必须是 (任意数量, 25, 19)")
    
    model = load_model('/root/DeepRole-master/battlefield/nn_train/my_model.h5')
    predictions = model.predict(input_data)
    #print(predictions)
    return predictions
    #找到最大的预测结果
    # predicted_class = np.argmax(predictions, axis=1)
    # return predicted_class


def main():
    #base_directory = "/root/DeepRole-master/battlefield/collect_information"
    base_directory = "/root/DeepRole-master/battlefield/test_data"
    data = load_data(base_directory)
    #print("data:",data)
    X_scaled, y = preprocess_data(data)
    #print("X_scaled:",X_scaled)
    #print("y:",y)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    # print("X_train.shape[1]:",X_train.shape[1])
    # print("X_train.shape[2]:",X_train.shape[2])
    model = build_model((X_train.shape[1], X_train.shape[2]))
    history = train_and_evaluate(model, X_train, y_train, X_test, y_test)

    # input_data = np.random.rand(1, 25, 19)
    # predictions = predict(input_data)
    # #print("Predicted class:", predicted_class)

if __name__ == "__main__":
    #python3 train.py
    main()






