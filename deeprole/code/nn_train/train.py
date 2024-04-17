# -*- coding: utf-8 -*-
import tensorflow as tf
#from tensorflow.contrib.layers.python.layers import multihead_attention
from tensorflow.keras.layers import MultiHeadAttention, Dense, LayerNormalization, Embedding, Input
import numpy as np
import random
import sys

from data import load_data, MATRIX, get_payoff_vector, get_viewpoint_vector

class CFVMaskAndAdjustLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(CFVMaskAndAdjustLayer, self).__init__(*args, **kwargs)

    def call(self, inps):
        inp, cfvs = inps

        touched_cfvs = tf.matmul(inp, tf.constant(MATRIX, dtype=tf.float32))
        mask = tf.cast(tf.math.greater(touched_cfvs, tf.constant(0.0)), tf.float32)

        masked_cfvs = tf.math.multiply(cfvs, mask)
        masked_sum = tf.reduce_sum(masked_cfvs)
        mask_sum = tf.reduce_sum(mask)

        return masked_cfvs - masked_sum / mask_sum * mask

#用于实现输出总和为零
class ZeroSumLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(ZeroSumLayer, self).__init__(*args, **kwargs)

    def call(self, inp):
        print("inp", inp.shape)
        full_sum = tf.reduce_sum(inp, 1)
        print("full_sum", full_sum.shape)
        stacked = tf.stack([full_sum]*75, axis=1)
        print("stacked", stacked.shape)
        return inp - stacked/75.0

#用于根据胜率计算CFV
class CFVFromWinProbsLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(CFVFromWinProbsLayer, self).__init__(*args, **kwargs)

    def call(self, inps):
        inp_probs = inps[0][:, 5:]
        #print("INP_PROBS", inp_probs.shape)
        win_probs = inps[1]
        #print("WIN_PROBS", win_probs.shape)
        factual_values = []
        for player in range(5):
            #print("==== player", player, "====")
            good_win_payoff_for_player = tf.constant(get_payoff_vector(player), dtype=tf.float32)
            #print("good_win_payoff", good_win_payoff_for_player.shape)
            player_payoff = 2.0 * good_win_payoff_for_player * win_probs - good_win_payoff_for_player
            #print("player_payoff", player_payoff.shape)
            factual_values_for_belief_and_player = inp_probs * player_payoff
            #print("factual_values", factual_values_for_belief_and_player.shape)

            factual_values_for_player = tf.matmul(factual_values_for_belief_and_player, tf.constant(get_viewpoint_vector(player), dtype=tf.float32))
            #print("factual_values_for_player", factual_values_for_player.shape)
            factual_values.append(factual_values_for_player)

        result = tf.concat(factual_values, 1)
        #print("result", result.shape)
        return result




def loss(y_true, y_pred):
    return tf.losses.mean_squared_error(y_true, y_pred) # + tf.losses.huber_loss(y_true, y_pred) #+ tf.losses.mean_pairwise_squared_error(y_true, y_pred)


def create_model_v1():
    inp = tf.keras.layers.Input(shape=(65,))
    x = tf.keras.layers.Dense(128, activation='relu')(inp)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(5*15)(x)
    out = CFVMaskAndAdjustLayer()([inp, x])

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

def check_sum(x, y):
    return np.sum(y)

#创建transformer模型下的神经网络
def create_model_transformer():
    # 输入层
    inp = tf.keras.layers.Input(shape=(65,))
    print(inp.shape)
    # Embedding层
    x = Embedding(input_dim=65, output_dim=128)(inp)
    print(x.shape)

    # 多头自注意力层
    attention_output = MultiHeadAttention(num_heads=2, key_dim=128)(x, x)

    # 层标准化和全连接层
    x = LayerNormalization(epsilon=1e-6)(attention_output)
    print(x.shape)
    x = Dense(128, activation='relu')(x)
    print(x.shape)
    x = Dense(75)(x)
    print(x.shape)

    # Flatten层和额外的Dense层
    x = tf.keras.layers.Flatten()(x)
    print(x.shape)
    x = tf.keras.layers.Dense(75)(x)
    print(x.shape)

    model = tf.keras.models.Model(inputs=inp, outputs=x)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

def create_model_v2():
    inp = tf.keras.layers.Input(shape=(65,))
    x = tf.keras.layers.Dense(80, activation='relu')(inp)
    x = tf.keras.layers.Dense(80, activation='relu')(x)
    win_probs = tf.keras.layers.Dense(60, activation='sigmoid')(x)
    out = CFVFromWinProbsLayer()([inp, win_probs])

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model


def create_model_unconstrained():
    inp = tf.keras.layers.Input(shape=(65,))
    x = tf.keras.layers.Dense(80, activation='relu')(inp)
    x = tf.keras.layers.Dense(80, activation='relu')(x)
    x = tf.keras.layers.Dense(5*15, activation='linear')(x)
    out = ZeroSumLayer()(x)
    # out = x

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse', metrics=['mse', check_sum])
    return model

#用于训练模型并返回训练数据、标签和训练好的模型
def train(num_succeeds, num_fails, propose_count, model_type):
    #根据参数 model_type 的值创建相应的模型对象
    if model_type == 'unconstrained':
        model = create_model_unconstrained()
    elif model_type == 'win_probs':
        model = create_model_v2()
    elif model_type == 'transformer':
        model = create_model_transformer()

    #加载数据
    print("Loading data...")
    _, X, Y = load_data(num_succeeds, num_fails, propose_count)

    #设置模型检查点的回调函数，用于保存在验证集上性能最好的模型
    if model_type == 'win_probs':
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            'models/{}_{}_{}.h5'.format(num_succeeds, num_fails, propose_count),
            save_best_only=True
        )
    elif model_type == 'transformer':
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            'transformer_models/{}_{}_{}.h5'.format(num_succeeds, num_fails, propose_count),
            save_best_only=True
        )

    #设置 TensorBoard 的回调函数，用于记录训练过程的日志。
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='logs/{}_{}_{}'.format(
            num_succeeds,
            num_fails,
            propose_count
        )
    )

    print("Fitting...")
    model.fit(
        x=X,
        y=Y,
        batch_size=4096,
        epochs=3000,
        validation_split=0.1,
        callbacks=[checkpoint_callback, tensorboard_callback]
    )
    return X, Y, model

#进行模型训练和验证
def held_out_loss_data(num_succeeds, num_fails, propose_count):
    #调用 load_data 函数加载数据
    _, X, Y = load_data(num_succeeds, num_fails, propose_count)
    #对数据进行随机排列，生成一个随机的索引数组 ind
    ind = np.random.permutation(len(X))
    #对输入数据 X 和输出数据 Y 进行重新排序
    X = X[ind]
    Y = Y[ind]
    #划分验证集和训练集
    val_X, train_X = X[:20000], X[20000:]
    val_Y, train_Y = Y[:20000], Y[20000:]
    # val_X, train_X = X[:2000], X[2000:]
    # val_Y, train_Y = Y[:2000], Y[2000:]
    #进行模型训练和验证的循环
    for model_type in ['unconstrained', 'win_probs']:
        for n_datapoints in [20000, 40000, 60000, 80000, 100000]:
        #for n_datapoints in [2000, 4000, 6000, 8000, 10000]:
            # epochs = int(3000 * 10000/n_datapoints)
            # patience = int(100 * 10000/n_datapoints)
            epochs = int(3000 * 100000/n_datapoints)
            patience = int(100 * 100000/n_datapoints)
            #根据模型类型创建相应的模型对象
            if model_type == 'unconstrained':
                model = create_model_unconstrained()
            elif model_type == 'win_probs':
                model = create_model_v2()
            elif model_type == 'transformer':
                model = create_model_transformer()
            #设置 Early Stopping 的参数 patience 和 TensorBoard 的回调函数，用于在训练过程中进行早停和记录日志
            early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir='logs/{}_{}'.format(model_type, n_datapoints)
            )

            #调用模型的 fit 方法进行训练
            model.fit(
                x=train_X[:n_datapoints],
                y=train_Y[:n_datapoints],
                batch_size=4000,
                epochs=epochs,
                validation_data=(val_X, val_Y),
                callbacks=[early_stopping, tensorboard_callback]
            )


def compare(a, b):
    a = np.abs(a)
    b = np.abs(b)
    m = max(np.max(a), np.max(b))
    increment = m/20
    print("{: <20}{: >20}".format('A', 'B'))
    for i in range(len(a)):
        print("{: <20}{: >20}".format('#' * int(a[i]/increment), '#' * int(b[i]/increment)))


def random_compare(X, Y, model):
    index = int(len(X) * random.random())
    a = Y[index]
    b = model.predict(np.array([X[index]]))[0]
    compare(a, b)

if __name__ == "__main__":
    # _, num_succeeds, num_fails, propose_count = sys.argv
    # held_out_loss_data(num_succeeds, num_fails, propose_count)
    # if len(sys.argv) != 5:
    #     print("Usage:")
    #     print("python train.py <num_succeeds> <num_fails> <propose_count> <nn_type>")
    _, num_succeeds, num_fails, propose_count, nn_type = sys.argv
    train(num_succeeds, num_fails, propose_count, nn_type)