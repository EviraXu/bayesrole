import tensorflow as tf
from train import CFVFromWinProbsLayer_v2

# 步骤 1: 加载模型
model_path = '/root/DeepRole-master/deepbayes/code/nn_train/RNN_models/2_2_4.h5'  # 修改为你的模型实际路径
model = tf.keras.models.load_model(model_path, custom_objects={'CFVFromWinProbsLayer_v2': CFVFromWinProbsLayer_v2})

# 步骤 2: 创建模型以获取每层的输入和输出
layer_data = []
for layer in model.layers:
    # 为每层创建一个模型，该模型的输入是原始模型的输入，输出是当前层的输出
    layer_model = tf.keras.models.Model(inputs=model.input, outputs=layer.output)
    layer_data.append((layer.name, layer_model))

# 步骤 3: 准备输入数据
input_data = [0.5850713, 0.1521401, 0.32881644, 0.536435, 0.19548236,
              0.80560553, 0.9057351, 0.63114804, 0.24308443, 0.8995358,
              0.58273137, 0.08613664, 0.44164747, 0.9777127, 0.6730262,
              0.3832715, 0.47151107, 0.96859837, 0.470616, 0.92077744,
              0.64998895, 0.5723444, 0.43639943, 0.4723748, 0.7397141,
              0.06065928, 0.3384151, 0.9180218, 0.14372388, 0.08235753,
              0.51579237, 0.9548887, 0.6805624, 0.15492418, 0.08350591,
              0.3799303, 0.6518955, 0.6772254, 0.63404936, 0.37938783,
              0.756582, 0.79151255, 0.42461476, 0.1434602, 0.92026955,
              0.58200324, 0.34088093, 0.40264636, 0.59304583, 0.74889493,
              0.72664607, 0.18046911, 0.63394207, 0.08368706, 0.98510534,
              0.5774527, 0.7079842, 0.86489314, 0.27096736, 0.34750062,
              0.696931, 0.24634585, 0.73587674, 0.32203126, 0.4584979]
input_tensor = tf.constant([input_data], shape=[1, 1, 65])

# 步骤 4: 预测每层的输出并打印
for layer_name, layer_model in layer_data:
    output = layer_model.predict(input_tensor)
    print(f"Layer Name: {layer_name}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}\n")
