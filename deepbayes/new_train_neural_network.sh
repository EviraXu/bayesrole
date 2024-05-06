#!/usr/bin/env bash

NS="$1"
NF="$2"
PC="$3"

cd code/nn_train

python3 train.py $NS $NF $PC RNN

python3 convert.py RNN_models/${NS}_${NF}_${PC}.h5 deepbayes_models/${NS}_${NF}_${PC}.json

mkdir -p /root/DeepRole-master/deepbayes/deepbayes_models

# 移动模型文件到指定目录
cp deepbayes_models/${NS}_${NF}_${PC}.json /root/DeepRole-master/deepbayes/deepbayes_models