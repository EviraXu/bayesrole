#!/usr/bin/env bash

NS="$1"
NF="$2"
PC="$3"

cd code/nn_train

python3 train.py $NS $NF $PC win_probs

python3 convert.py models/${NS}_${NF}_${PC}.h5 deeprole_models/${NS}_${NF}_${PC}.json

mkdir -p /root/DeepRole-master/deeprole/deeprole_models_v2

# 移动模型文件到指定目录
cp deeprole_models/${NS}_${NF}_${PC}.json /root/DeepRole-master/deeprole/deeprole_models_v2/