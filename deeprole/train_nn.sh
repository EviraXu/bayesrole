#!/usr/bin/env bash

NS="$1"
NF="$2"
PC="$3"

cd code/nn_train

python3 train.py $NS $NF $PC

python3 convert.py models/${NS}_${NF}_${PC}.h5 exported_models/${NS}_${NF}_${PC}.json
