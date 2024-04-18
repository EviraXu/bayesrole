#!/usr/bin/env bash

# 定义日志文件路径
LOG_FILE="training_log.txt"

function train_section() {
    NUM_SUCCEEDS="$1"
    NUM_FAILS="$2"
    PROPOSE_COUNT="$3"

    echo "$(date) ======== Starting training process for: Succeeds=$NUM_SUCCEEDS, Fails=$NUM_FAILS, Propose=$PROPOSE_COUNT" >> "$LOG_FILE"
    echo "$(date) ==== Generating datapoints..." >> "$LOG_FILE"

    # generate datapoints
    mkdir -p deeprole_output/${NUM_SUCCEEDS}_${NUM_FAILS}_${PROPOSE_COUNT}
    ./new_generate_deeprole_data.sh $NUM_SUCCEEDS $NUM_FAILS $PROPOSE_COUNT

    DATAPOINT_COUNT=$(cat deeprole_output/${NUM_SUCCEEDS}_${NUM_FAILS}_${PROPOSE_COUNT}/* | wc -l)
    echo "$(date) ($NUM_SUCCEEDS, $NUM_FAILS, $PROPOSE_COUNT) Datapoints: $DATAPOINT_COUNT" >> "$LOG_FILE"

    # train_nn
    echo "$(date) ==== Training neural network... (Takes 30-45 minutes)" >> "$LOG_FILE"
    ./new_train_neural_network.sh $NUM_SUCCEEDS $NUM_FAILS $PROPOSE_COUNT

    echo "$(date) ==== Done with Succeeds=$NUM_SUCCEEDS, Fails=$NUM_FAILS, Propose=$PROPOSE_COUNT" >> "$LOG_FILE"

    # push git
    git add .
    git commit -m "Update: Training completion for Succeeds=$NUM_SUCCEEDS, Fails=$NUM_FAILS, Propose=$PROPOSE_COUNT"
    git push
    echo "$(date) ==== Git push done with Succeeds=$NUM_SUCCEEDS, Fails=$NUM_FAILS, Propose=$PROPOSE_COUNT" >> "$LOG_FILE"
}

ITEMS="1 2
2 0
1 1
0 2
1 0
0 1
0 0"

IFS=$'\n'
for item in $ITEMS; do
    IFS=' ' read ns nf <<< "$item"
    for i in $(seq 4 -1 0); do
        if [[ "$ns" == "2" && "$nf" == "1" && ("$i" == "4" || "$i" == "3") ]]; then
            continue
        fi
        train_section $ns $nf $i
    done
done