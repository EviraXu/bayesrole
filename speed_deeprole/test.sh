#!/usr/bin/env bash
LOG_FILE="test.txt"

echo "$(date) ======== Starting training process for: Succeeds=2, Fails=2, Propose=3" >> "$LOG_FILE"
echo "$(date) ==== Generating datapoints..." >> "$LOG_FILE"

mkdir -p deeprole_output/2_2_3
./code/deeprole -n2500 -i1500 -w500 -p3 -s2 -f2 --modeldir=deeprole_models -d2

DATAPOINT_COUNT=$(cat deeprole_output/2_2_3/* | wc -l)
echo "$(date) (2, 2, 3) Datapoints: $DATAPOINT_COUNT" >> "$LOG_FILE"