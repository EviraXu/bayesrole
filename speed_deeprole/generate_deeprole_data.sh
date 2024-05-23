#!/bin/bash
NS="$1"
NF="$2"
PC="$3"

# 循环运行八次
for i in $(seq 1 8); do
    # 执行 deeprole 命令，并将其放入后台执行
    ./code/deeprole -n2500 -i1500 -w500 -p$PC -s$NS -f$NF --modeldir=deeprole_models &
    
    # 获取并保存后台进程的 PID
    pids[$i]=$!
done

# 可选：等待所有后台进程结束
for pid in ${pids[*]}; do
    wait $pid
done
