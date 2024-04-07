#!/bin/bash

# 循环运行八次
for i in $(seq 1 8); do
    # 执行 deeprole 命令，并将其放入后台执行
    echo "Start"
    ./code/deeprole -n250 -i1500 -w500 -p4 -s2 -f2 --modeldir=deeprole_models &
    
    # 获取并保存后台进程的 PID
    pids[$i]=$!
done

# 可选：等待所有后台进程结束
for pid in ${pids[*]}; do
    wait $pid
done

echo "Success"
