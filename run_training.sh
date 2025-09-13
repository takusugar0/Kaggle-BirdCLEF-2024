#!/bin/bash

# 現在の日付を取得
current_date=$(date +%Y-%m-%d)
model_name="cnn_b0ns"

nohup python train.py --stage train_ce --model_name $model_name > logs/${current_date}_${model_name}.log 2>&1 &

# バックグラウンドプロセスのIDを表示
echo "Training started with PID $!"
echo "Output is logged to logs/$current_date.log"