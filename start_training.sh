#!/bin/bash
# 启动训练脚本

echo "=========================================="
echo "中国A股预测 - V45 Transformer训练"
echo "=========================================="
echo "开始时间: $(date)"
echo "GPU: NVIDIA RTX PRO 6000 (94.97 GB)"
echo "=========================================="

cd /root/.trae-cn/china-a-share-prediction

# 使用nohup后台运行训练
nohup python train_v45_stock_transformer_fixed.py > train.log 2>&1 &

# 获取进程ID
TRAIN_PID=$!

echo "训练进程已启动"
echo "PID: $TRAIN_PID"
echo ""
echo "查看日志: tail -f train.log"
echo "查看进度: tail -20 train.log"
echo "停止训练: kill $TRAIN_PID"
echo ""
echo "=========================================="
