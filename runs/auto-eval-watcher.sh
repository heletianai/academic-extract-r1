#!/bin/bash
# 训练跑完自动评测+关机（执行会话#4 挂载，云端自治不依赖本地会话）。
# 前提：训练链尾 shutdown 已解除（杀 bash 父进程，python 被 init 收养继续训练），
#       收尾全权由本 watcher 接管。
# 逻辑：等训练进程退出 → lora 落盘=正常完成 → 跑评测一条龙（2h 保险丝）→ 关机；
#       lora 缺失=训练崩 → 跳过评测直接关机（现场留盘明天诊断）。
LOG=/root/academic-extract-r1/runs/auto-eval-watcher.log
RUN_DIR=/root/academic-extract-r1/outputs/grpo_mt/grpomt-20260722-170547
exec >> "$LOG" 2>&1
echo "[watcher] started $(date '+%F %T')"

while pgrep -f grpo_train_mt.py > /dev/null; do sleep 60; done
echo "[watcher] train process gone $(date '+%F %T')"
sleep 30

if [ -d "$RUN_DIR/lora" ]; then
  echo "[watcher] lora found -> eval pipeline"
  timeout 7200 bash /root/academic-extract-r1/runs/eval-mt-pipeline.sh
  echo "[watcher] eval exit=$? $(date '+%F %T')"
else
  echo "[watcher] NO lora (train crashed?) -> skip eval, keep scene"
fi

echo "[watcher] shutdown $(date '+%F %T')"
/usr/local/sbin/shutdown
