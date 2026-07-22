#!/bin/bash
# Stage C 多轮评测一条龙（执行会话#4 预置 7.22，开机后跑——新会话接手时本脚本即文档）
# 前提：GPU 实例开机；grpomt-20260722-170547 训练完成（250 步 penalty 0.8/0.3 版）
# 预算：holdout 200 多轮贪心推理 ~1h ≈¥2
set -e
cd /root/academic-extract-r1
RUN_DIR=outputs/grpo_mt/grpomt-20260722-170547
PY=/root/miniconda3/bin/python3

echo "=== ① 六标准终检：adapter md5 必须 ≠ SFT 4039f4df ==="
md5sum $RUN_DIR/lora/adapter_model.safetensors outputs/sft/sft-20260721-195436/lora/adapter_model.safetensors

echo "=== ② 多轮评测 holdout 200（temp0 贪心/max_turns 3/同 9000 池/exclude_id 自身屏蔽）==="
env PYTHONPATH=/root/academic-extract-r1 $PY src/eval/gen_predictions_mt.py \
  --lora $RUN_DIR/lora --mode mt_grpo --out runs/pred-mt_grpo.jsonl

echo "=== ③ 评分（五方第五格）==="
env PYTHONPATH=/root/academic-extract-r1 $PY -m src.eval.evaluate \
  --pred runs/pred-mt_grpo.jsonl --gold data/processed/holdout.jsonl \
  --out runs/eval-mt_grpo.json
cat runs/eval-mt_grpo.json

echo "=== ④ 行为统计（pred 文件自带列：answered/searches/turns）==="
$PY - << 'PYEOF'
import json
rows = [json.loads(l) for l in open("runs/pred-mt_grpo.jsonl") if l.strip()]
n = len(rows)
print(f"n={n} answered_rate={sum(r['answered'] for r in rows)/n:.3f} "
      f"search_rate={sum(r['searches']>0 for r in rows)/n:.3f} "
      f"mean_turns={sum(r['turns'] for r in rows)/n:.2f} "
      f"mean_searches={sum(r['searches'] for r in rows)/n:.2f}")
PYEOF

echo "[done] 数字进 run-log 五方对照表；之后：产物回传本地 → 关机 → 控制台存镜像"
