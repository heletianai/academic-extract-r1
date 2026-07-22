"""多轮评测预测生成（五方第五格：SFT+多轮 agentic GRPO 的 holdout 推理）。

与 gen_predictions_local.py 分工：那份单轮直推（base/SFT/单轮GRPO 三方），本份跑
MultiTurnRollout 多轮检索-抽取循环——与训练同一状态机/检索件/协议。口径纪律（#012）：
system = holdout messages[0] 原文 + MT_PROTOCOL_SUFFIX，与 build_mt_rows 逐字同构；
temperature 0 贪心 = 其他四方 temp0 口径对齐；exclude_id=paper_id 防检索命中自身。

输出对接 evaluate.py：{"id","raw_text","mode","model"}，raw_text = 轨迹最后一个
<answer> 块内容（未收针则空串 = 一枪口径公平计 0）。另带行为统计列
（answered/searches/turns/invalid_actions/queries）——行为对比线一份产物两用
（训后 answered_rate/search_rate vs 未训冒烟基线 0.7/0.5）。

用法（实例上）：
  PYTHONPATH=. python3 src/eval/gen_predictions_mt.py --lora outputs/grpo_mt/<run_id>/lora
  # 行为对照（未训 SFT policy 的多轮行为，GPU 时间富余才跑）：
  PYTHONPATH=. python3 src/eval/gen_predictions_mt.py \
      --lora outputs/sft/<id>/lora --mode mt_sft
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from src.training.grpo_train_mt import MT_PROTOCOL_SUFFIX

MT_MODES = ("mt_grpo", "mt_sft")
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_last_answer(text: str) -> str:
    """多轮完整文本 → 最后一个 <answer> 块内容；无 answer 返回空串。"""
    matches = ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else ""


def build_mt_pred_messages(row: dict) -> list[dict]:
    """holdout 行 → 多轮 prompt（与 grpo_train_mt.build_mt_rows 逐字同构）。"""
    msgs = row["messages"]
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    return [
        {"role": "system", "content": msgs[0]["content"] + MT_PROTOCOL_SUFFIX},
        {"role": "user", "content": msgs[1]["content"]},
    ]


def run_predictions(rollout, rows: list[dict], out_path: Path, mode: str,
                    model_label: str, batch_size: int) -> None:
    """批量多轮 rollout → 逐行落盘（断点续传由调用方过滤 done_ids）。"""
    t0 = time.time()
    n_done = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for lo in range(0, len(rows), batch_size):
            batch = rows[lo: lo + batch_size]
            prompts = [build_mt_pred_messages(r) for r in batch]
            paper_ids = [str(r["id"]) for r in batch]
            result = rollout.run_batch(prompts, paper_ids)
            for row, text, st in zip(batch, result.completion_texts, result.stats):
                out_f.write(json.dumps({
                    "id": row["id"], "raw_text": extract_last_answer(text),
                    "mode": mode, "model": model_label,
                    "answered": st.answered, "searches": st.searches,
                    "turns": st.turns, "invalid_actions": st.invalid_actions,
                    "queries": st.queries,
                }, ensure_ascii=False) + "\n")
            out_f.flush()
            n_done += len(batch)
            dt = time.time() - t0
            print(f"[{n_done}/{len(rows)}] {dt/n_done:.1f}s/条 "
                  f"剩余≈{dt/n_done*(len(rows)-n_done)/60:.0f}min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True, help="LoRA 目录（grpo_mt 或 sft）")
    ap.add_argument("--base-model", default="/root/autodl-tmp/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gold", default="data/processed/holdout.jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--mode", default="mt_grpo", choices=MT_MODES)
    ap.add_argument("--index-dir", default="data/retrieval_index")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    # 循环参数默认锁训练值（改动即口径漂移，评测报数须注记）
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max-obs-tokens", type=int, default=400)
    ap.add_argument("--max-new-per-turn", type=int, default=320)
    ap.add_argument("--max-completion", type=int, default=1024)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    from src.retrieval.bm25_index import BM25Index
    from src.training.multiturn_loop import MultiTurnRollout

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model = PeftModel.from_pretrained(model, args.lora)  # 推理态（无需 is_trainable）
    model.eval()

    index = BM25Index.load(args.index_dir)
    print(f"[index] {len(index.docs)} docs", flush=True)

    rollout = MultiTurnRollout(
        model=model, tokenizer=tokenizer, index=index,
        max_turns=args.max_turns, topk=args.topk,
        max_obs_tokens=args.max_obs_tokens,
        max_new_tokens_per_turn=args.max_new_per_turn,
        max_total_completion=args.max_completion,
        temperature=0.0,  # 贪心，评测口径
    )

    rows = []
    with open(args.gold, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out or f"runs/pred-{args.mode}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(str(json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [r for r in rows if str(r["id"]) not in done_ids]
    print(f"mode={args.mode} lora={args.lora} gold={len(rows)} "
          f"done={len(done_ids)} todo={len(todo)}", flush=True)
    if not todo:
        return

    run_predictions(rollout, todo, out_path, args.mode, args.lora, args.batch_size)
    print(f"[done] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
