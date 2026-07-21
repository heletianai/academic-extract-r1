"""本地推理版预测生成（GPU 实例上跑，不依赖 vLLM serve——版本地狱规避件）。

与 gen_predictions.py 的分工：那份走 OpenAI 兼容端点（API/vLLM serve），这份用
Unsloth 原生 generate 直推（prompt 构造复用同一 build_pred_messages，零口径漂移）。
输出格式与 evaluate.py 对接：{"id","raw_text","mode","model"}。

用法（实例上）：
  python3 src/eval/gen_predictions_local.py --mode base_fewshot \
      --model /root/autodl-tmp/models/Qwen3-4B-Instruct-2507
  python3 src/eval/gen_predictions_local.py --mode student_zeroshot \
      --model outputs/sft/<run_id>/lora
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.eval.gen_predictions import MODES, build_pred_messages

MAX_NEW_TOKENS = 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--model", required=True, help="基座路径（base_fewshot）或 LoRA 目录（student_zeroshot）")
    ap.add_argument("--gold", default="data/processed/holdout.jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-seq-length", type=int, default=4096)  # 4-shot 长 prompt 要 >2048
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        fast_inference=False,
    )
    FastLanguageModel.for_inference(model)

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
    print(f"mode={args.mode} model={args.model} gold={len(rows)} done={len(done_ids)} todo={len(todo)}", flush=True)

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, row in enumerate(todo):
            messages = build_pred_messages(args.mode, row)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,  # non-thinking 口径（与训练一致）
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, temperature=None, top_p=None, top_k=None,  # greedy=API temp0 对齐
                pad_token_id=tokenizer.eos_token_id,
            )
            raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            out_f.write(json.dumps({
                "id": row["id"], "raw_text": raw,
                "mode": args.mode, "model": args.model,
            }, ensure_ascii=False) + "\n")
            out_f.flush()
            if (i + 1) % 20 == 0:
                dt = time.time() - t0
                print(f"[{i+1}/{len(todo)}] {dt/(i+1):.1f}s/条 剩余≈{dt/(i+1)*(len(todo)-i-1)/60:.0f}min", flush=True)
    print(f"[done] {len(todo)} preds in {time.time()-t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
