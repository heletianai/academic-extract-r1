"""Stage A 单轮 SFT（Unsloth Qwen3-4B notebook 改造版，抄件对应见行内注）。

抄件：refs/unsloth-notebooks/nb/Kaggle-Qwen3_(4B)-GRPO.ipynb cell 0/1/8/10
适配决策（与 notebook 的差异，全部有出处）：
- 底座 Qwen3-4B-Instruct-2507 + enable_thinking=False（项目计划"统一 non-thinking"口径；
  thinking 默认开会污染 JSON 输出=已知坑；拉不到时 fallback unsloth/Qwen3-4B）
- fast_inference=False：vLLM 是 GRPO rollout 件，SFT 不需要，省显存（T4 16GB 友好）
- 不用 notebook 自定义 chat template（那是 GRPO reasoning 格式），用 Qwen3 原生模板
- train_on_responses_only：只对 assistant 段算 loss（SFT loss mask 标准做法，
  prompt/instruction token 不进梯度——面试考点"SFT 的 loss mask"实体）
- --max-samples：scaling 五点曲线（500/1k/1.5k/4k/8k）复用同一脚本

用法：
  python3 sft_train.py --smoke                          # Colab T4 冒烟（50条+20步）
  python3 sft_train.py --data data/processed/sft_train.jsonl          # 全量
  python3 sft_train.py --max-samples 1500               # scaling 点
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODEL_PRIMARY = "unsloth/Qwen3-4B-Instruct-2507"
MODEL_FALLBACK = "unsloth/Qwen3-4B"
MAX_SEQ_LENGTH = 2048          # notebook cell 1
LORA_RANK = 32                 # notebook cell 1（8/16/32/64 消融项）
SEED = 3407                    # notebook 全局 seed


def load_model(model_name: str, max_seq_length: int, lora_rank: int):
    from unsloth import FastLanguageModel

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=False,          # LoRA 16bit（notebook cell 1）
            fast_inference=False,        # SFT 不挂 vLLM（适配决策）
        )
    except Exception as e:
        if model_name == MODEL_PRIMARY:
            print(f"[fallback] {model_name} 加载失败({e})，退 {MODEL_FALLBACK}")
            return load_model(MODEL_FALLBACK, max_seq_length, lora_rank)
        raise

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],   # notebook cell 1
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    return model, tokenizer


def load_dataset(path: str, tokenizer, max_seq_length: int, max_samples: int = 0):
    from datasets import Dataset

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if max_samples:
        rows = rows[:max_samples]

    texts, n_dropped = [], 0
    for r in rows:
        text = tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=False,
            enable_thinking=False,  # non-thinking 口径（Qwen3 已知坑防线）
        )
        n_tok = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tok <= max_seq_length:   # notebook cell 8 的长度过滤，不截断只丢弃并计数
            texts.append(text)
        else:
            n_dropped += 1
    print(f"[data] kept={len(texts)} dropped_overlen={n_dropped}")
    return Dataset.from_list([{"text": t} for t in texts]), n_dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/sft_train.jsonl")
    ap.add_argument("--model", default=MODEL_PRIMARY)
    ap.add_argument("--output", default="outputs/sft")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=2.0)      # notebook cell 10
    ap.add_argument("--lr", type=float, default=2e-4)         # notebook cell 10（长训降 2e-5）
    ap.add_argument("--lora-r", type=int, default=LORA_RANK)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="50条+20步冒烟（Colab T4）")
    args = ap.parse_args()

    if args.smoke:
        args.max_samples = args.max_samples or 50

    model, tokenizer = load_model(args.model, MAX_SEQ_LENGTH, args.lora_r)
    dataset, n_dropped = load_dataset(args.data, tokenizer, MAX_SEQ_LENGTH, args.max_samples)

    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    run_id = f"sft-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(args.output) / run_id

    config = SFTConfig(                       # notebook cell 10 骨架
        output_dir=str(out_dir),
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
        num_train_epochs=args.epochs,
        max_steps=20 if args.smoke else -1,
        learning_rate=args.lr,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=SEED,
        save_strategy="no" if args.smoke else "epoch",
        report_to="none",
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                         train_dataset=dataset, args=config)

    # loss mask：只训 assistant 响应段（Qwen3 chat template 的分隔标记）
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    t0 = time.time()
    result = trainer.train()
    dt = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir / "lora"))
    tokenizer.save_pretrained(str(out_dir / "lora"))

    summary = {
        "run_id": run_id, "model": args.model, "data": args.data,
        "n_train": len(dataset), "n_dropped_overlen": n_dropped,
        "lora_r": args.lora_r, "lr": args.lr, "epochs": args.epochs,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "smoke": args.smoke, "train_seconds": round(dt, 1),
        "final_loss": result.training_loss if hasattr(result, "training_loss") else None,
        "seed": SEED,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[done] LoRA -> {out_dir}/lora")


if __name__ == "__main__":
    main()
