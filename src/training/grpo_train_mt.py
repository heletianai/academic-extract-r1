"""Stage C 多轮 agentic GRPO 入口（原生 TRL 路径专用——unsloth 线已判死 #011）。

ASSUMPTIONS（pre-run 校准清单，手册§十）：
  A1 起点=SFT LoRA（单轮抽取已 0.90）→ 多轮协议是新行为，冒烟重点看 search_rate
     与 answered_rate（模型会不会用工具、会不会收针）
  A2 温度 1.2 沿用 Stage B 实测（去重率 0.80 平稳）
  A3 action penalty 起点 0.2/0.5/0.1（SR++"硬门控为主 penalty 为辅"），数值参数化
  A4 max_turns=3 + 最后一轮强制收针（SR final rollout 语义）
  A5 检索库=harvest 池 BM25，exclude_id 屏蔽自身（防泄漏捷径）
  A6（#010 血泪）冒烟六标准：四标准 + 第五 adapter md5 必变 + 第六 loss_mask
     有效性（info 段 token 占比>0 且被屏蔽）

用法：
  冒烟   PYTHONPATH=. python3 src/training/grpo_train_mt.py --smoke --sft-lora outputs/sft/<id>/lora
  正式   PYTHONPATH=. python3 src/training/grpo_train_mt.py --sft-lora outputs/sft/<id>/lora
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MAX_SEQ_LENGTH = 2560   # prompt(~700 多轮协议) + completion(1024) + 检索注入余量
SEED = 3407

MT_SYSTEM_PROMPT = """Extract paper metadata as a JSON object with exactly these fields: task_type (classification|generation|retrieval|reasoning|multimodal|agent|other), modalities (list: text|image|audio|video|code), benchmarks (list of {name, metric, value}), open_source (bool), claims_sota (bool), method_keywords (3-5 short phrases), one_line_summary (string).

You may use a search tool to verify uncertain fields (especially benchmarks and claims_sota) against related papers:
- To search: <search>your query</search> — you will receive related paper abstracts in <information>...</information>.
- To answer: <answer>{...the JSON object...}</answer>
Act directly without lengthy reasoning. Search at most a few times, then give your final answer inside <answer> tags."""


def build_mt_rows(path: str, max_prompts: int = 0) -> list[dict]:
    """sft_train.jsonl → 多轮行：system 换协议版，gold/paper_id 带出。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            m = r["messages"]
            rows.append({
                "prompt": [
                    {"role": "system", "content": MT_SYSTEM_PROMPT},
                    m[1],                                  # user: 摘要原文
                ],
                "gold_extraction": m[2]["content"],
                "paper_id": r.get("id"),
            })
    if max_prompts:
        rows = rows[:max_prompts]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-lora", required=True)
    ap.add_argument("--base-model", default="/root/autodl-tmp/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", default="data/processed/sft_train.jsonl")
    ap.add_argument("--index-dir", default="data/retrieval_index")
    ap.add_argument("--output", default="outputs/grpo_mt")
    ap.add_argument("--max-prompts", type=int, default=1000)   # 多轮每步慢 3-4 倍，规模减半起步
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=1.2)   # A2
    ap.add_argument("--max-turns", type=int, default=3)         # A4
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max-obs-tokens", type=int, default=400)
    ap.add_argument("--max-new-per-turn", type=int, default=320)
    ap.add_argument("--max-completion", type=int, default=1024)
    ap.add_argument("--penalty-no-search", type=float, default=0.2)   # A3
    ap.add_argument("--penalty-no-answer", type=float, default=0.5)
    ap.add_argument("--penalty-invalid", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--smoke", action="store_true", help="16 prompts + 5 步冒烟")
    args = ap.parse_args()

    if args.smoke:
        args.max_prompts = min(args.max_prompts, 16) or 16
        args.max_steps = 5

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model = PeftModel.from_pretrained(model, args.sft_lora, is_trainable=True)  # #010: is_trainable 必须显式
    model.enable_input_require_grads()

    lora_b_l1 = float(sum(p.abs().sum() for n, p in model.named_parameters() if "lora_B" in n))
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[sft-load] lora_B L1={lora_b_l1:.2f} trainable={n_trainable:,}", flush=True)
    assert lora_b_l1 > 1.0 and n_trainable > 1_000_000

    from datasets import Dataset

    dataset = Dataset.from_list(build_mt_rows(args.data, args.max_prompts))
    print(f"[data] prompts={len(dataset)}", flush=True)

    from src.retrieval.bm25_index import BM25Index
    from src.training.multiturn_loop import MultiTurnRollout
    from src.training.grpo_multiturn import MultiTurnGRPOTrainer
    from trl import GRPOConfig

    index = BM25Index.load(args.index_dir)
    print(f"[index] {len(index.docs)} docs", flush=True)

    run_id = f"grpomt-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(args.output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = GRPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=MAX_SEQ_LENGTH - args.max_completion,
        max_completion_length=args.max_completion,
        temperature=args.temperature,
        beta=args.beta,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=5,
        max_steps=args.max_steps,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=100,
        save_strategy="no" if args.smoke else "steps",
        optim="adamw_8bit",
        seed=SEED,
        report_to="none",
        use_vllm=False,
        gradient_checkpointing=True,
    )

    def dummy_reward(completions, **kw):  # 满足父类签名，覆写后永不触发（#008: 带 __name__）
        raise RuntimeError("不应被调用：打分在 _generate_and_score_completions 覆写内")
    dummy_reward.__name__ = "reward_v1_mt"

    trainer = MultiTurnGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[dummy_reward],
        args=config,
        train_dataset=dataset,
        penalty_no_search=args.penalty_no_search,
        penalty_no_answer=args.penalty_no_answer,
        penalty_invalid=args.penalty_invalid,
        detail_path=str(out_dir / "reward_detail.jsonl"),
    )
    trainer.attach_rollout(MultiTurnRollout(
        model=None,  # 每次生成前由 trainer 注入 unwrap 后引用
        tokenizer=tokenizer,
        index=index,
        max_turns=args.max_turns,
        topk=args.topk,
        max_obs_tokens=args.max_obs_tokens,
        max_new_tokens_per_turn=args.max_new_per_turn,
        max_total_completion=args.max_completion,
        temperature=args.temperature,
    ))

    t0 = time.time()
    result = trainer.train()
    dt = time.time() - t0

    model.save_pretrained(str(out_dir / "lora"))
    tokenizer.save_pretrained(str(out_dir / "lora"))
    summary = {
        "run_id": run_id, "sft_lora": args.sft_lora, "n_prompts": len(dataset),
        "num_generations": args.num_generations, "lr": args.lr, "beta": args.beta,
        "temperature": args.temperature, "max_turns": args.max_turns, "topk": args.topk,
        "penalties": [args.penalty_no_search, args.penalty_no_answer, args.penalty_invalid],
        "smoke": args.smoke, "train_seconds": round(dt, 1),
        "final_loss": getattr(result, "training_loss", None), "seed": SEED,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[done] MT-GRPO LoRA -> {out_dir}/lora | 明细 -> {out_dir}/reward_detail.jsonl", flush=True)


if __name__ == "__main__":
    main()
