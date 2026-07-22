"""Stage B 单轮 GRPO（TRL GRPOTrainer + Unsloth，从 SFT checkpoint 续训）。

ASSUMPTIONS（pre-run 校准清单，手册§十——开跑前对上阶段实测数据逐条核）：
  A1 起点=SFT LoRA，其输出合法率已高（Stage A 实测 100%）→ gate 触发率应 <10%；
     冒烟若 >20% 说明加载/模板有问题，先查再放量
  A2 组内多样性只来自采样温度（默认 1.0）→ 冒烟看组内去重率 >50%；
     塌缩则升温/调 top_p（手册§六.3 观察线）
  A3 reward 有区分度：组内 std>0、全同分组占比 <30%（超线触发 DAPO 动态过滤预案）
  A4 beta=0.05 系 extract0 参照值（7B 调的）→ 看 KL 曲线再调，非最优先验
  A5 Qwen3-4B-Instruct-2507 模板无 thinking 块（2507 系纯 non-thinking）
  A6 lr 5e-6 / num_generations 8 / completion 512 为参照起点，全部参数化
  A7（#010 血泪）加载路径必须 base + get_peft_model + 灌 SFT 权重；冒烟第五标准=
     5 步后 adapter md5 ≠ SFT md5（权重真在动），md5 相同=空转立即停

五件套曲线：reward/组内std/KL/长度 由 TRL logging 输出（tee 落盘）；
字段熵+gate 分布+组内去重率 由 RewardLogger 落 runs/<run_id>/reward_detail.jsonl
（hacking 观察线实体：字段熵骤降=填高频值刷分信号）。

用法：
  python3 src/training/grpo_train.py --smoke --sft-lora outputs/sft/<run_id>/lora
  python3 src/training/grpo_train.py --sft-lora outputs/sft/<run_id>/lora [--fast-inference]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from src.reward.reward_v1 import compute_reward, extract_json_str

MAX_SEQ_LENGTH = 1536          # prompt(~500) + completion(512) 带余量
SEED = 3407


class RewardLogger:
    """TRL reward_funcs 兼容包装：算分同时收集明细，定期落盘。

    组内切片依据：TRL 按 [p0×G, p1×G, ...] 顺序传 completions（G=num_generations）。
    """

    def __init__(self, out_path: Path, num_generations: int, flush_every: int = 20):
        self.__name__ = "reward_v1"  # TRL 0.22 注册 reward_funcs 时取 .__name__，callable 实例须自带
        self.out = out_path
        self.G = num_generations
        self.flush_every = flush_every
        self.buf: list[dict] = []
        self.n_calls = 0
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, completions, **kwargs) -> list[float]:
        golds = kwargs.get("gold_extraction") or []
        rewards, details = [], []
        for i, comp in enumerate(completions):
            if isinstance(comp, list) and comp and isinstance(comp[0], dict):
                text = comp[0].get("content", "")
            elif isinstance(comp, dict):
                text = comp.get("content", "")
            else:
                text = str(comp)
            gold = golds[i] if i < len(golds) else {}
            if isinstance(gold, str):
                try:
                    gold = json.loads(gold)
                except json.JSONDecodeError:
                    gold = {}
            d = compute_reward(text, gold)
            # 字段熵观察线要的是模型输出的 task_type **值**分布（hacking=集中填高频值），
            # compute_reward 明细只有分数——logger 侧自行提取，不动 reward_v1 本体
            d["_tt"] = None
            if d["gate"] is None:
                try:
                    obj = json.loads(extract_json_str(text) or "")
                    if isinstance(obj, dict):
                        d["_tt"] = obj.get("task_type")
                except (json.JSONDecodeError, TypeError):
                    pass
            rewards.append(d["reward"])
            details.append(d)

        self.n_calls += 1
        stat = self._batch_stat(details)
        stat["call"] = self.n_calls
        stat["ts"] = time.strftime("%H:%M:%S")
        self.buf.append(stat)
        if len(self.buf) >= self.flush_every:
            self.flush()
        return rewards

    def _batch_stat(self, details: list[dict]) -> dict:
        gates = Counter(d["gate"] for d in details if d["gate"])
        # 组内去重率（A2/手册§六.3）：unique hash / 组大小，对本 batch 各组取均值
        dedup_rates = []
        for g0 in range(0, len(details) - self.G + 1, self.G):
            grp = details[g0:g0 + self.G]
            dedup_rates.append(len({d["completion_hash"] for d in grp}) / self.G)
        # 字段熵（hacking 观察线）：合法样本输出的 task_type 值分布香农熵
        tt_vals = [d["_tt"] for d in details if d.get("_tt")]
        ent = None
        if tt_vals:
            c = Counter(tt_vals)
            n = sum(c.values())
            ent = -sum((k / n) * math.log2(k / n) for k in c.values())
        rs = [d["reward"] for d in details]
        return {
            "n": len(details), "reward_mean": round(sum(rs) / max(1, len(rs)), 4),
            "gate_rate": round(sum(gates.values()) / max(1, len(details)), 3),
            "gates": dict(gates),
            "group_dedup_mean": round(sum(dedup_rates) / max(1, len(dedup_rates)), 3) if dedup_rates else None,
            "task_type_entropy": round(ent, 3) if ent is not None else None,
        }

    def flush(self) -> None:
        if not self.buf:
            return
        with open(self.out, "a", encoding="utf-8") as f:
            for row in self.buf:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.buf.clear()


def build_grpo_rows(path: str, max_prompts: int = 0) -> list[dict]:
    """sft_train.jsonl → GRPO 行：prompt=前两条消息（chat 格式），gold=assistant JSON 文本。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            m = r["messages"]
            rows.append({"prompt": m[:2], "gold_extraction": m[2]["content"]})
    if max_prompts:
        rows = rows[:max_prompts]
    return rows


def build_grpo_dataset(path: str, max_prompts: int = 0):
    from datasets import Dataset

    return Dataset.from_list(build_grpo_rows(path, max_prompts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-lora", required=True, help="Stage A 的 LoRA 目录（GRPO 起点）")
    ap.add_argument("--base-model", default="/root/autodl-tmp/models/Qwen3-4B-Instruct-2507",
                    help="基座路径（#010：须从基座 get_peft_model 再灌 SFT 权重，直接加载 LoRA 目录参数冻结）")
    ap.add_argument("--no-unsloth", action="store_true",
                    help="#010/#011：原生 TRL+PEFT 路径。unsloth 慢路 rollout 的态切换失守致参数空转，"
                         "vLLM 0.6.6 不支持 Qwen3——原生 TRL 是当前唯一可训路径。此分支严禁 import unsloth（会全局 patch trl）")
    ap.add_argument("--data", default="data/processed/sft_train.jsonl")
    ap.add_argument("--output", default="outputs/grpo")
    ap.add_argument("--max-prompts", type=int, default=2000)   # A6 参照起点，冒烟后按速率定
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8, help="per-device completion 数，须被 num-generations 整除")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.05)        # A4 extract0 参照
    ap.add_argument("--temperature", type=float, default=1.0)  # A2 组内多样性来源
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--fast-inference", action="store_true", help="vLLM rollout（默认 transformers 慢路）")
    ap.add_argument("--smoke", action="store_true", help="32 prompts + 5 步冒烟")
    args = ap.parse_args()

    if args.smoke:
        args.max_prompts = min(args.max_prompts, 32) or 32
        args.max_steps = 5

    if args.no_unsloth:
        # #011 原生 TRL+PEFT 路径（unsloth 慢路空转 + vLLM 0.6.6 无 Qwen3 后的唯一可训线）
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="cuda",
        )
        model = PeftModel.from_pretrained(model, args.sft_lora, is_trainable=True)  # is_trainable 必须显式
        model.enable_input_require_grads()  # PEFT+gradient checkpointing 组合的梯度流必需项
    else:
        from unsloth import FastLanguageModel

        # #010：不能 from_pretrained(LoRA 目录)——加载出的 adapter 是推理态冻结参数，
        # GRPOTrainer 空转 500 步（loss/KL 有值但权重逐位不变）。官方轨道：基座 + get_peft_model
        # 新建可训 adapter，再把 SFT 权重灌入（lora_B 范数 0→非零为灌入成功证据）。
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=False,
            fast_inference=args.fast_inference,
        )
        peft_cfg = json.load(open(Path(args.sft_lora) / "adapter_config.json"))
        model = FastLanguageModel.get_peft_model(
            model,
            r=peft_cfg["r"],
            lora_alpha=peft_cfg["lora_alpha"],
            lora_dropout=0.0,
            target_modules=peft_cfg["target_modules"],
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict

        sft_sd = load_file(str(Path(args.sft_lora) / "adapter_model.safetensors"))
        load_res = set_peft_model_state_dict(model, sft_sd)
        print(f"[sft-load] unexpected_keys={len(load_res.unexpected_keys)}", flush=True)

    lora_b_l1 = float(sum(p.abs().sum() for n, p in model.named_parameters() if "lora_B" in n))
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[sft-load] lora_B L1={lora_b_l1:.2f}（非零=SFT 权重在位） trainable={n_trainable:,}", flush=True)
    assert lora_b_l1 > 1.0, "SFT 权重未在位：lora_B 为零"
    assert n_trainable > 1_000_000, f"可训参数异常: {n_trainable}"

    dataset = build_grpo_dataset(args.data, args.max_prompts)
    print(f"[data] prompts={len(dataset)}", flush=True)

    from trl import GRPOConfig, GRPOTrainer

    run_id = f"grpo-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(args.output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    reward_logger = RewardLogger(out_dir / "reward_detail.jsonl", args.num_generations)

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
        use_vllm=args.fast_inference,
        gradient_checkpointing=args.no_unsloth,  # 原生路径控显存；unsloth 路径用自带 checkpointing
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_logger],
        args=config,
        train_dataset=dataset,
    )

    t0 = time.time()
    result = trainer.train()
    dt = time.time() - t0
    reward_logger.flush()

    model.save_pretrained(str(out_dir / "lora"))
    tokenizer.save_pretrained(str(out_dir / "lora"))
    summary = {
        "run_id": run_id, "sft_lora": args.sft_lora, "n_prompts": len(dataset),
        "num_generations": args.num_generations, "batch_size": args.batch_size,
        "grad_accum": args.grad_accum, "lr": args.lr, "beta": args.beta,
        "temperature": args.temperature, "fast_inference": args.fast_inference,
        "smoke": args.smoke, "train_seconds": round(dt, 1),
        "final_loss": getattr(result, "training_loss", None), "seed": SEED,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[done] GRPO LoRA -> {out_dir}/lora | 明细曲线 -> {out_dir}/reward_detail.jsonl", flush=True)


if __name__ == "__main__":
    main()
