"""MultiTurnGRPOTrainer——TRL 0.22.2 的多轮 agentic 接线（施工图 docs/stage-c-trl-wiring.md）。

双 mask 分离（本文件的存在理由）：
- completion_mask = attention 语义（检索段对 forward 可见）——保证训练 logprob
  口径与 rollout 真实上下文一致，否则重要性采样 ratio 系统性有偏
- info_mask = loss 语义（模型生成=1 / 检索与纠错注入=0 / pad=0）——只对模型
  自生成 token 求梯度（Search-R1 retrieved token masking 的 TRL 化）

版本锁定：_compute_loss 复制自 trl==0.22.2 并做 loss_mask 替换，升 trl 前必须重对。
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import torch
from trl import GRPOTrainer
from trl.trainer.utils import nanmax, nanmin

from src.reward.reward_v1 import compute_reward
from src.training.multiturn_loop import MultiTurnRollout

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def last_answer_content(text: str) -> str:
    """多轮全文 → 最后一个 <answer> 内容（reward 只对它打分，防检索段 JSON 污染）。"""
    hits = ANSWER_RE.findall(text)
    return hits[-1].strip() if hits else ""


class MultiTurnGRPOTrainer(GRPOTrainer):
    """覆写生成与 loss：多轮 rollout + 双 mask 分离。reward 打分在生成覆写内完成，
    不走父类 reward_funcs 通道（构造时传占位函数满足签名）。"""

    def __init__(
        self,
        *args,
        penalty_no_search: float = 0.2,
        penalty_no_answer: float = 0.5,
        penalty_invalid: float = 0.1,
        detail_path: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rollout: MultiTurnRollout | None = None  # attach_rollout 注入
        self.penalty_no_search = penalty_no_search
        self.penalty_no_answer = penalty_no_answer
        self.penalty_invalid = penalty_invalid
        self.detail_path = Path(detail_path) if detail_path else None
        self._detail_call = 0
        if self.detail_path:
            self.detail_path.parent.mkdir(parents=True, exist_ok=True)

    def attach_rollout(self, rollout: MultiTurnRollout) -> None:
        self.rollout = rollout

    # ------------------------------------------------------------------
    # 覆写 1：多轮生成 + 打分（原版 400 行巨石的单卡 transformers 直线版）
    # ------------------------------------------------------------------
    def _generate_and_score_completions(self, inputs):
        assert self.rollout is not None, "先 attach_rollout()"
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        golds = [x.get("gold_extraction") for x in inputs]
        paper_ids = [x.get("paper_id") for x in inputs]

        # --- prompt ids（左 pad，与 rollout 内 apply_chat_template 同模板同参） ---
        prompts_text = [
            self.processing_class.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
            for p in prompts
        ]
        pt = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True,
            padding_side="left", add_special_tokens=False,
        )
        prompt_ids = pt["input_ids"].to(device)
        prompt_mask = pt["attention_mask"].to(device)
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # --- 多轮 rollout（no_grad 在 run_batch 内；eval 态生成防 dropout） ---
        self.rollout.model = self.accelerator.unwrap_model(self.model)
        was_training = self.model.training
        self.model.eval()
        res = self.rollout.run_batch(prompts, paper_ids)
        if was_training:
            self.model.train()

        completion_ids = res.completion_ids.to(device)
        info_mask = res.completion_mask.to(device)  # loss 语义（rollout 侧命名）
        T = completion_ids.size(1)
        positions = torch.arange(T, device=device).unsqueeze(0)
        completion_mask = (positions < res.lengths.to(device).unsqueeze(1)).int()  # attention 语义

        # --- reward = validator 合成式 + action penalty 层（总纲§三③激活） ---
        rewards, gates = [], []
        for text, gold, st in zip(res.completion_texts, golds, res.stats):
            gold_obj = gold if isinstance(gold, dict) else (json.loads(gold) if gold else {})
            d = compute_reward(last_answer_content(text), gold_obj)
            r = d["reward"]
            if st.answered and st.searches == 0:
                r -= self.penalty_no_search
            if not st.answered:
                r -= self.penalty_no_answer
            r -= self.penalty_invalid * st.invalid_actions
            rewards.append(r)
            gates.append(d["gate"])
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

        # --- 组内 advantage（对齐 TRL scale_rewards=True 口径） ---
        G = self.num_generations
        grouped = rewards_t.view(-1, G)
        mean_g = grouped.mean(dim=1, keepdim=True)
        std_g = grouped.std(dim=1, keepdim=True)
        advantages = ((grouped - mean_g) / (std_g + 1e-4)).view(-1)

        # --- ref logps（beta>0）：PEFT disable_adapter → ref=裸基座。
        #     KL 语义=对基座的偏移（与 Stage B 同口径，#010 教训：解读时记住） ---
        ref_per_token_logps = None
        if self.beta != 0.0:
            input_ids_full = torch.cat([prompt_ids, completion_ids], dim=1)
            attn_full = torch.cat([prompt_mask, completion_mask], dim=1)
            unwrapped = self.accelerator.unwrap_model(self.model)
            with torch.no_grad(), unwrapped.disable_adapter():
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model, input_ids_full, attn_full, T, compute_entropy=False,
                )

        # --- 五件套 metrics + 多轮明细 jsonl（监控三线 + 六标准数据源） ---
        self._metrics[mode]["reward"].append(rewards_t.mean().item())
        self._metrics[mode]["reward_std"].append(std_g.mean().item())
        self._metrics[mode]["completions/mean_length"].append(info_mask.sum(1).float().mean().item())
        self._metrics[mode]["multiturn/search_rate"].append(
            sum(1 for s in res.stats if s.searches > 0) / len(res.stats))
        self._metrics[mode]["multiturn/answered_rate"].append(
            sum(1 for s in res.stats if s.answered) / len(res.stats))
        if self.detail_path:
            self._detail_call += 1
            gate_ct = Counter(g for g in gates if g)
            dedup = []
            for g0 in range(0, len(res.completion_texts) - G + 1, G):
                grp = res.completion_texts[g0:g0 + G]
                dedup.append(len(set(grp)) / G)
            tt_vals = []
            for text in res.completion_texts:
                try:
                    obj = json.loads(last_answer_content(text))
                    if isinstance(obj, dict) and obj.get("task_type"):
                        tt_vals.append(obj["task_type"])
                except (json.JSONDecodeError, TypeError):
                    pass
            ent = None
            if tt_vals:
                c = Counter(tt_vals)
                n = sum(c.values())
                ent = -sum((k / n) * math.log2(k / n) for k in c.values())
            row = {
                "n": len(rewards), "reward_mean": round(rewards_t.mean().item(), 4),
                "gate_rate": round(sum(1 for g in gates if g) / len(gates), 3),
                "gates": dict(gate_ct),
                "group_dedup_mean": round(sum(dedup) / max(1, len(dedup)), 3) if dedup else None,
                "task_type_entropy": round(ent, 3) if ent is not None else None,
                "search_rate": round(sum(1 for s in res.stats if s.searches > 0) / len(res.stats), 3),
                "answered_rate": round(sum(1 for s in res.stats if s.answered) / len(res.stats), 3),
                "mean_turns": round(sum(s.turns for s in res.stats) / len(res.stats), 2),
                "call": self._detail_call, "ts": time.strftime("%H:%M:%S"),
            }
            with open(self.detail_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        out = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "info_mask": info_mask,
            "advantages": advantages,
            "num_items_in_batch": info_mask.sum(),
        }
        if ref_per_token_logps is not None:
            out["ref_per_token_logps"] = ref_per_token_logps
        return out

    # ------------------------------------------------------------------
    # 覆写 2：_compute_loss——复制 trl 0.22.2 原体，attention 保持 completion_mask，
    # 所有 loss 加权/聚合/metrics 换 loss_mask = completion_mask * info_mask
    # ------------------------------------------------------------------
    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        loss_mask = completion_mask * inputs["info_mask"]  # ← 本覆写唯一新增物

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # attention：检索段可见
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, loss_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * loss_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * loss_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        mode = "train" if self.model.training else "eval"
        loss_token_count = loss_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # sequence 级 importance sampling
                return x.mean()
            return (x * loss_mask).sum() / loss_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss
