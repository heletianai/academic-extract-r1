"""多轮检索-抽取 rollout（Search-R1 run_llm_loop 的单卡 TRL 化移植）。

抄件对照（refs/Search-R1/search_r1/llm_agent/generation.py）：
- 动作协议 <search>q</search> / <answer>json</answer>、</search> 截断防幻觉  ← _postprocess_responses L54
- 正则动作解析                                                        ← postprocess_predictions L407
- 非法动作纠错提示（不终止，给改正机会）                                ← execute_predictions L396
- active_mask 状态机 + turns/valid_action/is_search 统计               ← run_llm_loop L220
- <information> 包裹检索结果                                          ← execute_predictions L391

与 Search-R1 的结构差异（设计决策，口述层）：
- 不搬 veRL DataProto/TensorHelper/多 GPU padding——单卡直接张量操作
- info_mask 语义由 per-token completion_mask 承载（TRL GRPO loss 原生消费
  completion_mask，检索/纠错注入段置 0 = loss mask 三类 token 答法的第②类）
- 检索走进程内 BM25Index（训练零 HTTP 开销）；service.py 是解耦部署形态

风险防线（手册 Stage C 行）：max_turns 3-4 / obs 截断 / mask 可视化单测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from src.retrieval.bm25_index import BM25Index, format_passages

ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)

INVALID_HINT = (
    "\nMy previous action is invalid. If I want to search, I should put the query "
    "between <search> and </search>. If I want to give the final answer, I should "
    "put the answer between <answer> and </answer>. Let me try again.\n"
)


def truncate_at_action(text: str) -> str:
    """生成文本截断到第一个动作闭合标签（防模型幻觉续写检索结果）。"""
    if "</search>" in text:
        return text.split("</search>")[0] + "</search>"
    if "</answer>" in text:
        return text.split("</answer>")[0] + "</answer>"
    return text


def parse_action(text: str) -> tuple[str | None, str]:
    """→ (action, content)；action ∈ {'search','answer',None}。"""
    m = ACTION_RE.search(text)
    if not m:
        return None, ""
    return m.group(1), m.group(2).strip()


@dataclass
class TrajStats:
    turns: int = 0
    valid_actions: int = 0
    searches: int = 0
    answered: bool = False
    invalid_actions: int = 0
    queries: list = field(default_factory=list)


@dataclass
class RolloutResult:
    """一个 batch 的多轮 rollout 产物（右 pad 对齐，TRL 契约）。"""

    completion_ids: torch.Tensor      # [B, T]
    completion_mask: torch.Tensor     # [B, T] 1=模型生成(算loss) 0=注入/pad
    completion_texts: list[str]       # 完整多轮文本（给 reward/日志）
    stats: list[TrajStats]
    lengths: torch.Tensor             # [B] 各轨迹真实长度（attention 语义重建依据）


class MultiTurnRollout:
    def __init__(
        self,
        model,
        tokenizer,
        index: BM25Index,
        max_turns: int = 3,
        topk: int = 3,
        max_obs_tokens: int = 400,
        max_new_tokens_per_turn: int = 320,
        max_total_completion: int = 1024,
        temperature: float = 1.2,
    ):
        self.model = model
        self.tok = tokenizer
        self.index = index
        self.max_turns = max_turns
        self.topk = topk
        self.max_obs_tokens = max_obs_tokens
        self.max_new_per_turn = max_new_tokens_per_turn
        self.max_total = max_total_completion
        self.temperature = temperature

    # ---------- 环境步（可独立单测，不碰 GPU） ----------
    def env_step(self, gen_text: str, stats: TrajStats, exclude_id: str | None) -> tuple[str | None, bool]:
        """模型生成段 → (注入观察文本 or None, done)。观察文本 mask 恒为 0。"""
        action, content = parse_action(gen_text)
        stats.turns += 1
        if action == "answer":
            stats.valid_actions += 1
            stats.answered = True
            return None, True
        if action == "search":
            stats.valid_actions += 1
            stats.searches += 1
            stats.queries.append(content)
            hits = self.index.search(content, topk=self.topk, exclude_id=exclude_id)
            obs = format_passages(hits) if hits else "No relevant documents found."
            # 截断只作用于内容，闭合标签永远保留——硬截整块会把 </information>
            # 切掉，未闭合标签进上下文诱导模型自己补标签（SR 原版 L87 同缺陷）
            inner = self.tok(obs, add_special_tokens=False)["input_ids"][: self.max_obs_tokens]
            obs = self.tok.decode(inner)
            return f"\n\n<information>{obs}</information>\n\n", False
        stats.invalid_actions += 1
        return INVALID_HINT, False

    # ---------- 主循环 ----------
    @torch.no_grad()
    def run_batch(self, prompts: list[list[dict]], paper_ids: list[str | None] | None = None) -> RolloutResult:
        """prompts: chat 消息列表的列表；paper_ids: 逐条对齐的自身屏蔽 id。"""
        B = len(prompts)
        paper_ids = paper_ids or [None] * B
        device = self.model.device

        # 逐条累积的 completion token/mask（python list，最后一次性右 pad）
        comp_ids: list[list[int]] = [[] for _ in range(B)]
        comp_mask: list[list[int]] = [[] for _ in range(B)]
        stats = [TrajStats() for _ in range(B)]
        # 滚动上下文 = prompt 模板 ids + 已生成/注入 ids
        ctx: list[list[int]] = []
        for msgs in prompts:
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            ctx.append(list(ids))
        active = [True] * B

        for _turn in range(self.max_turns + 1):  # +1: 最后一轮只许 answer（对齐 SR final rollout）
            act_idx = [i for i in range(B) if active[i] and len(comp_ids[i]) < self.max_total]
            if not act_idx:
                break
            texts = self._generate(ctx, act_idx)
            for i, gen_text in zip(act_idx, texts):
                gen_text = truncate_at_action(gen_text)
                gen_ids = self.tok(gen_text, add_special_tokens=False)["input_ids"]
                room = self.max_total - len(comp_ids[i])
                gen_ids = gen_ids[:room]
                comp_ids[i] += gen_ids            # 模型自生成段
                comp_mask[i] += [1] * len(gen_ids)
                ctx[i] += gen_ids

                if _turn == self.max_turns:       # 最后一轮不再执行 search（do_search=False 语义）
                    action, _ = parse_action(gen_text)
                    stats[i].turns += 1
                    if action == "answer":
                        stats[i].valid_actions += 1
                        stats[i].answered = True
                    active[i] = False
                    continue

                obs_text, done = self.env_step(gen_text, stats[i], paper_ids[i])
                if done or len(comp_ids[i]) >= self.max_total:
                    active[i] = False
                    continue
                if obs_text:
                    obs_ids = self.tok(obs_text, add_special_tokens=False)["input_ids"]
                    room = self.max_total - len(comp_ids[i])
                    obs_ids = obs_ids[:room]
                    comp_ids[i] += obs_ids        # 环境注入段：mask=0（info_mask 本体）
                    comp_mask[i] += [0] * len(obs_ids)
                    ctx[i] += obs_ids
                    if len(comp_ids[i]) >= self.max_total:
                        active[i] = False

        # 右 pad 对齐（TRL completion 布局）
        T = max(1, max(len(c) for c in comp_ids))
        pad = self.tok.pad_token_id
        ids_t = torch.full((B, T), pad, dtype=torch.long)
        mask_t = torch.zeros((B, T), dtype=torch.long)
        for i in range(B):
            n = len(comp_ids[i])
            if n:
                ids_t[i, :n] = torch.tensor(comp_ids[i], dtype=torch.long)
                mask_t[i, :n] = torch.tensor(comp_mask[i], dtype=torch.long)
        texts = [self.tok.decode(c, skip_special_tokens=False) for c in comp_ids]
        lengths = torch.tensor([len(c) for c in comp_ids], dtype=torch.long)
        return RolloutResult(ids_t.to(device), mask_t.to(device), texts, stats, lengths.to(device))

    # ---------- 生成子步（active 子集，左 pad batch） ----------
    def _generate(self, ctx: list[list[int]], act_idx: list[int]) -> list[str]:
        pad = self.tok.pad_token_id
        seqs = [ctx[i] for i in act_idx]
        L = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs):        # 左 pad（生成场景标准布局）
            input_ids[r, L - len(s):] = torch.tensor(s, dtype=torch.long)
            attn[r, L - len(s):] = 1
        input_ids = input_ids.to(self.model.device)
        attn = attn.to(self.model.device)
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=self.max_new_per_turn,
            do_sample=True,
            temperature=self.temperature,
            pad_token_id=pad,
        )
        new_tokens = out[:, L:]
        return self.tok.batch_decode(new_tokens, skip_special_tokens=True)
