"""多轮 loop 单测：动作解析边界 + env_step 状态机 + info_mask 逐 token 可视化断言。

手册 Stage C 风险线："info_mask 实现 bug→observation 进 loss 训飞——
照抄 _info_masked_concatenate_with_padding + 可视化单测一条轨迹的 mask 位置"。
本文件就是那条单测：ToyTok 字符=token，mask 位置可人眼逐段核对。
"""

import torch

from src.retrieval.bm25_index import BM25Index
from src.training.multiturn_loop import (
    INVALID_HINT,
    MultiTurnRollout,
    TrajStats,
    parse_action,
    truncate_at_action,
)

# ---------- 玩具件：字符级 tokenizer + 脚本化 model ----------


class ToyTok:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, **kw):
        return "".join(chr(i) for i in ids if i != 0)

    def batch_decode(self, tensor, **kw):
        return [self.decode(row) for row in tensor.tolist()]

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=True):
        return [ord(c) for c in "P:"]


class ScriptedModel:
    """每次 generate 按剧本返回下一段文本（对 batch 内所有行同文）。"""

    device = torch.device("cpu")

    def __init__(self, script: list[str]):
        self.script = script
        self.n_call = 0

    def generate(self, input_ids, attention_mask, **kw):
        text = self.script[self.n_call]
        self.n_call += 1
        new = torch.tensor([[ord(c) for c in text]] * input_ids.shape[0], dtype=torch.long)
        return torch.cat([input_ids, new], dim=1)


def tiny_index():
    return BM25Index.build([
        {"id": "p1", "title": "GRPO for reasoning", "abstract": "group relative policy optimization"},
        {"id": "p2", "title": "BM25 retrieval", "abstract": "classic lexical search baseline"},
        {"id": "p3", "title": "Schema extraction", "abstract": "structured output from papers"},
    ])


# ---------- 纯函数边界 ----------


def test_parse_action_variants():
    assert parse_action("<search>bm25</search>") == ("search", "bm25")
    assert parse_action("x<answer>{\"a\":1}</answer>y") == ("answer", '{"a":1}')
    assert parse_action("no tags here") == (None, "")
    # 双标签取第一个（对齐 SR re.search 语义）
    assert parse_action("<search>q1</search><answer>a</answer>")[0] == "search"


def test_truncate_at_action_stops_hallucinated_obs():
    text = "<search>q</search><information>fake self-written obs</information>"
    assert truncate_at_action(text) == "<search>q</search>"
    assert truncate_at_action("<answer>{}</answer>trailing") == "<answer>{}</answer>"
    assert truncate_at_action("plain") == "plain"


# ---------- env_step 状态机 ----------


def test_env_step_three_arms():
    rollout = MultiTurnRollout(ScriptedModel([]), ToyTok(), tiny_index())
    s = TrajStats()
    obs, done = rollout.env_step("<answer>{}</answer>", s, None)
    assert done and obs is None and s.answered and s.valid_actions == 1

    s2 = TrajStats()
    obs, done = rollout.env_step("<search>lexical search</search>", s2, None)
    assert not done and "<information>" in obs and s2.searches == 1

    s3 = TrajStats()
    obs, done = rollout.env_step("garbage output", s3, None)
    assert not done and obs == INVALID_HINT and s3.invalid_actions == 1


def test_search_excludes_self():
    idx = tiny_index()
    hits = idx.search("bm25 lexical search", topk=3, exclude_id="p2")
    assert all(h["id"] != "p2" for h in hits)


# ---------- info_mask 可视化断言（核心） ----------


def test_run_batch_mask_positions():
    """轨迹剧本：search → 注入 obs → answer。断言 mask 逐段 = 1/0/1，pad=0。"""
    gen1 = "<search>grpo</search>"
    gen2 = "<answer>{}</answer>"
    tok = ToyTok()
    rollout = MultiTurnRollout(
        ScriptedModel([gen1, gen2]), tok, tiny_index(),
        max_turns=3, topk=1, max_obs_tokens=64,
    )
    res = rollout.run_batch([[{"role": "user", "content": "extract"}]], ["p3"])

    ids = res.completion_ids[0].tolist()
    mask = res.completion_mask[0].tolist()
    text = res.completion_texts[0]

    n1 = len(gen1)
    # 段1：模型生成 <search>…</search> → mask 全 1
    assert mask[:n1] == [1] * n1, f"gen1 段应全 1: {mask[:n1]}"
    # 段2：<information> 注入 → mask 全 0
    obs_start = n1
    obs_str = text[n1:text.index(gen2)]
    assert obs_str.startswith("\n\n<information>") and "</information>" in obs_str
    n_obs = len(obs_str)
    assert mask[obs_start:obs_start + n_obs] == [0] * n_obs, "information 段应全 0"
    # 段3：模型生成 <answer> → mask 回到全 1
    a_start = obs_start + n_obs
    assert mask[a_start:a_start + len(gen2)] == [1] * len(gen2), "answer 段应全 1"
    # 段4：pad → 0（ids=pad 处 mask 必 0）
    for i, t in enumerate(ids):
        if t == tok.pad_token_id:
            assert mask[i] == 0

    # stats
    st = res.stats[0]
    assert st.answered and st.searches == 1 and st.turns == 2 and st.invalid_actions == 0

    # 人眼可视化（pytest -s 查看）：逐段染色输出
    seg = "".join("█" if m else "·" for m in mask[: a_start + len(gen2)])
    print(f"\n[mask 可视化] █=算loss ·=屏蔽\n{text[: a_start + len(gen2)]}\n{seg}")


def test_run_batch_invalid_hint_masked_and_recovers():
    """非法动作 → 纠错提示注入(mask=0) → 模型改正 answer(mask=1)。"""
    bad = "I will just talk."
    good = "<answer>{}</answer>"
    rollout = MultiTurnRollout(ScriptedModel([bad, good]), ToyTok(), tiny_index(), max_turns=3)
    res = rollout.run_batch([[{"role": "user", "content": "x"}]])
    mask = res.completion_mask[0].tolist()
    n_bad = len(bad)
    n_hint = len(INVALID_HINT)
    assert mask[:n_bad] == [1] * n_bad                       # 模型的废话也算它的 loss
    assert mask[n_bad:n_bad + n_hint] == [0] * n_hint        # 纠错提示=环境注入，屏蔽
    assert res.stats[0].invalid_actions == 1 and res.stats[0].answered


def test_run_batch_max_turns_forced_stop():
    """全程只 search 不 answer → max_turns 后强制收针，answered=False（action penalty 素材）。"""
    s = "<search>q</search>"
    rollout = MultiTurnRollout(ScriptedModel([s, s, s, s]), ToyTok(), tiny_index(), max_turns=3)
    res = rollout.run_batch([[{"role": "user", "content": "x"}]])
    st = res.stats[0]
    assert not st.answered
    assert st.searches == 3  # 最后一轮 do_search=False，只算前 3 次
