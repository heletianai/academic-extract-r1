"""reward v1 合成式单测：门控/α乘子/1:3 权重算术/TRL 签名。"""

import json

from src.reward.reward_v1 import (
    GATE_PENALTY,
    compute_reward,
    extract_json_str,
    has_duplicate_top_level_keys,
    reward_fn,
)


def EX(**kw):
    base = {
        "task_type": "classification", "modalities": ["text"],
        "benchmarks": [{"name": "MMLU", "metric": "accuracy", "value": 85.3}],
        "open_source": True, "claims_sota": False,
        "method_keywords": ["LoRA", "GRPO", "distillation"],
        "one_line_summary": "s", "limitation_mentioned": None,
    }
    base.update(kw)
    return base


def C(obj) -> str:
    return "Some preamble\n" + json.dumps(obj)


class TestGates:
    def test_no_json(self):
        r = compute_reward("no braces at all", EX())
        assert r["reward"] == GATE_PENALTY and r["gate"] == "no_json"

    def test_duplicate_keys(self):
        s = '{"task_type": "other", "task_type": "agent"}'
        assert has_duplicate_top_level_keys(s)
        r = compute_reward(s, EX())
        assert r["gate"] == "duplicate_keys"

    def test_schema_invalid_missing_key(self):
        o = EX()
        del o["claims_sota"]
        r = compute_reward(C(o), EX())
        assert r["reward"] == GATE_PENALTY and r["gate"] == "schema_invalid"

    def test_extra_key_gated(self):
        o = EX()
        o["bonus"] = 1
        assert compute_reward(C(o), EX())["gate"] == "schema_invalid"


class TestLegalLayer:
    def test_perfect_full_reward(self):
        r = compute_reward(C(EX()), EX())
        assert r["gate"] is None
        assert abs(r["reward"] - 1.0) < 1e-9
        assert r["alpha"] == 1.0

    def test_soft_violation_alpha(self):
        o = EX(method_keywords=["a", "b"])  # 数量违规 → α=0.8
        r = compute_reward(C(o), EX())
        assert r["alpha"] == 0.8
        # F_field: keywords 项掉分但其余 4 项满 → reward < 0.8 且 > 0.5
        assert 0.5 < r["reward"] < 0.8

    def test_bench_weight_3x(self):
        # benchmarks 全错、五字段全对：R = (1*1 + 3*0)/4 = 0.25
        o = EX(benchmarks=[{"name": "Fake", "metric": "x", "value": 1.0}])
        r = compute_reward(C(o), EX())
        assert abs(r["reward"] - 0.25) < 0.05  # Fake vs MMLU 可能有微小字符串相似度
        # 反向：benchmarks 全对、其余全错的情况 F_field=0 → R≈0.75
        o2 = EX(task_type="agent", modalities=["audio"], open_source=False,
                claims_sota=True, method_keywords=["x", "y", "z"])
        r2 = compute_reward(C(o2), EX())
        assert abs(r2["reward"] - 0.75) < 0.05

    def test_action_penalty_additive(self):
        r0 = compute_reward(C(EX()), EX())["reward"]
        r1 = compute_reward(C(EX()), EX(), action_penalty=-0.2)["reward"]
        assert abs((r0 - 0.2) - r1) < 1e-9


class TestTrlSignature:
    def test_str_and_chat_completions(self):
        golds = [EX(), EX()]
        comps = [C(EX()), [{"role": "assistant", "content": C(EX())}]]
        rs = reward_fn(comps, gold_extraction=golds)
        assert len(rs) == 2 and all(abs(x - 1.0) < 1e-9 for x in rs)

    def test_gold_as_json_string(self):
        rs = reward_fn([C(EX())], gold_extraction=[json.dumps(EX())])
        assert abs(rs[0] - 1.0) < 1e-9

    def test_batch_never_crashes(self):
        rs = reward_fn([None, 123, "garbage", C(EX())], gold_extraction=[EX()] * 4)
        assert len(rs) == 4
        assert rs[3] == 1.0
        assert all(isinstance(x, float) for x in rs)


class TestExtractJsonStr:
    def test_nested_and_string_braces(self):
        s = 'x {"a": {"b": "}"}, "c": 1} y'
        assert json.loads(extract_json_str(s)) == {"a": {"b": "}"}, "c": 1}
