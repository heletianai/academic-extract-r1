"""field_scores + evaluate 端到端单测。"""

import json

from src.eval.evaluate import _extract_json, bootstrap_ci, evaluate
from src.eval.field_scores import f_field, score_fields


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


class TestScoreFields:
    def test_perfect(self):
        s = score_fields(EX(), EX())
        assert all(s[k] == 1.0 for k in ("task_type", "modalities", "open_source",
                                          "claims_sota", "benchmarks_hard", "method_keywords"))
        assert f_field(s) == 1.0

    def test_task_type_case_insensitive(self):
        s = score_fields(EX(task_type="Classification"), EX())
        assert s["task_type"] == 1.0

    def test_modalities_partial(self):
        s = score_fields(EX(modalities=["text", "image"]), EX(modalities=["text"]))
        # P=1/2, R=1/1 → F1=2/3
        assert abs(s["modalities"] - 2 / 3) < 1e-6

    def test_bool_flip(self):
        s = score_fields(EX(claims_sota=True), EX())
        assert s["claims_sota"] == 0.0

    def test_dirty_pred_no_crash(self):
        s = score_fields({}, EX())
        assert s["benchmarks_hard"] == 0.0 and s["task_type"] == 0.0


class TestExtractJson:
    def test_json_in_prose(self):
        obj = _extract_json('Sure! Here is the result: {"a": 1, "b": {"c": "x}"}} extra')
        assert obj == {"a": 1, "b": {"c": "x}"}}

    def test_no_json(self):
        assert _extract_json("no braces here") is None


class TestBootstrapCI:
    def test_degenerate_constant(self):
        lo, hi = bootstrap_ci([1.0] * 50)
        assert lo == hi == 1.0

    def test_contains_mean(self):
        vals = [0.0, 1.0] * 25
        lo, hi = bootstrap_ci(vals)
        assert lo < 0.5 < hi


class TestEvaluateEndToEnd:
    def test_mixed_predictions(self, tmp_path):
        gold_p = tmp_path / "gold.jsonl"
        pred_p = tmp_path / "pred.jsonl"
        with open(gold_p, "w") as f:
            for i in range(3):
                f.write(json.dumps({"id": str(i), "extraction": EX()}) + "\n")
        with open(pred_p, "w") as f:
            # 0: 完美；1: raw_text 带噪声但可提取；2: 缺失（不写）
            f.write(json.dumps({"id": "0", "extraction": EX()}) + "\n")
            f.write(json.dumps({"id": "1", "raw_text": "junk " + json.dumps(EX())}) + "\n")
        r = evaluate(str(pred_p), str(gold_p))
        assert r["n"] == 3
        assert abs(r["valid_json_rate"] - 2 / 3) < 1e-3  # report 值经 round(4)
        assert r["fields"]["task_type"]["mean"] < 1.0  # 缺失那条拉低
        assert r["overall"]["ci95"][0] <= r["overall"]["mean"] <= r["overall"]["ci95"][1]

    def test_holdout_messages_format_as_gold(self, tmp_path):
        # gold 直接用 to_sft_format 产出的 messages 结构也能评
        gold_p = tmp_path / "gold.jsonl"
        pred_p = tmp_path / "pred.jsonl"
        msgs = {"id": "7", "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": json.dumps(EX())},
        ]}
        gold_p.write_text(json.dumps(msgs) + "\n")
        pred_p.write_text(json.dumps({"id": "7", "extraction": EX()}) + "\n")
        r = evaluate(str(pred_p), str(gold_p))
        assert r["overall"]["mean"] == 1.0


class TestRedTeamFixes:
    def test_bad_gold_line_does_not_crash(self, tmp_path):
        # P1：一条坏 gold（messages 尾部非 JSON / 缺 content）不许崩整个 eval
        gold_p, pred_p = tmp_path / "g.jsonl", tmp_path / "p.jsonl"
        rows = [
            {"id": "1", "extraction": EX()},
            {"id": "2", "messages": [{"role": "assistant", "content": "here you go"}]},
            {"id": "3", "messages": [{"role": "assistant"}]},
        ]
        gold_p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        pred_p.write_text(json.dumps({"id": "1", "extraction": EX()}) + "\n")
        r = evaluate(str(pred_p), str(gold_p))
        assert r["n"] == 1 and r["n_bad_gold"] == 2
        assert r["overall"]["mean"] == 1.0
