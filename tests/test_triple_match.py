"""triple_match.py 单测：10 类边界案例的代码版（与 docs/schema-checklist.md 一一对应）。"""

from src.reward.triple_match import (
    match_benchmark_triples,
    metric_similarity,
    triple_hit,
    triple_similarity,
)


def T(name, metric, value):
    return {"name": name, "metric": metric, "value": value}


class TestMetricSimilarity:
    def test_alias_group_hit(self):
        assert metric_similarity("acc", "accuracy") == 1.0
        assert metric_similarity("EM", "exact match") == 1.0
        assert metric_similarity("Pass@1", "pass at 1") == 1.0  # pass@1 与 passat1 同别名组
        assert metric_similarity("pass@1", "pass1") == 1.0

    def test_not_alias_not_equal(self):
        assert metric_similarity("accuracy", "f1") < 0.85


class TestTripleSimilarity:
    def test_identical_full(self):
        assert triple_similarity(T("MMLU", "accuracy", 85.3), T("MMLU", "accuracy", 85.3)) == 1.0

    def test_name_wrong_partial(self):
        # 边界案例 6：name 错、metric/value 对 → soft ≈ 2/3（部分分），hard 必不命中
        s = triple_similarity(T("HumanEval", "accuracy", 85.3), T("MMLU", "accuracy", 85.3))
        assert 0.6 < s < 0.75

    def test_missing_field_zero_component(self):
        s = triple_similarity({"name": "MMLU", "metric": "accuracy"}, T("MMLU", "accuracy", 85.3))
        assert abs(s - 2.0 / 3.0) < 1e-9


class TestTripleHit:
    def test_exact_hit(self):
        assert triple_hit(T("MMLU", "accuracy", 85.3), T("MMLU", "accuracy", 85.3))

    def test_name_variant_hit(self):
        # 边界案例 2：GSM-8K vs GSM8K
        assert triple_hit(T("GSM-8K", "accuracy", 92.0), T("GSM8K", "acc", 92.0))

    def test_metric_alias_hit(self):
        # 边界案例 3：EM vs exact match
        assert triple_hit(T("SQuAD", "EM", 88.0), T("SQuAD", "exact match", 88.0))

    def test_value_tolerance(self):
        # 边界案例 4：0.5% 容差内命中，之外不命中
        assert triple_hit(T("MMLU", "acc", 85.31), T("MMLU", "accuracy", 85.3))
        assert not triple_hit(T("MMLU", "acc", 86.0), T("MMLU", "accuracy", 85.3))

    def test_percent_scale_hit(self):
        # 边界案例 5：0.853 vs 85.3 刻度对齐
        assert triple_hit(T("MMLU", "acc", 0.853), T("MMLU", "accuracy", 85.3))

    def test_name_is_necessary_condition(self):
        # 边界案例 6：name 必要条件
        assert not triple_hit(T("HumanEval", "accuracy", 85.3), T("MMLU", "accuracy", 85.3))

    def test_missing_value_no_hit(self):
        assert not triple_hit({"name": "MMLU", "metric": "acc"}, T("MMLU", "accuracy", 85.3))


class TestMatchBenchmarkTriples:
    GOLD = [T("MMLU", "accuracy", 85.3), T("GSM8K", "accuracy", 92.1)]

    def test_perfect_prediction(self):
        pred = [T("MMLU", "acc", 85.3), T("GSM-8K", "acc", 92.1)]
        r = match_benchmark_triples(pred, self.GOLD)
        assert r["hard_f1"] == 1.0
        assert r["soft_f1"] > 0.95

    def test_hallucinated_extra_hurts_precision(self):
        # 边界案例 7：幻觉多抽 → precision 掉、recall 满
        pred = [T("MMLU", "acc", 85.3), T("GSM8K", "acc", 92.1), T("FakeBench", "acc", 99.0)]
        r = match_benchmark_triples(pred, self.GOLD)
        assert abs(r["hard_precision"] - 2 / 3) < 1e-6
        assert abs(r["hard_recall"] - 1.0) < 1e-6

    def test_missing_triple_hurts_recall(self):
        # 边界案例 8：漏抽 → recall 掉
        pred = [T("MMLU", "acc", 85.3)]
        r = match_benchmark_triples(pred, self.GOLD)
        assert abs(r["hard_precision"] - 1.0) < 1e-6
        assert abs(r["hard_recall"] - 0.5) < 1e-6

    def test_duplicate_pred_counted_once(self):
        # 边界案例 9：pred 重复两条 MMLU → gold 只被配一次，第二条压 precision
        pred = [T("MMLU", "acc", 85.3), T("MMLU", "accuracy", 85.3)]
        r = match_benchmark_triples(pred, [T("MMLU", "accuracy", 85.3)])
        assert abs(r["hard_precision"] - 0.5) < 1e-6
        assert abs(r["hard_recall"] - 1.0) < 1e-6

    def test_both_empty_full_score(self):
        # 边界案例 10a：论文无 benchmark 且模型输出空列表 → 满分
        r = match_benchmark_triples([], [])
        assert r["soft_f1"] == 1.0 and r["hard_f1"] == 1.0

    def test_hallucination_on_empty_gold_zero(self):
        # 边界案例 10b：论文无 benchmark 但模型幻觉输出 → 0 分
        r = match_benchmark_triples([T("MMLU", "acc", 85.3)], [])
        assert r["soft_f1"] == 0.0 and r["hard_f1"] == 0.0

    def test_dirty_input_no_crash(self):
        # reward 路径不许崩：非 list / 元素非 dict 全部过滤计分
        r = match_benchmark_triples("garbage", self.GOLD)
        assert r["soft_f1"] == 0.0 and r["n_pred"] == 0
        r2 = match_benchmark_triples([T("MMLU", "acc", 85.3), "junk", 42], self.GOLD)
        assert r2["n_pred"] == 1 and r2["n_pred_dropped"] == 2
        assert r2["hard_recall"] == 0.5

    def test_value_wrong_soft_partial_hard_zero(self):
        # 值抄错（差 >0.5%）：soft 仍给部分分（bipartite 配上），hard 记 0
        pred = [T("MMLU", "accuracy", 79.0)]
        r = match_benchmark_triples(pred, [T("MMLU", "accuracy", 85.3)])
        assert r["hard_f1"] == 0.0
        assert 0.5 < r["soft_f1"] < 1.0
        assert r["matches"][0]["hard_hit"] is False
