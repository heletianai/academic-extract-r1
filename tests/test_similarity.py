"""similarity.py 单测：规范化 / 字符串 / 数值 / bipartite 贪心。"""

from src.reward.similarity import (
    greedy_bipartite_f1,
    levenshtein,
    normalize_name,
    parse_value,
    string_similarity,
    value_hit,
    value_similarity,
)


class TestNormalizeName:
    def test_variants_collapse(self):
        assert normalize_name("GSM8K") == normalize_name("GSM-8K") == normalize_name("gsm 8k")

    def test_squad_version(self):
        assert normalize_name("SQuAD v1.1") == normalize_name("squad-v1.1") == "squadv11"

    def test_slash_and_underscore(self):
        assert normalize_name("MMLU_Pro") == normalize_name("MMLU/Pro") == "mmlupro"


class TestLevenshtein:
    def test_basics(self):
        assert levenshtein("", "") == 0
        assert levenshtein("abc", "abc") == 0
        assert levenshtein("abc", "abd") == 1
        assert levenshtein("abc", "") == 3


class TestStringSimilarity:
    def test_exact_after_norm(self):
        assert string_similarity("GSM-8K", "GSM8K") == 1.0

    def test_minor_typo_high(self):
        assert string_similarity("HellaSwag", "HelaSwag") > 0.85

    def test_disjoint_low(self):
        assert string_similarity("MMLU", "HumanEval") < 0.5

    def test_empty_sides(self):
        assert string_similarity("", "") == 1.0
        assert string_similarity("MMLU", "") == 0.0


class TestParseValue:
    def test_percent_string(self):
        assert parse_value("85.3%") == 85.3

    def test_plain_forms(self):
        assert parse_value(85.3) == 85.3
        assert parse_value("0.853") == 0.853
        assert parse_value(42) == 42.0

    def test_comma_thousands(self):
        assert parse_value("1,234.5") == 1234.5

    def test_garbage_and_bool(self):
        assert parse_value("N/A") is None
        assert parse_value(True) is None
        assert parse_value(None) is None


class TestValueSimilarity:
    def test_scale_alignment(self):
        assert value_similarity(0.853, 85.3) == 1.0
        assert value_similarity("85.3%", 0.853) == 1.0

    def test_gold_zero_special(self):
        assert value_similarity(0.0, 0.0) == 1.0
        assert value_similarity(0.5, 0.0) == 0.0

    def test_relative_error(self):
        assert abs(value_similarity(90.0, 100.0) - 0.9) < 1e-9

    def test_unparseable_falls_back_to_string(self):
        assert value_similarity("N/A", "N/A") == 1.0


class TestValueHit:
    def test_within_half_percent(self):
        assert value_hit(85.31, 85.3)

    def test_beyond_tolerance(self):
        assert not value_hit(86.0, 85.3)

    def test_scale_aligned_hit(self):
        assert value_hit(0.853, 85.3)

    def test_zero_gold(self):
        assert value_hit(0.0, 0.0)
        assert not value_hit(0.5, 0.0)


class TestGreedyBipartite:
    def test_both_empty_full_score(self):
        r = greedy_bipartite_f1([], [], sim_fn=lambda a, b: 1.0)
        assert r["soft_f1"] == 1.0 and r["hard_f1"] == 1.0

    def test_one_side_empty_zero(self):
        r = greedy_bipartite_f1(["x"], [], sim_fn=lambda a, b: 1.0)
        assert r["soft_f1"] == 0.0
        r = greedy_bipartite_f1([], ["x"], sim_fn=lambda a, b: 1.0)
        assert r["soft_f1"] == 0.0

    def test_each_side_used_once(self):
        # pred 两条同名争夺一条 gold：只允许一对配上
        sim = lambda a, b: 1.0 if a == b else 0.0
        r = greedy_bipartite_f1(["a", "a"], ["a"], sim_fn=sim)
        assert len(r["matches"]) == 1
        assert abs(r["soft_precision"] - 0.5) < 1e-6
        assert abs(r["soft_recall"] - 1.0) < 1e-6

    def test_threshold_cutoff(self):
        sim = lambda a, b: 0.3  # 全部低于 0.35 阈值
        r = greedy_bipartite_f1(["x"], ["y"], sim_fn=sim)
        assert r["soft_f1"] == 0.0 and r["matches"] == []

    def test_greedy_prefers_best_pair(self):
        # p1-g1=0.9, p1-g2=0.8, p2-g1=0.7：贪心应配 (p1,g1)，p2 只剩 g2
        table = {("p1", "g1"): 0.9, ("p1", "g2"): 0.8, ("p2", "g1"): 0.7, ("p2", "g2"): 0.6}
        r = greedy_bipartite_f1(["p1", "p2"], ["g1", "g2"], sim_fn=lambda a, b: table[(a, b)])
        pairs = {(m["pred_idx"], m["gold_idx"]) for m in r["matches"]}
        assert pairs == {(0, 0), (1, 1)}
