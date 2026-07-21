"""六字段逐字段打分件（eval 与 reward F_field 共用——训练即评测的实体）。

分派表（总纲§三，extract0 按类型分派口径）：
- task_type / open_source / claims_sota → exact（0/1）
- modalities → 集合 F1（元素规范化 exact）
- method_keywords → bipartite 贪心 soft F1（0.35）
- benchmarks → triple_match 双出口（soft 进 reward，hard 进报告）
"""

from __future__ import annotations

from src.reward.similarity import greedy_bipartite_f1, normalize_name, string_similarity
from src.reward.triple_match import match_benchmark_triples


def _set_f1(pred: list, gold: list) -> float:
    p = {normalize_name(str(x)) for x in (pred or [])}
    g = {normalize_name(str(x)) for x in (gold or [])}
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = len(p & g)
    if inter == 0:
        return 0.0
    precision, recall = inter / len(p), inter / len(g)
    return 2 * precision * recall / (precision + recall)


def score_fields(pred: dict, gold: dict) -> dict:
    """pred/gold 均为 extraction dict（已过 validator 的 parsed 形态）。

    返回逐字段分 + benchmarks 双出口。脏值不抛异常（reward 路径纪律）。
    """
    pred, gold = pred or {}, gold or {}
    bench = match_benchmark_triples(pred.get("benchmarks") or [], gold.get("benchmarks") or [])
    return {
        "task_type": float(
            normalize_name(str(pred.get("task_type", ""))) == normalize_name(str(gold.get("task_type", "")))
        ),
        "modalities": _set_f1(pred.get("modalities") or [], gold.get("modalities") or []),
        "open_source": float(bool(pred.get("open_source")) == bool(gold.get("open_source"))),
        "claims_sota": float(bool(pred.get("claims_sota")) == bool(gold.get("claims_sota"))),
        "method_keywords": greedy_bipartite_f1(
            [str(x) for x in (pred.get("method_keywords") or [])],
            [str(x) for x in (gold.get("method_keywords") or [])],
            sim_fn=string_similarity,
        )["soft_f1"],
        "benchmarks_soft": bench["soft_f1"],
        "benchmarks_hard": bench["hard_f1"],
    }


# reward F_field 的五字段（benchmarks 单列为 F_bench，权重 3——R1-RE 1:3）
FIELD_KEYS_FOR_REWARD = ("task_type", "modalities", "open_source", "claims_sota", "method_keywords")


def f_field(scores: dict) -> float:
    """reward 合法层的 F_field：五个可验证字段均权平均。"""
    return sum(scores[k] for k in FIELD_KEYS_FOR_REWARD) / len(FIELD_KEYS_FOR_REWARD)
