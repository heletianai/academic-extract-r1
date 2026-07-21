"""benchmarks 三元组匹配规则（7.19 终判点名的 reward 打分规则洞，落代码版）。

三元组 = {"name": str, "metric": str, "value": float|str}（立项卡 Schema 终版）
GT 侧兼容 PwC 四元组（dataset/task/metric/value）——降维到三元组在数据管线做，本模块只认三字段。

双出口设计（口径来源，均为总纲§三冻结结构）：
- soft：三元组内 name/metric/value 相似度均权平均（extract0 dict 分支写法）→
        bipartite 贪心 0.35 → soft F1，进 GRPO reward（连续信号，RL 早期不稀疏）
- hard：name 命中 AND metric 命中 AND value 容差内 → 配对计数 F1（R1-RE 集合 F1 口径），
        进 eval 报告（可解释：命中 X/Y 条）
name 是必要条件：name 错则三元组必不命中（关系抽取三元组匹配惯例，头实体错=整条错）。

冻结项（初始值可跑，正式冻结待用户 schema 过手，consts 见下）：
- NAME_SIM_THRESHOLD / METRIC_SIM_THRESHOLD = 0.85（规范化后 exact 是主路径，
  Levenshtein 只救拼写微差；0.85 ≈ 8 字符名称容 1 错位）
- VALUE_REL_TOL = 0.005（0.5% 相对容差）
- METRIC_ALIASES 别名组（规范化后查组）
"""

from __future__ import annotations

from .similarity import (
    greedy_bipartite_f1,
    normalize_name,
    string_similarity,
    value_hit,
    value_similarity,
)

NAME_SIM_THRESHOLD = 0.85
METRIC_SIM_THRESHOLD = 0.85
VALUE_REL_TOL = 0.005
BIPARTITE_THRESHOLD = 0.35  # extract0 原值，数值消融项

# 别名组：组内互相视为同一 metric（规范化后匹配）。checklist 冻结项，用户过手可增删。
METRIC_ALIASES: list[set[str]] = [
    {"acc", "accuracy", "top1", "top1acc", "top1accuracy", "avgacc", "averageaccuracy"},
    {"em", "exactmatch", "exactmatchscore"},
    {"f1", "f1score", "microf1", "macrof1"},  # micro/macro 并组是初始妥协，过手可拆
    {"bleu", "bleu4", "bleuscore"},
    {"rouge", "rouge1", "rouge2", "rougel", "rougescore"},
    {"pass@1", "passat1", "pass1"},
    {"ppl", "perplexity"},
    {"auc", "auroc", "rocauc"},
    {"mrr", "meanreciprocalrank"},
    {"ndcg", "ndcg@10", "ndcgat10"},
    {"wer", "worderrorrate"},
    {"mmluscore", "mmluacc"},
]

_ALIAS_LOOKUP: dict[str, int] = {}
for _gid, _group in enumerate(METRIC_ALIASES):
    for _name in _group:
        _ALIAS_LOOKUP[normalize_name(_name)] = _gid


def metric_similarity(pred_metric: str, gold_metric: str) -> float:
    """metric 相似度：别名组命中 → 1.0，否则规范化 Levenshtein。"""
    p, g = normalize_name(pred_metric), normalize_name(gold_metric)
    if p == g:
        return 1.0
    gid_p, gid_g = _ALIAS_LOOKUP.get(p), _ALIAS_LOOKUP.get(g)
    if gid_p is not None and gid_p == gid_g:
        return 1.0
    return string_similarity(pred_metric, gold_metric)


def _get(t: dict, key: str):
    return t.get(key) if isinstance(t, dict) else None


def triple_similarity(pred: dict, gold: dict) -> float:
    """soft 三元组相似度：三分量均权平均（extract0 dict 分支均权口径，权重消融项）。

    缺字段按 0 分量计（extract0：单边 None=0.0），不抛异常——reward 路径上
    任何脏输入都必须返回数值（训练中 batch 不能因单条脏样本崩，SR++ 稳定性口径）。
    """
    s_name = string_similarity(str(_get(pred, "name") or ""), str(_get(gold, "name") or ""))
    s_metric = metric_similarity(str(_get(pred, "metric") or ""), str(_get(gold, "metric") or ""))
    p_val, g_val = _get(pred, "value"), _get(gold, "value")
    s_value = value_similarity(p_val, g_val) if (p_val is not None and g_val is not None) else 0.0
    return (s_name + s_metric + s_value) / 3.0


def triple_hit(pred: dict, gold: dict) -> bool:
    """hard 三元组命中：name AND metric AND value 三条全过。"""
    name_ok = (
        string_similarity(str(_get(pred, "name") or ""), str(_get(gold, "name") or ""))
        >= NAME_SIM_THRESHOLD
    )
    if not name_ok:
        return False
    metric_ok = (
        metric_similarity(str(_get(pred, "metric") or ""), str(_get(gold, "metric") or ""))
        >= METRIC_SIM_THRESHOLD
    )
    if not metric_ok:
        return False
    p_val, g_val = _get(pred, "value"), _get(gold, "value")
    if p_val is None or g_val is None:
        return False
    return value_hit(p_val, g_val, rel_tol=VALUE_REL_TOL)


def match_benchmark_triples(pred_triples: list, gold_triples: list) -> dict:
    """benchmarks 字段总入口。

    返回：
      soft_f1  → reward 的 F_bench 分量（权重 3，R1-RE w_tri）
      hard_f1  → eval 报告的三元组 F1
      matches  → 逐对配对明细（badcase 分析 / trajectory 落盘用）
    输入脏数据（非 list / 元素非 dict）不抛异常，过滤后计分。
    """
    pred_clean = [t for t in pred_triples if isinstance(t, dict)] if isinstance(pred_triples, list) else []
    gold_clean = [t for t in gold_triples if isinstance(t, dict)] if isinstance(gold_triples, list) else []
    n_dropped = (
        (len(pred_triples) if isinstance(pred_triples, list) else 0) - len(pred_clean)
    )
    result = greedy_bipartite_f1(
        pred_clean,
        gold_clean,
        sim_fn=triple_similarity,
        threshold=BIPARTITE_THRESHOLD,
        hit_fn=triple_hit,
    )
    result["n_pred"] = len(pred_clean)
    result["n_gold"] = len(gold_clean)
    result["n_pred_dropped"] = n_dropped
    return result
