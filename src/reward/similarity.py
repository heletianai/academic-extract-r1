"""底层相似度组件：规范化 + 字符串/数值相似度 + bipartite 贪心匹配。

写法来源（构思总纲§三装配图，抄结构不抄数值）：
- 字符串相似度：MeXtract type_classes.Str.compare（1 - lev/max_len）
- 数值相似度：extract0 calculate_field_similarity（相对误差，gold=0 特判）
- bipartite 贪心：extract0 calculate_field_similarity 列表分支（降序贪心+阈值截断+soft F1）
纯标准库实现，零第三方依赖（上卡环境免冲突）。
"""

from __future__ import annotations

import re

# ---------- 规范化 ----------

_NORM_STRIP_RE = re.compile(r"[\s\-_./\\]+")


def normalize_name(s: str) -> str:
    """benchmark/metric 名称规范化：小写 + 去空白/连字符/下划线/点/斜杠。

    覆盖变体：GSM8K vs GSM-8K vs gsm 8k → gsm8k
    不做同义词展开（同义词属别名表，checklist 冻结项）。
    """
    if not isinstance(s, str):
        s = str(s)
    return _NORM_STRIP_RE.sub("", s.strip().lower())


# ---------- 字符串相似度 ----------

def levenshtein(a: str, b: str) -> int:
    """标准 DP 编辑距离（与 MeXtract 所用 python-Levenshtein 同公式）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def string_similarity(pred: str, gold: str) -> float:
    """规范化后编辑距离相似度 ∈ [0,1]。MeXtract Str.compare 写法，先规范化再比。"""
    p, g = normalize_name(pred), normalize_name(gold)
    if p == g:
        return 1.0
    if not p or not g:
        return 0.0
    return max(0.0, 1.0 - levenshtein(p, g) / max(len(p), len(g)))


# ---------- 数值解析与相似度 ----------

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_value(v) -> float | None:
    """把 85.3 / "85.3" / "85.3%" / "0.853" 解析为 float；失败返回 None。"""
    if isinstance(v, bool):  # bool 是 int 子类，先拦
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = _NUM_RE.search(v.replace(",", ""))
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


def _aligned_pairs(p: float, g: float) -> list[tuple[float, float]]:
    """原刻度 + 百分比刻度对齐（0.853 vs 85.3 → /100 对齐）两组候选。

    PwC/论文里分数刻度与百分比刻度混存是常态，取对齐后更优者。
    """
    pairs = [(p, g)]
    if 0.0 < abs(p) <= 1.0 < abs(g) <= 100.0:
        pairs.append((p, g / 100.0))
    elif 0.0 < abs(g) <= 1.0 < abs(p) <= 100.0:
        pairs.append((p / 100.0, g))
    return pairs


def value_similarity(pred, gold) -> float:
    """soft 数值相似度 ∈ [0,1]：extract0 相对误差公式 max(0, 1-|p-g|/|g|)。

    gold=0 特判（extract0 原味）；解析失败按字符串比对兜底。
    """
    p, g = parse_value(pred), parse_value(gold)
    if p is None or g is None:
        return string_similarity(str(pred), str(gold))
    best = 0.0
    for pp, gg in _aligned_pairs(p, g):
        if gg == 0.0:
            s = 1.0 if pp == 0.0 else 0.0
        else:
            s = max(0.0, 1.0 - abs(pp - gg) / abs(gg))
        best = max(best, s)
    return best


def value_hit(pred, gold, rel_tol: float = 0.005, abs_tol: float = 1e-4) -> bool:
    """hard 数值命中：刻度对齐后相对误差 ≤ rel_tol 或绝对误差 ≤ abs_tol。

    rel_tol=0.005（0.5%）为初始值，checklist 冻结项（85.3 vs 85.31 命中；85.3 vs 86.0 不命中）。
    """
    p, g = parse_value(pred), parse_value(gold)
    if p is None or g is None:
        return normalize_name(str(pred)) == normalize_name(str(gold))
    for pp, gg in _aligned_pairs(p, g):
        if abs(pp - gg) <= abs_tol:
            return True
        if gg != 0.0 and abs(pp - gg) / abs(gg) <= rel_tol:
            return True
    return False


# ---------- bipartite 贪心匹配（extract0 列表分支原味） ----------

def greedy_bipartite_f1(
    pred_items: list,
    gold_items: list,
    sim_fn,
    threshold: float = 0.35,
    hit_fn=None,
) -> dict:
    """extract0 写法：全组合相似度降序贪心配对，每边只用一次，阈值截断。

    返回 soft 与（可选）hard 两套 P/R/F1：
    - soft：matched_sum（相似度分数和）→ P=sum/m, R=sum/n（extract0 原味，进 reward）
    - hard：hit_fn 判定命中的配对计数 → P=cnt/m, R=cnt/n（R1-RE 集合 F1 口径，进 eval）
    双空=1.0、单边空=0.0（extract0 边界口径）。
    """
    m, n = len(pred_items), len(gold_items)
    if m == 0 and n == 0:
        return {
            "soft_f1": 1.0, "soft_precision": 1.0, "soft_recall": 1.0,
            "hard_f1": 1.0, "hard_precision": 1.0, "hard_recall": 1.0,
            "matches": [],
        }
    if m == 0 or n == 0:
        return {
            "soft_f1": 0.0, "soft_precision": 0.0, "soft_recall": 0.0,
            "hard_f1": 0.0, "hard_precision": 0.0, "hard_recall": 0.0,
            "matches": [],
        }

    pairs = []
    for i, p in enumerate(pred_items):
        for j, g in enumerate(gold_items):
            pairs.append((sim_fn(p, g), i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)

    used_pred, used_gold = set(), set()
    matched_sum = 0.0
    hard_count = 0
    matches = []
    for s, i, j in pairs:
        if s < threshold:
            break
        if i in used_pred or j in used_gold:
            continue
        used_pred.add(i)
        used_gold.add(j)
        matched_sum += max(0.0, min(1.0, s))
        hit = bool(hit_fn(pred_items[i], gold_items[j])) if hit_fn else None
        if hit:
            hard_count += 1
        matches.append({"pred_idx": i, "gold_idx": j, "sim": round(s, 4), "hard_hit": hit})

    def _f1(num: float) -> tuple[float, float, float]:
        # extract0 原式带 eps 防零除；此处空列表已在入口早退，m/n 必 >0，
        # 去掉 eps 使满分路径精确等于 1.0（eval 报告需要精确的"命中 X/Y"）
        precision = num / m
        recall = num / n
        if precision <= 0.0 and recall <= 0.0:
            return 0.0, 0.0, 0.0
        return (2.0 * precision * recall) / (precision + recall), precision, recall

    soft_f1, soft_p, soft_r = _f1(matched_sum)
    hard_f1, hard_p, hard_r = _f1(float(hard_count)) if hit_fn else (None, None, None)
    return {
        "soft_f1": float(soft_f1), "soft_precision": float(soft_p), "soft_recall": float(soft_r),
        "hard_f1": hard_f1, "hard_precision": hard_p, "hard_recall": hard_r,
        "matches": matches,
    }
