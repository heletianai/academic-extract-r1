"""reward v1 合成式（总纲§三，结构冻结、数值留消融）。

R(completion) =
  ① 硬门控层（extract0 三重门控）：
     JSON 提取失败 / 顶层重复键 / validator INVALID（缺键/多余键/类型错/枚举非法）
     → GATE_PENALTY（-1.0 起步；R1-RE 用 -3，数值消融项）
  ② 合法层（R1-RE 1:3 分层 + SO-Bench 乘子）：
     R = α_schema × (1·F_field + 3·F_bench_soft) / 4
       α_schema: 1.0 全合规 / 0.8 软违规（validator 三态）
       F_field : 五个可验证字段按类型分派均权（field_scores.f_field）
       F_bench : benchmarks 三元组 bipartite soft F1（权重 3 = R1-RE w_tri）
  ③ action penalty 层：Stage C 多轮接入（零检索作答/不作答/超 max_turns），
     本文件留接口 action_penalty 参数，单轮恒 0。

TRL 原生签名（extract0 reward_wrapper 写法）：completions + kwargs 里取 gold。
纪律：任何输入不抛异常（SR++ 稳定性——reward 崩一次整个 batch 报废）。
"""

from __future__ import annotations

import json

from src.eval.field_scores import f_field, score_fields
from src.schema_model import INVALID, validate_extraction

GATE_PENALTY = -1.0   # 数值消融项（R1-RE: -3）
W_FIELD = 1.0
W_BENCH = 3.0         # R1-RE w_tri=3


def extract_json_str(text: str) -> str | None:
    """extract0 括号平衡提取（in-string / escape 感知），返回原始 JSON 串。"""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def has_duplicate_top_level_keys(json_str: str) -> bool:
    """extract0 _has_duplicate_top_level_keys：object_pairs_hook 检测；解析失败视为重复（保守拒）。"""
    try:
        pairs = json.loads(json_str, object_pairs_hook=list)
    except (json.JSONDecodeError, ValueError):
        return True
    if isinstance(pairs, list):
        seen = set()
        for k, _ in pairs:
            if k in seen:
                return True
            seen.add(k)
    return False


def compute_reward(completion_text: str, gold_extraction: dict,
                   action_penalty: float = 0.0) -> dict:
    """单条 reward。返回明细 dict（五件套曲线与 hacking 观察线要逐分量落盘）。"""
    json_str = extract_json_str(completion_text or "")
    if json_str is None:
        return {"reward": GATE_PENALTY, "gate": "no_json", "alpha": None,
                "f_field": None, "f_bench": None}
    if has_duplicate_top_level_keys(json_str):
        return {"reward": GATE_PENALTY, "gate": "duplicate_keys", "alpha": None,
                "f_field": None, "f_bench": None}
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return {"reward": GATE_PENALTY, "gate": "parse_fail", "alpha": None,
                "f_field": None, "f_bench": None}

    v = validate_extraction(obj)
    if v["status"] == INVALID:
        return {"reward": GATE_PENALTY, "gate": "schema_invalid", "alpha": None,
                "f_field": None, "f_bench": None, "errors": v["errors"]}

    scores = score_fields(v["parsed"], gold_extraction)
    ff = f_field(scores)
    fb = scores["benchmarks_soft"]
    r = v["alpha"] * (W_FIELD * ff + W_BENCH * fb) / (W_FIELD + W_BENCH)
    r += action_penalty  # Stage C 接入；单轮恒 0
    return {"reward": float(r), "gate": None, "alpha": v["alpha"],
            "f_field": round(ff, 4), "f_bench": round(fb, 4),
            "soft_flags": v.get("soft_flags", []), "field_scores": scores}


def reward_fn(completions, **kwargs) -> list[float]:
    """TRL GRPOTrainer 签名入口。gold 从 dataset 列 `gold_extraction` 取（extract0 wrapper 同构）。

    completions 兼容 str / [{"content": ...}]（chat 格式）两种形态。
    """
    golds = kwargs.get("gold_extraction") or []
    rewards = []
    for i, comp in enumerate(completions):
        if isinstance(comp, list) and comp and isinstance(comp[0], dict):
            text = comp[0].get("content", "")
        elif isinstance(comp, dict):
            text = comp.get("content", "")
        else:
            text = str(comp)
        gold = golds[i] if i < len(golds) else {}
        if isinstance(gold, str):
            try:
                gold = json.loads(gold)
            except json.JSONDecodeError:
                gold = {}
        rewards.append(compute_reward(text, gold)["reward"])
    return rewards
