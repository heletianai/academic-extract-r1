"""held-out 评测（评测协议冻结版：训练前定死，五方对照与教师基线共用）。

输入格式统一（无关来源系统——base few-shot/API few-shot/SFT/GRPO/多轮全用它）：
- predictions.jsonl: {"id": ..., "extraction": {...}} 或 {"id": ..., "raw_text": "..."}
  （raw_text 走 JSON 提取+validator，统计合法率；extraction 视为已解析）
- gold.jsonl: {"id": ..., "extraction": {...}}

指标：合法 JSON 率 / 逐字段平均分 / overall(六字段均权) / benchmarks hard-F1 单列 /
     bootstrap 95% CI（1000 次重采样，percentile 法，标准库实现）。

用法：
    python3 -m src.eval.evaluate --pred runs/xxx/pred.jsonl --gold data/processed/holdout.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.eval.field_scores import score_fields
from src.reward.reward_v1 import extract_json_str
from src.schema_model import INVALID, validate_extraction

REPORT_FIELDS = (
    "task_type", "modalities", "open_source", "claims_sota",
    "method_keywords", "benchmarks_soft", "benchmarks_hard",
)


def _extract_json(text: str):
    """复用 reward 同一提取器（最后一个可解析对象），消除双实现漂移。"""
    s = extract_json_str(text or "")
    if s is None:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _load_gold_extraction(g: dict):
    """gold 双格式安全加载（红队 P1 修复：一条坏 gold 行不许崩整个 eval）。

    额外过 validator（红队潜伏#2：gold 里字符串 bool 会被 score_fields 的
    bool() 误翻——validator 的 lax cast 统一成真 bool）。失败返回 None，
    上游计入 n_bad_gold 并跳过该条，显式报告不静默。
    """
    try:
        ex = g.get("extraction")
        if ex is None and g.get("messages"):
            ex = json.loads(g["messages"][-1].get("content", ""))
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
        return None
    if ex is None:
        return None
    v = validate_extraction(ex)
    return v["parsed"] if v["status"] != INVALID else None


def load_jsonl(path: str | Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r["id"])] = r
    return out


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def evaluate(pred_path: str, gold_path: str) -> dict:
    preds = load_jsonl(pred_path)
    golds = load_jsonl(gold_path)

    per_paper: list[dict] = []
    n_invalid = 0
    n_bad_gold = 0
    for pid, g in golds.items():
        gold_ex = _load_gold_extraction(g)
        if gold_ex is None:
            n_bad_gold += 1
            continue  # gold 数据问题不算被测系统头上，单列计数显式报告
        p = preds.get(pid)
        if p is None:
            n_invalid += 1
            per_paper.append({k: 0.0 for k in REPORT_FIELDS} | {"id": pid, "valid": 0.0})
            continue
        pred_ex = p.get("extraction")
        if pred_ex is None and "raw_text" in p:
            pred_ex = _extract_json(p["raw_text"])
        v = validate_extraction(pred_ex)
        if v["status"] == INVALID:
            n_invalid += 1
            per_paper.append({k: 0.0 for k in REPORT_FIELDS} | {"id": pid, "valid": 0.0})
            continue
        s = score_fields(v["parsed"], gold_ex)
        s["id"] = pid
        s["valid"] = 1.0
        per_paper.append(s)

    n = len(per_paper)
    report: dict = {
        "n": n,
        "valid_json_rate": round(sum(r["valid"] for r in per_paper) / max(1, n), 4),
        "n_invalid_or_missing": n_invalid,
        "n_bad_gold": n_bad_gold,
        "fields": {},
    }
    overall_per_paper = []
    for r in per_paper:
        # overall = 六字段均权（benchmarks 用 soft；hard 单列展示）
        keys = ("task_type", "modalities", "open_source", "claims_sota", "method_keywords", "benchmarks_soft")
        overall_per_paper.append(sum(r[k] for k in keys) / len(keys))
    for k in REPORT_FIELDS:
        vals = [r[k] for r in per_paper]
        lo, hi = bootstrap_ci(vals)
        report["fields"][k] = {"mean": round(sum(vals) / max(1, n), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    lo, hi = bootstrap_ci(overall_per_paper)
    report["overall"] = {"mean": round(sum(overall_per_paper) / max(1, n), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    report["per_paper"] = per_paper
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    report = evaluate(args.pred, args.gold)
    slim = {k: v for k, v in report.items() if k != "per_paper"}
    print(json.dumps(slim, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"full report -> {args.out}")


if __name__ == "__main__":
    main()
