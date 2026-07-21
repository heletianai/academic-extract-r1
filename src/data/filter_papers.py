"""本地筛选：去重 → 主类 cs.CL/cs.LG → id 前缀≥2507 → 长度过滤 → 分层抽样。

时间窗口用 arXiv id 前缀而非 OAI created 字段（2026-07-21 issues-log #002）：
实测 OAI created 36.5% 与 id 前缀年月不一致（例 2407.19342 真实 v1=2024-07-27，
OAI created=2025-07-01 是 replace 日期）——用 created 筛会混入 Qwen3 cutoff 前
的老论文，击穿防污染设计。id 前缀 YYMM 由分配规则绑定 v1 提交年月，不可变。

分层依据：benchmarks 是权重 3 的难字段，摘要不含数字的论文对它零信号——
抽样时"含数字摘要"占比默认 ≥50%（--numeric-ratio），保证难字段有训练信号。
（extract0 generate_data 同样做难度分布控制；此为配比参数非 schema 变更。）

用法：
    python3 -m src.data.filter_papers --stats           # 只看统计
    python3 -m src.data.filter_papers --sample 2000     # 抽样输出 papers_pool.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "arxiv_cs_2025h2.jsonl"
OUT_DIR = ROOT / "data" / "processed"

PRIMARY_CATS = {"cs.CL", "cs.LG"}
MIN_ID_YYMM = 2507  # v1 提交 ≥ 2025-07（Qwen3 cutoff 后，防污染口径）
ABS_LEN_RANGE = (100, 3000)

_NUM_PAT = re.compile(r"\d+\.\d+|\d+%")
_NEW_ID_RE = re.compile(r"^(\d{4})\.\d{4,5}")


def id_yymm(paper_id: str) -> int | None:
    """新式 id（YYMM.NNNNN）前缀 → 提交年月整数；旧式 id 返回 None（一律拒）。"""
    m = _NEW_ID_RE.match(paper_id or "")
    return int(m.group(1)) if m else None


def has_numeric_signal(abstract: str) -> bool:
    """含 ≥2 个小数或百分数 → 可能报告了数值结果（只影响抽样配比，不影响标注）。"""
    return len(_NUM_PAT.findall(abstract)) >= 2


def load_and_filter(raw_path: Path = RAW_PATH) -> tuple[list[dict], dict]:
    by_id: dict[str, dict] = {}
    n_lines = 0
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_id[r["id"]] = r  # 后出现覆盖先出现（updated 更新记录在后）

    kept, stats = [], {
        "raw_lines": n_lines, "unique_ids": len(by_id),
        "primary_cat": 0, "id_window_ok": 0, "len_ok": 0, "numeric": 0,
    }
    for r in by_id.values():
        cats = r.get("categories", "").split()
        if not cats or cats[0] not in PRIMARY_CATS:
            continue
        stats["primary_cat"] += 1
        ym = id_yymm(r.get("id", ""))
        if ym is None or ym < MIN_ID_YYMM:
            continue
        stats["id_window_ok"] += 1
        alen = len(r.get("abstract", ""))
        if not (ABS_LEN_RANGE[0] <= alen <= ABS_LEN_RANGE[1]):
            continue
        stats["len_ok"] += 1
        r["numeric_signal"] = has_numeric_signal(r["abstract"])
        stats["numeric"] += int(r["numeric_signal"])
        kept.append(r)
    return kept, stats


def stratified_sample(pool: list[dict], n: int, numeric_ratio: float, seed: int,
                      plain_floor: float = 0.25) -> list[dict]:
    """plain_floor（审计 B7）：空 benchmarks 是权重 3+幻觉惩罚最狠的状态，
    不含数字的 plain 样本保底 ≥25%，否则 GRPO 学不到"该空就空"的克制。"""
    rng = random.Random(seed)
    numeric = [r for r in pool if r["numeric_signal"]]
    plain = [r for r in pool if not r["numeric_signal"]]
    rng.shuffle(numeric)
    rng.shuffle(plain)
    n_plain_min = min(len(plain), int(n * plain_floor))
    n_num = min(len(numeric), int(n * numeric_ratio), n - n_plain_min)
    n_plain = min(len(plain), n - n_num)
    # 仍不足时反向补齐
    if n_num + n_plain < n and len(numeric) > n_num:
        n_num = min(len(numeric), n - n_plain)
    sample = numeric[:n_num] + plain[:n_plain]
    rng.shuffle(sample)
    return sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--numeric-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR / "papers_pool.jsonl"))
    args = ap.parse_args()

    pool, stats = load_and_filter()
    print(json.dumps(stats, indent=2))
    print(f"filtered pool = {len(pool)} (numeric {stats['numeric']}, plain {len(pool)-stats['numeric']})")
    if args.stats or not args.sample:
        return

    sample = stratified_sample(pool, args.sample, args.numeric_ratio, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_num = sum(1 for r in sample if r["numeric_signal"])
    print(f"sampled {len(sample)} -> {out} (numeric {n_num} = {n_num/len(sample):.0%}, seed={args.seed})")


if __name__ == "__main__":
    main()
