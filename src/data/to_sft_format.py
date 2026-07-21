"""distilled.jsonl → SFT 训练格式（messages 结构，Unsloth 侧 apply_chat_template）。

两个口径决策（依据在 docstring，冒烟后可按数据回调）：
1. 只收 validator_status==VALID 的样本（rejection filter 精神：SOFT 样本会把软违规
   习惯教给学生；VALID 率 <80% 时再议放宽——冒烟看数据说话）。
2. 学生用短 system prompt（一行 schema 键表）：规则经标签内化进权重，不靠 prompt 背书——
   短 prompt 省 token 是"训小模型替代 API+长 prompt"论点本身（NuExtract/extract0 同款）。

held-out 切分：按 id 哈希无关的固定 seed 洗牌后切尾部 N 条，train/held-out 无重叠，
切分结果落盘保证后续任何阶段（GRPO 采样池/badcase 分析）同一 held-out。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "processed" / "distilled.jsonl"
OUT_DIR = ROOT / "data" / "processed"

STUDENT_SYSTEM = (
    "Extract paper metadata as a JSON object with exactly these fields: "
    "task_type (classification|generation|retrieval|reasoning|multimodal|agent|other), "
    "modalities (list: text|image|audio|video|code), "
    "benchmarks (list of {name, metric, value}), "
    "open_source (bool), claims_sota (bool), "
    "method_keywords (3-5 short phrases), one_line_summary (str), "
    "limitation_mentioned (str or null). Output JSON only."
)


def to_messages(row: dict) -> dict:
    return {
        "id": row["id"],
        "messages": [
            {"role": "system", "content": STUDENT_SYSTEM},
            {"role": "user", "content": f"Title: {row['title']}\nAbstract: {row['abstract']}"},
            {"role": "assistant", "content": json.dumps(row["extraction"], ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(IN_PATH))
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-soft", action="store_true", help="放宽收 SOFT_VIOLATION（默认只收 VALID）")
    args = ap.parse_args()

    blacklist_p = Path(args.input).parent / "blacklist.txt"
    blacklist = set(blacklist_p.read_text().split()) if blacklist_p.exists() else set()

    rows, n_soft_skipped = [], 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("id") in blacklist:  # 人工/盲测判错条目（issues-log #004）
                continue
            if r.get("validator_status") == "VALID":
                rows.append(r)
            elif r.get("validator_status") == "SOFT_VIOLATION":
                if args.include_soft:
                    rows.append(r)
                else:
                    n_soft_skipped += 1

    seen: set[str] = set()
    uniq = []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        uniq.append(r)

    rng = random.Random(args.seed)
    rng.shuffle(uniq)
    n_hold = min(args.holdout, max(0, len(uniq) - 1))
    holdout, train = uniq[:n_hold], uniq[n_hold:]
    # 红队 P2 修复：uniq < holdout 时原逻辑静默产出 train=1，SFT 在一条样本上开跑。
    # 守卫语义=请求的 holdout 未被满足（池子不够，切分退化），显式小规模冒烟合法放行
    if n_hold < args.holdout:
        import sys
        sys.exit(
            f"holdout 请求 {args.holdout} 但只能给 {n_hold}（unique={len(uniq)}, train={len(train)}）："
            f"蒸馏量不足。冒烟请显式传更小值（如 --holdout 10）。"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, subset in (("sft_train", train), ("holdout", holdout)):
        p = OUT_DIR / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(to_messages(r), ensure_ascii=False) + "\n")
        paths[name] = str(p)

    # 人工核验空壳（审计 B5：held-out GT 100% 是蒸馏的，独立验证/教师基线都以此为前置）
    # 优先队列 = sota_disagree ∪ grounding_flags 非空（#004：教师可疑输出浓度最高），不足补随机
    review_n = min(50, len(holdout))
    suspect = [r for r in holdout if r.get("sota_disagree") or r.get("grounding_flags")]
    rest = [r for r in holdout if not (r.get("sota_disagree") or r.get("grounding_flags"))]
    review_set = (suspect + rest)[:review_n]
    with open(OUT_DIR / "holdout_for_review.jsonl", "w", encoding="utf-8") as f:
        for r in review_set:
            f.write(json.dumps({
                "id": r["id"], "title": r["title"], "abstract": r["abstract"],
                "teacher_extraction": r["extraction"],
                "sota_disagree": r.get("sota_disagree", False),
                "grounding_flags": r.get("grounding_flags", []),
                "human_extraction": None,  # 人工填此字段后即成"真·黄金 held-out 子集"
            }, ensure_ascii=False) + "\n")

    soft_total = len(rows) + n_soft_skipped if not args.include_soft else len(rows)
    stats = {
        "input": args.input, "seed": args.seed,
        "valid_rows": len(rows), "unique": len(uniq),
        "soft_skipped": n_soft_skipped,
        "soft_ratio": round(n_soft_skipped / max(1, soft_total), 3),  # >0.15 须复议 include-soft（审计 B4）
        "sota_disagree_ratio": round(
            sum(1 for r in uniq if r.get("sota_disagree")) / max(1, len(uniq)), 3
        ),
        "train": len(train), "holdout": len(holdout),
        "review_shell": review_n,
        "grounding_flag_ratio": round(
            sum(1 for r in uniq if r.get("grounding_flags")) / max(1, len(uniq)), 3
        ),
        "benchmarks_nonempty_ratio": round(
            sum(1 for r in uniq if r["extraction"]["benchmarks"]) / max(1, len(uniq)), 3
        ),
    }
    (OUT_DIR / "sft_datacard.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if stats["soft_ratio"] > 0.15:
        print(f"[warn] SOFT 占比 {stats['soft_ratio']:.1%} > 15%：默认丢弃会造成分布左移，复议 --include-soft（审计 B4）")


if __name__ == "__main__":
    main()
