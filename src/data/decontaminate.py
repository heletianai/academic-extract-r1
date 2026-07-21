"""训练/评测集去污染：13-gram 词级重叠检测（数据管线三件之三）。

对象：train vs held-out（+后续黄金集）的摘要文本——防止近重复论文（同工作
不同版本/扩展版）横跨切分导致评测虚高。id 级去重在 filter 已做，本件抓
"不同 id 但文本近重复"。目标带：重叠率 <2%（手册§七）；命中对从 train 剔除。
embedding 双重校验为放量后可选增强（需 sentence-transformers，第一版不引依赖）。

用法：
    python3 -m src.data.decontaminate --train data/processed/sft_train.jsonl \
        --against data/processed/holdout.jsonl --apply
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NGRAM = 13  # 社区常用 13-gram（GPT-3/Llama 报告口径）

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _doc_text(row: dict) -> str:
    """兼容 messages 格式（user turn 含 title+abstract）与裸 abstract 格式。"""
    if "messages" in row:
        for m in row["messages"]:
            if m["role"] == "user":
                return m["content"]
    return f"{row.get('title', '')} {row.get('abstract', '')}"


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def find_contaminated(train_rows: list[dict], against_rows: list[dict],
                      n: int = NGRAM) -> list[dict]:
    ref_grams: dict[tuple, str] = {}
    for r in against_rows:
        for g in ngrams(tokenize(_doc_text(r)), n):
            ref_grams[g] = str(r.get("id"))

    hits = []
    for r in train_rows:
        grams = ngrams(tokenize(_doc_text(r)), n)
        overlap = [g for g in grams if g in ref_grams]
        if overlap:
            hits.append({
                "train_id": str(r.get("id")),
                "against_id": ref_grams[overlap[0]],
                "n_shared_ngrams": len(overlap),
                "sample": " ".join(overlap[0][:8]) + " ...",
            })
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/sft_train.jsonl")
    ap.add_argument("--against", nargs="+", default=["data/processed/holdout.jsonl"])
    ap.add_argument("--apply", action="store_true", help="从 train 剔除命中样本并回写")
    args = ap.parse_args()

    train_rows = load(args.train)
    against_rows = []
    for p in args.against:
        against_rows.extend(load(p))

    hits = find_contaminated(train_rows, against_rows)
    rate = len(hits) / max(1, len(train_rows))
    report = {
        "train": args.train, "against": args.against,
        "n_train": len(train_rows), "n_against": len(against_rows),
        "ngram": NGRAM, "n_contaminated": len(hits),
        "overlap_rate": round(rate, 4),
        "pass_2pct_gate": rate < 0.02,
        "hits": hits[:50],
    }
    out = Path(args.train).parent / "decontamination_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "hits"}, indent=2))

    if args.apply and hits:
        bad_ids = {h["train_id"] for h in hits}
        kept = [r for r in train_rows if str(r.get("id")) not in bad_ids]
        with open(args.train, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[apply] removed {len(train_rows) - len(kept)} -> {args.train}")


if __name__ == "__main__":
    main()
