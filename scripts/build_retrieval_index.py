"""构建 BM25 检索索引：papers_pool（开发 9k）或全量合格池（生产 49k）→ data/retrieval_index/。

用法: PYTHONPATH=. python3 scripts/build_retrieval_index.py [--source data/processed/papers_pool.jsonl] [--out data/retrieval_index]
"""
import argparse
import time

from src.retrieval.bm25_index import BM25Index

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="data/processed/papers_pool.jsonl")
ap.add_argument("--out", default="data/retrieval_index")
args = ap.parse_args()

t0 = time.time()
idx = BM25Index.build_from_jsonl(args.source)
idx.save(args.out)
print(f"[build] {len(idx.docs)} docs -> {args.out} ({time.time()-t0:.1f}s)")
