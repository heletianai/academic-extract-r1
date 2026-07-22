"""检索服务 FastAPI 壳——payload 协议兼容 Search-R1 官方检索服务。

请求:  {"queries": ["..."], "topk": 3, "return_scores": true, "exclude_ids": ["2507.xxxxx"]}
响应:  {"result": [[{"document": {"contents": "title\ntext"}, "score": ...}, ...], ...]}

exclude_ids 为我们的扩展字段（逐 query 对齐，训练时屏蔽被抽取论文自身）；
Search-R1 客户端不传时行为与官方一致。

启动: python3 -m uvicorn src.retrieval.service:app --port 8000
索引目录用环境变量 RETRIEVAL_INDEX_DIR 指定（默认 data/retrieval_index）。
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from src.retrieval.bm25_index import BM25Index

app = FastAPI()
_index: BM25Index | None = None


def get_index() -> BM25Index:
    global _index
    if _index is None:
        index_dir = os.environ.get("RETRIEVAL_INDEX_DIR", "data/retrieval_index")
        _index = BM25Index.load(index_dir)
    return _index


class SearchRequest(BaseModel):
    queries: list[str]
    topk: int = 3
    return_scores: bool = True
    exclude_ids: list[str] | None = None  # 逐 query 对齐；None=不屏蔽


@app.post("/retrieve")
def retrieve(req: SearchRequest):
    idx = get_index()
    excludes = req.exclude_ids or [None] * len(req.queries)
    result = []
    for q, ex in zip(req.queries, excludes):
        hits = idx.search(q, topk=req.topk, exclude_id=ex)
        result.append([
            {"document": {"contents": f"{h['title']}\n{h['abstract']}"}, "score": h["score"]}
            for h in hits
        ])
    return {"result": result}


@app.get("/health")
def health():
    idx = get_index()
    return {"status": "ok", "n_docs": len(idx.docs)}
