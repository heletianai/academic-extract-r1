"""BM25 检索索引（Stage C 检索端核心，手册防线：本地语料库不走外网）。

语料 = harvest 摘要池（title+abstract），任务语义 = 检索相关论文佐证
benchmarks/claims_sota 类信息不足型字段的判断（B 阶段 diff 定位的增量空间）。

设计约束：
- 纯逻辑零服务依赖，可单测（service.py 只是 FastAPI 壳）
- 索引落盘 = 纯 JSON（tokenized corpus + 元数据），加载重建 BM25——不用 pickle，
  格式安全可读，5 万条重建只需秒级
- 查询归一与建库归一必须同一函数（_tokenize），防训练/服务分词漂移
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, bm25: BM25Okapi, docs: list[dict]):
        self.bm25 = bm25
        self.docs = docs  # 每条: {"id": arxiv_id, "title": ..., "abstract": ...}

    # ---------- 构建 ----------
    @classmethod
    def build(cls, papers: list[dict]) -> "BM25Index":
        docs, corpus = [], []
        for p in papers:
            title = (p.get("title") or "").strip()
            abstract = (p.get("abstract") or "").strip()
            if not title and not abstract:
                continue
            docs.append({"id": p.get("id", ""), "title": title, "abstract": abstract})
            corpus.append(_tokenize(f"{title} {abstract}"))
        if not docs:
            raise ValueError("空语料，无法建索引")
        return cls(BM25Okapi(corpus), docs)

    @classmethod
    def build_from_jsonl(cls, path: str | Path) -> "BM25Index":
        papers = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        return cls.build(papers)

    # ---------- 查询 ----------
    def search(self, query: str, topk: int = 3, exclude_id: str | None = None) -> list[dict]:
        """返回 [{"id","title","abstract","score"}]，按分降序。

        exclude_id：训练时屏蔽被抽取论文自身（防检索到自己=信息泄漏式捷径）。
        """
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order:
            if exclude_id and self.docs[i]["id"] == exclude_id:
                continue
            out.append({**self.docs[i], "score": round(float(scores[i]), 4)})
            if len(out) >= topk:
                break
        return out

    # ---------- 持久化（纯 JSON，无 pickle） ----------
    def save(self, dir_path: str | Path) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "docs.jsonl", "w", encoding="utf-8") as f:
            for doc in self.docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, dir_path: str | Path) -> "BM25Index":
        """从 docs.jsonl 重建（tokenize+BM25 构建，5 万条秒级）。"""
        d = Path(dir_path)
        docs = [json.loads(l) for l in open(d / "docs.jsonl", encoding="utf-8")]
        corpus = [_tokenize(f"{doc['title']} {doc['abstract']}") for doc in docs]
        return cls(BM25Okapi(corpus), docs)


def format_passages(results: list[dict]) -> str:
    """检索结果 → 注入 <information> 块的文本（对齐 Search-R1 _passages2string 格式）。"""
    lines = []
    for idx, r in enumerate(results):
        lines.append(f"Doc {idx + 1}(Title: {r['title']}) {r['abstract']}")
    return "\n".join(lines)
