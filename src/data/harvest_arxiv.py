"""arXiv OAI-PMH 批量元数据获取（Kaggle 快照的替代实现，2026-07-21 换源）。

换源理由：本机无 Kaggle 凭证；OAI-PMH 是 arXiv 官方 bulk 通道，零凭证、
按时间窗精确拉取（只要 2025-07 后 cs 类，几十 MB），落地 jsonl 后管线仍是纯本地筛。
产物等价于"快照本地筛"，不引入运行时外网依赖。

用法：
    python3 -m src.data.harvest_arxiv            # 全量（断点续传，可 Ctrl-C 随时中断）
    python3 -m src.data.harvest_arxiv --pages 3  # 冒烟测试只拉 3 页

流控：OAI 规范用 503+Retry-After 限速，尊重之；正常页间 3s 礼貌延迟。
状态：data/raw/harvest_state.json 存 resumptionToken；输出 append 到 jsonl，
      中断重跑从 token 续，最终去重由 filter_papers.py 按 id 兜底。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_URL = "https://oaipmh.arxiv.org/oai"
FROM_DATE = "2025-07-01"   # 数据卡口径：created ≥ 2025H2（本地筛再按 created 精筛）
UNTIL_DATE = "2026-07-20"  # 快照右界定死，保证可复现
SET_SPEC = "cs"

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = RAW_DIR / "arxiv_cs_2025h2.jsonl"
STATE_PATH = RAW_DIR / "harvest_state.json"

NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}

POLITE_DELAY_S = 3.0
MAX_RETRIES = 5


def _fetch(url: str) -> bytes:
    """带 503 Retry-After 与指数退避的 GET。"""
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "academic-extract-r1/0.1 (research; single-user)"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 503:
                wait = int(e.headers.get("Retry-After", "10"))
                print(f"[flow-control] 503, wait {wait}s", flush=True)
                time.sleep(wait)
                continue
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"[retry {attempt+1}] {e}", flush=True)
            time.sleep(2 ** attempt * 2)
    raise RuntimeError("unreachable")


def _parse_records(xml_bytes: bytes) -> tuple[list[dict], str | None]:
    root = ET.fromstring(xml_bytes)
    err = root.find("oai:error", NS)
    if err is not None:
        code = err.get("code", "")
        if code == "noRecordsMatch":
            return [], None
        raise RuntimeError(f"OAI error {code}: {err.text}")

    records = []
    for rec in root.findall(".//oai:record", NS):
        meta = rec.find(".//arxiv:arXiv", NS)
        if meta is None:  # deleted 记录无 metadata
            continue

        def _text(tag: str) -> str:
            el = meta.find(f"arxiv:{tag}", NS)
            return (el.text or "").strip() if el is not None else ""

        records.append({
            "id": _text("id"),
            "created": _text("created"),
            "updated": _text("updated"),
            "title": " ".join(_text("title").split()),
            "abstract": " ".join(_text("abstract").split()),
            "categories": _text("categories"),
            "doi": _text("doi"),
            "license": _text("license"),
        })

    token_el = root.find(".//oai:resumptionToken", NS)
    token = token_el.text.strip() if (token_el is not None and token_el.text) else None
    # completeListSize 供进度显示
    if token_el is not None and token_el.get("completeListSize"):
        _parse_records.total = token_el.get("completeListSize")  # type: ignore[attr-defined]
    return records, token


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=0, help="仅拉 N 页（冒烟用），0=拉完")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    token: str | None = None
    n_written = 0
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
        token = state.get("resumptionToken")
        n_written = state.get("n_written", 0)
        print(f"[resume] token={str(token)[:40]}... already={n_written}", flush=True)

    page = 0
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        while True:
            if token:
                q = {"verb": "ListRecords", "resumptionToken": token}
            else:
                q = {
                    "verb": "ListRecords", "metadataPrefix": "arXiv",
                    "set": SET_SPEC, "from": FROM_DATE, "until": UNTIL_DATE,
                }
            url = f"{BASE_URL}?{urllib.parse.urlencode(q)}"
            xml_bytes = _fetch(url)
            try:
                records, token = _parse_records(xml_bytes)
            except ET.ParseError as e:
                print(f"[xml-error] {e}; retry once after 10s", flush=True)
                time.sleep(10)
                continue

            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n_written += len(records)
            page += 1

            STATE_PATH.write_text(json.dumps({"resumptionToken": token, "n_written": n_written}))
            total = getattr(_parse_records, "total", "?")
            print(f"[page {page}] +{len(records)} total={n_written}/{total} token={'yes' if token else 'DONE'}", flush=True)

            if token is None:
                print(f"[done] {n_written} records -> {OUT_PATH}", flush=True)
                STATE_PATH.write_text(json.dumps({"resumptionToken": None, "n_written": n_written, "done": True}))
                break
            if args.pages and page >= args.pages:
                print(f"[stop] --pages={args.pages} reached, token saved for resume", flush=True)
                break
            time.sleep(POLITE_DELAY_S)


if __name__ == "__main__":
    sys.exit(main())
