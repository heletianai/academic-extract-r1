"""DeepSeek-V3 异步并发蒸馏（教师=deepseek-chat 稳定版，7.21 口径：不追 V4 预览）。

管线：papers_pool.jsonl → 并发调用（semaphore 限流+指数退避）→ JSON 解析
     → validator 过滤（INVALID 升温重试一次，仍挂进 rejected.jsonl = rejection filter 留痕）
     → distilled.jsonl（断点续传：已有 id 跳过）
安全：key 只读环境变量 DEEPSEEK_API_KEY，缺失即退出（sk-14c605b4 旧 key 已泄漏严禁使用）。
成本：usage 累计+价目估算（价格快照 2026-07，cache hit/miss 细分，以账单为准）；
     北京时间 9-12/14-18 高峰两倍价，启动时警告不拦截（冒烟随时跑，放量锁低谷）。

用法：
    python3 -m src.data.distill --limit 50                 # 50 条冒烟
    python3 -m src.data.distill                            # 全量（断点续传）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

from src.data.grounding import grounding_check
from src.data.prompts import build_messages
from src.data.sota_regex import claims_sota_by_regex
from src.schema_model import INVALID, validate_extraction

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "processed" / "papers_pool.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "distilled.jsonl"
REJ_PATH = ROOT / "data" / "processed" / "rejected.jsonl"

MODEL = "deepseek-chat"  # V3 稳定版
BASE_URL = "https://api.deepseek.com"
MAX_TOKENS = 512
TEMP_MAIN = 0.0    # 蒸馏 GT 要确定性
TEMP_RETRY = 0.6   # INVALID 重试时升温换采样
CONCURRENCY = 8
MAX_API_RETRIES = 4

# 价格快照 2026-07（¥/M token，标准时段；低谷约半价）——估算用，以账单为准
PRICE = {"input_miss": 2.0, "input_hit": 0.5, "output": 8.0}


def peak_warning() -> None:
    bj = datetime.now(timezone(timedelta(hours=8)))
    if 9 <= bj.hour < 12 or 14 <= bj.hour < 18:
        print(f"[warn] 北京时间 {bj:%H:%M} 处于高峰时段(9-12/14-18)，API 两倍价——冒烟无所谓，放量请换低谷", flush=True)


class Usage:
    def __init__(self):
        self.hit = 0
        self.miss = 0
        self.out = 0
        self.calls = 0

    def add(self, u) -> None:
        if u is None:
            return
        self.calls += 1
        hit = getattr(u, "prompt_cache_hit_tokens", 0) or 0
        self.hit += hit
        self.miss += (u.prompt_tokens or 0) - hit
        self.out += u.completion_tokens or 0

    def cost_cny(self) -> float:
        return (self.miss * PRICE["input_miss"] + self.hit * PRICE["input_hit"]
                + self.out * PRICE["output"]) / 1e6

    def report(self) -> str:
        return (f"calls={self.calls} in_miss={self.miss} in_hit={self.hit} out={self.out} "
                f"≈¥{self.cost_cny():.3f}(标准价, 低谷约半)")


async def call_teacher(client: AsyncOpenAI, sem: asyncio.Semaphore, paper: dict,
                       usage: Usage, temperature: float) -> dict | None:
    async with sem:
        for attempt in range(MAX_API_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=build_messages(paper["title"], paper["abstract"]),
                    temperature=temperature,
                    max_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
                usage.add(resp.usage)
                text = resp.choices[0].message.content or ""
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"__parse_error__": text[:500]}
            except (RateLimitError, APITimeoutError) as e:
                wait = 2 ** attempt * 3
                print(f"[{paper['id']}] {type(e).__name__}, wait {wait}s", flush=True)
                await asyncio.sleep(wait)
            except APIError as e:
                if attempt == MAX_API_RETRIES - 1:
                    print(f"[{paper['id']}] APIError final: {e}", flush=True)
                    return None
                await asyncio.sleep(2 ** attempt * 2)
        return None


async def process_paper(client: AsyncOpenAI, sem: asyncio.Semaphore, paper: dict,
                        usage: Usage, out_f, rej_f, lock: asyncio.Lock, counters: dict) -> None:
    obj = await call_teacher(client, sem, paper, usage, TEMP_MAIN)
    v = validate_extraction(obj) if obj is not None else {"status": INVALID, "errors": ["api_failed"], "soft_flags": []}
    used_temp = TEMP_MAIN

    if v["status"] == INVALID:  # 升温重试一次（rejection sampling 精神）
        obj2 = await call_teacher(client, sem, paper, usage, TEMP_RETRY)
        v2 = validate_extraction(obj2) if obj2 is not None else {"status": INVALID, "errors": ["api_failed"], "soft_flags": []}
        if v2["status"] != INVALID:
            obj, v = obj2, v2
            used_temp = TEMP_RETRY  # provenance 记实际采纳温度（审计 fix）

    row = {
        "id": paper["id"], "created": paper["created"], "title": paper["title"],
        "abstract": paper["abstract"], "categories": paper["categories"],
        "teacher": MODEL, "temperature": used_temp,
        "validator_status": v["status"], "soft_flags": v.get("soft_flags", []),
    }
    if v["status"] != INVALID:
        # claims_sota 正则交叉核对（checklist 硬化#2）：不一致样本是抽检优先队列
        regex_sota = claims_sota_by_regex(f"{paper['title']}. {paper['abstract']}")
        row["sota_regex"] = regex_sota
        row["sota_disagree"] = bool(v["parsed"]["claims_sota"]) != regex_sota
        # grounding 自动核对（issues-log #004）：value/modality 编造粗筛，进 review 优先队列
        row["grounding_flags"] = grounding_check(v["parsed"], paper["title"], paper["abstract"])
        if row["grounding_flags"]:
            counters["flagged"] += 1
    async with lock:
        if v["status"] == INVALID:
            row["errors"] = v.get("errors", [])
            row["raw"] = obj if isinstance(obj, dict) else str(obj)[:500]
            rej_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rej_f.flush()
            counters["rejected"] += 1
        else:
            row["extraction"] = v["parsed"]
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            counters["ok"] += 1
        done = counters["ok"] + counters["rejected"]
        if done % 10 == 0:
            print(f"[{done}/{counters['total']}] ok={counters['ok']} rej={counters['rejected']} "
                  f"flagged={counters['flagged']} {usage.report()}", flush=True)


async def amain(args) -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        sys.exit("缺 DEEPSEEK_API_KEY 环境变量。请轮换新 key 后 export DEEPSEEK_API_KEY=...（不落盘）")
    if api_key.startswith("sk-14c605b4"):
        sys.exit("检测到已泄漏的旧 key（sk-14c605b4...），拒绝运行。请到 platform.deepseek.com 轮换新 key。")

    peak_warning()

    papers = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))

    done_ids: set[str] = set()
    for p in (OUT_PATH, REJ_PATH):
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    todo = [p for p in papers if p["id"] not in done_ids]
    if args.limit:
        todo = todo[: args.limit]
    print(f"total={len(papers)} done={len(done_ids)} todo={len(todo)}", flush=True)
    if not todo:
        return

    client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL, timeout=120)
    sem = asyncio.Semaphore(args.concurrency)
    usage = Usage()
    lock = asyncio.Lock()
    counters = {"ok": 0, "rejected": 0, "flagged": 0, "total": len(todo)}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(OUT_PATH, "a", encoding="utf-8") as out_f, open(REJ_PATH, "a", encoding="utf-8") as rej_f:
        await asyncio.gather(*[
            process_paper(client, sem, p, usage, out_f, rej_f, lock, counters) for p in todo
        ])
    dt = time.time() - t0
    print(f"[done] ok={counters['ok']} rejected={counters['rejected']} "
          f"grounding_flagged={counters['flagged']} in {dt:.0f}s | {usage.report()}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(IN_PATH))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
