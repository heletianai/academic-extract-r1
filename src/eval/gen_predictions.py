"""五方对照预测生成（eval-protocol.md FROZEN §一/§三：统一入口，Phase 0 清单件）。

三 mode（prompt 配置=协议表 1-3 行，修改须 Fable 会话裁决）：
  api_fewshot     DeepSeek-V3 + 教师长 prompt 4-shot + temp 0（=蒸馏教师同配置：
                  held-out 上即"教师即天花板"的显式化，报数必须带口径声明）
  base_fewshot    未训 Qwen3-4B + 同款教师 prompt（OpenAI 兼容端点：vLLM serve）
  student_zeroshot SFT/GRPO 模型 + STUDENT_SYSTEM 短 prompt 零示例（部署形态）

输入 holdout.jsonl 为 messages 格式：user turn 与 build_messages 末条一字相同，
直接透传（零反解漂移）。输出 {"id","raw_text",...}：被测方一次作答不过 validator
重试环（蒸馏 GT 生产才有容错，评测口径=公平一枪）；网络层重试保留。

用法：
    python3 -m src.eval.gen_predictions --mode api_fewshot --gold data/processed/holdout.jsonl
    python3 -m src.eval.gen_predictions --mode base_fewshot --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

from src.data.distill import Usage, peak_warning
from src.data.prompts import SYSTEM_PROMPT, build_messages
from src.data.to_sft_format import STUDENT_SYSTEM

ROOT = Path(__file__).resolve().parents[2]

MODES = ("api_fewshot", "base_fewshot", "student_zeroshot")
MAX_TOKENS = 512
MAX_API_RETRIES = 4


def user_content(row: dict) -> str:
    """gold 行 → 最终 user turn。messages 格式透传；裸 title/abstract 兜底同模板。"""
    for m in row.get("messages", []):
        if m["role"] == "user":
            return m["content"]
    return f"Title: {row.get('title', '')}\nAbstract: {row.get('abstract', '')}"


def build_pred_messages(mode: str, row: dict) -> list[dict]:
    uc = user_content(row)
    if mode in ("api_fewshot", "base_fewshot"):
        # 教师长 prompt + 4-shot：build_messages 末条模板与 holdout user turn 同构，直接替换
        msgs = build_messages("", "")[:-1]
        assert msgs[0]["content"] == SYSTEM_PROMPT
        return msgs + [{"role": "user", "content": uc}]
    return [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": uc}]


async def call_model(client: AsyncOpenAI, sem: asyncio.Semaphore, model: str,
                     messages: list[dict], usage: Usage) -> str | None:
    async with sem:
        for attempt in range(MAX_API_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=model, messages=messages, temperature=0.0,
                    max_tokens=MAX_TOKENS, response_format={"type": "json_object"},
                )
                usage.add(resp.usage)
                return resp.choices[0].message.content or ""
            except (RateLimitError, APITimeoutError) as e:
                print(f"{type(e).__name__}, wait {2 ** attempt * 3}s", flush=True)
                await asyncio.sleep(2 ** attempt * 3)
            except APIError as e:
                if attempt == MAX_API_RETRIES - 1:
                    print(f"APIError final: {e}", flush=True)
                    return None
                await asyncio.sleep(2 ** attempt * 2)
        return None


async def amain(args) -> None:
    if args.mode == "api_fewshot":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            sys.exit("api_fewshot 需要 DEEPSEEK_API_KEY 环境变量（不落盘）")
        if api_key.startswith("sk-14c605b4"):
            sys.exit("检测到已泄漏旧 key，拒绝运行（platform.deepseek.com 轮换）")
        base_url, model = "https://api.deepseek.com", args.model or "deepseek-chat"
        peak_warning()
    else:
        api_key = os.environ.get("LOCAL_API_KEY", "EMPTY")  # vLLM serve 不校验
        base_url = args.base_url
        if not args.model:
            sys.exit(f"{args.mode} 需要 --model（端点上的模型名）")
        model = args.model

    rows = []
    with open(args.gold, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    out_path = Path(args.out or ROOT / "runs" / f"pred-{args.mode}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(str(json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [r for r in rows if str(r["id"]) not in done_ids]
    if args.limit:
        todo = todo[: args.limit]
    print(f"mode={args.mode} model={model} gold={len(rows)} done={len(done_ids)} todo={len(todo)}", flush=True)
    if not todo:
        return

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=180)
    sem = asyncio.Semaphore(args.concurrency)
    usage = Usage()
    lock = asyncio.Lock()
    n_done = 0

    async def one(row: dict) -> None:
        nonlocal n_done
        text = await call_model(client, sem, model, build_pred_messages(args.mode, row), usage)
        async with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "id": row["id"], "raw_text": text if text is not None else "",
                    "mode": args.mode, "model": model, "api_failed": text is None,
                }, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 20 == 0:
                print(f"[{n_done}/{len(todo)}] {usage.report()}", flush=True)

    t0 = time.time()
    await asyncio.gather(*[one(r) for r in todo])
    print(f"[done] {len(todo)} preds in {time.time() - t0:.0f}s -> {out_path} | {usage.report()}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--gold", default=str(ROOT / "data" / "processed" / "holdout.jsonl"))
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
