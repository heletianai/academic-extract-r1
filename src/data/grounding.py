"""教师输出 grounding 自动核对（issues-log #004 待办兑现：预检脚本组件化）。

定位：机器粗筛信号，生成 review 优先队列——不是审判者，不改 validator 三态。
核对目标 = 冒烟抽检实锤的两类教师系统性错误 + 顺带可验证项：
  1. 占位/编造三元组（#8 型）：value 数字不在原文 / metric·name 是占位词
  2. modalities 过度外推（#33/#1/#22 型）：非 text 模态无文本词面证据
  3. benchmark name 未按 surface form 复制（prompt 规则 4"原样复制"的可查半边）
  4. open_source=true 但全文无开源词面证据
词表口径对齐 prompts.SYSTEM_PROMPT 规则 3（"multimodal 泛指不算证据"）——
visualization/frame/clip 等高撞车词刻意排除，宁漏勿滥（flag 是人工队列入口，
误报率高会把队列淹掉）。误报率以 50 条冒烟校准为准（blacklist 4 条应命中）。

用法：
    python3 -m src.data.grounding --input data/processed/distilled.jsonl            # 统计报告
    python3 -m src.data.grounding --input data/processed/distilled.jsonl --update   # 回写 grounding_flags 字段
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# ---- 数字抽取（value grounding 的参照集）----
# 前缀拒 \w 和 .（防 Qwen2.5 的 2.5、285.3 的 85.3）；后缀只拒数字（允许 "72.4." 句末）。
_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?)(?!\d)")
_SCI_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?[eE][+-]?\d+)(?![\w.])")

# ---- 占位词（metric/name 编造检测，#8 型三元组的词面半边）----
_PLACEHOLDER = {"unspecified", "n/a", "na", "unknown", "none", "not specified", "null", ""}

# ---- 非 text 模态的词面证据表（窄表：宁漏勿滥）----
# visual 不带 \w*（visualization≠image 证据）；video 只认本词干（frame/clip 撞车）。
_MODALITY_EVIDENCE = {
    "image": re.compile(
        r"\b(images?|visual(?:ly)?|vision|photo\w*|pictures?|pixels?|ocr|scan(?:ned|s)?|imagery)\b",
        re.IGNORECASE),
    "audio": re.compile(
        r"\b(audio|speech|acoustic\w*|voice|spoken|asr|tts|phonemes?|music(?:al)?)\b",
        re.IGNORECASE),
    "video": re.compile(r"\b(videos?)\b", re.IGNORECASE),
}
# code 单列：词组白名单直接算证据；裸 "code" 需排除 release 语境
# （"code is available at github" 是 open_source 证据不是 code 模态证据）
_CODE_PHRASE = re.compile(
    r"\b(code\s+(?:generation|completion|search|review|repair|synthesis|translation)"
    r"|source\s+code|programming|coding|codebases?|program\s+synthesis|sql)\b",
    re.IGNORECASE)
_CODE_WORD = re.compile(r"\bcodes?\b", re.IGNORECASE)
_RELEASE_NEAR = re.compile(
    r"(available|released?|release|public(?:ly)?|open[-\s]?sourced?|github|gitlab"
    r"|repositor\w+|will\s+be|huggingface)", re.IGNORECASE)
_CODE_CTX_WINDOW = 50

# ---- open_source=true 的证据表（宽表：true 才查，证据宽 → 少误报）----
# 任意 URL 即证据：prompt 规则 5 "true if a code/data link is given"——含匿名平台
# （anonymous.4open.science 等）与 "The full code is <URL>" 直给模式（冒烟校准误报修复）
_OPEN_SOURCE_EVIDENCE = re.compile(
    r"(https?://\S+"
    r"|github\.com|gitlab|huggingface\.co"
    r"|open[-\s]?sourc\w+"
    r"|(?:code|data|dataset|model|checkpoint)s?\s+(?:is|are|will\s+be)\s+(?:made\s+)?(?:publicly\s+)?(?:available|released)"
    r"|will\s+be\s+(?:made\s+)?(?:publicly\s+)?(?:available|released)"
    r"|(?:publicly|freely)\s+available|available\s+at|released?\s+(?:at|our)"
    r"|we\s+(?:release|open[-\s]?source))", re.IGNORECASE)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _text_numbers(text: str) -> set[float]:
    nums: set[float] = set()
    for m in _NUM_RE.finditer(text):
        try:
            nums.add(round(float(m.group(1).replace(",", "")), 6))
        except ValueError:
            continue
    for m in _SCI_RE.finditer(text):
        try:
            nums.add(round(float(m.group(1)), 6))
        except ValueError:
            continue
    return nums


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _stem(tok: str) -> str:
    # 轻量单复数归一（冒烟校准误报修复："split" vs "splits"）。双方同规则剥尾 s，
    # 等价类只会粗化 → 只多放行不多 flag，与"宁漏勿滥"同向
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok


def _contains_seq(haystack: list[str], needle: list[str]) -> bool:
    if not needle:
        return True
    hay = [_stem(t) for t in haystack]
    ndl = [_stem(t) for t in needle]
    n = len(ndl)
    return any(hay[i:i + n] == ndl for i in range(len(hay) - n + 1))


def check_benchmarks(benchmarks: list, text: str) -> list[str]:
    flags: list[str] = []
    nums = _text_numbers(text)
    toks = _tokens(text)
    for i, t in enumerate(benchmarks):
        name = str(t.get("name", "")).strip()
        metric = str(t.get("metric", "")).strip()
        value = t.get("value")
        if name.lower() in _PLACEHOLDER or metric.lower() in _PLACEHOLDER:
            flags.append(f"bench[{i}] placeholder name/metric: {name!r}/{metric!r}")
        if isinstance(value, str):
            try:
                value = float(value.strip().rstrip("%"))
            except ValueError:
                continue  # 非数值 str 已有 SOFT flag 兜着，不重复
        if isinstance(value, (int, float)):
            if round(float(value), 6) not in nums:
                flags.append(f"bench[{i}].value {value} not found in text")
        if name and _tokens(name) and not _contains_seq(toks, _tokens(name)):
            flags.append(f"bench[{i}].name {name!r} not found in text")
    return flags


def check_modalities(modalities: list, text: str) -> list[str]:
    flags: list[str] = []
    for mod in modalities:
        if mod == "text":
            continue  # 全库论文默认涉文本，不设证据要求
        if mod == "code":
            if _CODE_PHRASE.search(text):
                continue
            evid = False
            for m in _CODE_WORD.finditer(text):
                ctx = text[max(0, m.start() - _CODE_CTX_WINDOW): m.end() + _CODE_CTX_WINDOW]
                if not _RELEASE_NEAR.search(ctx):
                    evid = True
                    break
            if not evid:
                flags.append("modalities: 'code' lacks textual evidence (only release-context mentions)")
        elif mod in _MODALITY_EVIDENCE:
            if not _MODALITY_EVIDENCE[mod].search(text):
                flags.append(f"modalities: {mod!r} lacks textual evidence")
    return flags


def check_open_source(open_source, text: str) -> list[str]:
    if open_source is True and not _OPEN_SOURCE_EVIDENCE.search(text):
        return ["open_source=true without textual evidence"]
    return []


def grounding_check(extraction: dict, title: str, abstract: str) -> list[str]:
    """核对入口：返回 flag 列表，空 = 全过。永不抛异常（与 validator 同纪律）。"""
    try:
        text = f"{title}. {abstract}"
        flags = check_benchmarks(extraction.get("benchmarks") or [], text)
        flags += check_modalities(extraction.get("modalities") or [], text)
        flags += check_open_source(extraction.get("open_source"), text)
        return flags
    except Exception as e:  # 粗筛信号不许砍断蒸馏主流程
        return [f"grounding_check_error: {type(e).__name__}: {e}"]


def _flag_category(flag: str) -> str:
    if "placeholder" in flag:
        return "placeholder"
    if ".value" in flag:
        return "bench_value"
    if ".name" in flag:
        return "bench_name"
    if "modalities" in flag:
        return "modality"
    if "open_source" in flag:
        return "open_source"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/distilled.jsonl")
    ap.add_argument("--update", action="store_true", help="回写 grounding_flags 字段")
    args = ap.parse_args()

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    n_checked = 0
    n_flagged = 0
    by_cat: dict[str, int] = {}
    flagged_ids: list[str] = []
    examples: list[dict] = []
    for r in rows:
        ex = r.get("extraction")
        if not isinstance(ex, dict):
            continue
        n_checked += 1
        flags = grounding_check(ex, r.get("title", ""), r.get("abstract", ""))
        r["grounding_flags"] = flags
        if flags:
            n_flagged += 1
            flagged_ids.append(str(r.get("id")))
            for fl in flags:
                by_cat[_flag_category(fl)] = by_cat.get(_flag_category(fl), 0) + 1
            if len(examples) < 10:
                examples.append({"id": r.get("id"), "flags": flags})

    report = {
        "input": args.input, "n_checked": n_checked, "n_flagged": n_flagged,
        "flag_rate": round(n_flagged / max(1, n_checked), 3),
        "by_category": by_cat,
    }
    # blacklist 召回校准：已人工判错的条目应被 flag 命中（冒烟校准的黄金标准）
    bl_path = Path(args.input).parent / "blacklist.txt"
    if bl_path.exists():
        bl = set(bl_path.read_text().split())
        if bl:
            hit = sorted(bl & set(flagged_ids))
            report["blacklist_recall"] = {"total": len(bl), "flagged": len(hit),
                                          "hit": hit, "missed": sorted(bl - set(flagged_ids))}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    for e in examples:
        print(f"  [{e['id']}] {e['flags']}")

    if args.update:
        with open(args.input, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[update] grounding_flags 已回写 -> {args.input}")


if __name__ == "__main__":
    main()
