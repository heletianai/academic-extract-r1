"""蒸馏 prompt（教师=DeepSeek-V3，规则口径与 docs/schema-checklist.md FROZEN 版一字对齐）。

few-shot 三例为虚构摘要（真实论文进 few-shot = 该论文相对训练集泄漏，且虚构可精确
覆盖边界案例：多指标拆条/空 benchmarks/相对提升不抽/词面白名单/will be released）。
"""

SYSTEM_PROMPT = """You are a precise academic metadata extraction engine. Given a paper's title and abstract, output ONLY a JSON object with EXACTLY these 8 fields:

{
  "task_type": one of ["classification","generation","retrieval","reasoning","multimodal","agent","other"],
  "modalities": non-empty array from ["text","image","audio","video","code"],
  "benchmarks": array of {"name": str, "metric": str, "value": number},
  "open_source": boolean,
  "claims_sota": boolean,
  "method_keywords": array of 3-5 short method noun phrases,
  "one_line_summary": one sentence,
  "limitation_mentioned": str or null
}

RULES (follow exactly; when in doubt, follow the rule, not intuition):
1. Base every judgment ONLY on the given text. Do NOT use external knowledge about the paper, its authors, or its code — even if you recognize the paper.
2. task_type: the MAIN contribution's task. "multimodal" means the task itself is multimodal understanding/generation (e.g. VQA); a vision-language model doing pure classification -> "classification". Benchmark/evaluation papers -> the capability domain they evaluate. General-purpose LLM with no single task -> "other".
3. modalities: data modalities the paper processes. Text-only LLM -> ["text"]. Code datasets or code generation -> include "code".
   - Include a modality ONLY if the text explicitly indicates that data type is processed. A paper being "about multimodality" in general, or symbolic/tabular/time-series data, does NOT justify adding image/audio/video/code.
4. benchmarks: ONLY absolute numeric results of THIS paper's proposed method stated in the text.
   - NEVER output placeholder triples: if a benchmark is named but no absolute value is given, do NOT invent {"metric": "unspecified", "value": 0} — omit it entirely.
   - Copy name/metric/value in their original surface form (keep original scale: 85.3 or 0.853 as written).
   - One benchmark with N metrics -> N separate triples.
   - Do NOT extract: baseline/competitor numbers, relative improvements ("+3.2%", "improves by 5 points"), ranges, or vague claims.
   - No qualifying numbers in text -> [] (empty array, NOT null).
5. open_source: true if a code/data link is given OR the text states code is (or will be) released/open-sourced. Otherwise false.
6. claims_sota: true ONLY if the text ASSERTS this paper's method achieves/establishes/sets a result using whitelist phrases (case-insensitive): "state-of-the-art"/"SOTA", "outperforms all"/"surpasses all", "previous best". If such a phrase appears only as a comparison reference ("comparable to state-of-the-art", "close to SOTA", "matches SOTA") -> false. "outperforms strong baselines", "competitive" -> false.
7. method_keywords: 3-5 method/technique noun phrases (<=4 words each), prefer abbreviations (write "LoRA" not "Low-Rank Adaptation"). Exclude task names, benchmark names, and generic terms like "deep learning".
8. limitation_mentioned: quote/paraphrase the self-stated limitation if present, else null.
9. Output the JSON object only. No markdown, no explanation, no code fences."""

# ---- few-shot（虚构摘要，覆盖边界案例）----

FEW_SHOT_1_USER = """Title: RankFuse: Listwise Reranking with Distilled Cross-Encoders
Abstract: We present RankFuse, a listwise reranking method that distills cross-encoder signals into a lightweight bi-encoder. On MS MARCO, RankFuse achieves 41.2 MRR@10 and 89.5 Recall@1k, establishing a new state-of-the-art among sub-100M-parameter rerankers. Our code is available at https://github.com/example/rankfuse."""

FEW_SHOT_1_ASSISTANT = """{"task_type": "retrieval", "modalities": ["text"], "benchmarks": [{"name": "MS MARCO", "metric": "MRR@10", "value": 41.2}, {"name": "MS MARCO", "metric": "Recall@1k", "value": 89.5}], "open_source": true, "claims_sota": true, "method_keywords": ["listwise reranking", "knowledge distillation", "cross-encoder", "bi-encoder"], "one_line_summary": "RankFuse distills cross-encoder signals into a lightweight bi-encoder for listwise reranking.", "limitation_mentioned": null}"""

FEW_SHOT_2_USER = """Title: On the Epistemology of Benchmark Saturation
Abstract: We argue that benchmark saturation is a category error rather than an empirical milestone. Through a conceptual analysis of evaluation practices, we identify three fallacies in how the community interprets leaderboard progress. We discuss implications for the design of future evaluations, and note that our analysis is limited to English-language benchmarks."""

FEW_SHOT_2_ASSISTANT = """{"task_type": "other", "modalities": ["text"], "benchmarks": [], "open_source": false, "claims_sota": false, "method_keywords": ["conceptual analysis", "evaluation methodology", "benchmark design"], "one_line_summary": "A position paper arguing benchmark saturation is a category error, identifying three interpretive fallacies.", "limitation_mentioned": "analysis is limited to English-language benchmarks"}"""

FEW_SHOT_3_USER = """Title: GlyphVQA: Visual Question Answering over Historical Manuscripts
Abstract: We introduce GlyphVQA, a vision-language framework for question answering over scanned historical manuscripts. GlyphVQA couples a glyph-aware visual encoder with retrieval-augmented decoding, improving accuracy by 6.8 points over the strongest baseline and reaching 72.4% accuracy on DocVQA. Performance is comparable to state-of-the-art general VQA systems twice its size. Code and the GlyphBench dataset will be released."""

FEW_SHOT_3_ASSISTANT = """{"task_type": "multimodal", "modalities": ["text", "image"], "benchmarks": [{"name": "DocVQA", "metric": "accuracy", "value": 72.4}], "open_source": true, "claims_sota": false, "method_keywords": ["vision-language model", "retrieval-augmented decoding", "glyph-aware encoder"], "one_line_summary": "GlyphVQA answers questions over historical manuscripts via a glyph-aware encoder and retrieval-augmented decoding.", "limitation_mentioned": null}"""

# 第 4 例：克制案例（审计 B1——前三例全是"果断填满"，缺"该空就空"的示范：
# 模糊定量表述不抽数字、方法框架类骑墙 task_type 走 other、keywords 不硬凑）

FEW_SHOT_4_USER = """Title: Adaptive Curriculum Scheduling for Efficient LLM Post-Training
Abstract: We revisit data ordering in LLM post-training and propose an adaptive curriculum scheduler that reorders training samples by an online difficulty estimate. Across several instruction-following and reasoning suites, our schedule yields consistently strong results and notable efficiency gains over static curricula, while remaining simple to integrate into existing pipelines. We discuss connections to importance sampling and leave a full theoretical treatment to future work."""

FEW_SHOT_4_ASSISTANT = """{"task_type": "other", "modalities": ["text"], "benchmarks": [], "open_source": false, "claims_sota": false, "method_keywords": ["adaptive curriculum", "data scheduling", "difficulty estimation"], "one_line_summary": "An adaptive curriculum scheduler reorders LLM post-training data by online difficulty estimates for efficiency gains.", "limitation_mentioned": "full theoretical treatment left to future work"}"""


def build_messages(title: str, abstract: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_1_USER},
        {"role": "assistant", "content": FEW_SHOT_1_ASSISTANT},
        {"role": "user", "content": FEW_SHOT_2_USER},
        {"role": "assistant", "content": FEW_SHOT_2_ASSISTANT},
        {"role": "user", "content": FEW_SHOT_3_USER},
        {"role": "assistant", "content": FEW_SHOT_3_ASSISTANT},
        {"role": "user", "content": FEW_SHOT_4_USER},
        {"role": "assistant", "content": FEW_SHOT_4_ASSISTANT},
        {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"},
    ]
