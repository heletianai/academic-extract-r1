"""grounding 核对单测。fixture 直接取 prompts.py few-shot（教师口径的正样本），
反例对齐冒烟抽检实锤的两类错误模式（issues-log #004：占位三元组/modalities 外推）。"""

import json

import pytest

from src.data.grounding import (
    _text_numbers,
    check_benchmarks,
    check_modalities,
    check_open_source,
    grounding_check,
)
from src.data.prompts import (
    FEW_SHOT_1_ASSISTANT,
    FEW_SHOT_1_USER,
    FEW_SHOT_3_ASSISTANT,
    FEW_SHOT_3_USER,
)


def _split_fewshot(user_text):
    title, abstract = user_text.split("\nAbstract: ", 1)
    return title.removeprefix("Title: "), abstract


# ---- 正样本：few-shot 教师标准答案应零 flag（口径对齐回归锚）----

def test_fewshot1_clean():
    title, abstract = _split_fewshot(FEW_SHOT_1_USER)
    ex = json.loads(FEW_SHOT_1_ASSISTANT)
    assert grounding_check(ex, title, abstract) == []


def test_fewshot3_clean():
    # GlyphVQA：image 证据=visual/scanned；open_source 证据="will be released"
    title, abstract = _split_fewshot(FEW_SHOT_3_USER)
    ex = json.loads(FEW_SHOT_3_ASSISTANT)
    assert grounding_check(ex, title, abstract) == []


# ---- 数字抽取 ----

def test_number_extraction_variants():
    nums = _text_numbers("achieves 41.2 MRR and 89.5% recall, 1,234 samples, ending 72.4.")
    assert {41.2, 89.5, 1234.0, 72.4} <= nums


def test_number_extraction_rejects_version_and_inner():
    nums = _text_numbers("Qwen2.5 improves over GPT-4 to 285.3 points")
    assert 285.3 in nums
    assert 2.5 not in nums          # 版本号前缀是字母
    assert 85.3 not in nums         # 285.3 内部不截断


def test_number_extraction_scientific():
    assert 0.0001 in _text_numbers("learning rate of 1e-4 works best")


# ---- 占位/编造三元组（#8 型）----

def test_fabricated_value_flagged():
    text = "We evaluate on GLUE and achieve strong results."
    flags = check_benchmarks([{"name": "GLUE", "metric": "accuracy", "value": 85.0}], text)
    assert any(".value 85.0 not found" in f for f in flags)


def test_placeholder_metric_flagged():
    text = "We evaluate on GLUE and achieve 85.0 accuracy."
    flags = check_benchmarks([{"name": "GLUE", "metric": "unspecified", "value": 0}], text)
    assert any("placeholder" in f for f in flags)
    assert any(".value 0 not found" in f for f in flags)


def test_name_not_in_text_flagged():
    text = "Our method reaches 91.0 F1 on the benchmark suite."
    flags = check_benchmarks([{"name": "SuperGLUE", "metric": "F1", "value": 91.0}], text)
    assert any(".name" in f for f in flags)
    assert not any(".value" in f for f in flags)


def test_name_hyphen_variant_matches():
    # MS-MARCO vs "MS MARCO"：tokenize 天然吸收连字符差异
    text = "On MS MARCO, we reach 41.2 MRR@10."
    assert check_benchmarks([{"name": "MS-MARCO", "metric": "MRR@10", "value": 41.2}], text) == []


def test_name_plural_variant_matches():
    # 冒烟校准实测误报（2509.04482）："hard abstention split" vs 原文 "splits"
    text = "across both easy and hard abstention splits, AUROC reaches 0.961"
    assert check_benchmarks(
        [{"name": "hard abstention split", "metric": "AUROC", "value": 0.961}], text) == []


def test_str_value_parsed_and_matched():
    text = "reaching 72.4% accuracy on DocVQA"
    assert check_benchmarks([{"name": "DocVQA", "metric": "accuracy", "value": "72.4%"}], text) == []


# ---- modalities 外推（#33/#1/#22 型）----

def test_modality_image_without_evidence_flagged():
    text = "We propose a text classification method with strong results on GLUE."
    flags = check_modalities(["text", "image"], text)
    assert any("'image' lacks" in f for f in flags)


def test_visualization_not_image_evidence():
    text = "We provide a visualization of attention weights for interpretability."
    assert check_modalities(["image"], text) != []


def test_modality_image_with_evidence_clean():
    assert check_modalities(["text", "image"], "a vision-language model for scanned documents") == []


def test_modality_audio_speech_evidence():
    assert check_modalities(["text", "audio"], "end-to-end speech recognition") == []


def test_code_release_context_not_evidence():
    # "code is available" 是 open_source 证据，不是 code 模态证据
    text = "A text summarization model. Our code is available at github.com/x/y."
    flags = check_modalities(["text", "code"], text)
    assert any("'code' lacks" in f for f in flags)
    assert check_open_source(True, text) == []


def test_code_generation_is_evidence():
    assert check_modalities(["text", "code"], "We study code generation with LLMs.") == []


# ---- open_source ----

def test_open_source_true_without_evidence_flagged():
    assert check_open_source(True, "We propose a new method.") != []


def test_open_source_false_never_flagged():
    assert check_open_source(False, "We propose a new method.") == []


@pytest.mark.parametrize("phrase", [
    "Code and data will be released.",
    "Our implementation is publicly available.",
    "We open-source all checkpoints.",
    "available at https://github.com/x/y",
    # 冒烟校准实测误报（2604.17323）：URL 直给+匿名平台
    "The full code is https://anonymous.4open.science/r/2026_ACL_Universal/.",
])
def test_open_source_evidence_variants(phrase):
    assert check_open_source(True, f"We propose a method. {phrase}") == []


# ---- 纪律：永不抛 ----

def test_never_raises_on_malformed():
    flags = grounding_check({"benchmarks": "not-a-list", "modalities": None}, "t", "a")
    assert isinstance(flags, list)
