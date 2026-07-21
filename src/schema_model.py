"""Schema validator 三用件：蒸馏 rejection filter / GRPO reward 硬门控 / eval 合法性判定。

范式签名（总纲§五-8）：同一个 validator 贯穿合成→训练→评测（verifier-as-reward）。
写法来源：MeXtract schema.py `ConfigDict(extra='forbid', strict=False)` + validate_a 的
cast 容错精神；三态判定对齐 reward v1 门控/SO-Bench α 乘子（总纲§三）。

三态：
- INVALID        → reward 门控层固定负分（结构崩：缺键/多余键/类型错/枚举非法）
- SOFT_VIOLATION → 合法但软违规，α_schema=0.8（数量约束出界/value 非数值/modalities 空）
- VALID          → α_schema=1.0
cast 容错（不算违规）：枚举大小写、首尾空格、"true"/"85.3" 字符串数值——结果导向，
大小写噪声不该主导 reward（extract0/MeXtract 均规范化后比较）。
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

TASK_TYPES = ("classification", "generation", "retrieval", "reasoning", "multimodal", "agent", "other")
MODALITIES = ("text", "image", "audio", "video", "code")

VALID = "VALID"
SOFT_VIOLATION = "SOFT_VIOLATION"
INVALID = "INVALID"


class BenchmarkTriple(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=False)
    name: str
    metric: str
    value: Union[float, int, str]  # str 记 soft（matcher 的 parse_value 兜底解析）


class PaperExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=False)

    task_type: Literal[TASK_TYPES]  # type: ignore[valid-type]
    modalities: list[Literal[MODALITIES]]  # type: ignore[valid-type]
    benchmarks: list[BenchmarkTriple]
    open_source: bool
    claims_sota: bool
    method_keywords: list[str]
    one_line_summary: str
    limitation_mentioned: Union[str, None]

    @field_validator("task_type", mode="before")
    @classmethod
    def _norm_task(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("modalities", mode="before")
    @classmethod
    def _norm_modalities(cls, v):
        if isinstance(v, list):
            return [x.strip().lower() if isinstance(x, str) else x for x in v]
        return v

    @field_validator("method_keywords", mode="before")
    @classmethod
    def _norm_keywords(cls, v):
        if isinstance(v, list):
            return [x.strip() if isinstance(x, str) else x for x in v]
        return v


def validate_extraction(obj) -> dict:
    """三态判定入口。永不抛异常（reward 路径纪律）。

    返回 {"status": VALID|SOFT_VIOLATION|INVALID,
          "alpha": 1.0|0.8|None,
          "errors": [str], "soft_flags": [str],
          "parsed": dict|None}
    """
    if not isinstance(obj, dict):
        return {"status": INVALID, "alpha": None,
                "errors": [f"not a JSON object: {type(obj).__name__}"],
                "soft_flags": [], "parsed": None}
    try:
        model = PaperExtraction.model_validate(obj)
    except ValidationError as e:
        errors = [f"{'.'.join(str(x) for x in err['loc'])}: {err['type']}" for err in e.errors()]
        return {"status": INVALID, "alpha": None, "errors": errors, "soft_flags": [], "parsed": None}

    soft_flags: list[str] = []
    if not (3 <= len(model.method_keywords) <= 5):
        soft_flags.append(f"method_keywords count {len(model.method_keywords)} not in 3-5")
    if len(model.modalities) == 0:
        soft_flags.append("modalities empty")
    if len(model.modalities) != len(set(model.modalities)):
        soft_flags.append("modalities has duplicates")
    for i, t in enumerate(model.benchmarks):
        if isinstance(t.value, str):
            soft_flags.append(f"benchmarks[{i}].value is str not number")

    if soft_flags:
        return {"status": SOFT_VIOLATION, "alpha": 0.8, "errors": [],
                "soft_flags": soft_flags, "parsed": model.model_dump()}
    return {"status": VALID, "alpha": 1.0, "errors": [], "soft_flags": [],
            "parsed": model.model_dump()}
