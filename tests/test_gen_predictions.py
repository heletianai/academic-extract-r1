"""gen_predictions prompt 构造单测（协议表 1-3 行的形状锁定；IO 主流程为脚本级不单测）。"""

import json

from src.data.prompts import SYSTEM_PROMPT, build_messages
from src.data.to_sft_format import STUDENT_SYSTEM, to_messages
from src.eval.gen_predictions import build_pred_messages, user_content

GOLD_ROW = {
    "id": "2507.00001",
    "messages": [
        {"role": "system", "content": STUDENT_SYSTEM},
        {"role": "user", "content": "Title: T\nAbstract: A"},
        {"role": "assistant", "content": "{}"},
    ],
}


def test_user_content_from_messages_passthrough():
    assert user_content(GOLD_ROW) == "Title: T\nAbstract: A"


def test_user_content_bare_fallback_same_template():
    # 裸格式兜底与 to_sft_format 模板一字相同（零反解漂移）
    bare = {"id": "x", "title": "T", "abstract": "A"}
    row = {"id": "x", "title": "T", "abstract": "A", "extraction": {}}
    assert user_content(bare) == to_messages({**row})["messages"][1]["content"]


def test_api_fewshot_is_teacher_config():
    # 协议表 2 行：教师长 prompt + 4-shot，与 build_messages 完全同构
    msgs = build_pred_messages("api_fewshot", GOLD_ROW)
    ref = build_messages("T", "A")
    assert msgs == ref  # 同 system、同 4-shot、同末条 user —— 教师即天花板显式化


def test_base_fewshot_same_prompt_as_api():
    assert build_pred_messages("base_fewshot", GOLD_ROW) == build_pred_messages("api_fewshot", GOLD_ROW)


def test_student_zeroshot_short_prompt():
    msgs = build_pred_messages("student_zeroshot", GOLD_ROW)
    assert len(msgs) == 2
    assert msgs[0] == {"role": "system", "content": STUDENT_SYSTEM}
    assert msgs[1]["content"] == "Title: T\nAbstract: A"
    assert SYSTEM_PROMPT not in json.dumps(msgs)  # 部署形态不带教师长 prompt
