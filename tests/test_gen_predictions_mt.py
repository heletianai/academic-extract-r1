"""多轮评测件单测：answer 抽取边界 + prompt 口径对拍（#012 防回归）+ 贪心分支 + e2e。"""

import json

from src.eval.gen_predictions_mt import (
    build_mt_pred_messages,
    extract_last_answer,
    run_predictions,
)
from src.training.grpo_train_mt import build_mt_rows
from src.training.multiturn_loop import MultiTurnRollout
from tests.test_multiturn_loop import ScriptedModel, ToyTok, tiny_index


def test_extract_last_answer_variants():
    assert extract_last_answer("") == ""
    assert extract_last_answer("no tags") == ""
    assert extract_last_answer('<answer>{"a":1}</answer>') == '{"a":1}'
    # 多 answer 取最后（纠错重试后以最终作答为准）
    two = "<answer>draft</answer> mid <answer>final</answer>"
    assert extract_last_answer(two) == "final"
    # DOTALL 跨行
    assert extract_last_answer("<answer>{\n \"k\": 1\n}</answer>") == '{\n "k": 1\n}'


def test_build_mt_pred_messages_matches_training(tmp_path):
    """#012 钉子：评测 prompt 必须与 build_mt_rows 训练构造逐字一致。"""
    row = {
        "id": "2507.00001",
        "messages": [
            {"role": "system", "content": "SYS with full schema rules"},
            {"role": "user", "content": "Title: X\nAbstract: Y"},
            {"role": "assistant", "content": '{"gold": true}'},
        ],
    }
    p = tmp_path / "train.jsonl"
    p.write_text(json.dumps(row) + "\n")
    train_row = build_mt_rows(str(p))[0]

    pred_msgs = build_mt_pred_messages(row)
    assert pred_msgs[0]["content"] == train_row["prompt"][0]["content"]
    assert pred_msgs[1]["content"] == train_row["prompt"][1]["content"]


class RecordingModel(ScriptedModel):
    """记录 generate 收到的采样 kwargs（贪心分支断言用）。"""

    def __init__(self, script):
        super().__init__(script)
        self.calls: list[dict] = []

    def generate(self, input_ids, attention_mask, **kw):
        self.calls.append(kw)
        return super().generate(input_ids, attention_mask, **kw)


def test_generate_greedy_when_temp0():
    model = RecordingModel(["<answer>{}</answer>"])
    rollout = MultiTurnRollout(model, ToyTok(), tiny_index(), temperature=0.0)
    rollout.run_batch([[{"role": "user", "content": "q"}]])
    assert all(c["do_sample"] is False for c in model.calls)
    assert all("temperature" not in c for c in model.calls)

    model2 = RecordingModel(["<answer>{}</answer>"])
    rollout2 = MultiTurnRollout(model2, ToyTok(), tiny_index(), temperature=1.2)
    rollout2.run_batch([[{"role": "user", "content": "q"}]])
    assert all(c["do_sample"] is True and c["temperature"] == 1.2 for c in model2.calls)


def test_run_predictions_e2e(tmp_path):
    """剧本 search→answer：输出行含最终 answer 与行为统计。"""
    model = ScriptedModel(["<search>grpo</search>", '<answer>{"f": 1}</answer>'])
    rollout = MultiTurnRollout(model, ToyTok(), tiny_index(), temperature=0.0)
    rows = [{
        "id": "p9",
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ],
    }]
    out = tmp_path / "pred.jsonl"
    run_predictions(rollout, rows, out, mode="mt_grpo", model_label="lora-x", batch_size=8)

    got = json.loads(out.read_text().strip())
    assert got["id"] == "p9"
    assert got["raw_text"] == '{"f": 1}'
    assert got["mode"] == "mt_grpo"
    assert got["answered"] is True
    assert got["searches"] == 1
    assert got["queries"] == ["grpo"]
