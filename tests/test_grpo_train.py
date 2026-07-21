"""grpo_train 纯函数层单测（GPU 路径留给实机冒烟——手册§十：本地兜纯函数，冒烟兜实机）。"""

import json

from src.training.grpo_train import RewardLogger, build_grpo_rows

GOOD = json.dumps({
    "task_type": "retrieval", "modalities": ["text"],
    "benchmarks": [{"name": "MS MARCO", "metric": "MRR@10", "value": 41.2}],
    "open_source": True, "claims_sota": False,
    "method_keywords": ["listwise reranking", "distillation", "bi-encoder"],
    "one_line_summary": "s", "limitation_mentioned": None,
})


def _mk_logger(tmp_path, G=2):
    return RewardLogger(tmp_path / "detail.jsonl", num_generations=G, flush_every=1)


def test_reward_logger_returns_floats_and_flushes(tmp_path):
    lg = _mk_logger(tmp_path)
    rewards = lg([GOOD, "not json at all"], gold_extraction=[GOOD, GOOD])
    assert len(rewards) == 2
    assert rewards[0] > 0.9        # 完全命中 gold
    assert rewards[1] == -1.0      # 硬门控
    lines = [json.loads(l) for l in open(tmp_path / "detail.jsonl")]
    assert lines[0]["gate_rate"] == 0.5
    assert lines[0]["gates"] == {"no_json": 1}


def test_group_dedup_rate(tmp_path):
    lg = _mk_logger(tmp_path, G=2)
    # 组内两条完全相同 → 去重率 0.5（多样性塌缩信号，A2 观察线）
    lg([GOOD, GOOD], gold_extraction=[GOOD, GOOD])
    row = json.loads(open(tmp_path / "detail.jsonl").readline())
    assert row["group_dedup_mean"] == 0.5


def test_task_type_entropy(tmp_path):
    lg = _mk_logger(tmp_path, G=2)
    other = GOOD.replace('"retrieval"', '"classification"')
    lg([GOOD, other], gold_extraction=[GOOD, GOOD])
    row = json.loads(open(tmp_path / "detail.jsonl").readline())
    assert row["task_type_entropy"] == 1.0   # 两值均分 → 1 bit；全同 → 0（hacking 信号）


def test_chat_format_completions(tmp_path):
    lg = _mk_logger(tmp_path)
    rewards = lg([[{"role": "assistant", "content": GOOD}], [{"role": "assistant", "content": GOOD}]],
                 gold_extraction=[GOOD, GOOD])
    assert all(r > 0.9 for r in rewards)


def test_build_grpo_dataset(tmp_path):
    p = tmp_path / "sft.jsonl"
    row = {"id": "x", "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Title: T\nAbstract: A"},
        {"role": "assistant", "content": GOOD},
    ]}
    p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    rows = build_grpo_rows(str(p), max_prompts=1)
    assert len(rows) == 1
    assert len(rows[0]["prompt"]) == 2                    # 只保留 system+user
    assert rows[0]["prompt"][1]["role"] == "user"
    assert json.loads(rows[0]["gold_extraction"])["task_type"] == "retrieval"
