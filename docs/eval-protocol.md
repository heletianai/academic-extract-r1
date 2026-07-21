# 评测协议（FROZEN 2026-07-21，与 schema 冻结同级——修改须 Fable 会话裁决）

> 审计 B3 抓出的未闭合变量：五方对照"各方用什么 prompt 生成预测"决定第一个对照数字的
> 公平性与可复现性，必须在跑数字前冻结。

## 一、五方对照生成配置

| 方 | 模型 | prompt 配置 | 依据 |
|---|---|---|---|
| 1 base few-shot | Qwen3-4B（未训） | **教师长 prompt + 4-shot**（`prompts.build_messages` 同款） | 给 base 满配援助——SFT 增量是"内化了长 prompt 规则"后的净增量，不靠削弱基线 |
| 2 API few-shot | DeepSeek-V3（deepseek-chat） | 同上（=教师配置，temp 0） | API 上限参照；与蒸馏教师同配置=教师即天花板的显式化 |
| 3 SFT | Qwen3-4B+LoRA | **学生短 prompt zero-shot**（`STUDENT_SYSTEM`） | 部署形态评测：短 prompt 省 token 是选型论点本身 |
| 4 SFT+GRPO | 同 3 | 同 3 | — |
| 5 +多轮 agentic | 同 3 | 短 prompt + 检索工具（Stage C） | — |

**口径声明（报告/口述必须带）**：1/2 方享受 few-shot 长 prompt、3/4/5 方零示例短 prompt——
这是**刻意的不对称**：对照回答的问题是"满配 prompt 的通用模型 vs 规则内化的专用小模型"，
即成本-效果选型问题本身，不是同 prompt 下的模型裸能力对比。防"故意弱基线"质疑的一手答案。

## 二、评测集三层

| 层 | 规模 | GT 来源 | 用途 |
|---|---|---|---|
| held-out | 200 | 教师蒸馏 | 主数字（口径：对教师参照系的拟合） |
| 人工核验子集 | 50 | `holdout_for_review.jsonl` 人工填 `human_extraction` | 真·黄金：打断教师自证循环的独立数字，与 held-out 并排报告 |
| MOLE 外部 | 可映射子集 | MOLE 官方标注 | 外部考场；**内外 gap 显式报告**（gap=教师偏差的量，主动讲=诚实叙事素材） |

## 三、指标与工具

- 统一入口 `src/eval/evaluate.py`：合法 JSON 率 / 逐字段 / overall（六字段均权，benchmarks 用 soft）/
  benchmarks hard-F1 单列 / bootstrap 95% CI
- 预测生成统一走 `src/eval/gen_predictions.py`（待写，Phase 0 清单件）：
  `--mode {base_fewshot | api_fewshot | student_zeroshot}` → 统一 `predictions.jsonl`
- run-log 纪律：每次报数必须 held-out / 人工子集 / MOLE 三列并排，禁止单报 held-out
