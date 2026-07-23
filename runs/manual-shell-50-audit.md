# 独立复核子集 50 条审计报告（manual shell audit）

**日期**：2026-07-22　**对象**：`data/processed/holdout_for_review.jsonl`（50 条，**可疑优先采样**：sota_disagree ∪ grounding_flags 优先入壳，31 条带机器信号，19 条无信号）

## 口径声明（读数前必看）

- **复核者**：异构 LLM 三层交叉，非纯人工——机器 grounding flag（在线组件）→ 2×Opus 盲审（各 25 条，**不可见机器 flag**，互相独立）→ Fable 终裁（全部 HARD 判定与分歧条逐条读 abstract 原文裁决）。复核模型（Claude 家族）与教师（DeepSeek-V3）**非同家族**，与本项目"异构 judge 修自证循环"同一方法论。
- **判级**：HARD＝实质错误（编造/占位数值、无依据模态、与原文相反的布尔）；SOFT＝可辩护偏差（合成命名、delta 当绝对值、范围端点拆分、无依据但领域合理的模态推断）；OK＝完全忠实。
- **采样偏置**：本壳按可疑优先构造，是**偏坏样本**——下述错误率是全量数据的保守上界（一致率是保守下界）。

## Headline

| 指标 | 数值 |
|---|---|
| **条目级无实质错误（无 HARD）** | **47/50 = 94%** |
| 完全忠实（OK） | 33/50 = 66% |
| HARD | 3 条（全部集中在带 grounding flag 的 20 条内） |
| SOFT | 14 条 |

与冒烟期 50 条抽检（92%，#004）同带宽——放量后教师质量分布稳定的第二证据。

## HARD 3 条（逐条原文实锤）

| 行 | id | 错误 | 原文证据 |
|---|---|---|---|
| 4 | 2604.21223 | 占位三元组：`{DetectRL, "detection performance", value=0}` | 原文仅 "achieve superior detection performance, outperforms existing..."——**零数字**；metric 填的是性能描述非度量名，value=0 系编造 |
| 7 | 2606.27984 | 模态外推：`[text, image, audio, video]` 四模态零依据 | 原文是锅炉燃烧监测的**传感器**多模态（"inconsistent sensor sampling frequencies"），通篇无 text/image/audio/video 任何一词——外推最恶劣案例 |
| 14 | 2509.18154 | 占位三元组：`{VideoMME, "state-of-the-art performance", value=0}` | 原文对 VideoMME 仅给效率数字（"46.7% GPU memory cost and 8.7% inference time"），**无准确率分数**；value=0 + 非度量 metric 同行 4 模式 |

**三条 HARD 全部 = 机器 flag ∩ 盲审独立命中**（两路互盲信号收敛后再经终裁原文验证）。教师 HARD 错误就两个模式：①有 benchmark 名无数字时编 `value=0` + 拿性能描述当 metric；②非视听文本域喷标准模态——与冒烟期结论（占位 + 模态外推）完全一致。

## SOFT 14 条分类

| 模式 | 条数 | 例 |
|---|---|---|
| 合成/规范化 benchmark 名（值真实、实体或事实在原文，但名字是构造的） | 4 | 行 3"gold standard spot-check"、行 11"rectal cancer LN metastasis"（任务描述当名）、行 30"ICLR/NeurIPS peer review dataset" |
| 模态领域推断（原文未明示，但领域事实为图像等） | 6 | 行 12/15/17/18/24（TTA/SSL/disentanglement 惯例推断）、行 9（image 数据集标 text） |
| delta 当绝对值（"+X% 提升"记为分数，数字本身真实） | 3 | 行 36（OOD 均值拆到各 benchmark）、行 45（+0.17/+0.66/+0.43 记为 F1 绝对值）、行 30 |
| 范围端点拆双值 / claims_sota 过度归因 / task_type 偏窄 | 3 | 行 22（"78-85%"拆成两条并复制到两个 benchmark）、行 16（"first framework"≠SOTA 声明）、行 48 |

（行 30 同时命中两类，计入合成名类。）

## 交叉验证矩阵（组件质量的副产品）

- **grounding flag（20 条命中）**：对 HARD **召回 3/3 = 100%**（盲审在无 flag 的 30 条中 0 HARD），**precision 3/20 = 15%**——flag 是可靠的复核优先队列，不是错误判定器（设计定位即如此）。
- **误报源第三类新发现**：行 2 "S&P 500" 被 flag，原文实为 "S and P 500"（arXiv 对 `&` 的转义变体）——字面匹配的符号转义盲区，与冒烟期两类误报源（名词单复数、URL 直给开源证据）并列，已值得进 grounding 迭代清单。
- **sota_regex 分歧（16 条）**：15 条终裁教师正确（regex 侧误报：否定/上下文盲），1 条教师过度归因（行 16，novelty 当 SOTA，SOFT）——sota_disagree 信号的教师侧错误率仅 1/16。

## 全量外推（推断非测量，标注区间）

HARD 全部落在 flag 命中区 → 全量 HARD 率 ≈ flag 率 12.5% × precision 15% ≈ **1.9%**；无 flag 区 0/30 佐证。全量条目级无实质错误率估计 **≈98%（点估计，推断）**；本壳实测 94% 是可疑优先采样下的保守下界。

## 逐条判定表

OK（33）：1, 2, 5, 6, 8, 10, 13, 19, 20, 21, 23, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 49, 50
SOFT（14）：3, 9, 11, 12, 15, 16, 17, 18, 22, 24, 30, 36, 45, 48
HARD（3）：4, 7, 14

明细（含每条理由与修正建议）已回填至 `data/processed/holdout_for_review.jsonl` 的 `human_extraction` 字段（本地，不入库）。

## 三条 HARD 的修正

- 行 4 / 行 14：`benchmarks` 应为 `[]`（原文无可抽数字；效率百分比与准确率 metric 语义不同，不构成三元组）
- 行 7：四模态均无文本依据；传感器时序域不在标准模态枚举内，应不标注（或引入 sensor/timeseries 枚举——schema 迭代项）
