# Schema 六字段判定规则 Checklist

> 底稿 = 立项卡 Schema 终版（7.4，零设计决策）。本文件展开成蒸馏/抽检/reward 三用的判定规则。
> 状态：**FROZEN（2026-07-21）** — 用户委托按推荐案拍板（"你来做，继续"），7 项全按推荐案。
> 冻结时 Claude 代审补三处硬化（防标签噪声，未改变任何已定语义）：
> 1. **全字段判定仅基于给定文本片段，禁用教师模型的外部知识**（否则教师"知道"某论文开源→标 true→标签噪声；此条进蒸馏 system prompt）
> 2. claims_sota 白名单落成正则（大小写/连字符变体归一）：`sota | state[- ]of[- ]the[- ]art | outperforms? +all | surpass(es|ing)? +all | previous best`；**排除子句**：白名单词仅作为比较基准出现（comparable to / close to / matches / approaching + SOTA）→ false（达成断言才 true）
> 3. task_type=multimodal 消歧：指**任务本身**是多模态理解/生成（VQA/图文生成）；VLM 做纯分类任务 → classification（任务优先于模型架构）
> 之后任何修改 = 重大方案变更（须 Fable 会话裁决）。

## 〇、你只需要拍板的 7 件事（20 分钟的全部内容）

| # | 拍板项 | 推荐案 | 备选 |
|---|---|---|---|
| 1 | `open_source`：只说 "code will be released" 没给链接，算不算开源 | **true**（摘要里链接常缺失，只认链接会大量漏判） | false（严格版：见链接才算） |
| 2 | `claims_sota` 判定方式 | **词面白名单**（可复核、蒸馏一致性高）：SOTA / state-of-the-art / outperform(s) all / surpass(es) all / previous best 命中即 true | 语义判定（召回高但蒸馏易漂移） |
| 3 | benchmarks 抽取范围 | **只抽本文方法的结果**，baseline 数字不抽；相对提升（"+3.2%"）不抽（无绝对值） | 连 baseline 一起抽（标注歧义大，不推荐） |
| 4 | 数值命中容差 `VALUE_REL_TOL` | **0.5%**（85.31 vs 85.3 命中；86.0 vs 85.3 不命中） | 1% 放宽 |
| 5 | 名称/指标相似阈值 | **0.85**（规范化后 exact 是主路径，Levenshtein 只救拼写微差） | 0.9 收紧 |
| 6 | METRIC_ALIASES 12 组别名表（见 `src/reward/triple_match.py`）| **过一遍增删**：acc=accuracy、EM=exact match、F1 组（micro/macro 暂并组）、BLEU、ROUGE、pass@1、PPL、AUC、MRR、NDCG、WER | — |
| 7 | task_type 枚举英文值 | **classification / generation / retrieval / reasoning / multimodal / agent / other**（JSON 输出用英文，中文枚举 tokenizer 不友好） | — |

BIPARTITE_THRESHOLD=0.35 不需要拍板（extract0 原值照抄，数值消融项）。

## 一、可验证组六字段判定规则（进 reward）

### 1. task_type — 枚举单选
- 值域：`classification | generation | retrieval | reasoning | multimodal | agent | other`
- 判定：按 abstract 声称的**主贡献任务**归类；多任务论文取主贡献；benchmark/评测型论文按其评测的能力域；通用 LLM 无单一任务 → `other`
- 评分：规范化后 exact（对=1 错=0）

### 2. modalities — 多选列表
- 值域：`text | image | audio | video | code`
- 判定：论文**处理的数据模态**。纯文本 LLM → `["text"]`；VLM → `["text","image"]`；出现代码数据集/代码生成任务 → 含 `"code"`
- 评分：集合 F1（元素 exact）

### 3. benchmarks — 三元组列表（核心难字段，权重 3）
- 元素：`{"name": str, "metric": str, "value": number}`
- 判定：
  - 范围：**仅本文方法**在摘要中报告的绝对数值结果（拍板项 3）
  - name/metric 照抄原文形式（大小写/连字符不用统一——匹配器做规范化）
  - value 保持原文刻度（85.3 或 0.853 都行——匹配器做刻度对齐）
  - 一个 benchmark 报多个指标 → 拆多条三元组
  - 相对提升、区间值、"接近 X" → 不抽
- 空值语义：论文无 benchmark 数字 → `[]`（**不是 null**）；gold 空 + pred 空 = 满分，gold 空 + pred 非空 = 0 分（幻觉惩罚）
- 评分：soft F1 进 reward（bipartite 贪心 0.35）/ hard F1 进 eval（name AND metric AND value 三条全过，name 是必要条件）

### 4. open_source — bool
- true：给出代码链接，或明确声明（已）开源 / will be released（拍板项 1 推荐案）
- false：无任何表述
- 评分：exact

### 5. claims_sota — bool
- 词面白名单命中 → true（拍板项 2）：`state-of-the-art / SOTA / outperform(s|ing) all / surpass(es|ing) all / previous best`
- 仅 "outperforms (strong) baselines" / "comparable to SOTA" / "competitive" → false
- 评分：exact

### 6. method_keywords — 3-5 个方法名词列表
- 判定：方法/技术专名（GRPO、LoRA、contrastive learning），名词短语 ≤4 词
- 排除：任务名（task_type 已覆盖）、benchmark 名、泛词（deep learning / neural network）
- 形式：**优先缩写形式**（LoRA 不写 Low-Rank Adaptation）——缩写 vs 全称是字符串匹配的已知限制，规则端收敛
- 评分：bipartite 贪心 soft F1（权重 1，在 F_field 内）

## 二、展示组（不进 reward，零标注压力）
- `one_line_summary`: str — 一句话概括
- `limitation_mentioned`: str | null — 论文自述局限（无则 null）

## 三、10 条边界案例（蒸馏 few-shot 与人工抽检共用，对应单测名）

| # | 输入情形 | 正确标注 | 对应单测 |
|---|---|---|---|
| 1 | "GSM-8K 上取得 92.0"（gold 写 GSM8K） | 照抄 "GSM-8K"，匹配器归一命中 | `test_name_variant_hit` |
| 2 | "比 baseline 提升 3.2%" | 不进三元组（相对值） | —（蒸馏规则） |
| 3 | "GPT-4 得 86.4，我们的方法得 85.3" | 只抽 85.3（本文方法） | —（蒸馏规则） |
| 4 | "MMLU 上 acc 85.3、F1 0.90" | 两条三元组 | —（蒸馏规则） |
| 5 | 论文写 0.853，GT 是 85.3 | 照抄 0.853，匹配器刻度对齐 | `test_percent_scale_hit` |
| 6 | "comparable to SOTA" / "achieves new SOTA" | false / true | —（词面白名单） |
| 7 | "code will be released" | open_source=true（待拍板） | — |
| 8 | 分类+生成多任务论文 | task_type=主贡献任务 | — |
| 9 | 视觉语言模型 | modalities=["text","image"] | — |
| 10 | position paper 无任何数字 | benchmarks=[]（非 null） | `test_both_empty_full_score` |
| 10b | 无 benchmark 论文但模型幻觉输出 | 0 分（precision 惩罚） | `test_hallucination_on_empty_gold_zero` |

## 四、与代码的对应关系
- 规则实现：`src/reward/similarity.py`（规范化/字符串/数值/bipartite）+ `src/reward/triple_match.py`（三元组双出口）
- 单测：`tests/` 45 条全绿（2026-07-21）
- 本文件冻结后：常量表（阈值/别名组/枚举值）以代码为唯一真相源，文档只描述语义
