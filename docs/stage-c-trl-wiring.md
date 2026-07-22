# Stage C × TRL 0.22.2 接线设计（7.22 冻结，照此施工）

> 源码考古已完成（云端 trl/trainer/grpo_trainer.py），本文档=施工图。
> 核心发现：**TRL 的 completion_mask 身兼两职**——loss 加权 + attention 拼接
> （L1319/L1493/L1546 `attention_mask = cat([prompt_mask, completion_mask])`）。
> 直接把检索段置 0 会让训练 forward 的 attention 屏蔽检索段 → logprob 口径与
> rollout 上下文不一致 → 重要性采样 ratio 系统性偏差。SR 原版是双 mask 分离
> （attention_mask 全可见 + info_mask 管 loss，generation.py L335-342），veRL 消费
> 两个；TRL 单轮设计把它们耦合了。**修复：双 mask 分离的 TRL 化。**

## 覆写策略（新文件 src/training/grpo_multiturn.py，class MultiTurnGRPOTrainer(GRPOTrainer)）

### 覆写 1：`_generate_and_score_completions(inputs)` 精简版（~150 行）
原版 400+ 行巨石（vLLM/paged/VLM/FSDP 三分支全在里面），我们单卡 transformers+PEFT
只需直线路径。契约（_compute_loss 消费的键，已逐键核对）：

| 键 | 语义 | 我们的来源 |
|---|---|---|
| prompt_ids / prompt_mask | 左 pad prompt | tokenizer(apply_chat_template)，对齐原版 |
| completion_ids | 多轮完整 completion（含注入段） | MultiTurnRollout.run_batch |
| completion_mask | **attention 语义**：非 pad 全 1 | rollout 产物 ids≠pad |
| **info_mask（新增键）** | **loss 语义**：模型生成=1 / 注入=0 / pad=0 | rollout 的 completion_mask 字段 |
| advantages | 组内 (r-mean)/std | reward_v1 + action penalty → 组内归一 |
| old_per_token_logps | None（单 iteration 对齐时 TRL 自动 detach 当前值，源码 L1578 注释确认） | None |
| ref_per_token_logps | beta>0 必供：PEFT `model.disable_adapter()` 上下文下 forward | 参照原版 `_get_per_token_logps_and_entropies` 调用形状 |

流程：inputs（已被 RepeatSampler 重复 num_generations 次）→ 抽 prompt/gold/paper_id
→ run_batch → reward（compute_reward(text,gold) + action_penalty(stats)）→ 组内
advantage → ref logps → 返回 dict。**杂键全不做**（VLM 的 pixel_* 等）。

### 覆写 2：`_compute_loss(model, inputs)` 复制改三行
复制原版体（L1541-尾），改动：
```python
loss_mask = completion_mask * inputs["info_mask"]   # 新增
# attention 构造行不动（用 completion_mask——检索段对 forward 可见）
# 之后所有加权/聚合处 completion_mask → loss_mask：
#   - entropy_mask 调用（L1563）
#   - sequence 级 importance weights（L1586）
#   - per-token loss 最终聚合（尾段 ×mask/求和处，施工时逐处 grep 替换）
# KL 项本身 per-token，聚合同走 loss_mask
```
风险自知：复制体锁死 trl==0.22.2（环境已 pin，requirements 注记）。

## action penalty 接线（reward v1 §三③层激活）
- stats.searches == 0 且 answered → 扣（零检索直答）
- not stats.answered → 扣（answer avoidance / 超 max_turns）
- stats.invalid_actions > 0 → 每次小扣
- 数值参数化（--penalty-*），起点参照 SR++ 结论"硬门控为主 penalty 为辅"

## 多轮冒烟六标准（过闸新规）
四标准（跑完/gate<20%/组内去重>50%/reward 方差）+ 第五（adapter md5 变，#010 立规）
+ **第六：loss_mask 有效性**——冒烟日志打印一条轨迹的
`(info_mask==0).sum()`>0 且该段 per-token loss 贡献=0（数值断言进冒烟代码）。

## 依赖前提（已全绿）
- MultiTurnRollout：160 测全绿，info_mask 逐段断言+可视化过
- BM25 检索件：SR 协议兼容 + exclude_id 自身屏蔽；生产索引在云端用全量合格池建
  （build_retrieval_index.py --source <全量池> 待跑，49309 条）
- 数据：sft_train.jsonl 的 prompt 复用（gold=蒸馏 GT），paper_id 从样本 id 带出
