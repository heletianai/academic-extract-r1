# run-log（手册§五：每 run 一条——run_id/阶段/配置/指标/成本，run 结束 5 分钟内记）

## distill-full-20260721
- 阶段：Phase 0 数据线 / 放量蒸馏
- 配置：deepseek-chat(V3) temp0（INVALID 升温 0.6 重试一次）/ 并发 16 / 教师长 prompt 4-shot / 代码 commit 3a1d932
- 数据：papers_pool 9000（numeric 50% 分层，arXiv id 前缀≥2507）→ ok 8916（含冒烟 50）/ rejected 121（全部 validator_invalid=rejection filter 留痕，0 条 api_failed）/ grounding flag 率 12.5%（by_category: bench_name 683/modality 593/bench_value 349/placeholder 230/open_source 111；blacklist 召回 4/4）
- 切分：train 8648（13-gram 去污染剔 1，重叠率 0.01% 过 2% 闸）/ holdout 200 / 人工壳 50（sota_disagree∪grounding_flags 可疑优先）；SOFT 0.7% 丢弃（远低于 15% 复议线）；benchmarks 非空率 26.1%
- 成本：**全程实测 ¥18.75 清零**（≈¥2.1/千条；高峰段 1047 条学费差价 ~¥1.5；⚠️"低谷约半价"假设证伪——实测约标准价，估算器 PRICE 快照下次按实测校准）
- 用时：第二轮 7940 条 759s（10.5 条/s @并发 16）

## eval-api_fewshot-20260721
- 阶段：五方对照第 2 方（API few-shot）——评测链路首次全程实跑
- 配置：deepseek-chat + 教师同配置（长 prompt 4-shot temp0，协议§一表 2 行）/ holdout 200 / 代码 commit 5e27c16
- 指标：overall **0.9753** [CI95 0.9636-0.9844] / 合法 JSON 率 99.5%（1 条 INVALID）/ benchmarks hard-F1 0.9313 [0.8946-0.9633] / benchmarks soft 0.9586 / task_type 0.985 / modalities 0.995 / method_keywords 0.9335
- ⚠️ 口径声明（协议§一，报数必带）：教师同配置对蒸馏 GT 的重放＝**自一致性上界（"教师即天花板"的显式化），非独立能力数字**；与 1.0 的 2.5pp 缺口＝temp0 下 API 非确定性＝任务/评分器噪声地板参照。三列纪律：held-out 0.9753 / 人工核验子集 50 未填 / MOLE 未映射——独立数字待后两件闭环
- 成本：¥0.41（标准估）
