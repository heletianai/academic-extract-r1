## 2026-07-21 #001 pytest 退出码被 pipe 掩盖导致挂测先 commit
- 现象：`pytest | tail && git commit` 在 1 个测试失败时仍然 commit（pipe 退出码=tail 的 0）
- 假设→验证：单跑确认挂因=断言容差 1e-6 撞上 report round(4)，非逻辑 bug；顺带发现 zsh 下 PIPESTATUS 需用小写 pipestatus
- 结论：测试→提交一律分两步跑，不再 pipe 后直接 &&；断言对 round 值用 1e-3 容差

## 2026-07-21 #002 【P0】OAI created 字段不可靠，防污染时间窗被击穿（key 测试暴露→已修复）
- 现象：key 全链测试返回的"2025H2 论文"id 为 2407.19342/2408.01933（id 前缀=2024 年 7/8 月提交）
- 假设：OAI <created> 非 v1 提交日（可能为 replace/版本相关日期），用它筛 2025H2 会混入 cutoff 前老论文
- 验证：arXiv export API 对照 2407.19342 → published=2024-07-27，OAI created=2025-07-01（v2 日期）；全量 111311 条新式 id 统计 36.5% created 与 id 前缀年月不一致
- 结论：时间窗改用 arXiv id 前缀 YYMM≥2507（分配规则绑定 v1 提交年月，不可变）；旧式 id 一律拒；created 保留仅供参考。filter_papers.py 已改+单测锁定。数据卡口径同步："训练/测试窗口=arXiv id 前缀≥2507"

## 2026-07-21 #003 红队(Opus A)七项确认修复
- P1 evaluate 坏 gold 行崩全局 → _load_gold_extraction 防护+n_bad_gold 显式计数+gold 过 validator(顺带修 str-bool 潜伏)
- P1 parse_fail 门控死代码(语法错全标 duplicate_keys) → 门控重排:先 loads 后查重复键,诊断标签修正
- P2 extract_json_str 取第一个对象惩罚推理文本 → 改取最后一个可解析对象(Stage C 多轮前置修复)
- P2 sota_regex 否定盲+插入语击穿窗口 → 否定词组+窗口60+子句边界截断
- P2 to_sft_format holdout≥unique 时 train 静默缩到 1 → 阈值保护退出
- P2 value=True 被 cast 成 1.0 拿满分 / NaN·Inf 穿透 VALID → before-validator 转 str 落 SOFT;parse_value 拒非有限值(防御化,原靠 max(0.0,nan) 参数顺序偶然防住)
- (temperature provenance 上批已修,红队审的旧树)

## 2026-07-21 #004 冒烟抽检闭环（Claude 预检+Opus 独立盲测交叉验证）
- 现象：教师 50 条冒烟数据质量待裁决（闸 0.5）
- 验证：机器粗筛（三元组数字 grounding+漏抽正则+sota 分歧）→ Claude 人工裁决可疑项 → Opus 盲测 agent 独立核 8 条（不带预检结论）→ modalities 全量补筛
- 结论：50 条已知 4 错（#8 占位三元组幻觉/#33·#1·#22 modalities 过度外推）→ 条目级 92%，目标带（87-92%）内，**闸 0.5 过**。盲测价值实证：独立复现 #8 且抓出预检漏掉的 #33
- 处置：blacklist.txt 4 条（to_sft_format 自动剔除）；prompt 加两句对症防御（禁占位三元组/模态须文本明示）；教师错误模式集中=占位+模态外推两类，放量后抽检重点盯这两类
- 待办（放量前）：distill.py 加 value grounding 自动核对 flag（预检脚本组件化，放量时自动生成 review 优先队列）

## 2026-07-21 #005 【P0-repo】gitignore 'data/' 误吞 src/data/ 整目录
- 现象：改完 prompts.py 后 git 报 nothing to commit
- 假设→验证：git show HEAD:src/data/prompts.py → "exists on disk, but not in HEAD"——gitignore 无前导斜杠的 `data/` 递归匹配任意层级同名目录，数据管线 7 个源码文件从未入库；此前 commit message 声称提交系失真
- 结论：.gitignore 目录规则全部改带前导 /（/data/ 只匹配根）；补提交入库。教训：①gitignore 目录模式一律带前导斜杠 ②commit 后 spot-check git ls-files ③"nothing to commit"类小异常必追根因

## 2026-07-21 #006 【P1-repo】gitignore '/runs/' 整目录忽略，日志体系从未入库（#005 续集）
- 现象：放量收官 commit 报 nothing to commit，runs/ 下 8 个新文件全部未被跟踪
- 假设→验证：git status --ignored → runs/ 整目录命中 /runs/ 规则；git ls-files 确认 issues-log.md/smoke-review 自创建起从未入库，此前 commit message 提及它们＝失真（#005 同款第二例）
- 结论：改 /runs/* + !/runs/*.{md,json,jsonl,log}（顶层小件放行；否定行置于全局 *.jsonl 之后防覆盖；子目录留给 checkpoint/trajectory 大件）。纪律升级：#005 只 spot-check 了出事目录——今后**每个新目录首次产出文件的 commit 后，git ls-files 扫该目录**

## 2026-07-21 #007 torchao>=0.16 与镜像 torch 2.5.1 冲突（开卡日环境搭建）
- 现象：pip 装完后 import unsloth 爆 AttributeError: module 'torch' has no attribute 'int1'
- 假设→验证：torch.int1 是 2.6+ 新类型；requirements 的 torchao>=0.16 pin 来自 notebook（torch2.8 环境），在 4090 实例镜像（PyTorch 2.5.1/cu124，主机驱动 560 只支持≤12.6 所以没法用 2.8 镜像）下不兼容
- 结论：16bit LoRA 不需要 torchao，直接卸载→unsloth 2026.7.4 导入 OK。requirements-train.txt 注记：torchao 仅 torch≥2.6 环境装。教训=分环境 pin：镜像 torch 版本决定周边件版本域，跨环境 pin 表要带条件
- 环境快照：4090 24GB/驱动560.35.03/PyTorch2.5.1+cu124/py3.12/unsloth 2026.7.4/transformers 4.56.2/trl 0.22.2

## 2026-07-22 #008 TRL 0.22 GRPOTrainer 要求 reward callable 带 __name__（GRPO 冒烟首崩）
- 现象：冒烟 90 秒关卡抓崩：AttributeError: 'RewardLogger' object has no attribute '__name__'（grpo_trainer.py:301 注册 reward_func_names）
- 假设→验证：TRL 取 reward_funcs[i].__name__ 当日志名——函数有、类实例无；本地 153 单测未实例化真 GRPOTrainer（需 GPU）故未暴露
- 结论：RewardLogger.__init__ 加 self.__name__ = "reward_v1" 一行修复，二次冒烟通过。教训：trainer 集成层的 API 契约（TRL 对 reward callable 的隐式要求）单测覆盖不到，只能真环境冒烟抓——90 秒关卡纪律再次兑现（4.29 立规，今日第 3 次抓真崩）
## 2026-07-22 #009 冒烟组内去重率尾部俯冲（0.688→0.281，均值 0.55 压线）
- 现象：temp1.0 冒烟 5 步 gate 全 0、reward_mean 冲 0.99+，但 group_dedup_mean 逐 call 下滑，末 call 0.281
- 假设：非权重漂移（lr5e-6×5 步动不了权重）——是 prompt 难度分层：高置信 prompt 上 temp1.0 采样 8 条全收敛到同一正确 JSON（抽取任务答案唯一，SFT 后模型高置信）→ 全对全同组 advantage=0，正式 run 中这类组白耗算力（手册§六.3"同质化=advantage 失效"预警线命中，但方向是塌向正确而非 hacking：reward 高+gate 0 佐证）
- 验证：temp 1.2 二次冒烟对照（A2 校准预案"塌缩则升温"），观察去重率回升幅度与 gate 是否仍 0
- 结论：t1.2 冒烟去重率 0.938→0.781（均值 0.86 vs t1.0 的 0.55），gate 仍全 0——升温对症无副作用，正式 run 采 t1.2。留观：正式 run 后段去重率走势（reward_detail 每 20 call 一行）

## 2026-07-22 #010 【P0】GRPO 500 步权重空转——unsloth 加载 LoRA 目录=推理态冻结
- 现象：GRPO 评测 overall 0.901 与 SFT 每字段每 CI 完全一致→查实 200/200 条预测逐字相同、最终 lora 与全部 checkpoint(100-500) adapter md5 均=SFT 逐位拷贝；但训练日志 KL 恒 0.8-1.35 波动
- 假设→验证：KL 非零但无增长趋势=TRL PEFT 模式 ref=禁用 adapter 的 base，KL 量的是"SFT 相对 base 的固定距离"非训练偏移；loss 有值仅证前向可算。md5 全程不变=optimizer 从未写入参数（任何非零更新必变浮点位）。根因=from_pretrained(LoRA 目录) 加载的 adapter 为推理态，非官方 GRPO notebook 轨道（官方=基座+get_peft_model）
- 修复 1（不够）：基座+get_peft_model+set_peft_model_state_dict 灌 SFT 权重（lora_B L1 0→146628 灌入成功）——第五标准仍 ❌，md5 仍不变，且 reward 五行与 t1.2 冒烟逐字重放=权重仍未更新
- 结论：加载侧修复不充分，空转根源在 unsloth 对 GRPOTrainer 的 patch 层（见 #011）。**教训三条：①冒烟四标准缺"权重在动"维度——第五标准（N 步后 adapter md5≠起点）自此为 RL 类训练必检项 ②评测数字与基线完全一致=第一优先怀疑"同一模型被测两次" ③KL 非零≠在训练（PEFT ref=base 时 KL 起点即非零）**
## 2026-07-22 #011 unsloth GRPO 慢路态切换失守 + vLLM 0.6.6 无 Qwen3 → 原生 TRL 定案
- 现象：#010 修复 1 后权重仍空转；unsloth/models/rl.py 存在 for_inference/for_training 切换与 prepare_for_training_mode 装饰器（L203/222/399-444）——unsloth GRPO 支持围绕 fast_inference=True(vLLM colocate) 官方场景设计，transformers 慢路 rollout 的训练态恢复失守（机制假设，行号留档未逐行验证——修复优先于考古）
- 验证：vLLM 线：0.6.6 装成且 torch 2.5.1 未动，但 ValueError: Qwen3ForCausalLM not supported（0.6.6 系 2024.12 版早于 Qwen3，设计期 pin 误配；升 0.8.x 需连升 torch=4.29 版本地狱复刻，30min 闸内不可行→放弃）。原生 TRL+PEFT 线：AutoModel+PeftModel.from_pretrained(is_trainable=True)+enable_input_require_grads+gradient_checkpointing，绝不 import unsloth（会全局 patch trl）→ 冒烟第五标准 ✅（md5 4039f4df→88cfa5af）+四标准全绿+reward 序列开始漂移（权重逐步更新痕迹）
- 结论：**Stage B 定案原生 TRL 慢路**（16.7s/it 反快于 unsloth 慢路 18.2——其优化在未走通路径上本无效）。unsloth 保留给 SFT（实证有效）；Stage C 若需 vLLM 则版本仗在镜像保护下重打（升 torch 全链）。PEFT 加载纪律：is_trainable=True 必须显式（PEFT 默认推理态，与 #010 同族坑）

## 2026-07-22 #012 多轮冒烟 gate 100%——MT prompt 从截断输出抄 schema 漏字段
- 现象：多轮冒烟机制指标全活（search_rate 0.5/answered 0.85/mean_turns 2.6/第五标准 PASS）但 gate_rate 100%（schema_invalid 27/32 量级）
- 假设→验证：GPU 取样看 answer 肉眼完全合法→本地零 GPU 复现→reward 明细 errors 钉死：limitation_mentioned: missing——gold 8 字段，MT_SYSTEM_PROMPT 只写 7（schema 说明系从 head -c 400 的截断输出抄写，limitation_mentioned 在截断点外）；模型忠实照 prompt 输出 7 字段
- 结论：修复=MT prompt 的 schema 段逐字复用数据内 SFT system 原文（与权重内化口径恒一致、数据变自动跟），协议段做后缀追加。教训：**关键配置（schema/接口/协议）禁止从截断输出重建，必须读原文**——与"索引行≠内容，命中必 Read 全文"同族纪律的代码版

## 2026-07-22 #013 多轮正式 run 行为塌缩(search avoidance)——penalty 配比失衡
- 现象：正式 run(penalty 0.2/0.5/0.1)warmup 一结束(step 7,lr 到满)6 步内 search_rate 0.5→0.03、answered_rate→1.0、组内 reward_std 1.0→0.03、entropy 0.20→0.06、reward 稳收 0.65 附近="全员零检索直答"稳态；聚合明细同步佐证(gate 0.22→0,no_json 归零=无人检索自然无人超时)
- 假设→验证：直答期望=F1(0.85-0.9)−0.2(零检索罚)≈0.65 无风险；检索轨迹背"未收针=gate −1.0−0.5 不作答罚=−1.5"下行风险→组内对比直答 advantage 恒正,GRPO 快速淘汰检索=Search-R1++ answer avoidance 的镜像(search avoidance)。方差归零+熵塌=梯度信号已死,剩 232 步(¥12/5.5h)为空烧→step ~25 止损杀(损失 ¥1.2/35min),曲线存档 grpomt-full-0722-collapsed-013.log。诚实注脚：杀前末 3 call search 回升 0.22-0.41(疑似 β·KL 往 SFT 先验拉回的对抗效应),持续性未观察即截断
- 结论：penalty 数值消融(预案框架内)：no_search 0.2→0.8(直答封顶 F1−0.8≈0.15 必亏)/no_answer 0.5→0.3(检索下行 −1.5→−1.3)；冒烟扩展至 20 步——**5 步冒烟对行为塌缩是盲区**(塌缩在 step 7+,四连冒烟全未见)。教训：①行为型 reward 的冒烟窗口必须覆盖 lr 满血后 ≥10 步 ②action penalty 是天平不是开关,两侧罚值之差=对行为的定价,失衡即塌向 riskless 臂 ③RL 空转判定升级：方差归零+熵塌=行为空转(权重在动但学习信号已死),与 #010 权重空转同族——第五标准(md5)抓不到这型,曲线三线(std/熵/行为率)是唯一预警
