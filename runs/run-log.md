# run-log（手册§五：每 run 一条——run_id/阶段/配置/指标/成本，run 结束 5 分钟内记）

## distill-full-20260721
- 阶段：Phase 0 数据线 / 放量蒸馏
- 配置：deepseek-chat(V3) temp0（INVALID 升温 0.6 重试一次）/ 并发 16 / 教师长 prompt 4-shot / 代码 commit 3a1d932
- 数据：papers_pool 9000（numeric 50% 分层，arXiv id 前缀≥2507）→ ok 8916（含冒烟 50）/ rejected 121（全部 validator_invalid=rejection filter 留痕，0 条 api_failed）/ grounding flag 率 12.5%（by_category: bench_name 683/modality 593/bench_value 349/placeholder 230/open_source 111；blacklist 召回 4/4）
- 切分：train 8648（13-gram 去污染剔 1，重叠率 0.01% 过 2% 闸）/ holdout 200 / 人工壳 50（sota_disagree∪grounding_flags 可疑优先）；SOFT 0.7% 丢弃（远低于 15% 复议线）；benchmarks 非空率 26.1%
- 成本：**实际账单 ¥5.85 全程**（平台用量页 7.22 对账：9,482 请求/19.01M tokens，≈¥0.65/千条）。⚠️勘误：跑批刚结束查 balance 显示 0.00 系并发预扣费瞬时假象，结算后余额 ¥12.93；昨晚"¥18.75 清零/低谷半价证伪"两结论作废。实际账单≈脚本标准价估算的 26%（PRICE 快照偏高 ~4 倍，与 memory 早前"便宜 4 倍"实测一致）——估算器当安全上界用，真实成本以账单为准
- 用时：第二轮 7940 条 759s（10.5 条/s @并发 16）

## eval-api_fewshot-20260721
- 阶段：五方对照第 2 方（API few-shot）——评测链路首次全程实跑
- 配置：deepseek-chat + 教师同配置（长 prompt 4-shot temp0，协议§一表 2 行）/ holdout 200 / 代码 commit 5e27c16
- 指标：overall **0.9753** [CI95 0.9636-0.9844] / 合法 JSON 率 99.5%（1 条 INVALID）/ benchmarks hard-F1 0.9313 [0.8946-0.9633] / benchmarks soft 0.9586 / task_type 0.985 / modalities 0.995 / method_keywords 0.9335
- ⚠️ 口径声明（协议§一，报数必带）：教师同配置对蒸馏 GT 的重放＝**自一致性上界（"教师即天花板"的显式化），非独立能力数字**；与 1.0 的 2.5pp 缺口＝temp0 下 API 非确定性＝任务/评分器噪声地板参照。三列纪律：held-out 0.9753 / 人工核验子集 50 未填 / MOLE 未映射——独立数字待后两件闭环
- 成本：¥0.41（标准估）

## sft-smoke-20260721-195019（冒烟）
- 阶段：Phase 2 开卡日 / SFT 冒烟（90s 关卡）
- 环境：AutoDL 内蒙B区 183机 RTX 4090 24GB ¥2.08/时 / 驱动 560.35.03 / PyTorch 2.5.1+cu124 镜像（主机 CUDA≤12.6，12.8 镜像被拒后降级）/ unsloth 2026.7.4 / transformers 4.56.2 / trl 0.22.2 / Xformers 0.0.29.post1
- 配置：Qwen3-4B-Instruct-2507（ModelScope 本地）/ LoRA r32 α64 / bs2×accum4 / lr 2e-4 / 50 条 20 步 / train_on_responses_only loss mask / non-thinking
- 结果：**loss 1.18→0.36→0.24→0.14（20 步）**，29.8s（1.15s/step），可训参数 66M/4.09B=1.62%，padding-free 自动启用——全链路通，过闸
- 事故：#007 torchao 版本冲突（已修）；tmux 无包改 nohup（手册备胎）

## sft-full-20260721-195436（全量，已完成——收官补记 7.21 深夜）
- 配置：同冒烟，全量 8648 条 × 2 epochs = 2162 步 / LoRA r32 / lr 2e-4 / seed 3407
- 结果：**final_loss 0.1594**，train_seconds 2594.8（43.2 min，与 ETA 吻合）/ n_dropped_overlen 0
- 产物：LoRA 272M 本地+云端双份（runs/gpu-artifacts/sft-lora-8648/ + 云端 outputs/sft/sft-20260721-195436/lora）

## eval 三方对照 + scaling 五点（20260721，holdout 200，收官补记）
- **base_fewshot 0.6713**（valid JSON 94%）/ **SFT 0.9010** [CI95 0.8893-0.9129]（valid 100%）/ **API few-shot 0.9753**（教师自证口径声明照旧）
- SFT 分字段：claims_sota 0.99 / open_source 0.985 / modalities 0.968 / task_type 0.86 / benchmarks_hard 0.823 / method_keywords 0.715（最弱项）
- scaling：500→0.8718 [0.8512-0.8908] / 1000→0.8871 / 1500→0.8861（平台）/ 4000→0.8945 / 8648→0.9010——边际递减但未饱和，8k 拍板被曲线背书
- 泄漏排查：train/holdout 13-gram 重叠 0.01% 过 2% 闸；SFT(0.901)<API 天花板(0.975)，无"学生超教师"警报（手册§七）

## grpo pre-run 校准（20260721 深夜，手册§十——执行会话#3 第 0 步，开跑前记）
对 grpo_train.py ASSUMPTIONS A1-A6 逐条核 Stage A 实测：
- **A1 ✅**：SFT valid_json_rate=1.0（200/200），gate 率 <10% 预期成立；冒烟闸 20% 维持
- **A2 ⏳设计假设**：Stage A 全程 temp0 无多样性实测——正是冒烟四标准"组内去重率>50%"的验证对象，temp 1.0 起点不动
- **A3 ✅**：per-sample 均分 std=0.0868、满分仅 8%、零分 0——分数连续分布非全同；API-SFT 有 7.4pp 提升空间；最弱两项 method_keywords 0.715 / benchmarks_hard 0.823 恰是 reward 高权重项（F_bench×3），区分度前提成立
- **A4 ⏳无 A 阶段可核项**：beta=0.05 维持 extract0 参照，KL 曲线出来再调（数值参数化，改参零风险）
- **A5 ✅**：thinking 块 0/200，2507 纯 non-thinking 实证
- **A6 ✅**：SFT 输出长度 p99=725 字符≈242 token，max 786≈262 token——completion 512 有 ~2 倍余量（temp1.0 发散加长也够；超长截断→JSON 不闭合→gate 负分=自然惩罚）；lr 5e-6 / num_gen 8 维持参照起点
- **结论：零修改开跑**（零修改也记"已审视"）。冒烟过闸四标准不变：5 步跑完 / gate_rate<20% / 组内去重率>50% / reward 非零方差

## grpo-smoke-20260722（冒烟三连：崩→t1.0→t1.2）
- 阶段：Stage B / GRPO 冒烟（32 prompts × 5 步）
- 首跑 90 秒关卡抓崩：#008 TRL 0.22 要求 reward callable 带 __name__（RewardLogger 实例缺）→ 一行修复
- t1.0（081446）：91.1s 跑完，gate_rate 全 0，reward 方差在（0.635-0.996），但组内去重率 0.688→0.281 尾部俯冲（#009 高置信 prompt 全对全同→advantage=0 白耗）
- t1.2（082027）：去重率 0.938→0.781 全程>50% 线，gate 仍全 0，reward 方差在——四标准全绿，过闸
- 决策：正式 run 采 temp 1.2（A2 校准预案"塌缩则升温"框架内，非新决策）

## grpo-full-20260722（❌ 作废 #010：500 步权重空转，产物已改名 *-invalid-010）
- 配置：起点 SFT LoRA（sft-20260721-195436）/ temp 1.2 / lr 5e-6 / beta 0.05 / num_gen 8 / batch 8×accum4 / 2000 prompts × 1 epoch = 500 步 / save_steps 100 / seed 3407 / transformers 慢路
- vLLM 仗推迟：慢路实测 ~19s/it 已达经济闸，且实例无镜像保护（昨晚 GPU 被抢没存成）——不动 working 环境；vllm==0.6.6 包已预下载 /root/vllm-pkgs 待镜像后再打
- 启动：7.22 08:26 北京，nohup 链尾 shutdown（跑完/训崩皆自动关机，防空转）；RUN_LAUNCHED 已 touch，watchdog 解除
- 起跑关卡：显存 17.4/24G ✅ 速率 17.6-24.2s/it，ETA ≈2h40m ≈ ¥5.6（余额 ¥25 大幅盈余）
- 备注：generation_config 被 Qwen3 默认覆盖 top_p=0.8（冒烟同条件跑的，无新变量）；明细 outputs/grpo/grpo-20260722-08*/reward_detail.jsonl 每 20 call flush
- 产物回收：跑完自动关机，产物留盘——下次**无卡模式开机**拉回（0.1 元/时，不抢卡）
- 作废详情：unsloth from_pretrained(LoRA 目录)=推理态冻结，optimizer 全程未写参数（全 checkpoint md5=SFT）；KL 0.8-1.35 系"SFT vs base 固定距离"假信号；2.8h/¥5.9 学费换 #010 三教训+第五标准立规

## grpo-smoke-native-20260722-115238（原生 TRL 冒烟，#011 修复验证）
- 配置：--no-unsloth（AutoModel+PeftModel is_trainable=True+enable_input_require_grads+gradient_checkpointing）/ temp 1.2 / 其余同参
- 结果：**第五标准 ✅ md5 4039f4df→88cfa5af（权重真更新）**+四标准全绿（gate 0/去重 0.63-1.0/reward 方差在/5 步 83.4s=16.7s/it 反快于 unsloth 慢路）+reward 序列相对前两轮冒烟开始漂移（步间更新痕迹）
- 环境注记：vllm 0.6.6 已装但不用（无 Qwen3 支持，#011）；torch 2.5.1 未动；unsloth 保留给 SFT 线

## grpo-full-native-20260722（正式 run v2，进行中）
- 配置：--no-unsloth / temp 1.2 / lr 5e-6 / beta 0.05 / num_gen 8 / 2000 prompts=500 步 / save_steps 100 / seed 3407
- 启动：7.22 11:57 北京，nohup 无 shutdown（用户在场今日不关机）；显存 15.2/24G；17.2s/it ETA≈2h20m≈¥4.9
- 预期判读入口：五件套+第五标准终检（跑完 lora md5≠SFT）+ holdout 200 评测进五方对照

## grpo-full-native-20260722（正式 run v2，✅ 完成 14:32）
- 结果：500 步 2h35m，final_loss 同量级；**第五标准终检 PASS**（lora md5 ff98cc7c ≠ SFT 4039f4df）；五件套全程零事故（gate≈0/去重 0.80-0.81/熵平稳/reward 尾段真实上抬 0.943→0.951）
- **eval（holdout 200，自动链 14:46 完成）**：overall **0.9044 [0.8923-0.9166]** vs SFT 0.9010 [0.8893-0.9129] = **+0.34pp，CI 重叠不显著**；valid JSON 100%；分字段 task_type +1.5pp / method_keywords +0.45pp / claims_sota -0.5pp / 其余 ±0.2pp 内
- **diff 归因（钉死）**：GRPO 实质改写 143/200 条（72%）输出，但净胜仅 7（赢 32/输 25/平 86），赢输主战场同为 method_keywords（24 vs 20）=蒸馏 GT 噪声最大的主观字段——**训练真实有效（md5+72% 改动），撞到的是 reward 信号区分度天花板**：validator+F1 能保结构（gate 0%）不能裁"哪个关键词更对"
- 判读：单轮 GRPO 收官。增量空间判定在信息侧（多轮检索）而非对齐侧——Stage C 的实证论据，比"涨 5pp"更硬的假设检验叙事
- 成本：v2 run+评测 ≈¥5.6；今日累计 ≈¥14（含 #010 学费 ¥5.9）
