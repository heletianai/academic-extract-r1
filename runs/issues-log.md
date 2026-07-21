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
