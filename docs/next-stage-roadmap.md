# 后续提分路线图

更新时间：2026-07-28

## 当前事实

当前最佳线上分数为 `0.89071`：HuBERT Large + 音素 CTC + Temporal Adapter。
第二名是 WavLM Large 的独立训练轨迹，线上 `0.89054`。两者只差 `0.00017`，
因此不能把任一分支视为绝对支配模型，应优先挖掘互补性。

已验证的主要收益：

| 改动 | 已验证线上收益 | 结论 |
|---|---:|---|
| WavLM + 注册文本字符 CTC | `+0.15125` | 强语音表征加文本先验是路线基础 |
| 字符 CTC 改音素 CTC | `+0.02836` | 音素目标更贴近英文关键词判别 |
| Temporal Adapter | `+0.03005` | 冻结 encoder 后仍需时序建模 |
| WavLM Large 替换 Base+ | `+0.01611` | 更强底模有明确收益 |
| HuBERT Large | `+0.00516` | 不同预训练目标仍有额外信息 |
| 全量数据和单纯增加 epoch | 千分位到半百分点 | 不是当前优先级 |

## 实验纪律

- 新路线默认：256 条 smoke test 后，使用 `--subset 100000` 的 CTC 横测。
- 帧级 matcher 以 `50000 pair` 作为同量级对照。
- eval 只能推理和最终提交，不能参与训练、选 checkpoint、选融合权重或调参。
- 每日最多 3 次提交：优先提交 Dev 明确优于单模型的方案。
- 全量 `1,000,000 utterance` 仅在小规模路线明确胜出且单独确认后运行。
- 新训练必须带显式 `--seed`；同一 checkpoint 恢复训练必须保持 seed 不变。

## P0：立即执行

### 1. WavLM Large + HuBERT Large 秩融合

状态：代码已支持，优先级最高。

- 在 dev 的 seen/unseen 内分别转换为 percentile rank。
- 搜索全局 HuBERT 权重，目标为宏平均 AUC。
- 将固定权重应用到 eval，两份 eval CSV 仅作为无标签输入。
- 只有融合 Dev 高于两个单模型时才占用当天第三次提交。

预期：两条线上分数接近但底模不同，通常比继续训练任一单模型更有希望获得
千分位到百分点收益。

### 2. 三模型融合是否值得

候选为 HuBERT E015、WavLM E016、WavLM E014。E014 较弱但来自不同噪声随机
轨迹，可能仍提供排序互补性。

- 先完成双模型融合。
- 若双模型 Dev 提升，扩展为三模型网格搜索；权重只在 dev 选择。
- 若 E014 加入后 Dev 不提升，立即剔除，不提交。

## P1：高价值新路线

### 3. 可控 seed ensemble

目的：把当前偶然的 PID 噪声差异改为可追溯的模型多样性。

- 固定所有结构和数据规模，仅改 `--seed 43`、`--seed 44`。
- 每个分支保存最佳 checkpoint，并先看 dev 是否与现有模型互补。
- 不需要每个 seed 都单独提交；优先将其加入 Dev 融合。

这是低风险的集成方法。它不会改变赛规边界：冻结开源 encoder，训练的仍是自研
Temporal CTC 头。

### 4. 更强的时序 Adapter

当前 Adapter 是两层 kernel=5 的深度可分离卷积。可进行单因素小网格：

- Adapter dim：256 -> 384。
- Adapter 层数：2 -> 4。
- 时间卷积 dilation：`1, 2, 4`，扩大感受野以覆盖连读和音素过渡。

评价规则：保持 WavLM Large、100K、音素 CTC、增强不变。只有 Dev 至少高于
WavLM E016 `0.8880` 且 unseen 不下降时，才进入融合池。

### 5. 字符 + 音素多任务 CTC

已有 Base+ 字符/音素后验融合带来过收益，说明两种标签有互补性。下一步不是重跑
两个完全独立的弱模型，而是在同一个 Large encoder 后训练共享 Temporal Adapter，
接两个自研 CTC head：一个字符 head，一个音素 head。

- 训练损失为字符 CTC 与音素 CTC 的加权和。
- 推理时对 enroll text 同时计算两条分数，在 dev 中搜索融合权重。
- 重点观察 unseen，因为字符拼写信息可能补充音素词典的发音归纳。

### 6. CTC 难负样本和似音词判别

赛题明确包含 `hi` / `haier` 类似音误唤醒，而当前 CTC 只优化“目标文本的绝对
对数概率”。可建立额外的自研判别头：

- 用 CMUdict/g2p 音素编辑距离，为每个训练关键词检索相似但不同的文本。
- 构造目标文本与似音文本的 hard negative 对。
- 输入目标 CTC 分数、相似词 CTC 分数差、文本音素长度等特征，训练二分类或
  pairwise ranking head。

这是直接针对比赛误唤醒痛点的路线。先在训练/dev 内闭环验证，不读取 eval 的
隐藏 query 文本。

### 7. 注册音频分支：重做音频匹配器

当前最佳 CTC 分支使用注册文本和 query 音频，没有直接利用注册音频。旧 Base+
matcher 的 unseen 很低，但它使用的模型容量和训练目标都较弱，不能证明注册音频
无价值。

更合理的重做方式：

- 冻结 WavLM Large 或 HuBERT Large，提取逐帧表征。
- 用带 mask 的对称 soft alignment / cross-attention 对齐 enroll 与 query。
- 从上述音素近邻词中挖 hard negative，而不是随机负样本。
- 将 audio-match posterior 与 CTC posterior 做 Dev 监督融合。

这是高上限路线，因为它补充了文本 CTC 无法表达的说话人、发音和录音条件信息。
但实现量大，优先级在 Large 融合之后。

## P2：条件性路线

### 8. 域增强与课程训练

当前仅使用 DEMAND 噪声，概率 0.5，SNR `[-10, 5]` dB。可补充：

- MUSAN/AudioSet 开源人声、音乐、环境噪声。
- RIR 混响、带宽限制、轻微速度扰动。
- 从 clean/高 SNR 逐步到低 SNR 的课程训练。

目的不是制造更多随机噪声，而是覆盖测试中的真实人声干扰和远场声学条件。使用
外部数据前必须在 README 中记录来源与许可证。

### 9. 外部带文本英语语音数据

可用 LibriSpeech、Common Voice 等开源带转写语料，训练自研 CTC head 或多任务
head，重点服务 unseen 关键词。此路线成本较高，先在 100K 内部数据上确认模型
结构上限后再启动。

### 10. 参数高效微调

可尝试只训练 WavLM/HuBERT 顶部少数层的 LoRA/Adapter，而不是直接微调整个
开源 encoder。它可能提升领域适应性，但赛规对“开源权重仅用于特征提取”的边界
需要先向主办方确认；未确认前保持 encoder 冻结。

## 当前不优先

- 直接跑全量训练：历史收益远小于强底模和时序结构收益。
- 将已经见顶的 HuBERT/WavLM 配置继续堆到更高 epoch。
- 只用高斯噪声替换 DEMAND，或只改 `pos_weight`。
- 使用 eval 样本、eval 文本假设或线上成绩反复调融合权重。

## 推荐顺序

1. 双模型 Dev 秩融合并提交。
2. 若融合有效，尝试包含 WavLM E014 的三模型 Dev 融合。
3. 固定新 seed 训练一个 WavLM Large 分支，作为可控 ensemble 候选。
4. Temporal Adapter 的 dilation/深度单因素对照。
5. 音素 hard negative 判别分支。
6. Large 注册音频 matcher 与 CTC 的监督融合。
7. 只有上述至少一条路线在 100K 明确胜出，才讨论全量训练。
