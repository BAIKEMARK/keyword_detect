# 后续提分路线图

更新时间：2026-08-06

## 当前结论

当前线上最佳为 `0.90796`：HuBERT Large hard-negative、W2V-BERT 2.0 与
forced-alignment rescorer 的秩融合。加入 forced alignment 前为 `0.90721`，实际只
提升 `0.00075`，该方向已经达到停止条件，不再继续调 rescorer。

距离约 `0.98` 仍差 `0.07204`。这个量级不能合理地寄希望于更多 epoch、固定
seed 重跑或直接扩到全量。当前系统的结构性限制是：训练阶段只学习正确文本的
CTC 转写概率，推理阶段只看目标文本的绝对 CTC 分数，没有显式学习“这段语音更像
目标词，还是另一个词”。此外，所有有效 CTC 实验都冻结了底模，只训练约 41 万
参数的 Temporal Adapter，底模还没有针对本任务适配。

已验证收益：

| 改动 | 线上收益 | 结论 |
|---|---:|---|
| WavLM + 注册文本字符 CTC | `+0.15125` | 强表征、文本先验和任务重构是基础 |
| 字符改音素 CTC | `+0.02836` | 音素目标更适合英文关键词判别 |
| Temporal Adapter | `+0.03005` | 冻结 encoder 后仍需局部时序建模 |
| WavLM Large 替换 Base+ | `+0.01611` | 更强底模有明确收益 |
| WavLM Large + HuBERT Large 融合 | `+0.01052` | 跨底模互补仍有价值 |
| HuBERT hard-negative + W2V-BERT 2.0 | `+0.0060` | 新底模有互补，但仍不是跨越式收益 |
| forced-alignment rescorer | `+0.00075` | 已到停止条件，不再继续调参 |
| 全量数据或单纯增加 epoch | 千分位到半百分点 | 不是当前突破口 |

最近关键结果：W2V-BERT 2.0 单模型线上 `0.89638`；HuBERT hard-negative 与
W2V-BERT 融合 `0.90721`；加入 alignment 后 `0.90796`。Whisper Large-v3 冻结
底模的最佳 Dev Mean 为 `0.8815`，不续训到 15 epoch。注册音频 align 头 Dev
约 `0.5083`，当前实现不可用。

## 当前优先实验：顶层部分微调

下一次结构性横测改为 W2V-BERT 2.0 最后一个 Transformer block 部分微调：

- Temporal Adapter 继续使用 `1e-3`；
- 底模最后一层使用独立小学习率 `1e-5`；
- 其余底模参数保持冻结和 eval；
- checkpoint 同时保存 Adapter、最后一层底模和 optimizer，可断点恢复；
- 先跑 256 条 smoke，确认层定位、梯度、显存和 checkpoint 重载，再跑 100K；
- 首轮只解冻一层，不同时改数据量、hard-negative 参数或 Adapter 结构。

成功标准：W2V-BERT 单模型 Dev Mean 至少超过原 `0.8937` 达到 `0.9000`，或与
当前 HuBERT 分支融合至少提升 `0.005`。达不到就停止扩大解冻层数，转向训练集误报
驱动的在线 hard-negative 和音素竞争词似然比。

## 新底模判断

语音领域没有一个已经可以断言会像 DINOv3 一样，在本赛题中无条件跨越式提分的
公开模型。语音识别模型的 WER、通用音频模型的分类成绩和短关键词的似音判别能力
不是同一个指标。候选底模必须保留细粒度帧特征，并且只能作为冻结特征提取器，
最终判决仍由自研模型完成。

| 优先级 | 候选 | 值得测的原因 | 风险与结论 |
|---:|---|---|---|
| 1 | `facebook/w2v-bert-2.0` | 约 6 亿参数、超大规模多语种自监督语音预训练；在公开可用 encoder 中最接近“新一代通用语音表征底座” | 不是英语专用，且输入是声学特征而非当前 raw-waveform 接口；先做 100K 严格横测 |
| 2 | NVIDIA Parakeet 0.6B encoder | 强英语监督 ASR encoder，可能比纯自监督模型更直接编码音素和单词边界 | NeMo 接入工作量较大；只能冻结 encoder 并训练自研头，不能直接拿其转写结果判决 |
| 3 | `openai/whisper-large-v3` encoder | 大规模弱监督 ASR 训练，英语、口音和噪声鲁棒性强，可能与 WavLM/HuBERT 互补 | 不能使用 decoder 直接判决；输入前端和当前代码不同，模型更重，未必优于专门的帧级 SSL encoder |
| 4 | XEUS encoder | 约 5.8 亿参数、超大规模跨语言 SSL，适合作为 WavLM/HuBERT 之外的表征多样性候选 | 英语不是唯一优化目标，生态和部署成熟度低于 Transformers 主线；前三项无收益后再做 |
| 5 | MMS/XLS-R/Data2Vec Audio/UniSpeech-SAT | 都是合法的公开帧级 encoder，可提供额外预训练目标 | 代际或任务优势不够明确，不优先逐个穷举 |

暂不投入 BEATs/EAT 等通用音频事件模型。它们擅长声音事件和场景语义，时频 patch
通常比音素边界粗，不是当前短词辨音的首选。也不投入音频 LLM、完整 Whisper
decoder 或云端 ASR API：它们的生成结果既不符合“开源模型只作特征提取”的当前
合规边界，也不适合作为 10 万条短音频的轻量帧级前端。

## P0：先攻打分目标

### 1. CTC 解码信息诊断

不重新训练 encoder，先从 E015/E016 的现有输出中同时导出：

- 目标注册词的长度归一化 CTC 分数 `s_target`；
- query 的 greedy CTC 音素序列；
- greedy 序列与注册词音素序列的归一化编辑相似度；
- `s_target - s_greedy` 似然差；
- blank 比例、解码置信度、目标/解码长度差和音频时长。

先分别计算每个特征的 seen/unseen AUC，再用官方训练 pair 训练一个很小的
logistic/MLP 判别头，在 dev 上只做一次验证。eval 仅应用固定模型。这一步能直接
判断：当前 CTC 头是否已经识别出 query 的音素，只是绝对分数没有把信息利用好。

成功标准：单个 Large 分支 Dev Mean 至少提升 `0.01`，或与现有融合后提升
`0.005`。若达不到，停止在分数校准上继续调参。

### 2. 音素难负样本挖掘

训练 CSV 已给出 `query_txt`，因此可以在训练集内合法构造难负样本：

1. 把全部训练词转换成无重音 ARPAbet 音素序列。
2. 按归一化音素编辑距离，为每个真实 query 词检索 5 至 20 个不同但最接近的词。
3. 保留官方负 pair，并额外加入这些似音词作为错误 enroll 文本。
4. 第一轮按音素距离挖掘；第二轮用当前模型在训练集上的高分误报做在线 hard mining。
5. 单独报告随机负样本 AUC、近音负样本 AUC 和总体 seen/unseen AUC。

不要只把难负样本塞进普通 CTC 转写。CTC 转写损失只知道“正确文本是什么”，并不
直接惩罚某个错误目标分数过高。应增加 pairwise margin：

```text
L = L_ctc(true_text)
    + lambda * softplus(margin + score(hard_negative) - score(true_text))
```

推理时使用目标分数相对近音竞争词的差值，而不是只用目标绝对分数。该目标直接
对应 `hi/haier` 一类误唤醒，比继续加深两层 Temporal Adapter 更有潜力。

### 3. 训练自研关键词判别头

最终头只接收自研 CTC/匹配特征，使用训练 pair 的 `label` 监督：

- CTC 目标分数、greedy 编辑相似度和近音词 margin；
- WavLM/HuBERT 两分支分数及差异；
- 后续注册音频匹配分数；
- 仅用于校准的长度和置信度特征。

优先 logistic regression 或两层 MLP，不先上复杂 GBDT。模型容量不是瓶颈，避免
在 1 万条 dev 上反复选特征造成泄漏。训练集拟合，dev 只选一次版本和融合权重。

## P1：补回注册音频

当前最佳系统只使用 `enroll_txt + query audio`，完全丢掉了官方提供的 enroll
audio。旧 Base+ matcher unseen 接近随机，只能证明旧全局匹配头失败，不能证明
注册音频没有价值。

新的 matcher 应使用 WavLM/HuBERT Large 冻结逐帧特征：

- enroll/query 帧之间做带 mask 的 cross-attention、soft-DTW 或单调软对齐；
- 加入音素序列作为对齐条件，而不是无条件比较两段全局 embedding；
- 训练 batch 由正 pair、随机负 pair、音素近邻 hard negative 共同组成；
- 使用 supervised contrastive/triplet margin，让真实同词对高于近音异词对；
- 最终与 CTC 判别分支融合，不单独替代当前 `0.90123`。

这是比单纯再换一个底模更高上限的路线，因为文本 CTC 和注册音频条件匹配的错误
来源不同。先用 50K pair；只有 Dev 明确提升才讨论 500K 全量。

## P2：底模横测

先为代码增加统一 backbone adapter，隔离三种输入/长度接口：raw waveform
Transformers、Whisper log-Mel、NeMo acoustic encoder。所有横测保持相同：

- 256 条 smoke；
- 100,000 utterance；
- 音素 Temporal Adapter，2 层、dim 256；
- batch 按显存调整，但有效训练样本和 epoch 相同；
- encoder 全冻结，DEMAND 配置不变；
- 同时比较单模型 Dev、与 E015/E016 的融合增益、峰值显存和推理速度。

顺序为 W2v-BERT 2.0、Parakeet encoder、Whisper Large-v3 encoder。新模型即使单独
只持平，只要能使融合 Dev 提升 `0.005` 也可保留。不要在 100K 未胜出前跑全量。

## P3：后续王牌

按潜力排序：

1. 音素 hard negative + CTC 似然比/判别头。
2. Large 注册音频条件 matcher + hard negative。
3. W2v-BERT 2.0 / Parakeet encoder 新底模及跨底模融合。
4. 字符、音素和音节/发音变体多任务头。
5. 顶层少量 LoRA/解冻；须先向主办方确认合规边界。
6. MUSAN 人声干扰、RIR、codec、带宽和速度扰动。
7. 外部英语带转写语料训练自研 CTC 头。

固定 seed ensemble、三模型同架构融合、继续增加 epoch、Adapter dim/层数小网格均
降为低优先级。它们可能带来千分位，但不具备填补 0.90 到 0.98 差距的结构性潜力。

## 执行顺序

1. 实现 CTC greedy/似然差特征导出，先用现有 checkpoint 做零重训诊断。
2. 实现训练词表音素近邻挖掘和训练集判别头，在 dev 验证。
3. 将 hard-negative margin 加入音素 CTC 训练，100K 横测。
4. 并行准备 W2v-BERT 2.0 backbone adapter，完成 100K 横测。
5. 重做 Large 注册音频 matcher，使用 50K pair + hard negatives。
6. 只把明确提升的分支加入现有 WavLM/HuBERT 融合池。
7. 某条路线在小规模至少提升一个百分点后，再单独确认是否跑全量。
