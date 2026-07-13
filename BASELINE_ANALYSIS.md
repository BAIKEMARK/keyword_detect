# 讯飞基于语音注册的关键词检测：Baseline 阅读记录

记录日期：2026-07-13

## 1. 任务本质

每个样本包含注册音频、注册文本和测试音频。模型需要输出测试音频与注册关键词是否同文本/同关键词的分数，并提交 `id,posterior` 两列。

- 测试分为 `seen` 和 `unseen`，平台分别计算 AUC，再取二者平均。
- 测试场景包含 clean 与约 -10 至 5 dB 噪声，并包含发音相似的负样本。
- 允许使用开源模型做特征提取，但开源模型权重或 API 不能直接完成关键词判决；最终判决模型必须由参赛者训练。
- 测试集只能用于最终推理，不能参与训练。
- 截止时间为 2026-08-27 17:00，每队每天最多提交 3 次。

因此，这不是固定类别分类，而是 Query-by-example 二元匹配：给定一个新的注册词，也要判断另一段音频是否在说它。

## 2. 当前仓库和数据

仓库来源：`https://www.modelscope.cn/datasets/LoveFishO/keyword_detect.git`

当前文件没有解压。代码期望解压后存在以下路径：

```text
keyword_detect/
├── train_subset/train_label.csv
├── train_subset/wav.zip
├── dev/dev_seen/dev_seen_label.csv
├── dev/dev_seen/wav.zip
├── dev/dev_unseen/dev_unseen_label.csv
├── dev/dev_unseen/wav.zip
├── eval/eval_seen/wav.zip
└── eval/eval_unseen/wav.zip
```

实测数据：

| 子集 | 样本数 | 正样本 | 负样本 | 备注 |
| --- | ---: | ---: | ---: | --- |
| `train_subset` | 50,000 | 9,956 | 40,044 | 当前本地 baseline 子集，约 1:4 |
| `dev_seen` | 5,000 | 2,000 | 3,000 | 注册词基本在训练词表中 |
| `dev_unseen` | 5,000 | 2,000 | 3,000 | 注册词不在训练词表中 |
| `eval_seen` | 50,000 | 未提供 | 未提供 | 仅有 `id,enroll_txt` |
| `eval_unseen` | 50,000 | 未提供 | 未提供 | 仅有 `id,enroll_txt` |

这里的 5 万对只描述当前 ModelScope 仓库中的 `train_subset`，不代表赛事官方完整训练集只有 5 万对。赛事页面标注的 50 万对完整训练数据需要从官网下载并另行整理；当前本地尚未包含它，后续应以官网文件为准。

训练 CSV 实际字段为：

```csv
id,enroll_txt,query_txt,label
pair_335244,inspiring,inspiring,1
pair_058370,philosophers,cleopatra,0
```

每个 ID 在 `wav.zip` 中对应：

```text
wav/{id}_enroll.wav
wav/{id}_query.wav
```

训练包共有 100,000 条 WAV，均为 16 kHz 单声道。时长中位数约 0.55 秒，99 分位约 1.02 秒，最长约 2.04 秒；1,184 条超过 1 秒。

值得注意：有 69 个正样本的两列文本拼写不同，例如 `markets/market's`、`principal/principle`、`lawrence/laurence`。它们通常是同音词、所有格/复数变体或等价读音。这说明不能把原始字符串相等直接当作最终判决规则，文本分支需要做发音层面的建模或规范化。

## 3. Baseline 的完整链路

### 3.1 输入特征

`baseline/data.py` 完成以下步骤：

1. 从嵌套 `wav.zip` 按 ID 读取注册音频和测试音频。
2. 转为单声道，必要时重采样到 16 kHz。
3. 计算 40 维 log-Mel 频谱：400 点窗，160 点帧移。
4. 截断或补齐到 100 帧，得到 `(1, 40, 100)` 输入。

### 3.2 模型

`baseline/model.py` 是共享权重的孪生网络：

```text
注册音频 -> log-Mel -> 两层 2D CNN -> 全局平均池化 -> 64 维 L2 embedding
测试音频 -> log-Mel -> 同一个 CNN  -> 全局平均池化 -> 64 维 L2 embedding
                                              ↓
                         cosine similarity * 可学习 scale + bias
                                              ↓
                                            logit
```

模型约 2.3 万参数，非常小。`n_mels` 构造参数实际没有参与网络结构。

### 3.3 训练

`baseline/train.py`：

- 从固定配对 CSV 中随机选最多 `--subset` 条；对当前本地 `train_subset` 而言只有 5 万条，所以默认的 `500000` 不会产生更多训练对。换成官网完整训练 CSV 后才可能使用更多样本。
- 使用 `BCEWithLogitsLoss(pos_weight=4)`，对应约 1:4 的正负比例。
- 优化器为 Adam，学习率 `1e-3`，默认 10 epoch、batch size 128。
- 每轮分别计算 `dev_seen` 和 `dev_unseen` AUC，以两者平均值保存最佳 checkpoint。

### 3.4 推理

`baseline/infer.py`：

- 分别对 `eval_seen` 和 `eval_unseen` 预测。
- 对 logit 做 sigmoid，添加 `seen_` / `unseen_` 前缀。
- 合并为一个 100,000 行的 `id,posterior` CSV。

这与官方提交格式一致。

## 4. 这个 Baseline 在学什么

它学习的是“两个短语音的全局声学相似度”。正样本拉近 embedding，负样本推远 embedding；共享编码器使模型可以处理训练词表外的新注册词。

它没有使用注册文本，也没有固定关键词分类头，因此思路简单且符合 QbE 任务。它适合作为数据管线和提交格式的烟雾测试，但不是有竞争力的终点模型。

## 5. 主要能力边界

1. **时间顺序损失严重。** 两层卷积后同时对频率和时间做全局平均，模型更接近统计局部声学纹理，难以区分音素相近但顺序或局部细节不同的词。
2. **完全忽略注册文本。** 测试明确提供注册文本，尤其在低信噪比和 unseen 词上，这是被浪费的强先验。
3. **没有噪声增强。** 赛题明确覆盖 -10 至 5 dB，当前训练代码没有在线混噪、混响、增益、速度或频谱增强。
4. **没有困难负样本策略。** 固定随机负对无法重点学习 `hi/haier` 一类发音相似词，而这正是测试重点。
5. **固定 100 帧且无 mask。** 超过约 1 秒的音频被截断；短音频在 log-Mel 后补零，并参与全局平均，模型可能利用时长/补零比例而非纯发音内容。
6. **当前本地运行只会使用 5 万个固定对。** 这是本地只下载了 `train_subset` 所致，不是赛题数据上限；赛题允许使用官网完整训练数据并自行重新配对更多正负样本。
7. **输出并非严格校准概率。** `pos_weight=4` 会改变概率校准；不过指标只看 AUC 排序，所以首要目标仍是提升排序质量，而不是先做阈值校准。

## 6. 官方给出的三档优化案例

以下内容来自赛事官方的 baseline 优化说明。它们不是互斥方案，可以按成本从低到高逐步叠加。

### 第一档：容易上手

1. **增加训练轮次。** Baseline 默认只训练 10 轮，官方示例是先尝试 `python train.py --epochs 15`。
2. **调整正负样本权重。** 当前约 1:4，默认 `pos_weight=4.0`；可以对比 `3.0`、`4.0`、`5.0`，以双子集平均 AUC 而不是训练 loss 选值。
3. **加入简单噪声增强。** 训练时按 -10 至 5 dB 采样 SNR，把高斯噪声混入音频，使训练分布先覆盖测试噪声范围。
4. **使用全量 50 万对训练数据。** 从[赛事官网下载页](https://challenge.xfyun.cn/topic/info?type=KDBURI&option=stsj)下载并解压 `train.zip`，然后把配置从 `train_subset` 改为完整的 `train`：

```python
self.train_zip = os.path.join(r, "train", "wav.zip")
self.train_csv = os.path.join(r, "train", "train_label.csv")
```

官方提供的全量训练包直链为：

```text
https://static-op.xfyun.cn/aicontest/2026%E7%AE%97%E6%B3%95%E8%B5%9B/757%E5%9F%BA%E4%BA%8E%E7%94%A8%E6%88%B7%E6%B3%A8%E5%86%8C%E7%9A%84%E5%85%B3%E9%94%AE%E8%AF%8D%E6%A3%80%E6%B5%8B%E6%8C%91%E6%88%98%E8%B5%9B/train.zip
```

### 第二档：需要修改数据或模型

1. **帧级匹配。** 不再把整段音频直接压成一个向量，保留逐帧 embedding，并使用对称 max-mean 做软对齐。设帧级相似度矩阵为 `S`，可使用：

```text
score = 0.5 * (mean_i max_j S[i,j] + mean_j max_i S[i,j])
```

2. **使用真实噪声。** 用官方噪声或开源真实噪声替代纯高斯噪声，仍按 -10 至 5 dB 混合，使训练场景更接近设备端人声干扰和环境噪声。

### 第三档：进阶方案

1. **预训练模型提取特征。** 使用 Whisper 等开源预训练 encoder 替代 log-Mel 前端，以改善 unseen 泛化。赛规允许特征提取，但禁止直接使用开源模型输出完成最终判决，因此应冻结 encoder，并在其输出之上训练自己的匹配模型。
2. **难负样本挖掘。** 根据发音相似度构造额外负对，补充 `hi/haier` 一类测试重点；还可以使用当前模型的高分误报进行迭代挖掘。

## 7. 我们的方案：基模优先，公平横测

官方第三档明确允许“冻结预训练 encoder + 自研匹配头”。结合此前视觉比赛中强基模直接突破传统模型上限的经验，我们不应先在两层 CNN 上投入大量调参，而应尽早确认哪一个语音基模提供了最适合本任务的声学-音素表征。

### 7.1 合规模型边界

```text
注册音频 ─> 冻结的预训练 encoder ─> 帧级特征 ─┐
                                                  ├─> 自研时序匹配网络 ─> 自研判决头 ─> posterior
测试音频 ─> 冻结的预训练 encoder ─> 帧级特征 ─┘
```

- 预训练 encoder 只负责特征提取，不使用其转写、原始相似度或 API 输出直接判决。
- 匹配网络和最终判决头必须使用赛事训练标签训练。
- 不把预训练 embedding 的原始余弦相似度直接作为提交分数。
- 记录基模名称、权重来源、冻结方式、自研模块结构和训练数据来源，供最终复现与审核。

### 7.2 第一轮基模候选

| 基模 | 主要假设 | 风险 |
| --- | --- | --- |
| WavLM Large | 自监督语音表征和噪声鲁棒性较适合低 SNR 匹配 | 参数量和特征缓存较大 |
| HuBERT Large | 离散语音单元预训练可能更利于细粒度音素区分 | 极低 SNR 表现需要实测 |
| Whisper Encoder | 大规模弱监督训练可能带来较强抗噪和 unseen 泛化 | 较重，且 ASR 导向未必最适合短词帧级匹配 |

第一轮先冻结全部基模，使用相同数据、增强、Adapter、匹配头、训练轮数和随机种子。这样测到的差异主要来自基模表征，而不是训练配方差异。

### 7.3 自研匹配头

不采用“全局向量直接余弦”的弱判决方式。推荐保留帧级特征，依次比较：

1. 官方建议的对称 max-mean，作为最小可用匹配头。
2. 小型 Cross-Attention 或 Conformer Adapter，学习注册帧与测试帧的局部对齐。
3. 在 BCE 之外加入 supervised contrastive loss，但最终 posterior 仍由自研二分类头输出。

### 7.4 推荐实验顺序

1. **锁定基线：** 全量 50 万对、15 epoch、真实噪声增强，记录 seen/unseen AUC。
2. **快速选基模：** 用固定的 5% 至 10% 训练子集横测 WavLM、HuBERT、Whisper，保持匹配头完全一致。
3. **确定匹配方式：** 对优胜基模比较全局池化、对称 max-mean、Cross-Attention。
4. **针对测试难点：** 加入真实噪声、相似音 hard negatives、动态重新配对和 batch 内负样本。
5. **训练最终模型：** 在完整数据上训练前两名方案，最后再考虑模型融合。
6. **可选文本分支：** 将 `enroll_txt` 转为字符或音素序列，训练自己的文本编码器，与音频匹配分数融合；对同音、所有格和复数变体做发音规范化，禁止原始字符串硬匹配。

模型选择至少比较以下指标：

- `dev_seen AUC`、`dev_unseen AUC` 及两者平均值。
- 人工构造的 -10、-5、0、5 dB 验证集 AUC。
- 发音相似负样本 AUC。
- 单样本推理时间、峰值显存和特征缓存体积。

这条路线的判断原则是：**基模决定表征上限，匹配头决定能否利用表征，数据构造决定最终判别边界。**

## 8. 当前可复现命令

数据解压到上述目录后，在 `keyword_detect` 目录执行：

```bash
python -m pip install -r requirements.txt
python baseline/train.py
python baseline/infer.py --ckpt baseline/checkpoints/best.pt --out submission.csv
```

提交前至少检查：

- 表头严格为 `id,posterior`。
- 总行数为 100,001（含表头）。
- seen 与 unseen 各 50,000 条，ID 唯一且前缀正确。
- `posterior` 全部是有限数且位于 `[0, 1]`。

## 9. 一句话判断

这个 baseline 的价值是把“音频对读取 -> 孪生匹配 -> 双子集 AUC -> 提交 CSV”闭环跑通；它最大的结构性问题是丢失时间顺序、没有训练期噪声建模、也没有使用注册文本。后续最合理的路线不是在两层 CNN 上反复调参，而是先建立保留时序的强声学匹配模型，再加入 hard negative、低 SNR 增强和文本/音素辅助。
