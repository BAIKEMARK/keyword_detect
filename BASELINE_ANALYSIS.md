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
| `train_subset` | 50,000 | 9,956 | 40,044 | 约 1:4，并非赛题描述的 50 万对 |
| `dev_seen` | 5,000 | 2,000 | 3,000 | 注册词基本在训练词表中 |
| `dev_unseen` | 5,000 | 2,000 | 3,000 | 注册词不在训练词表中 |
| `eval_seen` | 50,000 | 未提供 | 未提供 | 仅有 `id,enroll_txt` |
| `eval_unseen` | 50,000 | 未提供 | 未提供 | 仅有 `id,enroll_txt` |

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

- 从固定配对 CSV 中随机选最多 `--subset` 条；当前数据只有 5 万条，所以默认的 `500000` 不会产生更多训练对。
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
6. **只使用 5 万个固定对。** 赛题允许自行重新配对，当前实现没有从相同语音池生成更多正负对。
7. **输出并非严格校准概率。** `pos_weight=4` 会改变概率校准；不过指标只看 AUC 排序，所以首要目标仍是提升排序质量，而不是先做阈值校准。

## 6. 建议的升级顺序

### P0：先复现并锁定 baseline

- 用支持 RAR5 的解压器解压三个数据包。
- 安装 `requirements.txt` 后跑通训练、seen/unseen AUC 和 10 万行提交文件。
- 固定随机种子、保存训练配置和每轮双子集指标。
- 当前 `eval.rar` 的 SHA-256 与 Git LFS 记录一致；macOS 自带 `bsdtar` 报错更可能是解压器兼容问题，不应直接认定文件损坏。

### P1：先修正声学表征

- 保留孪生范式，换成能保留时间结构的 CNN/TDNN/Conformer 编码器。
- 使用 attentive statistics pooling 或带 mask 的时间池化，不要直接对整个时频平面平均。
- 加入长度 mask、CMVN 和合理的裁剪/补齐策略。

这是最直接、最容易通过消融验证的一步。

### P2：针对评测场景训练

- 按 -10 至 5 dB 在线混入官方噪声和开源噪声，并加入混响、增益、速度扰动。
- 根据文本、音素距离或当前模型高分误报挖掘 hard negatives。
- 动态重新配对，扩大训练对数量；batch 内构造额外负样本。
- 在 BCE 外尝试 supervised contrastive / triplet / prototypical loss，但保留一个由自己训练的匹配判决头。

### P3：利用注册文本

- 将 `enroll_txt` 转成字符或音素序列，训练自己的文本编码器。
- 训练音频 embedding 与文本/音素 embedding 对齐，再将“音频-音频分数”和“测试音频-注册文本分数”输入自己训练的融合判决头。
- 对同音、所有格、复数等变体做发音规范化，避免原始字符串硬匹配。

这条路线对低 SNR 和 unseen 词最有价值，也最贴合测试时额外提供注册文本的设计。

### P4：合规使用开源声学模型

可以用 Whisper、WavLM 等开源权重抽取帧级特征，但不要直接用其转写结果、相似度或 API 输出完成唤醒判决。合规且更稳妥的形式是：开源模型只产出特征，后面连接并训练自己的时序编码器、匹配头和融合模型，同时在方案中明确数据与权重来源。

## 7. 当前可复现命令

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

## 8. 一句话判断

这个 baseline 的价值是把“音频对读取 -> 孪生匹配 -> 双子集 AUC -> 提交 CSV”闭环跑通；它最大的结构性问题是丢失时间顺序、没有训练期噪声建模、也没有使用注册文本。后续最合理的路线不是在两层 CNN 上反复调参，而是先建立保留时序的强声学匹配模型，再加入 hard negative、低 SNR 增强和文本/音素辅助。
