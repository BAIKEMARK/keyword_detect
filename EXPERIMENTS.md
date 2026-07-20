# 实验记录

最近更新：2026-07-19

## 记录约定

- `Dev Mean` = `(Dev Seen + Dev Unseen) / 2`，checkpoint 按该值选择。
- `pair` 指一组 enroll/query；字符或音素 CTC 会把 5 万 pair 展开为最多
  10 万条音频文本 `utterance`。
- `线上` 是比赛平台返回的提交分数；`-` 表示没有提交或没有留存成绩。
- checkpoint、日志、数据集、噪声和提交 CSV 不进入 GitHub，表中路径均相对
  项目根目录。
- smoke test 只用于检查流程，不与正式实验直接比较。

## 结果总表

| ID | 状态 | 方案 | 训练规模 | Dev Seen | Dev Unseen | Dev Mean | 线上 | 最佳 epoch | Checkpoint | 代码 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| E001 | 历史结果，有缺陷 | CNN `frame_maxmean`，未修 padding mask | 50,000 pair | 0.6053 | 0.5019 | 0.5536 | - | 15 | `baseline/checkpoints/frame_demand_50k_w2.pt` | `1f5c914` 之前 |
| E002 | 诊断实验 | CNN `frame_maxmean`，已修 padding mask | 50,000 pair | 0.5772 | 0.5031 | 0.5402 | - | 5 | `baseline/checkpoints/frame_masked_demand_50k_e5.pt` | `32f1b4b` |
| E003 | 已完成 | 冻结 WavLM Base+ 音频匹配 | 50,000 pair | 0.7132 | 0.4950 | 0.6041 | - | 1/3 | `baseline/checkpoints/wavlm_base_plus_50k_e3.pt` | `4ec43c2` |
| E004 | 当前主模型 | 冻结 WavLM Base+ 字符 CTC | 100,000 utterance | 0.8104 | 0.8117 | 0.8111 | **0.81103** | 10 | `baseline/checkpoints/wavlm_char_ctc_100k_e10.pt` | `418add6` |
| E005 | 已完成 | 冻结 WavLM Base+ 音素 CTC | 100,000 utterance | 0.8392 | 0.8419 | 0.8406 | **0.83939** | 8 | `baseline/checkpoints/wavlm_phoneme_ctc_100k_e10.pt` | `2d40979` |
| E006 | 已完成 | 字符 + 音素 CTC 秩融合 | 100,000 utterance 两分支 | 0.8362 | 0.8375 | 0.8369 | **0.84202** | - | 字符 E004 + 音素 E005 | `7f2d962` |
| E007 | 当前最佳 | 全量冻结 WavLM Base+ 音素 CTC | 1,000,000 utterance | 0.8421 | 0.8444 | 0.8433 | **0.84335** | 2 | `baseline/checkpoints/wavlm_phoneme_ctc_full_e3.pt` | `1303514` |
| E008 | 已完成，不提交 | 音素 CTC 20 epoch，batch size 300 | 100,000 utterance | 0.8368 | 0.8389 | 0.8378 | - | 20 | `baseline/checkpoints/wavlm_phoneme_ctc_100k_e20.pt` | `2d40979` |
| E009 | 已完成，待线上验证 | 全量冻结 WavLM Base+ 音素 CTC，从零训练 10 epoch，batch size 256 | 1,000,000 utterance | 0.8408 | 0.8456 | 0.8432 | - | 8 | `baseline/checkpoints/wavlm_phoneme_ctc_full_scratch_e10.pt` | `9b735c1` |

## 线上提交记录

以下表格按比赛平台返回结果原样记录，当前最高分为 `0.84335`。

| ID | 状态 | 评分 | 提交文件名 | 提交者 | 提交时间 |
|---:|---|---:|---|---|---|
| 1 | 返回分数 | **0.84335** | `submission_wavlm_phoneme_ctc_full_epoch2.csv` | Mark | 2026-07-18 23:13:29 |
| 2 | 返回分数 | 0.84202 | `submission_wavlm_ctc_rank_fusion.csv` | Mark | 2026-07-18 09:04:18 |
| 3 | 返回分数 | 0.83939 | `submission_wavlm_phoneme_ctc_100k.csv` | Mark | 2026-07-17 23:11:50 |
| 4 | 返回分数 | 0.81103 | `submission_wavlm_char_ctc_100k.csv` | Mark | 2026-07-17 19:42:00 |
| 5 | 返回分数 | 0.65978 | `submission_frame_noise_50k.csv` | Mark | 2026-07-13 20:20:18 |
| 6 | 返回分数 | 0.62547 | `submission.csv` | Mark | 2026-07-11 20:38:53 |

## E001：帧级 CNN，padding mask 修复前

配置：

- `model=frame_maxmean`
- 50,000 pair，15 epoch，batch size 128，学习率 `1e-3`
- `pos_weight=4.0`
- DEMAND 真实噪声 144 条，混噪概率 0.5，SNR `[-10, 5]` dB
- 训练设备：Apple MPS
- 参数量：22,978

复现命令：

```bash
python baseline/train.py \
  --subset 50000 \
  --epochs 15 \
  --model frame_maxmean \
  --pos-weight 4.0 \
  --noise-prob 0.5 \
  --noise-dir noise/DEMAND_16k/wav \
  --out baseline/checkpoints/frame_demand_50k_w2.pt
```

结论：seen 随训练提升，但 unseen 基本等于随机。该 checkpoint 在帧级 padding
mask 修复前产生，不能作为当前实现的严格基线，只保留作历史参考。

## E002：帧级 CNN，padding mask 修复后诊断

配置与 E001 基本相同，但只训练 5 epoch，并使用修复后的有效帧 mask。
本次日志显示 `real noise files: 0`，因此虽然传入了 DEMAND 路径，实际没有加载
真实噪声，不能作为“修 mask + DEMAND”的正式对照。

结论：修复后 5 epoch 的 unseen 仍接近 0.5。该实验用于确认代码流程，不用于
判断真实噪声或完整 15 epoch 的最终上限。

## E003：冻结 WavLM Base+ 音频匹配

配置：

- 冻结 `microsoft/wavlm-base-plus`，只训练 100,062 参数的匹配头
- 50,000 pair，3 epoch，batch size 128，学习率 `1e-3`
- 最长音频 2.5 秒，CUDA AMP
- query 混入 DEMAND：概率 0.5，SNR `[-10, 5]` dB，真实噪声 144 条
- 训练设备：NVIDIA A10 24GB

复现命令：

```bash
python3 baseline/train_wavlm.py \
  --model-id /mnt/workspace/models/wavlm-base-plus \
  --subset 50000 \
  --epochs 3 \
  --bs 128 \
  --workers 8 \
  --device cuda \
  --noise-prob 0.5 \
  --noise-dir noise/DEMAND_16k/wav \
  --out baseline/checkpoints/wavlm_base_plus_50k_e3.pt
```

结论：强预训练底模把 seen 提高到 0.7132，但 unseen 仍为 0.4950。仅比较
两段音频无法解决训练词表与 unseen 词表零重叠的问题。

## E004：冻结 WavLM Base+ 字符 CTC

配置：

- 5 万 pair 展开为 10 万条音频文本，训练时同时使用 enroll/query 音频文本
- 冻结 WavLM Base+，只训练 21,545 参数的层加权和字符 CTC 头
- blank + `a-z` + apostrophe，共 28 类
- 10 epoch，batch size 128，学习率 `1e-3`，最长音频 2.5 秒
- DEMAND 混噪概率 0.5，SNR `[-10, 5]` dB，真实噪声 144 条
- 每轮有 3 条过短音频因 CTC 无法对齐而跳过
- 推理只使用测试提供的 `enroll_txt` 和 query 音频

复现命令：

```bash
python3 baseline/train_wavlm_ctc.py \
  --model-id /mnt/workspace/models/wavlm-base-plus \
  --units char \
  --subset 100000 \
  --epochs 10 \
  --bs 128 \
  --workers 8 \
  --device cuda \
  --noise-prob 0.5 \
  --noise-dir noise/DEMAND_16k/wav \
  --out baseline/checkpoints/wavlm_char_ctc_100k_e10.pt
```

结论：Dev Mean 0.8111 与线上 0.81103 几乎一致，说明开发集评估可靠。
字符 CTC 同时解决 seen 和 unseen，是当前主干方案。

## E005：冻结 WavLM Base+ 音素 CTC

实际配置与结果：

- 与 E004 使用相同的 10 万条 utterance 和冻结 WavLM Base+
- `g2p_en==2.1.0`，blank + 39 个无重音 ARPAbet 音素，共 40 类
- `numpy>=1.24,<2`，避免 `g2p_en` 在 NumPy 2.x 下的 OOV 数值溢出
- 10 epoch，batch size 128，学习率 `1e-3`
- DEMAND 混噪概率 0.5，SNR `[-10, 5]` dB

- 最佳 epoch 8：seen `0.8392`，unseen `0.8419`，mean `0.8406`
- 线上提交：`submission_wavlm_phoneme_ctc_100k.csv`，AUC `0.83939`

复现命令：

```bash
export NLTK_DATA=/mnt/workspace/nltk_data

python3 baseline/train_wavlm_ctc.py \
  --model-id /mnt/workspace/models/wavlm-base-plus \
  --units phoneme \
  --subset 100000 \
  --epochs 10 \
  --bs 128 \
  --workers 8 \
  --device cuda \
  --noise-prob 0.5 \
  --noise-dir noise/DEMAND_16k/wav \
  --out baseline/checkpoints/wavlm_phoneme_ctc_100k_e10.pt
```

结论：音素监督比字符监督更适合该英文关键词任务，线上比 E004 提高
`0.02836`。该模型随后参与 E006 融合。

## E006：分数融合

实际输入：

- E004 字符 CTC 分数
- E005 音素 CTC 分数
- E003 音频匹配分数未加入，因其 unseen 接近随机

在 seen/unseen 内分别做平均秩归一化，并在 dev 搜索一个全局权重。最终权重为
音素 `0.666`、字符 `0.334`。dev mean `0.8369`，线上
`submission_wavlm_ctc_rank_fusion.csv` 得分 `0.84202`。虽然该次 dev 导出的
音素分数未完全复现 E005，但线上结果确认两分支具有互补性。

## E007：全量50万 pair 音素 CTC

配置与已知事实：

- `train/train_label.csv` 含50万 pair，展开为100万 utterance
- 冻结 WavLM Base+，音素 CTC，DEMAND 概率0.5，SNR `[-10, 5]` dB
- epoch 1：seen `0.8310`，unseen `0.8342`，mean `0.8326`
- epoch 2：seen `0.8421`，unseen `0.8444`，mean `0.8433`
- epoch 3 中途停止，epoch 2 checkpoint 已完整保存
- 线上 `submission_wavlm_phoneme_ctc_full_epoch2.csv` 得分 **`0.84335`**

该日志曾被两个训练进程同时写入：`3906` step/epoch 对应 batch size 256，
`7812` step/epoch 对应 batch size 128。epoch 2 完整评估前的日志为
`7812` step 且峰值显存 `4.41GB`，因此结果高概率来自 batch size 128；但旧
checkpoint 未保存 batch size，不能作为绝对证明。后续 checkpoint 必须记录完整
训练配置和优化器状态。

当前决定：按 batch size 128 从 epoch 2 checkpoint 继续到目标 epoch 5。旧文件
缺少 optimizer 和 AMP scaler，因此首次续训会恢复模型头和历史最佳指标，并重建
optimizer/scaler；之后每轮生成的 `.last.pt` 包含完整训练状态，可继续无损恢复。

本次断点续训改造后，checkpoint 同时记录 batch size、学习率、数据路径、训练
utterance 数、增强参数、目标轮数、optimizer、AMP scaler、随机数状态和
best/current epoch 指标。`--out` 保留最佳模型，`.last.pt` 保留最近完整 epoch。

## E009：全量音素 CTC 从零训练 10 epoch

配置：

- 全量 500,000 pair，展开为 1,000,000 utterance
- 冻结 WavLM Base+，只训练 30,773 参数的音素 CTC 头
- batch size 256，学习率 `1e-3`，10 epoch，DEMAND 概率 0.5，SNR `[-10, 5]` dB
- `3906` step/epoch，峰值显存 `8.44GB`

Dev 结果：

| Epoch | Seen | Unseen | Mean |
|---:|---:|---:|---:|
| 1 | 0.8316 | 0.8348 | 0.8332 |
| 2 | 0.8380 | 0.8410 | 0.8395 |
| 3 | 0.8388 | 0.8438 | 0.8413 |
| 4 | 0.8404 | 0.8451 | 0.8427 |
| 5 | 0.8413 | 0.8451 | 0.8432 |
| 6 | 0.8413 | 0.8448 | 0.8430 |
| 7 | 0.8410 | 0.8448 | 0.8429 |
| 8 | 0.8408 | 0.8456 | **0.8432** |
| 9 | 0.8399 | 0.8437 | 0.8418 |
| 10 | 0.8410 | 0.8449 | 0.8430 |

结论：从零训练的最佳 Dev Mean 为 `0.8432`，低于 E007 续训得到的 `0.8451`，
暂不替换当前最佳模型。`full_scratch_e10.pt` 保存的是最佳 epoch 8，
`full_scratch_e10.last.pt` 保存的是最后 epoch 10。

## E008：100K 音素 CTC 20 epoch

该实验使用 batch size 300，20轮最佳为 epoch 20：seen `0.8368`、unseen
`0.8389`、mean `0.8378`。由于原10轮 E005 使用 batch size 128，两者不是只改变
epoch 的严格对照。该模型低于 E005，不提交。

## 后续记录模板

新增实验时复制下面字段，并在结果总表追加一行，不覆盖旧结果：

```text
ID：
状态：
目标：
代码 commit：
训练数据与规模：
模型与可训练参数：
epoch / batch size / learning rate：
增强与 SNR：
checkpoint：
Dev Seen：
Dev Unseen：
Dev Mean：
线上成绩：
提交 CSV：
结论与下一步：
```
