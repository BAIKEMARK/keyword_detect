# 实验记录

最近更新：2026-08-01

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
| E007 | 已完成 | 全量冻结 WavLM Base+ 音素 CTC | 1,000,000 utterance | 0.8421 | 0.8444 | 0.8433 | **0.84335** | 2 | `baseline/checkpoints/wavlm_phoneme_ctc_full_e3.pt` | `1303514` |
| E008 | 已完成，不提交 | 音素 CTC 20 epoch，batch size 300 | 100,000 utterance | 0.8368 | 0.8389 | 0.8378 | - | 20 | `baseline/checkpoints/wavlm_phoneme_ctc_100k_e20.pt` | `2d40979` |
| E009 | 历史线上最佳 | 全量冻结 WavLM Base+ 音素 CTC，从零训练 10 epoch，batch size 256 | 1,000,000 utterance | 0.8408 | 0.8456 | 0.8432 | **0.84446** | 8 | `baseline/checkpoints/wavlm_phoneme_ctc_full_scratch_e10.pt` | `9b735c1` |
| E010 | 已完成，不采用 | E007 旧 checkpoint 兼容续训，batch size 128，optimizer/scaler 重建 | 1,000,000 utterance | 0.8443 | 0.8460 | 0.8451 | **0.84074** | 4 | `baseline/checkpoints/wavlm_phoneme_ctc_full_e3.pt` | `9b735c1` |
| E011 | 代码就绪，待训练 | 全量冻结 WavLM Base+ 帧级音频匹配 | 500,000 pair | - | - | - | - | - | `baseline/checkpoints/wavlm_matcher_full_e3.pt` | `14db8ae` |
| E012 | 已完成 | 冻结 WavLM Base+ 音素 CTC + Temporal Adapter，batch size 128 | 100,000 utterance | 0.8725 | 0.8723 | **0.8724** | **0.86944** | 3 | `baseline/checkpoints/wavlm_base_plus_phoneme_temporal_100k_e3.pt` | `7fc28b7` |
| E013 | 历史线上最佳 | 冻结 WavLM Base+ 音素 CTC + Temporal Adapter，batch size 256，15 epoch | 100,000 utterance | 0.8773 | 0.8710 | **0.8742** | **0.87676** | 13 | `baseline/checkpoints/wavlm_base_plus_phoneme_temporal_100k_bs256_e15.pt` | `7fc28b7` |
| E014 | 历史线上最佳 | 冻结 WavLM Large 音素 CTC + Temporal Adapter，batch size 128 | 100,000 utterance | 0.8893 | 0.8802 | **0.8847** | **0.88555** | 1 | `baseline/checkpoints/wavlm_large_phoneme_temporal_100k_e3.pt` | `7fc28b7` |
| E015 | 已完成，融合分支 | 冻结 HuBERT Large 音素 CTC + Temporal Adapter，batch size 128 | 100,000 utterance | 0.8930 | 0.8803 | **0.8866** | **0.89071** | 10 | `baseline/checkpoints/hubert_large_phoneme_temporal_100k_e3.pt` | `7fc28b7` |
| E016 | 已完成 | 冻结 WavLM Large 音素 CTC + Temporal Adapter，从零重跑 10 epoch | 100,000 utterance | 0.8900 | 0.8860 | **0.8880** | **0.89054** | 10 | `baseline/checkpoints/wavlm_large_phoneme_temporal_100k_e10.pt` | `7fc28b7` |
| E017 | 当前线上最佳 | WavLM Large + HuBERT Large CTC 秩融合 | 100,000 utterance 两分支 | - | - | - | **0.90123** | - | E015 + E016 | `c3f1b5c` |
| E018 | 诊断完成，淘汰替代打分 | Large CTC greedy、似然差和编辑相似度零重训诊断 | E015 + E016 | 0.8930 / 0.8900 | 0.8803 / 0.8860 | 0.8866 / 0.8880 | - | - | E015 + E016 | `7c1fe7b` |
| E019 | 已完成，停止路线 | 冻结 HuBERT Large 注册音频 soft alignment matcher | 50,000 pair | 0.5182 | 0.4985 | 0.5083 | - | 1 | `baseline/checkpoints/hubert_large_align_50k_e3.pt` | `7c1fe7b` |
| E020 | 已完成，待融合 | HuBERT Large 音素 CTC + Temporal Adapter + hard-negative margin，batch size 128，10 epoch | 100,000 utterance | 0.8960 | 0.8824 | **0.8892** | - | 8 | `baseline/checkpoints/hubert_large_phoneme_temporal_hardneg_100k_e10.pt` | `9f24b06` |

## 线上提交记录

以下表格按比赛平台返回结果原样记录，当前最高分为 `0.90123`。

| ID | 状态 | 评分 | 提交文件名 | 提交者 | 提交时间 |
|---:|---|---:|---|---|---|
| 1 | 返回分数 | **0.90123** | `submission_wavlm_hubert_large_rank_fusion.csv` | Mark | 2026-07-28，具体时间未记录 |
| 2 | 返回分数 | 0.89054 | `submission_wavlm_large_phoneme_temporal_100k_e10_best_epoch10.csv` | Mark | 2026-07-28 10:37:04 |
| 3 | 返回分数 | 0.89071 | `submission_hubert_large_phoneme_temporal_100k_best_epoch10.csv` | Mark | 2026-07-28 10:34:29 |
| 4 | 返回分数 | 0.88555 | `submission_wavlm_large_phoneme_temporal_100k_best_epoch1.csv` | Mark | 2026-07-27 20:23:18 |
| 5 | 返回分数 | 0.87676 | `submission_wavlm_base_plus_phoneme_temporal_100k_bs256_e15.csv` | Mark | 2026-07-27 12:21:32 |
| 6 | 返回分数 | 0.86944 | `submission_wavlm_base_plus_phoneme_temporal_100k_e3.csv` | Mark | 2026-07-27 12:18:44 |
| 7 | 返回分数 | 0.84446 | `submission_wavlm_phoneme_ctc_full_scratch_epoch8.csv` | Mark | 2026-07-20 22:36:13 |
| 8 | 返回分数 | 0.84074 | `submission_wavlm_phoneme_ctc_full_epoch4.csv` | Mark | 2026-07-20 12:04:42 |
| 9 | 返回分数 | 0.84335 | `submission_wavlm_phoneme_ctc_full_epoch2.csv` | Mark | 2026-07-18 23:13:29 |
| 10 | 返回分数 | 0.84202 | `submission_wavlm_ctc_rank_fusion.csv` | Mark | 2026-07-18 09:04:18 |
| 11 | 返回分数 | 0.83939 | `submission_wavlm_phoneme_ctc_100k.csv` | Mark | 2026-07-17 23:11:50 |
| 12 | 返回分数 | 0.81103 | `submission_wavlm_char_ctc_100k.csv` | Mark | 2026-07-17 19:42:00 |
| 13 | 返回分数 | 0.65978 | `submission_frame_noise_50k.csv` | Mark | 2026-07-13 20:20:18 |
| 14 | 返回分数 | 0.62547 | `submission.csv` | Mark | 2026-07-11 20:38:53 |

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

后续按 batch size 128 从 epoch 2 checkpoint 继续到 epoch 5。旧文件缺少
optimizer 和 AMP scaler，因此兼容续训恢复模型头和历史最佳指标，并重建
optimizer/scaler。该续训对应 E010，线上结果低于 epoch 2，不再采用。

本次断点续训改造后，checkpoint 同时记录 batch size、学习率、数据路径、训练
utterance 数、增强参数、目标轮数、optimizer、AMP scaler、随机数状态和
best/current epoch 指标。`--out` 保留最佳模型，`.last.pt` 保留最近完整 epoch。

## E008：100K 音素 CTC 20 epoch

该实验使用 batch size 300，20轮最佳为 epoch 20：seen `0.8368`、unseen
`0.8389`、mean `0.8378`。由于原10轮 E005 使用 batch size 128，两者不是只改变
epoch 的严格对照。该模型低于 E005，不提交。

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

结论：从零训练的最佳 Dev Mean 为 `0.8432`。虽然低于 E010 的 dev `0.8451`，
线上却得到当前最佳 **`0.84446`**。`full_scratch_e10.pt` 保存的是最佳 epoch 8，
`full_scratch_e10.last.pt` 保存的是最后 epoch 10。这说明千分位 dev 差异不足以
稳定预测线上排序，不能继续只按单次 dev 最优值判断模型。

## E010：旧 checkpoint 兼容续训到 epoch 5

- batch size 128，从 E007 已完成的 epoch 2 开始
- 旧 checkpoint 没有 optimizer/scaler，续训时两者从头初始化
- epoch 3：seen `0.8442`，unseen `0.8459`，mean `0.8451`
- epoch 4：seen `0.8443`，unseen `0.8460`，mean `0.8451`
- epoch 5：seen `0.8434`，unseen `0.8443`，mean `0.8439`
- epoch 4 线上得分：`0.84074`

结论：该实验 dev 高于 E007/E009，但线上明显更低。旧模型头配合全新 optimizer
继续训练改变了原有优化轨迹，不属于无损续训；该 checkpoint 不再用于最终方案。

## E011：全量 WavLM 帧级音频匹配

commit `14db8ae` 已让 `train_wavlm.py` 支持显式传入全量 `train_csv`、
`train_zip` 和 500,000 pair，并加入最佳/最新 checkpoint 与断点续训。模型仍使用
既有的冻结 WavLM Base+ 帧级特征、对称 max-mean 和自研二分类头。

这表示“全量帧级音频匹配”的训练代码已经就绪，但实验尚未运行，也尚未加入
难负样本、CTC 分数融合或监督式融合头。因此不能把 `14db8ae` 记作该路线已经
验证有效，只能记作基础训练能力完成。

## E012：100K 音素 CTC + Temporal Adapter

配置：

- 冻结 WavLM Base+，在 encoder 时序输出后加入两层带 mask 的卷积 Adapter
- Adapter dim 256，训练参数 344,373；batch size 128，学习率 `1e-3`
- 100,000 utterance，3 epoch，DEMAND 概率 0.5，SNR `[-10, 5]` dB
- 峰值显存 `4.41GB`

Dev 结果：

| Epoch | Seen | Unseen | Mean |
|---:|---:|---:|---:|
| 1 | 0.8526 | 0.8504 | 0.8515 |
| 2 | 0.8681 | 0.8625 | 0.8653 |
| 3 | 0.8725 | 0.8723 | **0.8724** |

与同为 100,000 utterance 的线性音素 CTC E005 相比，Dev Mean 从 `0.8406`
提升至 `0.8724`，绝对提升 **`0.0318`**；Seen 提升 `0.0333`，Unseen 提升
`0.0304`。线上得分 `0.86944`，相对 E005 的 `0.83939` 提升 **`0.03005`**，
证明 Dev 收益基本保留到测试集。这是目前最强的小规模架构收益，且两类子集同步
提升。

## E013：100K Temporal Adapter，batch size 256，15 epoch

配置沿用 E012，但将 batch size 从 128 增加到 256，并将训练目标从 3 epoch
增加到 15 epoch。checkpoint 为
`baseline/checkpoints/wavlm_base_plus_phoneme_temporal_100k_bs256_e15.pt`，
提交文件为 `submission_wavlm_base_plus_phoneme_temporal_100k_bs256_e15.csv`。

Dev 结果：

| Epoch | Seen | Unseen | Mean |
|---:|---:|---:|---:|
| 1 | 0.8513 | 0.8504 | 0.8508 |
| 2 | 0.8578 | 0.8492 | 0.8535 |
| 3 | 0.8608 | 0.8606 | 0.8607 |
| 4 | 0.8655 | 0.8669 | 0.8662 |
| 5 | 0.8718 | 0.8658 | 0.8688 |
| 6 | 0.8733 | 0.8661 | 0.8697 |
| 7 | 0.8738 | 0.8704 | 0.8721 |
| 8 | 0.8716 | 0.8656 | 0.8686 |
| 9 | 0.8746 | 0.8664 | 0.8705 |
| 10 | 0.8754 | 0.8694 | 0.8724 |
| 11 | 0.8724 | 0.8676 | 0.8700 |
| 12 | 0.8728 | 0.8687 | 0.8708 |
| 13 | 0.8773 | 0.8710 | **0.8742** |
| 14 | 0.8768 | 0.8679 | 0.8724 |
| 15 | 0.8763 | 0.8700 | 0.8732 |

线上得分 **`0.87676`**，比 E012 提升 `0.00732`，比此前全量线性 CTC 最佳
E009 提升 `0.03230`。由于该实验同时改变了 batch size 和训练轮数，不能把收益
单独归因于增加 epoch。最佳 checkpoint 来自 epoch 13；其 Dev Mean 仅比 E012
高 `0.0018`，但线上高 `0.00732`。此外，batch size 256 的 epoch 3 Dev Mean
`0.8607` 低于 E012 使用 batch size 128 的 `0.8724`，说明更大的 batch 本身没有
带来早期收益，而是需要更多 epoch 才能补足较少的参数更新次数。

## E014：100K WavLM Large + Temporal Adapter

配置与 E012 保持一致，只将冻结底模从 WavLM Base+ 换为 WavLM Large：100,000
utterance、batch size 128、3 epoch、两层 dim 256 Temporal Adapter、DEMAND
概率 0.5、SNR `[-10, 5]` dB。冻结参数 315,453,120，可训练参数 410,433，
峰值显存 `6.83GB`。

| Epoch | Seen | Unseen | Mean |
|---:|---:|---:|---:|
| 1 | 0.8893 | 0.8802 | **0.8847** |
| 2 | 0.8851 | 0.8828 | 0.8839 |
| 3 | 0.8861 | 0.8822 | 0.8842 |

最佳 checkpoint 来自 epoch 1。与同配置 Base+ E012 相比，Dev Mean 提升
`0.0123`；Seen 提升 `0.0168`，Unseen 提升 `0.0079`。线上得分 **`0.88555`**，
比 E012 提升 **`0.01611`**，比此前最佳 E013 提升 `0.00879`。训练 loss 在后
两轮继续下降，但 Dev 未超过第一轮，说明 Large 收敛更快并出现早期过拟合迹象，
不直接扩展到 15 epoch。

## E015：100K HuBERT Large + Temporal Adapter

配置与 E014 相同，只将冻结底模替换为 HuBERT Large LL60K。冻结参数
315,438,720，可训练参数 410,433，峰值显存 `6.83GB`。

| Epoch | Seen | Unseen | Mean |
|---:|---:|---:|---:|
| 1 | 0.8596 | 0.8474 | 0.8535 |
| 2 | 0.8729 | 0.8585 | 0.8657 |
| 3 | 0.8862 | 0.8722 | 0.8792 |
| 4 | 0.8830 | 0.8741 | 0.8786 |
| 5 | 0.8853 | 0.8786 | 0.8820 |
| 6 | 0.8837 | 0.8734 | 0.8785 |
| 7 | 0.8894 | 0.8802 | 0.8848 |
| 8 | 0.8846 | 0.8740 | 0.8793 |
| 9 | 0.8884 | 0.8767 | 0.8825 |
| 10 | 0.8930 | 0.8803 | **0.8866** |
| 11 | 0.8872 | 0.8754 | 0.8813 |
| 12 | 0.8901 | 0.8760 | 0.8831 |
| 13 | 0.8956 | 0.8773 | 0.8864 |
| 14 | 0.8905 | 0.8785 | 0.8845 |
| 15 | 0.8931 | 0.8764 | 0.8848 |

从 epoch 3 的 `.last.pt` 保留 optimizer/scaler 续训，日志确认从 epoch 4 开始，
不是重新训练。最佳 checkpoint 来自 epoch 10，Dev Mean 比 WavLM Large E014
高 `0.0019`；epoch 13 接近但低 `0.0002`。训练到 epoch 15 后未继续改善，当前
配置已经进入平台期。使用保存最佳模型的 `.pt` 进行线上验证，得到 **`0.89071`**，
比 WavLM Large E014 提升 `0.00516`，并成为当前线上最佳；不使用保存 epoch 15
的 `.last.pt`。

## E016：100K WavLM Large 从零重跑 10 epoch

配置与 E014 相同，但使用新 checkpoint 从零训练到 10 epoch，不是从 E014
续训。冻结参数 315,453,120，可训练参数 410,433，峰值显存 `6.83GB`。

| Epoch | Seen | Unseen | Mean |
|---:|---:|---:|---:|
| 1 | 0.8854 | 0.8787 | 0.8821 |
| 2 | 0.8889 | 0.8844 | 0.8866 |
| 3 | 0.8864 | 0.8792 | 0.8828 |
| 4 | 0.8862 | 0.8807 | 0.8834 |
| 5 | 0.8872 | 0.8827 | 0.8850 |
| 6 | 0.8878 | 0.8790 | 0.8834 |
| 7 | 0.8926 | 0.8804 | 0.8865 |
| 8 | 0.8889 | 0.8811 | 0.8850 |
| 9 | 0.8925 | 0.8820 | 0.8873 |
| 10 | 0.8900 | 0.8860 | **0.8880** |

最佳 checkpoint 来自 epoch 10，Dev Mean 比 E014 高 `0.0033`，比 HuBERT
E015 高 `0.0014`。虽然训练入口固定全局 seed=42，但当前噪声增强按进程 PID
派生随机数，不同运行的增强序列并不完全可复现，因此 E016 应视为同配置的另一条
随机轨迹，而不是 E014 的严格确定性复现。先线上验证最佳 `.pt`；后续需要将增强
seed 改为显式、与 PID 无关的可控种子。线上得分 **`0.89054`**，比 E014 提升
`0.00499`，仅比 HuBERT E015 低 `0.00017`。

## E018：Large CTC 零重训打分诊断

直接复用 E015/E016 checkpoint，在 10,000 条有标签 dev pair 上比较目标文本
CTC 分数、greedy 路径、目标与 greedy 的似然差及音素编辑相似度。

| 模型 | 特征 | Seen | Unseen | Mean/All |
|---|---|---:|---:|---:|
| HuBERT Large | `target_score` | 0.8930 | 0.8803 | **0.8866** |
| HuBERT Large | `likelihood_margin` | 0.7634 | 0.7438 | 0.7537 |
| HuBERT Large | `edit_similarity` | 0.7477 | 0.7331 | 0.7404 |
| WavLM Large | `target_score` | 0.8900 | 0.8860 | **0.8877** |
| WavLM Large | `likelihood_margin` | 0.7739 | 0.7496 | 0.7616 |
| WavLM Large | `edit_similarity` | 0.7549 | 0.7448 | 0.7493 |

两模型的 greedy score、帧置信度、blank 比例和解码长度 AUC 都接近 `0.5`。
结论：不能用 greedy、似然差或编辑相似度替换现有目标 CTC 分数；后续改为在训练期
加入近音负词和判别 margin，而不是继续调整零训练推理公式。

## E019：HuBERT Large 注册音频对齐头

冻结 HuBERT Large LL60K，训练双向 soft frame alignment 头；50,000 pair，
batch size 32，dev batch size 8，3 epoch，DEMAND 加噪概率 0.5。可训练参数
133,499，峰值显存 `4.02GB`。

| Epoch | Seen | Unseen | Mean |
|---:|---:|---:|---:|
| 1 | 0.5182 | 0.4985 | **0.5083** |
| 2 | 0.5182 | 0.4985 | 0.5083 |
| 3 | 0.5182 | 0.4985 | 0.5083 |

训练损失长期停留在约 `1.109`，三个 epoch 的 AUC 排序完全不变，符合加权 BCE
常数预测。实现测试已覆盖参数梯度、padding mask 和 enroll/query 对称性，因此当前
结论是无条件音频相似度特征缺乏判别信息。停止扩大数据和增加 epoch；只有在加入
音素条件或 hard negative 后才重新考虑注册音频分支。

## E020：HuBERT Large 音素 CTC + hard-negative margin

在 E015 的 HuBERT Large Temporal CTC 上加入音素近邻负词和
`softplus(margin + negative_score - true_score)`，配置为 hard-negative weight
`0.25`、margin `0.5`、100,000 utterance、batch size 128、10 epoch。训练前为
8,335 个训练词建立了可区分音素近邻，峰值显存 `6.83GB`。

| Epoch | Seen | Unseen | Mean |
|---:|---:|---:|---:|
| 1 | 0.8651 | 0.8493 | 0.8572 |
| 2 | 0.8829 | 0.8646 | 0.8738 |
| 3 | 0.8856 | 0.8748 | 0.8802 |
| 4 | 0.8905 | 0.8723 | 0.8814 |
| 5 | 0.8922 | 0.8766 | 0.8844 |
| 6 | 0.8904 | 0.8748 | 0.8826 |
| 7 | 0.8932 | 0.8791 | 0.8862 |
| 8 | 0.8960 | 0.8824 | **0.8892** |
| 9 | 0.8941 | 0.8801 | 0.8871 |
| 10 | 0.8958 | 0.8814 | 0.8886 |

相对原 HuBERT Large E015，Dev Mean 提升 `0.0026`；相对 WavLM Large E016，
提升 `0.0012`。hard-negative margin loss 从约 `0.96` 下降到 `0.43`，说明训练
确实在压低近音目标分数。下一步先测试它替换原 HuBERT 分支后的 WavLM/HuBERT 秩融合，
暂不提交单模型结果。

## 线上收益拆解

| 改动 | 对照 | 线上变化 | 判断 |
|---|---|---:|---|
| 帧级 CNN + DEMAND + 15 epoch | 官方 baseline | `+0.03431` | 多项基础优化的合计收益，无法拆开归因 |
| WavLM + 注册文本字符 CTC | 增强 CNN | **`+0.15125`** | 最大单步收益，来自强底模、文本先验和任务重构的组合 |
| 字符改为音素 CTC | 100K 字符 CTC | **`+0.02836`** | 最清晰的单因素大收益，证明音素监督更匹配任务 |
| 字符/音素秩融合 | 100K 音素 CTC | `+0.00263` | 有互补性，但属于千分位收益 |
| 全量数据 epoch 2 | 100K 音素 CTC | `+0.00396` | 全量数据有效，但边际收益远小于模型/目标变化 |
| 全量从零 epoch 8 | 100K 音素 CTC | `+0.00507` | 当前最佳，包含数据量、batch 和完整优化轨迹差异 |
| 全量从零 epoch 8 | 全量 epoch 2 | `+0.00111` | 更多训练仅有千分位提升 |
| 旧 checkpoint 兼容续训 epoch 4 | 全量 epoch 2 | `-0.00261` | dev 上升但线上下降，重建 optimizer 的续训不可靠 |
| Temporal Adapter，100K/3 epoch | 100K 线性音素 CTC | **`+0.03005`** | 时序建模带来明确且可迁移的架构收益 |
| Temporal Adapter，100K/15 epoch，batch 256 | 100K Temporal/3 epoch，batch 128 | `+0.00732` | 有额外收益，但 batch 和 epoch 同时变化，无法单独归因 |
| WavLM Large Temporal，100K/3 epoch | 同配置 WavLM Base+ Temporal | **`+0.01611`** | 严格单因素底模升级，Dev 和线上均确认有效 |
| HuBERT Large Temporal，100K/15 epoch | WavLM Large Temporal E014 | `+0.00516` | 不同预训练目标带来额外收益，当前单模型最佳 |
| WavLM Large Temporal 重跑，100K/10 epoch | WavLM Large Temporal E014 | `+0.00499` | 随机增强轨迹本身可带来可观差异，适合作为融合分支 |
| WavLM Large + HuBERT Large 秩融合 | 最佳单模型 HuBERT E015 | **`+0.01052`** | 不同底模错误具有显著互补性，是目前除架构升级外最大的近期收益 |

从官方 baseline `0.62547` 到当前最佳 `0.90123`，累计绝对提升为
**`0.27576`**。收益主次已经明确：任务重构/强底模、音素监督和 Temporal
Adapter 是三个主要台阶；WavLM/HuBERT 的跨底模融合也获得了 `+0.01052`，说明
错误互补性值得保留。继续增加当前模型的训练轮数、固定 seed 重跑和直接扩到全量
都不足以解释到 `0.98` 的差距，下一阶段转向判别目标和难负样本。

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
