---
license: Apache License 2.0
---
数据集文件元信息以及数据文件，请浏览“数据集文件”页面获取。

当前数据集卡片使用的是默认模版，数据集的贡献者未提供更加详细的数据集介绍，但是您可以通过如下GIT Clone命令，或者ModelScope SDK来下载数据集

#### 下载方法 
:modelscope-code[]{type="sdk"}
:modelscope-code[]{type="git"}

## 运行增强版 baseline

本仓库默认训练配置已切到增强版 baseline：

- 15 epoch
- `frame_maxmean` 帧级匹配
- `pos_weight=4.0`
- 训练期按 -10 至 5 dB 混噪
- 如果提供 `--noise-dir`，使用真实噪声；否则退回高斯噪声

本地 5 万子集训练：

```bash
python baseline/train.py \
  --subset 50000 \
  --model frame_maxmean \
  --noise-prob 0.5 \
  --noise-dir noise/DEMAND_16k/wav \
  --out baseline/checkpoints/frame_demand_50k.pt
```

生成提交文件：

```bash
python baseline/infer.py \
  --ckpt baseline/checkpoints/frame_demand_50k.pt \
  --out submission_frame_demand_50k.csv
```

## DEMAND 噪声数据

`noise/` 目录被 `.gitignore` 忽略，推送到 GitHub 后不会自动包含 DEMAND 噪声文件。新机器拉仓库后，需要单独下载：

```bash
bash scripts/download_demand_16k.sh
```

脚本会下载并解压 9 个 16 kHz DEMAND 场景到：

```text
noise/DEMAND_16k/wav
```

包含场景：

```text
TCAR, TBUS, TMETRO, STRAFFIC, PCAFETER, OMEETING, OOFFICE, DKITCHEN, DLIVING
```

如果服务器需要代理，可以先设置环境变量：

```bash
export HTTPS_PROXY=socks5h://127.0.0.1:10808
bash scripts/download_demand_16k.sh
```

DEMAND 来源：<https://zenodo.org/records/1227121>

## 冻结 WavLM Base+

WavLM 只作为冻结特征提取器，最终 posterior 由训练得到的层加权、投影和
对称帧级匹配头输出。首次运行会下载 `microsoft/wavlm-base-plus`。

```bash
python3 -m pip install "transformers>=4.40,<5"
```

先在服务器做一轮小规模 smoke test：

```bash
python baseline/train_wavlm.py \
  --subset 256 \
  --epochs 1 \
  --bs 8 \
  --device cuda \
  --noise-dir noise/DEMAND_16k/wav \
  --out baseline/checkpoints/wavlm_smoke.pt
```

确认日志显示 `real noise files: 144`，并完成 seen/unseen 评估后，运行 50K
首轮实验：

```bash
python baseline/train_wavlm.py \
  --subset 50000 \
  --epochs 3 \
  --bs 16 \
  --workers 8 \
  --device cuda \
  --noise-dir noise/DEMAND_16k/wav \
  --out baseline/checkpoints/wavlm_base_plus_50k.pt
```

生成提交文件：

```bash
python baseline/infer_wavlm.py \
  --ckpt baseline/checkpoints/wavlm_base_plus_50k.pt \
  --device cuda \
  --out submission_wavlm_base_plus_50k.csv
```

### 全量50万 pair 音频匹配

`train_wavlm.py` 的默认路径仍指向 5 万子集。全量实验必须显式传入官方训练
CSV 和 wav ZIP；这里的 `--subset 500000` 表示 50 万个 enroll/query pair，
不会像 CTC 那样展开为 100 万条独立音频：

```bash
python3 -u baseline/train_wavlm.py \
  --model-id /mnt/workspace/models/wavlm-base-plus \
  --train-csv train/train_label.csv \
  --train-zip train/wav.zip \
  --subset 500000 \
  --epochs 3 \
  --bs 128 \
  --workers 8 \
  --device cuda \
  --pos-weight 4.0 \
  --noise-prob 0.5 \
  --noise-dir noise/DEMAND_16k/wav \
  --out baseline/checkpoints/wavlm_matcher_full_e3.pt
```

`--out` 保存 Dev Mean 最佳模型，默认派生出的
`wavlm_matcher_full_e3.last.pt` 每轮保存最新完整训练状态。中断后把目标总轮数
设大，并增加：

```bash
--resume baseline/checkpoints/wavlm_matcher_full_e3.last.pt
```

checkpoint 会校验模型、投影维度、数据路径、pair 数、batch size、学习率、
`pos_weight` 和噪声参数，防止恢复到不兼容的实验。

## 字符 CTC：使用注册文本处理 unseen

字符 CTC 使用训练集提供的音频文本监督，推理时只读取测试提供的
`enroll_txt` 和 query 音频，不读取未知的 query 文本。先运行 smoke test：

```bash
python baseline/train_wavlm_ctc.py \
  --subset 256 \
  --epochs 1 \
  --bs 64 \
  --workers 8 \
  --device cuda \
  --noise-dir noise/DEMAND_16k/wav \
  --out baseline/checkpoints/wavlm_char_ctc_smoke.pt
```

smoke test 完成 seen/unseen AUC 后，运行完整 10 万条训练音频实验：

```bash
python baseline/train_wavlm_ctc.py \
  --subset 100000 \
  --epochs 10 \
  --bs 128 \
  --workers 8 \
  --device cuda \
  --noise-dir noise/DEMAND_16k/wav \
  --out baseline/checkpoints/wavlm_char_ctc_100k.pt
```

生成 CTC-only 提交文件：

```bash
python baseline/infer_wavlm_ctc.py \
  --ckpt baseline/checkpoints/wavlm_char_ctc_100k.pt \
  --device cuda \
  --out submission_wavlm_char_ctc.csv
```

该字符 CTC 配置在 5 万 pair（展开为 10 万条音频文本）上取得开发集
mean AUC 0.8111，线上 AUC 0.81103。

## 音素 CTC：补充发音先验

音素模式使用 `g2p_en` 将英文注册文本转换为固定的 39 类 ARPAbet
音素。CMUdict 用于常见词，`g2p_en` 的 OOV 回退用于未登录词；G2P
输出只作为自研 CTC 头的训练目标和输入特征，不直接参与唤醒判决。

安装依赖并准备 NLTK 资源：

```bash
python3 -m pip install "numpy>=1.24,<2" "g2p-en==2.1.0"
mkdir -p /mnt/workspace/nltk_data
python3 -m nltk.downloader -d /mnt/workspace/nltk_data \
  cmudict \
  averaged_perceptron_tagger averaged_perceptron_tagger_eng
export NLTK_DATA=/mnt/workspace/nltk_data
```

`/mnt/workspace/nltk_data` 是 DSW 持久化目录，新实例继续设置同一个
`NLTK_DATA` 即可复用。外部发音资源来源：
[g2p_en](https://github.com/Kyubyong/g2p) 和
[CMUdict](https://github.com/cmusphinx/cmudict)。

`g2p_en 2.1.0` 与 NumPy 2.x 存在 OOV 推理数值溢出，因此音素模式固定
使用 NumPy 1.x；字符 CTC 不受该问题影响。

先在 A10 上运行 256 条 smoke test：

```bash
python3 baseline/train_wavlm_ctc.py \
  --model-id /mnt/workspace/models/wavlm-base-plus \
  --units phoneme \
  --subset 256 \
  --epochs 1 \
  --bs 64 \
  --workers 8 \
  --device cuda \
  --noise-dir noise/DEMAND_16k/wav \
  --out baseline/checkpoints/wavlm_phoneme_ctc_smoke.pt
```

smoke test 完成 seen/unseen 评估后，训练全部 10 万条音频：

```bash
python3 baseline/train_wavlm_ctc.py \
  --model-id /mnt/workspace/models/wavlm-base-plus \
  --units phoneme \
  --subset 100000 \
  --epochs 10 \
  --bs 128 \
  --workers 8 \
  --device cuda \
  --noise-dir noise/DEMAND_16k/wav \
  --out baseline/checkpoints/wavlm_phoneme_ctc_100k_e10.pt
```

生成音素 CTC-only 提交文件：

```bash
python3 baseline/infer_wavlm_ctc.py \
  --ckpt baseline/checkpoints/wavlm_phoneme_ctc_100k_e10.pt \
  --model-id /mnt/workspace/models/wavlm-base-plus \
  --bs 128 \
  --workers 8 \
  --device cuda \
  --out submission_wavlm_phoneme_ctc_100k.csv
```

## 字符与音素 CTC 秩融合

融合只需要额外导出两个模型在有标签 dev 上的逐样本分数。测试集直接复用
已经生成的字符和音素提交 CSV。先导出 dev 分数：

```bash
mkdir -p scores
export NLTK_DATA=/mnt/workspace/nltk_data

python3 baseline/export_wavlm_ctc_dev.py \
  --ckpt baseline/checkpoints/wavlm_char_ctc_100k_e10.pt \
  --model-id /mnt/workspace/models/wavlm-base-plus \
  --bs 128 \
  --workers 8 \
  --device cuda \
  --out scores/dev_wavlm_char_ctc_100k.csv

python3 baseline/export_wavlm_ctc_dev.py \
  --ckpt baseline/checkpoints/wavlm_phoneme_ctc_100k_e10.pt \
  --model-id /mnt/workspace/models/wavlm-base-plus \
  --bs 128 \
  --workers 8 \
  --device cuda \
  --out scores/dev_wavlm_phoneme_ctc_100k.csv
```

搜索全局音素权重并生成融合提交：

```bash
python3 baseline/fuse_ctc_scores.py \
  --char-dev scores/dev_wavlm_char_ctc_100k.csv \
  --phoneme-dev scores/dev_wavlm_phoneme_ctc_100k.csv \
  --char-eval submission_wavlm_char_ctc_100k.csv \
  --phoneme-eval submission_wavlm_phoneme_ctc_100k.csv \
  --out submission_wavlm_ctc_rank_fusion.csv
```

脚本分别在 seen/unseen 内对两个分支做平均秩归一化，在 dev 上以 0.001
步长搜索一个全局音素权重，然后将固定权重应用于 eval。实验参数和 dev
AUC 同时写入 `submission_wavlm_ctc_rank_fusion.csv.json`。

## 全量50万 pair 训练

官方全量包包含 `train/train_label.csv` 和嵌套的 `train/wav.zip`。只解压
外层 `train.zip`，不要继续解压16GB的 `wav.zip`：

```bash
unzip -n train.zip -d .

ls -lh train/train_label.csv train/wav.zip
python3 -c "import csv; print(sum(1 for _ in csv.DictReader(open('train/train_label.csv', encoding='utf-8'))))"
```

50万 pair 会展开为约100万条音频文本。先运行3个全量 epoch 的音素 CTC：

```bash
export NLTK_DATA=/mnt/workspace/nltk_data

python3 -u baseline/train_wavlm_ctc.py \
  --model-id /mnt/workspace/models/wavlm-base-plus \
  --units phoneme \
  --train-csv train/train_label.csv \
  --train-zip train/wav.zip \
  --epochs 3 \
  --bs 128 \
  --workers 8 \
  --device cuda \
  --noise-prob 0.5 \
  --noise-dir noise/DEMAND_16k/wav \
  --out baseline/checkpoints/wavlm_phoneme_ctc_full_e3.pt
```

省略 `--subset` 表示使用 CSV 中的全部音频；仍可显式传入该参数做 smoke
test。checkpoint 会记录实际训练 CSV、wav ZIP 和音频数量。根据前三轮 dev
走势决定是否继续到5轮，不默认直接训练10轮。

### 从 checkpoint 继续训练

`--epochs` 是目标总轮数，不是额外轮数。下面从已完成的全量 epoch 2 继续，
训练到 epoch 5：

```bash
export NLTK_DATA=/mnt/workspace/nltk_data

python3 -u baseline/train_wavlm_ctc.py \
  --model-id /mnt/workspace/models/wavlm-base-plus \
  --units phoneme \
  --train-csv train/train_label.csv \
  --train-zip train/wav.zip \
  --epochs 5 \
  --bs 128 \
  --workers 8 \
  --device cuda \
  --noise-prob 0.5 \
  --noise-dir noise/DEMAND_16k/wav \
  --resume baseline/checkpoints/wavlm_phoneme_ctc_full_e3.pt \
  --out baseline/checkpoints/wavlm_phoneme_ctc_full_e3.pt
```

`--out` 保存历史最佳 Dev Mean checkpoint；默认派生出的
`wavlm_phoneme_ctc_full_e3.last.pt` 每轮保存最新状态。旧 checkpoint 没有
optimizer 和 AMP scaler，首次恢复会以兼容模式重建这两项；之后服务器中断时，
改用 `--resume baseline/checkpoints/wavlm_phoneme_ctc_full_e3.last.pt` 即可完整恢复
最近一个已完成 epoch 的训练状态。
