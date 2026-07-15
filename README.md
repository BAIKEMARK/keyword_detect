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
