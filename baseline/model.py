"""孪生网络：共享 CNN 编码器 + 音频对匹配打分。

支持两种匹配头：
  global: 兼容原 baseline，全局池化后做余弦相似度。
  frame_maxmean: 保留 CNN 时间帧，用对称 max-mean 做软对齐。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """两层卷积编码器，支持全局和帧级 embedding。"""

    def __init__(self, n_mels: int, embed_dim: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, embed_dim)

    def forward_global(self, x):
        h = self.pool(self.cnn(x)).flatten(1)
        emb = self.fc(h)
        return F.normalize(emb, dim=-1)

    def forward_frames(self, x):
        h = self.cnn(x).mean(dim=2).transpose(1, 2)
        emb = self.fc(h)
        return F.normalize(emb, dim=-1)

    def forward(self, x):
        return self.forward_global(x)


class SiameseKWS(nn.Module):
    def __init__(self, n_mels: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(n_mels, embed_dim)
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, enroll, query):
        e = self.encoder.forward_global(enroll)
        q = self.encoder.forward_global(query)
        sim = (e * q).sum(dim=-1)       # 余弦相似度（已 L2 归一化）
        return self.scale * sim + self.bias


class FrameMaxMeanKWS(nn.Module):
    def __init__(self, n_mels: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(n_mels, embed_dim)
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, enroll, query):
        e = self.encoder.forward_frames(enroll)
        q = self.encoder.forward_frames(query)
        sim = torch.bmm(e, q.transpose(1, 2))
        score_e = sim.max(dim=2).values.mean(dim=1)
        score_q = sim.max(dim=1).values.mean(dim=1)
        score = 0.5 * (score_e + score_q)
        return self.scale * score + self.bias


def build_model(model_name: str, n_mels: int, embed_dim: int = 64) -> nn.Module:
    if model_name == "global":
        return SiameseKWS(n_mels, embed_dim)
    if model_name == "frame_maxmean":
        return FrameMaxMeanKWS(n_mels, embed_dim)
    raise ValueError(f"unknown model: {model_name}")
