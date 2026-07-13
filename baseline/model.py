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

    def forward_frames(self, x, lengths=None):
        if lengths is not None:
            steps = torch.arange(x.shape[-1], device=x.device)
            valid = steps.unsqueeze(0) < lengths.unsqueeze(1)
            x = x.masked_fill(~valid[:, None, None, :], 0.0)
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

    def forward(self, enroll, query, enroll_lengths=None, query_lengths=None):
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

    @staticmethod
    def _pooled_lengths(lengths, batch_size, output_frames, device):
        if lengths is None:
            return torch.full(
                (batch_size,), output_frames, dtype=torch.long, device=device)
        lengths = lengths.to(device=device)
        lengths = torch.div(lengths, 2, rounding_mode="floor")
        lengths = torch.div(lengths, 2, rounding_mode="floor")
        return lengths.clamp(min=1, max=output_frames)

    @staticmethod
    def _mask(lengths, frames):
        steps = torch.arange(frames, device=lengths.device)
        return steps.unsqueeze(0) < lengths.unsqueeze(1)

    @staticmethod
    def _masked_mean(values, mask):
        values = values.masked_fill(~mask, 0.0)
        return values.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def forward(self, enroll, query, enroll_lengths=None, query_lengths=None):
        e = self.encoder.forward_frames(enroll, enroll_lengths)
        q = self.encoder.forward_frames(query, query_lengths)
        e_lengths = self._pooled_lengths(
            enroll_lengths, enroll.shape[0], e.shape[1], e.device)
        q_lengths = self._pooled_lengths(
            query_lengths, query.shape[0], q.shape[1], q.device)
        e_mask = self._mask(e_lengths, e.shape[1])
        q_mask = self._mask(q_lengths, q.shape[1])

        sim = torch.bmm(e, q.transpose(1, 2))
        fill = torch.finfo(sim.dtype).min
        best_e = sim.masked_fill(~q_mask[:, None, :], fill).max(dim=2).values
        best_q = sim.masked_fill(~e_mask[:, :, None], fill).max(dim=1).values
        score_e = self._masked_mean(best_e, e_mask)
        score_q = self._masked_mean(best_q, q_mask)
        score = 0.5 * (score_e + score_q)
        return self.scale * score + self.bias


def build_model(model_name: str, n_mels: int, embed_dim: int = 64) -> nn.Module:
    if model_name == "global":
        return SiameseKWS(n_mels, embed_dim)
    if model_name == "frame_maxmean":
        return FrameMaxMeanKWS(n_mels, embed_dim)
    raise ValueError(f"unknown model: {model_name}")
