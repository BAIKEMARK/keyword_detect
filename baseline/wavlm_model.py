from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def length_mask(lengths: torch.Tensor, frames: int) -> torch.Tensor:
    lengths = lengths.clamp(min=1, max=frames)
    steps = torch.arange(frames, device=lengths.device)
    return steps.unsqueeze(0) < lengths.unsqueeze(1)


class SymmetricFrameMatchHead(nn.Module):
    def __init__(self, hidden_size: int, num_hidden_states: int,
                 projection_dim: int = 128):
        super().__init__()
        self.num_hidden_states = num_hidden_states
        self.layer_logits = nn.Parameter(torch.zeros(num_hidden_states))
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def _combine_layers(self, hidden_states: Sequence[torch.Tensor]):
        if len(hidden_states) != self.num_hidden_states:
            raise ValueError(
                f"expected {self.num_hidden_states} hidden states, "
                f"got {len(hidden_states)}")
        weights = torch.softmax(self.layer_logits, dim=0)
        combined = hidden_states[0] * weights[0]
        for weight, hidden in zip(weights[1:], hidden_states[1:]):
            combined = combined + hidden * weight
        return combined

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor):
        expanded_mask = mask
        while expanded_mask.ndim < values.ndim:
            expanded_mask = expanded_mask.unsqueeze(-1)
        values = values.masked_fill(~expanded_mask, 0.0)
        return values.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

    def forward(self, hidden_states: Sequence[torch.Tensor],
                enroll_lengths: torch.Tensor, query_lengths: torch.Tensor):
        combined = self._combine_layers(hidden_states)
        projected = F.normalize(self.projection(combined), dim=-1)
        batch_size = enroll_lengths.shape[0]
        if projected.shape[0] != batch_size * 2:
            raise ValueError("hidden-state batch must contain enroll then query")
        enroll, query = projected[:batch_size], projected[batch_size:]

        e_mask = length_mask(enroll_lengths, enroll.shape[1])
        q_mask = length_mask(query_lengths, query.shape[1])
        similarity = torch.bmm(enroll, query.transpose(1, 2))
        fill = torch.finfo(similarity.dtype).min

        best_e = similarity.masked_fill(
            ~q_mask[:, None, :], fill).max(dim=2).values
        best_q = similarity.masked_fill(
            ~e_mask[:, :, None], fill).max(dim=1).values
        score_e = self._masked_mean(best_e.unsqueeze(-1), e_mask).squeeze(-1)
        score_q = self._masked_mean(best_q.unsqueeze(-1), q_mask).squeeze(-1)

        enroll_global = F.normalize(self._masked_mean(enroll, e_mask), dim=-1)
        query_global = F.normalize(self._masked_mean(query, q_mask), dim=-1)
        global_cosine = (enroll_global * query_global).sum(dim=-1)

        features = torch.stack([
            0.5 * (score_e + score_q),
            torch.abs(score_e - score_q),
            global_cosine,
        ], dim=-1)
        return self.classifier(features).squeeze(-1)


class FrozenWavLMMatcher(nn.Module):
    def __init__(self, model_id: str = "microsoft/wavlm-base-plus",
                 projection_dim: int = 128):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required: pip install 'transformers>=4.40,<5'") \
                from exc

        self.model_id = model_id
        try:
            self.backbone = AutoModel.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load frozen WavLM backbone: {model_id}") from exc

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

        config = self.backbone.config
        self.head = SymmetricFrameMatchHead(
            hidden_size=config.hidden_size,
            num_hidden_states=config.num_hidden_layers + 1,
            projection_dim=projection_dim,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, enroll: torch.Tensor, query: torch.Tensor,
                enroll_lengths: torch.Tensor, query_lengths: torch.Tensor):
        waveforms = torch.cat([enroll, query], dim=0)
        sample_lengths = torch.cat([enroll_lengths, query_lengths], dim=0)
        attention_mask = length_mask(sample_lengths, waveforms.shape[1]).long()

        with torch.no_grad():
            output = self.backbone(
                waveforms,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        if output.hidden_states is None:
            raise RuntimeError("WavLM did not return hidden states")

        feature_lengths = self.backbone._get_feat_extract_output_lengths(
            sample_lengths).long()
        batch_size = enroll.shape[0]
        return self.head(
            output.hidden_states,
            feature_lengths[:batch_size],
            feature_lengths[batch_size:],
        )

    def head_state_dict(self):
        return {
            key: value.detach().cpu()
            for key, value in self.head.state_dict().items()
        }

    def load_head_state_dict(self, state_dict):
        self.head.load_state_dict(state_dict, strict=True)
