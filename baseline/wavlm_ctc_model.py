from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavlm_model import length_mask


class CharacterCTCHead(nn.Module):
    def __init__(self, hidden_size: int, num_hidden_states: int,
                 vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.num_hidden_states = num_hidden_states
        self.layer_logits = nn.Parameter(torch.zeros(num_hidden_states))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: Sequence[torch.Tensor]):
        if len(hidden_states) != self.num_hidden_states:
            raise ValueError(
                f"expected {self.num_hidden_states} hidden states, "
                f"got {len(hidden_states)}")
        weights = torch.softmax(self.layer_logits, dim=0)
        combined = hidden_states[0] * weights[0]
        for weight, hidden in zip(weights[1:], hidden_states[1:]):
            combined = combined + hidden * weight
        return self.classifier(self.dropout(combined))


class FrozenWavLMCTC(nn.Module):
    def __init__(self, vocab_size: int,
                 model_id: str = "microsoft/wavlm-base-plus",
                 dropout: float = 0.1):
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
        self.head = CharacterCTCHead(
            hidden_size=config.hidden_size,
            num_hidden_states=config.num_hidden_layers + 1,
            vocab_size=vocab_size,
            dropout=dropout,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, waveforms: torch.Tensor,
                sample_lengths: torch.Tensor):
        attention_mask = length_mask(
            sample_lengths, waveforms.shape[1]).long()
        with torch.no_grad():
            output = self.backbone(
                waveforms,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        if output.hidden_states is None:
            raise RuntimeError("WavLM did not return hidden states")
        logits = self.head(output.hidden_states)
        output_lengths = self.backbone._get_feat_extract_output_lengths(
            sample_lengths).long().clamp(max=logits.shape[1])
        return logits, output_lengths

    def log_probs(self, waveforms: torch.Tensor,
                  sample_lengths: torch.Tensor):
        logits, output_lengths = self(waveforms, sample_lengths)
        return F.log_softmax(logits.float(), dim=-1), output_lengths

    def head_state_dict(self):
        return {
            key: value.detach().cpu()
            for key, value in self.head.state_dict().items()
        }

    def load_head_state_dict(self, state_dict):
        self.head.load_state_dict(state_dict, strict=True)
