from __future__ import annotations

from typing import Mapping, Sequence

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

    def forward(self, hidden_states: Sequence[torch.Tensor],
                output_lengths: torch.Tensor | None = None):
        if len(hidden_states) != self.num_hidden_states:
            raise ValueError(
                f"expected {self.num_hidden_states} hidden states, "
                f"got {len(hidden_states)}")
        weights = torch.softmax(self.layer_logits, dim=0)
        combined = hidden_states[0] * weights[0]
        for weight, hidden in zip(weights[1:], hidden_states[1:]):
            combined = combined + hidden * weight
        return self.classifier(self.dropout(combined))


class TemporalConvBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.depthwise = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=5, padding=2,
            groups=hidden_size)
        self.pointwise = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor):
        residual = hidden
        output = self.norm(hidden).masked_fill(~mask, 0.0)
        output = self.depthwise(output.transpose(1, 2))
        output = F.gelu(output)
        output = self.pointwise(output).transpose(1, 2)
        output = self.dropout(output)
        return (residual + output).masked_fill(~mask, 0.0)


class TemporalCTCHead(nn.Module):
    def __init__(self, hidden_size: int, num_hidden_states: int,
                 vocab_size: int, adapter_dim: int = 256,
                 adapter_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if adapter_dim <= 0 or adapter_layers <= 0:
            raise ValueError("adapter_dim and adapter_layers must be positive")
        self.num_hidden_states = num_hidden_states
        self.layer_logits = nn.Parameter(torch.zeros(num_hidden_states))
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, adapter_dim)
        self.blocks = nn.ModuleList([
            TemporalConvBlock(adapter_dim, dropout)
            for _ in range(adapter_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(adapter_dim, vocab_size)

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

    def forward(self, hidden_states: Sequence[torch.Tensor],
                output_lengths: torch.Tensor | None = None):
        combined = self._combine_layers(hidden_states)
        if output_lengths is None:
            output_lengths = torch.full(
                (combined.shape[0],), combined.shape[1],
                dtype=torch.long, device=combined.device)
        mask = length_mask(output_lengths, combined.shape[1]).unsqueeze(-1)
        output = self.input_projection(self.input_norm(combined))
        output = output.masked_fill(~mask, 0.0)
        for block in self.blocks:
            output = block(output, mask)
        return self.classifier(self.dropout(output)).masked_fill(~mask, 0.0)


def checkpoint_head_config(checkpoint: Mapping):
    config = checkpoint.get("training_config", {})
    return {
        "head_type": config.get(
            "head_type", checkpoint.get("head_type", "linear")),
        "adapter_dim": int(config.get(
            "adapter_dim", checkpoint.get("adapter_dim", 256))),
        "adapter_layers": int(config.get(
            "adapter_layers", checkpoint.get("adapter_layers", 2))),
    }


class FrozenWavLMCTC(nn.Module):
    def __init__(self, vocab_size: int,
                 model_id: str = "microsoft/wavlm-base-plus",
                 dropout: float = 0.1, head_type: str = "linear",
                 adapter_dim: int = 256, adapter_layers: int = 2):
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
        self.head_type = head_type
        if head_type == "linear":
            self.head = CharacterCTCHead(
                hidden_size=config.hidden_size,
                num_hidden_states=config.num_hidden_layers + 1,
                vocab_size=vocab_size,
                dropout=dropout,
            )
        elif head_type == "temporal":
            self.head = TemporalCTCHead(
                hidden_size=config.hidden_size,
                num_hidden_states=config.num_hidden_layers + 1,
                vocab_size=vocab_size,
                adapter_dim=adapter_dim,
                adapter_layers=adapter_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"unsupported CTC head type: {head_type!r}")

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
        output_lengths = self.backbone._get_feat_extract_output_lengths(
            sample_lengths).long().clamp(max=output.hidden_states[0].shape[1])
        logits = self.head(output.hidden_states, output_lengths)
        output_lengths = output_lengths.clamp(max=logits.shape[1])
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
