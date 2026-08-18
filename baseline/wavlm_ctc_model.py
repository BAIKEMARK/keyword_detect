from __future__ import annotations

import inspect
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavlm_model import length_mask


BACKBONE_TYPES = ("auto", "raw", "w2v-bert", "whisper", "parakeet")


def _find_nemo_checkpoint(model_id: str) -> str | None:
    path = Path(model_id)
    if path.is_file() and path.suffix == ".nemo":
        return str(path)
    if path.is_dir():
        checkpoints = sorted(path.rglob("*.nemo"))
        if checkpoints:
            return str(checkpoints[0])
    return None


def resolve_backbone_type(model_id: str, backbone_type: str = "auto") -> str:
    if backbone_type not in BACKBONE_TYPES:
        raise ValueError(f"unsupported backbone type: {backbone_type!r}")
    if backbone_type != "auto":
        return backbone_type
    if _find_nemo_checkpoint(model_id) is not None:
        return "parakeet"
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id)
        model_type = str(getattr(config, "model_type", "")).lower()
    except (ImportError, AttributeError, OSError, ValueError):
        return "raw"
    if model_type == "whisper":
        return "whisper"
    if model_type in {"wav2vec2-bert", "wav2vec2_bert"}:
        return "w2v-bert"
    return "raw"


def checkpoint_backbone_type(checkpoint: Mapping) -> str:
    config = checkpoint.get("training_config", {})
    return config.get(
        "backbone_type", checkpoint.get("backbone_type", "auto"))


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return torch.device("cpu")
    return parameter.device


def _scaled_lengths(lengths: torch.Tensor, source_width: int,
                    target_width: int) -> torch.Tensor:
    if source_width <= 0 or target_width <= 0:
        raise ValueError("feature widths must be positive")
    lengths = lengths.to(dtype=torch.long)
    return torch.div(
        lengths * target_width + source_width - 1,
        source_width,
        rounding_mode="floor",
    ).clamp(min=1, max=target_width)


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


def checkpoint_backbone_tune_config(checkpoint: Mapping):
    config = checkpoint.get("training_config", {})
    return {
        "unfreeze_layers": int(config.get(
            "unfreeze_layers", checkpoint.get("unfreeze_layers", 0))),
    }


def checkpoint_model_config(checkpoint: Mapping):
    return {
        **checkpoint_head_config(checkpoint),
        **checkpoint_backbone_tune_config(checkpoint),
    }


def load_ctc_checkpoint_state(model, checkpoint: Mapping):
    model.load_head_state_dict(checkpoint["head"])
    model.load_trainable_backbone_state_dict(
        checkpoint.get("backbone_trainable"))


class FrozenWavLMCTC(nn.Module):
    def __init__(self, vocab_size: int,
                 model_id: str = "microsoft/wavlm-base-plus",
                 dropout: float = 0.1, head_type: str = "linear",
                 adapter_dim: int = 256, adapter_layers: int = 2,
                 backbone_type: str = "auto", unfreeze_layers: int = 0):
        super().__init__()
        if unfreeze_layers < 0:
            raise ValueError("unfreeze_layers must be non-negative")
        self.model_id = model_id
        self.backbone_type = resolve_backbone_type(model_id, backbone_type)
        self.unfreeze_layers = int(unfreeze_layers)
        self._unfrozen_backbone_modules = []
        self.feature_extractor = None
        self.preprocessor = None
        self._parakeet_hidden_size = None
        self._load_backbone(model_id)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if self.preprocessor is not None:
            for parameter in self.preprocessor.parameters():
                parameter.requires_grad = False
        self.backbone.eval()
        if self.preprocessor is not None:
            self.preprocessor.eval()
        if self.unfreeze_layers:
            layers = self._backbone_layers()
            if self.unfreeze_layers > len(layers):
                raise ValueError(
                    f"unfreeze_layers={self.unfreeze_layers} exceeds "
                    f"backbone layers={len(layers)}")
            self._unfrozen_backbone_modules = list(
                layers[-self.unfreeze_layers:])
            for layer in self._unfrozen_backbone_modules:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        hidden_size, num_hidden_states = self._backbone_dimensions()
        self.head_type = head_type
        if head_type == "linear":
            self.head = CharacterCTCHead(
                hidden_size=hidden_size,
                num_hidden_states=num_hidden_states,
                vocab_size=vocab_size,
                dropout=dropout,
            )
        elif head_type == "temporal":
            self.head = TemporalCTCHead(
                hidden_size=hidden_size,
                num_hidden_states=num_hidden_states,
                vocab_size=vocab_size,
                adapter_dim=adapter_dim,
                adapter_layers=adapter_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"unsupported CTC head type: {head_type!r}")

    def _load_backbone(self, model_id: str):
        if self.backbone_type == "parakeet":
            self._load_parakeet(model_id)
            return
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required: pip install 'transformers>=4.40,<5'") \
                from exc
        try:
            if self.backbone_type == "whisper":
                from transformers import AutoFeatureExtractor, WhisperModel

                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    model_id)
                whisper = WhisperModel.from_pretrained(model_id)
                self.backbone = whisper.encoder
                del whisper
            else:
                self.backbone = AutoModel.from_pretrained(model_id)
                if self.backbone_type == "w2v-bert":
                    from transformers import AutoFeatureExtractor

                    self.feature_extractor = \
                        AutoFeatureExtractor.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load frozen {self.backbone_type} backbone: "
                f"{model_id}") from exc

    def _backbone_layers(self):
        candidates = (
            ("encoder", "layers"),
            ("encoder", "layer"),
            ("layers",),
            ("transformer", "layers"),
            ("conformer", "layers"),
        )
        for path in candidates:
            module = self.backbone
            for name in path:
                module = getattr(module, name, None)
                if module is None:
                    break
            if module is not None and isinstance(
                    module, (nn.ModuleList, list, tuple)):
                return module
        raise RuntimeError(
            f"cannot locate transformer layers for {self.backbone_type} "
            "backbone; use unfreeze_layers=0")

    def trainable_backbone_state_dict(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self.backbone.named_parameters()
            if value.requires_grad
        }

    def load_trainable_backbone_state_dict(self, state_dict):
        expected = {
            key for key, value in self.backbone.named_parameters()
            if value.requires_grad
        }
        if not expected:
            if state_dict:
                raise ValueError(
                    "checkpoint contains trainable backbone parameters but "
                    "current model has no unfrozen layers")
            return
        if state_dict is None:
            raise ValueError(
                "checkpoint is missing trainable backbone parameters")
        actual = set(state_dict)
        if actual != expected:
            raise ValueError(
                "trainable backbone checkpoint parameters do not match "
                f"current model (missing={sorted(expected - actual)[:3]}, "
                f"unexpected={sorted(actual - expected)[:3]})")
        parameters = dict(self.backbone.named_parameters())
        with torch.no_grad():
            for key in expected:
                parameters[key].copy_(state_dict[key].to(
                    device=parameters[key].device,
                    dtype=parameters[key].dtype))

    def _backbone_context(self):
        if self.training and self.unfreeze_layers:
            return torch.enable_grad()
        return torch.no_grad()

    def _load_parakeet(self, model_id: str):
        checkpoint = _find_nemo_checkpoint(model_id)
        if checkpoint is None:
            raise FileNotFoundError(
                f"no .nemo checkpoint found under: {model_id}")
        try:
            from nemo.collections.asr.models import ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "Parakeet requires NeMo ASR; run: "
                "bash scripts/install_parakeet_runtime.sh") from exc
        try:
            model = ASRModel.restore_from(
                restore_path=checkpoint, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"failed to restore Parakeet checkpoint: {checkpoint}") from exc
        self.preprocessor = model.preprocessor
        self.backbone = model.encoder
        encoder_config = getattr(getattr(model, "cfg", None), "encoder", None)
        candidates = (
            getattr(encoder_config, "d_model", None),
            getattr(self.backbone, "d_model", None),
            getattr(self.backbone, "_feat_out", None),
            getattr(self.backbone, "feat_out", None),
        )
        self._parakeet_hidden_size = next(
            (int(value) for value in candidates if value is not None), None)
        if self._parakeet_hidden_size is None:
            raise RuntimeError("cannot determine Parakeet encoder hidden size")
        del model

    def _backbone_dimensions(self):
        if self.backbone_type == "parakeet":
            return self._parakeet_hidden_size, 1
        config = self.backbone.config
        if self.backbone_type == "whisper":
            hidden_size = int(config.d_model)
            layers = int(getattr(
                config, "encoder_layers",
                getattr(config, "num_hidden_layers", 0)))
        else:
            hidden_size = int(config.hidden_size)
            layers = int(config.num_hidden_layers)
        if layers <= 0:
            raise RuntimeError("cannot determine backbone layer count")
        return hidden_size, layers + 1

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        for layer in self._unfrozen_backbone_modules:
            layer.train(mode)
        if self.preprocessor is not None:
            self.preprocessor.eval()
        return self

    def _forward_raw(self, waveforms: torch.Tensor,
                     sample_lengths: torch.Tensor):
        device = _module_device(self.backbone)
        waveforms = waveforms.to(device, non_blocking=True)
        sample_lengths = sample_lengths.to(device, non_blocking=True)
        attention_mask = length_mask(
            sample_lengths, waveforms.shape[1]).long()
        with self._backbone_context():
            output = self.backbone(
                waveforms,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        if output.hidden_states is None:
            raise RuntimeError("raw speech backbone did not return hidden states")
        output_lengths = self.backbone._get_feat_extract_output_lengths(
            sample_lengths).long().clamp(max=output.hidden_states[0].shape[1])
        return output.hidden_states, output_lengths

    def _extract_transformer_features(self, waveforms, sample_lengths):
        lengths = [int(length) for length in sample_lengths.detach().cpu()]
        cpu_waveforms = waveforms.detach().float().cpu()
        audio = [
            cpu_waveforms[index, :length].numpy()
            for index, length in enumerate(lengths)
        ]
        kwargs = {
            "sampling_rate": 16000,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }
        if self.backbone_type == "whisper":
            kwargs.update({
                "padding": "max_length",
                "max_length": max(lengths),
                "truncation": True,
            })
        else:
            kwargs.update({"padding": True, "truncation": False})
        features = self.feature_extractor(audio, **kwargs)
        input_features = features["input_features"]
        attention_mask = features.get("attention_mask")
        feature_width = (
            input_features.shape[-1]
            if self.backbone_type == "whisper"
            else input_features.shape[1]
        )
        if attention_mask is None:
            feature_lengths = _scaled_lengths(
                torch.tensor(lengths), max(lengths), feature_width)
        else:
            mask_lengths = attention_mask.long().sum(dim=-1)
            feature_lengths = _scaled_lengths(
                mask_lengths, attention_mask.shape[-1], feature_width)
        device = _module_device(self.backbone)
        return (
            input_features.to(device, non_blocking=True),
            feature_lengths.to(device, non_blocking=True),
        )

    def _forward_w2v_bert(self, waveforms, sample_lengths):
        input_features, feature_lengths = self._extract_transformer_features(
            waveforms, sample_lengths)
        attention_mask = length_mask(
            feature_lengths, input_features.shape[1]).long()
        with self._backbone_context():
            output = self.backbone(
                input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        if output.hidden_states is None:
            raise RuntimeError("W2V-BERT did not return hidden states")
        hidden_width = output.hidden_states[0].shape[1]
        output_lengths = _scaled_lengths(
            feature_lengths, input_features.shape[1], hidden_width)
        return output.hidden_states, output_lengths

    @staticmethod
    def _call_whisper_layer(layer, hidden_states, attention_mask):
        parameters = inspect.signature(layer.forward).parameters
        kwargs = {}
        if "attention_mask" in parameters:
            kwargs["attention_mask"] = attention_mask
        if "layer_head_mask" in parameters:
            kwargs["layer_head_mask"] = None
        if "output_attentions" in parameters:
            kwargs["output_attentions"] = False
        output = layer(hidden_states, **kwargs)
        return output[0] if isinstance(output, (tuple, list)) else output

    def _forward_whisper(self, waveforms, sample_lengths):
        input_features, feature_lengths = self._extract_transformer_features(
            waveforms, sample_lengths)
        with self._backbone_context():
            hidden = F.gelu(self.backbone.conv1(input_features))
            hidden = F.gelu(self.backbone.conv2(hidden)).permute(0, 2, 1)
            output_lengths = torch.div(
                feature_lengths + 1, 2, rounding_mode="floor").clamp(
                    max=hidden.shape[1])
            if hidden.shape[1] > self.backbone.embed_positions.weight.shape[0]:
                raise ValueError("Whisper input exceeds positional embedding size")
            positions = self.backbone.embed_positions.weight[
                :hidden.shape[1]].to(dtype=hidden.dtype)
            hidden = hidden + positions
            hidden = F.dropout(
                hidden, p=float(getattr(self.backbone.config, "dropout", 0.0)),
                training=False)
            valid = length_mask(output_lengths, hidden.shape[1])
            attention_mask = hidden.new_zeros(
                (hidden.shape[0], 1, hidden.shape[1], hidden.shape[1]))
            attention_mask.masked_fill_(
                ~valid[:, None, None, :], torch.finfo(hidden.dtype).min)
            hidden_states = [hidden]
            for layer in self.backbone.layers:
                hidden = self._call_whisper_layer(
                    layer, hidden, attention_mask)
                hidden_states.append(hidden)
            hidden = self.backbone.layer_norm(hidden)
            hidden_states[-1] = hidden
        return tuple(hidden_states), output_lengths

    def _forward_parakeet(self, waveforms, sample_lengths):
        device = _module_device(self.backbone)
        waveforms = waveforms.to(device, non_blocking=True)
        sample_lengths = sample_lengths.to(device, non_blocking=True)
        with self._backbone_context():
            processed, processed_lengths = self.preprocessor(
                input_signal=waveforms, length=sample_lengths)
            output = self.backbone(
                audio_signal=processed, length=processed_lengths)
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError("Parakeet encoder returned an unexpected output")
        hidden, output_lengths = output[:2]
        if hidden.ndim != 3:
            raise RuntimeError("Parakeet encoder output must be three-dimensional")
        if hidden.shape[1] == self._parakeet_hidden_size:
            hidden = hidden.transpose(1, 2)
        elif hidden.shape[2] != self._parakeet_hidden_size:
            raise RuntimeError("Parakeet encoder hidden dimension is unexpected")
        return (hidden,), output_lengths.long()

    def forward(self, waveforms: torch.Tensor,
                sample_lengths: torch.Tensor):
        if self.backbone_type == "raw":
            hidden_states, output_lengths = self._forward_raw(
                waveforms, sample_lengths)
        elif self.backbone_type == "w2v-bert":
            hidden_states, output_lengths = self._forward_w2v_bert(
                waveforms, sample_lengths)
        elif self.backbone_type == "whisper":
            hidden_states, output_lengths = self._forward_whisper(
                waveforms, sample_lengths)
        elif self.backbone_type == "parakeet":
            hidden_states, output_lengths = self._forward_parakeet(
                waveforms, sample_lengths)
        else:  # pragma: no cover - guarded by resolve_backbone_type
            raise RuntimeError(f"unsupported backbone: {self.backbone_type}")
        logits = self.head(hidden_states, output_lengths)
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
