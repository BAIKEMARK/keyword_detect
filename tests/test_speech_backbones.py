from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))

from wavlm_ctc_model import (FrozenWavLMCTC, checkpoint_backbone_type,
                             load_ctc_checkpoint_state,
                             resolve_backbone_type)  # noqa: E402


class FakeFeatureExtractor:
    @classmethod
    def from_pretrained(cls, model_id):
        return cls()

    def __call__(self, audio, **kwargs):
        lengths = [len(waveform) for waveform in audio]
        width = (max(lengths) + 1) // 2
        feature_lengths = [(length + 1) // 2 for length in lengths]
        mask = torch.zeros(len(audio), width, dtype=torch.long)
        for index, length in enumerate(feature_lengths):
            mask[index, :length] = 1
        if kwargs.get("max_length") is not None:
            features = torch.zeros(len(audio), 3, width)
        else:
            features = torch.zeros(len(audio), width, 3)
        return {"input_features": features, "attention_mask": mask}


class SpeechBackboneTest(unittest.TestCase):
    def test_checkpoint_defaults_to_auto(self):
        self.assertEqual(checkpoint_backbone_type({}), "auto")
        self.assertEqual(checkpoint_backbone_type({
            "training_config": {"backbone_type": "whisper"},
        }), "whisper")

    def test_w2v_bert_feature_frontend(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.config = types.SimpleNamespace(
                    hidden_size=6, num_hidden_layers=1)

            def forward(self, input_features, attention_mask,
                        output_hidden_states, return_dict):
                base = input_features.mean(dim=-1, keepdim=True).repeat(
                    1, 1, 6)
                return types.SimpleNamespace(
                    hidden_states=(base, base + self.weight))

        backbone = Backbone()
        transformers = types.ModuleType("transformers")
        transformers.AutoConfig = types.SimpleNamespace(
            from_pretrained=lambda model_id: types.SimpleNamespace(
                model_type="wav2vec2-bert"))
        transformers.AutoModel = types.SimpleNamespace(
            from_pretrained=lambda model_id: backbone)
        transformers.AutoFeatureExtractor = FakeFeatureExtractor
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            model = FrozenWavLMCTC(
                5, "fake/w2v-bert", dropout=0.0, head_type="temporal",
                adapter_dim=4, adapter_layers=1)

        log_probs, lengths = model.log_probs(
            torch.randn(2, 12), torch.tensor([12, 8]))
        self.assertEqual(model.backbone_type, "w2v-bert")
        self.assertEqual(log_probs.shape, (2, 6, 5))
        torch.testing.assert_close(lengths, torch.tensor([6, 4]))

    def test_only_last_backbone_layer_is_trainable_and_checkpointed(self):
        class ScaleLayer(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(value))

            def forward(self, hidden):
                return hidden * self.scale

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([
                    ScaleLayer(1.1), ScaleLayer(1.2),
                ])

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.config = types.SimpleNamespace(
                    hidden_size=4, num_hidden_layers=2)

            def forward(self, waveforms, attention_mask,
                        output_hidden_states, return_dict):
                hidden = waveforms[:, ::2].unsqueeze(-1).repeat(1, 1, 4)
                hidden_states = [hidden]
                for layer in self.encoder.layers:
                    hidden = layer(hidden)
                    hidden_states.append(hidden)
                return types.SimpleNamespace(
                    hidden_states=tuple(hidden_states))

            def _get_feat_extract_output_lengths(self, lengths):
                return torch.div(lengths + 1, 2, rounding_mode="floor")

        transformers = types.ModuleType("transformers")
        transformers.AutoModel = types.SimpleNamespace(
            from_pretrained=lambda model_id: Backbone())
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            model = FrozenWavLMCTC(
                5, "fake/raw", dropout=0.0, backbone_type="raw",
                unfreeze_layers=1)

        first, last = model.backbone.encoder.layers
        self.assertFalse(first.scale.requires_grad)
        self.assertTrue(last.scale.requires_grad)
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertFalse(first.training)
        self.assertTrue(last.training)

        log_probs, _ = model.log_probs(
            torch.randn(2, 12), torch.tensor([12, 8]))
        (-log_probs[..., 1].mean()).backward()
        self.assertIsNone(first.scale.grad)
        self.assertIsNotNone(last.scale.grad)

        saved_backbone = model.trainable_backbone_state_dict()
        self.assertEqual(set(saved_backbone), {
            "encoder.layers.1.scale",
        })
        expected = saved_backbone["encoder.layers.1.scale"].clone()
        with torch.no_grad():
            last.scale.fill_(9.0)
        load_ctc_checkpoint_state(model, {
            "head": model.head_state_dict(),
            "backbone_trainable": saved_backbone,
        })
        torch.testing.assert_close(last.scale, expected)

        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            with self.assertRaisesRegex(ValueError, "exceeds backbone layers"):
                FrozenWavLMCTC(
                    5, "fake/raw", backbone_type="raw", unfreeze_layers=3)

    def test_whisper_variable_length_encoder(self):
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(0.1))

            def forward(self, hidden_states, attention_mask=None,
                        layer_head_mask=None, output_attentions=False):
                if attention_mask.shape[-1] != hidden_states.shape[1]:
                    raise AssertionError("Whisper mask width does not match")
                return (hidden_states + self.bias,)

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(
                    d_model=4, encoder_layers=1, dropout=0.0)
                self.conv1 = torch.nn.Conv1d(3, 4, 3, padding=1)
                self.conv2 = torch.nn.Conv1d(4, 4, 3, stride=2, padding=1)
                self.embed_positions = torch.nn.Embedding(20, 4)
                self.layers = torch.nn.ModuleList([Layer()])
                self.layer_norm = torch.nn.LayerNorm(4)

        encoder = Encoder()
        transformers = types.ModuleType("transformers")
        transformers.AutoConfig = types.SimpleNamespace(
            from_pretrained=lambda model_id: types.SimpleNamespace(
                model_type="whisper"))
        transformers.AutoModel = types.SimpleNamespace()
        transformers.AutoFeatureExtractor = FakeFeatureExtractor
        transformers.WhisperModel = types.SimpleNamespace(
            from_pretrained=lambda model_id: types.SimpleNamespace(
                encoder=encoder))
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            model = FrozenWavLMCTC(
                5, "fake/whisper", dropout=0.0, head_type="temporal",
                adapter_dim=4, adapter_layers=1)

        log_probs, lengths = model.log_probs(
            torch.randn(2, 12), torch.tensor([12, 8]))
        self.assertEqual(model.backbone_type, "whisper")
        self.assertEqual(log_probs.shape, (2, 3, 5))
        torch.testing.assert_close(lengths, torch.tensor([3, 2]))

    def test_parakeet_nemo_encoder(self):
        class Preprocessor(torch.nn.Module):
            def forward(self, input_signal, length):
                return input_signal.unsqueeze(1), length

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, audio_signal, length):
                hidden = audio_signal[:, :, ::2].repeat(1, 4, 1)
                output_lengths = torch.div(
                    length + 1, 2, rounding_mode="floor")
                return hidden * self.weight, output_lengths

        restored = types.SimpleNamespace(
            preprocessor=Preprocessor(),
            encoder=Encoder(),
            cfg=types.SimpleNamespace(
                encoder=types.SimpleNamespace(d_model=4)),
        )
        models = types.ModuleType("nemo.collections.asr.models")
        models.ASRModel = types.SimpleNamespace(
            restore_from=lambda **kwargs: restored)
        asr = types.ModuleType("nemo.collections.asr")
        asr.models = models
        collections = types.ModuleType("nemo.collections")
        collections.asr = asr
        nemo = types.ModuleType("nemo")
        nemo.collections = collections
        modules = {
            "nemo": nemo,
            "nemo.collections": collections,
            "nemo.collections.asr": asr,
            "nemo.collections.asr.models": models,
        }
        with tempfile.NamedTemporaryFile(suffix=".nemo") as checkpoint:
            with mock.patch.dict(sys.modules, modules):
                model = FrozenWavLMCTC(
                    5, checkpoint.name, dropout=0.0,
                    backbone_type="parakeet")

        log_probs, lengths = model.log_probs(
            torch.randn(2, 12), torch.tensor([12, 8]))
        self.assertEqual(model.backbone_type, "parakeet")
        self.assertEqual(log_probs.shape, (2, 6, 5))
        torch.testing.assert_close(lengths, torch.tensor([6, 4]))

    def test_explicit_backbone_validation(self):
        self.assertEqual(resolve_backbone_type("unused", "raw"), "raw")
        with self.assertRaises(ValueError):
            resolve_backbone_type("unused", "unknown")


if __name__ == "__main__":
    unittest.main()
