from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))

from config import AudioConfig  # noqa: E402
from data import (WavePairDataset, collate_wave_pairs, normalize_waveform,
                  truncate_waveform)  # noqa: E402
from train_wavlm import (default_last_checkpoint_path, parse_args,  # noqa: E402
                         training_config, validate_resume_checkpoint)
from wavlm_model import SymmetricFrameMatchHead  # noqa: E402
from wavlm_model import FrozenWavLMMatcher  # noqa: E402


class WaveDataTest(unittest.TestCase):
    def test_training_cli_supports_full_data_and_resume(self):
        args = parse_args([
            "--train-zip", "train/wav.zip",
            "--train-csv", "train/train_label.csv",
            "--subset", "500000",
            "--resume", "checkpoint.pt",
            "--last-out", "latest.pt",
        ])
        self.assertEqual(args.train_zip, "train/wav.zip")
        self.assertEqual(args.train_csv, "train/train_label.csv")
        self.assertEqual(args.subset, 500000)
        self.assertEqual(args.resume, "checkpoint.pt")
        self.assertEqual(args.last_out, "latest.pt")

    def test_resume_checkpoint_paths_and_compatibility(self):
        self.assertEqual(
            default_last_checkpoint_path("checkpoints/model.pt"),
            "checkpoints/model.last.pt")

        args = parse_args([
            "--model-id", "local/wavlm",
            "--train-zip", "train/wav.zip",
            "--train-csv", "train/train_label.csv",
            "--subset", "500000",
            "--bs", "128",
        ])
        args.workers = 0
        config = training_config(
            args, max_samples=40000, train_pairs=500000,
            amp_enabled=False, device=torch.device("cpu"))
        legacy_checkpoint = {
            "model_id": "local/wavlm",
            "projection_dim": 128,
            "max_samples": 40000,
            "pos_weight": 4.0,
        }
        validate_resume_checkpoint(legacy_checkpoint, config)

        new_checkpoint = dict(legacy_checkpoint)
        new_checkpoint["training_config"] = dict(config)
        new_checkpoint["training_config"]["batch_size"] = 256
        with self.assertRaisesRegex(ValueError, "batch_size"):
            validate_resume_checkpoint(new_checkpoint, config)

    def test_truncate_waveform(self):
        waveform = np.arange(12, dtype=np.float32)
        result = truncate_waveform(waveform, max_samples=8)
        self.assertEqual(result.shape, (8,))
        torch.testing.assert_close(result, torch.arange(8, dtype=torch.float32))

    def test_normalize_waveform(self):
        result = normalize_waveform(torch.arange(10, dtype=torch.float32))
        self.assertAlmostEqual(result.mean().item(), 0.0, places=6)
        self.assertAlmostEqual(result.var(unbiased=False).item(), 1.0, places=6)

    def test_dynamic_padding_uses_longest_waveform(self):
        batch = [
            (torch.ones(5), torch.ones(3), 1, "a", 5, 3),
            (torch.ones(2), torch.ones(7), 0, "b", 2, 7),
        ]
        enroll, query, labels, ids, e_lens, q_lens = collate_wave_pairs(batch)
        self.assertEqual(enroll.shape, (2, 7))
        self.assertEqual(query.shape, (2, 7))
        self.assertEqual(ids, ["a", "b"])
        torch.testing.assert_close(labels, torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(e_lens, torch.tensor([5, 2]))
        torch.testing.assert_close(q_lens, torch.tensor([3, 7]))
        self.assertEqual(enroll[0, 5:].abs().sum().item(), 0.0)
        self.assertEqual(query[0, 3:].abs().sum().item(), 0.0)

    @mock.patch("data.read_wav")
    def test_augmentation_changes_query_only(self, read_wav):
        read_wav.return_value = np.arange(10, dtype=np.float32)

        class MaskFirstHalf:
            def __call__(self, waveform):
                waveform = waveform.copy()
                waveform[:5] = 0.0
                return waveform

        dataset = WavePairDataset(
            [{"id": "pair_1", "label": 1}],
            "unused.zip",
            AudioConfig(),
            max_samples=10,
            query_augment=MaskFirstHalf(),
        )
        enroll, query, *_ = dataset[0]
        expected_enroll = normalize_waveform(torch.arange(10, dtype=torch.float32))
        torch.testing.assert_close(enroll, expected_enroll)
        self.assertFalse(torch.allclose(query, enroll))


class WavLMHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.head = SymmetricFrameMatchHead(
            hidden_size=8, num_hidden_states=3, projection_dim=4)
        self.hidden = tuple(torch.randn(4, 7, 8) for _ in range(3))
        self.e_lens = torch.tensor([4, 6])
        self.q_lens = torch.tensor([5, 3])

    def test_padded_hidden_values_do_not_change_score(self):
        changed = [layer.clone() for layer in self.hidden]
        lengths = torch.cat([self.e_lens, self.q_lens])
        for layer in changed:
            for index, length in enumerate(lengths.tolist()):
                layer[index, length:] = 100.0 + index

        expected = self.head(self.hidden, self.e_lens, self.q_lens)
        actual = self.head(tuple(changed), self.e_lens, self.q_lens)
        torch.testing.assert_close(actual, expected)

    def test_matching_is_symmetric(self):
        swapped = tuple(torch.cat([layer[2:], layer[:2]], dim=0)
                        for layer in self.hidden)
        expected = self.head(self.hidden, self.e_lens, self.q_lens)
        actual = self.head(swapped, self.q_lens, self.e_lens)
        torch.testing.assert_close(actual, expected)

    def test_all_head_parameters_receive_finite_gradients(self):
        self.head(self.hidden, self.e_lens, self.q_lens).sum().backward()
        for name, parameter in self.head.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())


class FrozenWavLMWrapperTest(unittest.TestCase):
    def test_wrapper_freezes_backbone_and_saves_head_only(self):
        class FakeBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.config = types.SimpleNamespace(
                    hidden_size=8, num_hidden_layers=2)

            def forward(self, waveforms, attention_mask,
                        output_hidden_states, return_dict):
                self.last_attention_mask = attention_mask
                base = waveforms[:, ::2].unsqueeze(-1).repeat(1, 1, 8)
                hidden = tuple(base * self.weight + i for i in range(3))
                return types.SimpleNamespace(hidden_states=hidden)

            def _get_feat_extract_output_lengths(self, lengths):
                return torch.div(lengths + 1, 2, rounding_mode="floor")

        fake_backbone = FakeBackbone()

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(model_id):
                self.assertEqual(model_id, "fake/wavlm")
                return fake_backbone

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModel = FakeAutoModel
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            model = FrozenWavLMMatcher("fake/wavlm", projection_dim=4)

        self.assertTrue(all(not p.requires_grad
                            for p in model.backbone.parameters()))
        model.train()
        self.assertFalse(model.backbone.training)

        enroll = torch.randn(2, 12)
        query = torch.randn(2, 12)
        e_lens = torch.tensor([12, 8])
        q_lens = torch.tensor([10, 6])
        logits = model(enroll, query, e_lens, q_lens)
        self.assertEqual(logits.shape, (2,))
        logits.sum().backward()
        self.assertIsNone(model.backbone.weight.grad)
        self.assertTrue(all(parameter.grad is not None
                            for parameter in model.head.parameters()))

        state = model.head_state_dict()
        self.assertTrue(state)
        self.assertTrue(all(not key.startswith("backbone.") for key in state))
        model.load_head_state_dict(state)


if __name__ == "__main__":
    unittest.main()
