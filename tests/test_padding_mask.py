from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))

from data import pad_spec  # noqa: E402
from model import build_model  # noqa: E402


class PaddingMaskTest(unittest.TestCase):
    def test_pad_spec_returns_capped_valid_length(self):
        padded, length = pad_spec(torch.ones(40, 7), max_frames=10)
        self.assertEqual(padded.shape, (1, 40, 10))
        self.assertEqual(length, 7)

        truncated, length = pad_spec(torch.ones(40, 12), max_frames=10)
        self.assertEqual(truncated.shape, (1, 40, 10))
        self.assertEqual(length, 10)

    def test_frame_score_ignores_padded_values(self):
        torch.manual_seed(7)
        model = build_model("frame_maxmean", n_mels=40).eval()
        enroll = torch.randn(2, 1, 40, 20)
        query = torch.randn(2, 1, 40, 20)
        e_lens = torch.tensor([12, 16])
        q_lens = torch.tensor([10, 14])

        changed_enroll = enroll.clone()
        changed_query = query.clone()
        changed_enroll[0, :, :, 12:] = 100.0
        changed_enroll[1, :, :, 16:] = -100.0
        changed_query[0, :, :, 10:] = -100.0
        changed_query[1, :, :, 14:] = 100.0

        with torch.no_grad():
            expected = model(enroll, query, e_lens, q_lens)
            actual = model(changed_enroll, changed_query, e_lens, q_lens)
        torch.testing.assert_close(actual, expected)

    def test_frame_score_uses_valid_values(self):
        model = build_model("frame_maxmean", n_mels=40).eval()
        with torch.no_grad():
            for layer in model.encoder.modules():
                if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                    layer.weight.fill_(0.01)
                    layer.bias.zero_()

        enroll = torch.ones(1, 1, 40, 20)
        query = torch.ones(1, 1, 40, 20)
        changed_query = query.clone()
        changed_query[:, :, :, :12] = 0.0
        lengths = torch.tensor([12])

        with torch.no_grad():
            expected = model(enroll, query, lengths, lengths)
            actual = model(enroll, changed_query, lengths, lengths)
        self.assertFalse(torch.allclose(actual, expected))

    def test_models_support_length_aware_backward(self):
        torch.manual_seed(11)
        enroll = torch.randn(2, 1, 40, 20)
        query = torch.randn(2, 1, 40, 20)
        e_lens = torch.tensor([20, 13])
        q_lens = torch.tensor([17, 11])

        for model_name in ("global", "frame_maxmean"):
            with self.subTest(model=model_name):
                model = build_model(model_name, n_mels=40)
                model(enroll, query, e_lens, q_lens).sum().backward()
                grad = model.encoder.cnn[0].weight.grad
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all())


if __name__ == "__main__":
    unittest.main()
