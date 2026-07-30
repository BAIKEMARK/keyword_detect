from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))

from ctc_diagnostics import (  # noqa: E402
    diagnostic_features,
    edit_similarity,
    greedy_ctc_decode,
    sequence_edit_distance,
)


class CTCDiagnosticsTest(unittest.TestCase):
    def test_greedy_decode_removes_blanks_and_repeats(self):
        ids = torch.tensor([
            [0, 1, 1, 0, 2, 2, 0],
            [1, 0, 1, 1, 0, 0, 2],
        ])
        log_probs = torch.full((2, 7, 3), -10.0)
        log_probs.scatter_(2, ids.unsqueeze(-1), 0.0)
        decoded = greedy_ctc_decode(
            log_probs, torch.tensor([7, 7]), blank_id=0)
        self.assertEqual(decoded, [[1, 2], [1, 1, 2]])

    def test_edit_similarity_is_normalized(self):
        self.assertEqual(sequence_edit_distance([1, 2], [1, 2]), 0)
        self.assertEqual(sequence_edit_distance([], [1, 2]), 2)
        self.assertAlmostEqual(edit_similarity([1, 2], [1, 3]), 0.5)
        self.assertEqual(edit_similarity([], []), 1.0)

    def test_diagnostic_features_include_empty_greedy_path(self):
        logits = torch.tensor([
            [[3.0, 0.0], [3.0, 0.0], [3.0, 0.0]],
            [[0.0, 3.0], [0.0, 3.0], [3.0, 0.0]],
        ])
        features = diagnostic_features(
            logits.log_softmax(dim=-1), torch.tensor([3, 3]),
            torch.tensor([[1], [1]]), torch.tensor([1, 1]), blank_id=0)
        self.assertEqual(features["decoded"], [[], [1]])
        self.assertTrue(torch.isfinite(features["greedy_score"]).all())
        self.assertTrue(torch.isfinite(features["likelihood_margin"]).all())


if __name__ == "__main__":
    unittest.main()
