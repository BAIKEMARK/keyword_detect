from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))

from ctc_alignment_rescorer import (FEATURES, cross_validated_scores,  # noqa: E402
                                    decision_scores, fit_rescorer,
                                    rows_to_matrix, subset_auc)
from ctc_diagnostics import forced_alignment_features  # noqa: E402


def _log_probs(path, classes=3):
    logits = torch.full((1, len(path), classes), -8.0)
    for frame, token in enumerate(path):
        logits[0, frame, token] = 8.0
    return logits.log_softmax(dim=-1)


def _rows():
    rows = []
    for subset in ("seen", "unseen"):
        for label in (0, 1):
            for index in range(10):
                signal = float(label * 20 + index) / 10.0
                rows.append({
                    "id": f"{subset}_{label}_{index}",
                    "subset": subset,
                    "label": label,
                    **{
                        name: signal + feature_index * 0.01
                        for feature_index, name in enumerate(FEATURES)
                    },
                })
    return rows


class ForcedAlignmentTest(unittest.TestCase):
    def test_tracks_ordered_tokens_and_path_statistics(self):
        features = forced_alignment_features(
            _log_probs([0, 1, 0, 2, 0]),
            torch.tensor([5]),
            torch.tensor([[1, 2]]),
            torch.tensor([2]),
            blank_id=0,
        )
        self.assertGreater(features["token_peak_min"].item(), -0.01)
        self.assertGreater(features["aligned_token_logprob"].item(), -0.01)
        self.assertAlmostEqual(
            features["viterbi_blank_ratio"].item(), 3 / 5, places=5)
        self.assertAlmostEqual(
            features["token_span_ratio"].item(), 3 / 5, places=5)
        self.assertAlmostEqual(
            features["frames_per_token"].item(), 2.5, places=5)
        self.assertAlmostEqual(
            features["token_duration_cv"].item(), 0.0, places=5)

    def test_repeated_tokens_require_an_intermediate_blank(self):
        features = forced_alignment_features(
            _log_probs([1, 0, 1], classes=2),
            torch.tensor([3]),
            torch.tensor([[1, 1]]),
            torch.tensor([2]),
            blank_id=0,
        )
        self.assertGreater(features["viterbi_score"].item(), -0.01)
        self.assertAlmostEqual(
            features["viterbi_blank_ratio"].item(), 1 / 3, places=5)

    def test_marks_ctc_invalid_alignment_without_dropping_the_row(self):
        features = forced_alignment_features(
            _log_probs([1, 1], classes=2),
            torch.tensor([2]),
            torch.tensor([[1, 1]]),
            torch.tensor([2]),
            blank_id=0,
        )
        self.assertEqual(features["alignment_valid"].item(), 0.0)
        self.assertEqual(features["viterbi_score"].item(), 0.0)
        self.assertEqual(features["frames_per_token"].item(), 1.0)

    def test_mixed_batch_aligns_only_valid_rows(self):
        log_probs = torch.cat([
            _log_probs([0, 1, 0]),
            _log_probs([1, 1, 0]),
        ])
        features = forced_alignment_features(
            log_probs,
            torch.tensor([3, 2]),
            torch.tensor([[1, 0], [1, 1]]),
            torch.tensor([1, 2]),
            blank_id=0,
        )
        torch.testing.assert_close(
            features["alignment_valid"], torch.tensor([1.0, 0.0]))
        self.assertGreater(features["viterbi_score"][0].item(), -0.01)
        self.assertEqual(features["viterbi_score"][1].item(), 0.0)


class AlignmentRescorerTest(unittest.TestCase):
    def test_feature_order_and_fit(self):
        rows = _rows()
        matrix = rows_to_matrix(rows[:1])
        np.testing.assert_allclose(
            matrix[0], [rows[0][name] for name in FEATURES])
        model = fit_rescorer(rows)
        scores = decision_scores(rows, model)
        self.assertGreater(subset_auc(rows, scores, "seen"), 0.95)

    def test_cross_validation_is_complete_and_deterministic(self):
        rows = _rows()
        first = cross_validated_scores(rows, folds=5, seed=42)
        second = cross_validated_scores(rows, folds=5, seed=42)
        np.testing.assert_allclose(first, second)
        self.assertEqual(first.shape, (len(rows),))
        self.assertTrue(np.isfinite(first).all())
        self.assertGreater(subset_auc(rows, first, "unseen"), 0.9)


if __name__ == "__main__":
    unittest.main()
