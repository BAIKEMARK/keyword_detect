from __future__ import annotations

import sys
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))

from diagnose_registered_ctc import FEATURES  # noqa: E402
from fuse_ctc_scores import read_dev_scores  # noqa: E402
from registered_reranker import (decision_scores, fit_reranker, rows_to_matrix,
                                 subset_auc)  # noqa: E402
from train_registered_ctc_reranker import _write_dev_scores  # noqa: E402


def _rows():
    rows = []
    for label in (0, 1):
        for index in range(8):
            base = float(index) + (8.0 if label else 0.0)
            rows.append({
                "id": f"pair_{label}_{index}",
                "label": label,
                "subset": "seen" if index % 2 == 0 else "unseen",
                **{name: base + 0.01 * feature_index
                   for feature_index, name in enumerate(FEATURES)},
            })
    return rows


class RegisteredRerankerTest(unittest.TestCase):
    def test_matrix_uses_declared_feature_order(self):
        rows = _rows()[:1]
        actual = rows_to_matrix(rows)
        expected = np.array([[rows[0][name] for name in FEATURES]])
        np.testing.assert_allclose(actual, expected)

    def test_fit_and_score_improve_over_random(self):
        rows = _rows()
        model = fit_reranker(rows)
        scores = decision_scores(rows, model)
        self.assertGreater(subset_auc(rows, scores, "all"), 0.95)
        self.assertGreater(scores[-1], scores[0])

    def test_subset_auc_rejects_missing_subset(self):
        rows = _rows()
        scores = np.arange(len(rows), dtype=np.float64)
        with self.assertRaises(ValueError):
            subset_auc(rows, scores, "missing")

    def test_dev_writer_uses_fusion_id_prefixes(self):
        rows = [_rows()[0], _rows()[9]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev.csv"
            _write_dev_scores(path, rows, np.array([-1.0, 1.0]))
            with path.open(newline="", encoding="utf-8") as file:
                output = list(csv.DictReader(file))
            self.assertEqual(
                [row["id"] for row in output],
                ["seen_pair_0_0", "unseen_pair_1_1"],
            )
            self.assertEqual(set(read_dev_scores(str(path))), {
                "seen_pair_0_0", "unseen_pair_1_1"})


if __name__ == "__main__":
    unittest.main()
