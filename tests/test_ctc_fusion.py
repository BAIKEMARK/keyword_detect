from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))

from fuse_ctc_scores import (ScoreRow, align_dev_rows,  # noqa: E402
                             align_eval_rows, average_percentile_ranks,
                             fuse_eval_rows, main as fusion_main,
                             read_dev_scores,
                             search_phoneme_weight)


class RankFusionTest(unittest.TestCase):
    def test_average_percentile_ranks_handles_ties_and_constants(self):
        actual = average_percentile_ranks(np.array([1.0, 1.0, 3.0]))
        np.testing.assert_allclose(actual, [0.25, 0.25, 1.0])
        np.testing.assert_allclose(
            average_percentile_ranks(np.ones(4)), np.full(4, 0.5))

    def test_aligns_rows_by_id_in_official_order(self):
        character = {
            "unseen_pair_2": ScoreRow(
                "unseen_pair_2", "unseen", 0.2, 0),
            "seen_pair_2": ScoreRow("seen_pair_2", "seen", 0.8, 1),
            "seen_pair_1": ScoreRow("seen_pair_1", "seen", 0.1, 0),
        }
        phoneme = {
            "seen_pair_1": ScoreRow("seen_pair_1", "seen", 0.3, 0),
            "unseen_pair_2": ScoreRow(
                "unseen_pair_2", "unseen", 0.4, 0),
            "seen_pair_2": ScoreRow("seen_pair_2", "seen", 0.7, 1),
        }
        rows = align_dev_rows(character, phoneme)
        self.assertEqual(
            [row.pair_id for row in rows],
            ["seen_pair_1", "seen_pair_2", "unseen_pair_2"],
        )

    def test_rejects_missing_ids_and_label_disagreement(self):
        row = ScoreRow("seen_pair_1", "seen", 0.1, 0)
        with self.assertRaises(ValueError):
            align_dev_rows({row.pair_id: row}, {})
        mismatch = ScoreRow("seen_pair_1", "seen", 0.2, 1)
        with self.assertRaises(ValueError):
            align_dev_rows({row.pair_id: row}, {mismatch.pair_id: mismatch})

    def test_weight_search_finds_complementary_fusion(self):
        labels = np.array([0, 0, 1, 1] * 2)
        subsets = np.array(["seen"] * 4 + ["unseen"] * 4)
        character = np.array([0, 2, 1, 3] * 2, dtype=np.float64)
        phoneme = np.array([2, 0, 3, 1] * 2, dtype=np.float64)
        result = search_phoneme_weight(
            character, phoneme, labels, subsets, step=0.001)
        self.assertAlmostEqual(result.mean_auc, 1.0)
        self.assertGreater(result.weight, 0.0)
        self.assertLess(result.weight, 1.0)

    def test_weight_search_favors_phoneme_on_exact_tie(self):
        labels = np.array([0, 0, 1, 1] * 2)
        subsets = np.array(["seen"] * 4 + ["unseen"] * 4)
        scores = np.array([0, 1, 2, 3] * 2, dtype=np.float64)
        result = search_phoneme_weight(
            scores, scores, labels, subsets, step=0.001)
        self.assertEqual(result.weight, 1.0)


class FusionCsvTest(unittest.TestCase):
    def _csv(self, fieldnames, rows):
        temporary = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", newline="", encoding="utf-8",
            delete=False)
        with temporary:
            writer = csv.DictWriter(temporary, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(Path(temporary.name).unlink)
        return temporary.name

    def test_rejects_duplicate_and_nonfinite_dev_scores(self):
        duplicate = self._csv(
            ["id", "subset", "score", "label"],
            [
                {"id": "seen_pair_1", "subset": "seen",
                 "score": "0.1", "label": "0"},
                {"id": "seen_pair_1", "subset": "seen",
                 "score": "0.2", "label": "0"},
            ],
        )
        with self.assertRaises(ValueError):
            read_dev_scores(duplicate)

        nonfinite = self._csv(
            ["id", "subset", "score", "label"],
            [{"id": "seen_pair_1", "subset": "seen",
              "score": "nan", "label": "0"}],
        )
        with self.assertRaises(ValueError):
            read_dev_scores(nonfinite)

    def test_fused_eval_rows_are_bounded_and_ordered(self):
        character = {
            "unseen_pair_2": ScoreRow(
                "unseen_pair_2", "unseen", 0.7, None),
            "seen_pair_2": ScoreRow("seen_pair_2", "seen", 0.9, None),
            "seen_pair_1": ScoreRow("seen_pair_1", "seen", 0.2, None),
            "unseen_pair_1": ScoreRow(
                "unseen_pair_1", "unseen", 0.1, None),
        }
        phoneme = {
            pair_id: ScoreRow(row.pair_id, row.subset, 1.0 - row.score, None)
            for pair_id, row in character.items()
        }
        aligned = align_eval_rows(character, phoneme)
        fused = fuse_eval_rows(aligned, phoneme_weight=0.7)
        self.assertEqual(
            [pair_id for pair_id, _ in fused],
            ["seen_pair_1", "seen_pair_2",
             "unseen_pair_1", "unseen_pair_2"],
        )
        self.assertTrue(all(0.0 <= posterior <= 1.0
                            for _, posterior in fused))

    def test_cli_writes_submission_and_json_report(self):
        dev_fields = ["id", "subset", "score", "label"]
        char_dev_rows = []
        phone_dev_rows = []
        for subset in ("seen", "unseen"):
            labels = [0, 0, 1, 1]
            for index, (char_score, phone_score, label) in enumerate(zip(
                    [0, 2, 1, 3], [2, 0, 3, 1], labels), 1):
                pair_id = f"{subset}_pair_{index}"
                base = {"id": pair_id, "subset": subset, "label": label}
                char_dev_rows.append({**base, "score": char_score})
                phone_dev_rows.append({**base, "score": phone_score})
        char_dev = self._csv(dev_fields, char_dev_rows)
        phone_dev = self._csv(dev_fields, phone_dev_rows)
        eval_fields = ["id", "posterior"]
        char_eval = self._csv(eval_fields, [
            {"id": "seen_pair_1", "posterior": 0.1},
            {"id": "seen_pair_2", "posterior": 0.9},
            {"id": "unseen_pair_1", "posterior": 0.2},
            {"id": "unseen_pair_2", "posterior": 0.8},
        ])
        phone_eval = self._csv(eval_fields, [
            {"id": "unseen_pair_2", "posterior": 0.3},
            {"id": "seen_pair_2", "posterior": 0.2},
            {"id": "unseen_pair_1", "posterior": 0.7},
            {"id": "seen_pair_1", "posterior": 0.8},
        ])

        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "submission.csv")
            with mock.patch("sys.argv", [
                    "fuse_ctc_scores.py",
                    "--char-dev", char_dev,
                    "--phoneme-dev", phone_dev,
                    "--char-eval", char_eval,
                    "--phoneme-eval", phone_eval,
                    "--out", output,
            ]), mock.patch("builtins.print"):
                fusion_main()
            with open(output, encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(
                [row["id"] for row in rows],
                ["seen_pair_1", "seen_pair_2",
                 "unseen_pair_1", "unseen_pair_2"],
            )
            with open(f"{output}.json", encoding="utf-8") as file:
                report = json.load(file)
            self.assertEqual(report["rows"], 4)
            self.assertAlmostEqual(report["fusion"]["mean_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
