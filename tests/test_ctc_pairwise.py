from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))

from ctc_data import (collate_ctc_training_pairs,  # noqa: E402
                      load_ctc_training_pairs)
from ctc_hard_negative import (  # noqa: E402
    build_phoneme_hard_negative_candidates,
    build_phoneme_hard_negatives,
)
from ctc_text import CharacterVocabulary, PhonemeVocabulary  # noqa: E402
from train_ctc_pairwise import (pairwise_discriminative_objective,  # noqa: E402
                                parse_args)


class CTCPairDataTest(unittest.TestCase):
    def test_loads_official_pair_label_and_query_text(self):
        temporary = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", newline="", encoding="utf-8",
            delete=False)
        with temporary:
            writer = csv.DictWriter(
                temporary,
                fieldnames=["id", "enroll_txt", "query_txt", "label"],
            )
            writer.writeheader()
            writer.writerow({
                "id": "pair_1", "enroll_txt": "cat",
                "query_txt": "bat", "label": "0",
            })
        self.addCleanup(Path(temporary.name).unlink)
        self.assertEqual(load_ctc_training_pairs(temporary.name), [{
            "id": "pair_1",
            "enroll_text": "cat",
            "query_text": "bat",
            "label": 0,
        }])

    def test_collates_multiple_hard_negative_candidates(self):
        vocabulary = CharacterVocabulary()
        batch = [
            (torch.ones(5), "cat", "cat", 1, "p1", 5),
            (torch.ones(3), "cat", "dog", 0, "p2", 3),
        ]
        result = collate_ctc_training_pairs(
            batch,
            vocabulary,
            hard_negative_candidates={
                "cat": ("bat", "cap"),
                "dog": ("fog", "dig"),
            },
        )
        self.assertEqual(result[0].shape, (2, 5))
        self.assertEqual(result[8], 2)
        torch.testing.assert_close(result[9], torch.tensor([1.0, 0.0]))
        self.assertEqual(result[10], ["p1", "p2"])
        self.assertEqual(result[6].shape[0], 4)
        torch.testing.assert_close(
            result[6][0, :3], vocabulary.encode("bat"))
        torch.testing.assert_close(
            result[6][2, :3], vocabulary.encode("fog"))

    def test_top_k_candidates_preserve_single_neighbor_compatibility(self):
        pronunciations = {
            "cat": ["K", "AE1", "T"],
            "bat": ["B", "AE1", "T"],
            "cap": ["K", "AE1", "P"],
            "dog": ["D", "AO1", "G"],
        }
        vocabulary = PhonemeVocabulary(
            converter=lambda text: pronunciations[text])
        candidates = build_phoneme_hard_negative_candidates(
            vocabulary, ["cat"], pronunciations, neighbors_per_anchor=3)
        nearest = build_phoneme_hard_negatives(
            vocabulary, ["cat"], pronunciations)
        self.assertEqual(len(candidates["cat"]), 3)
        self.assertEqual(nearest["cat"], candidates["cat"][0])
        self.assertNotIn("cat", candidates["cat"])

    def test_cli_uses_pair_count_not_utterance_count(self):
        args = parse_args([
            "--ckpt", "base.pt", "--out", "pairwise.pt",
            "--subset", "256", "--hard-negative-k", "3",
        ])
        self.assertEqual(args.subset, 256)
        self.assertEqual(args.hard_negative_k, 3)
        self.assertEqual(args.pair_weight, 1.0)


class CTCPairObjectiveTest(unittest.TestCase):
    @staticmethod
    def _log_probs(first_class, second_class):
        logits = torch.full((2, 3, 3), -3.0)
        logits[0, :, first_class] = 3.0
        logits[1, :, second_class] = 3.0
        return logits.log_softmax(dim=-1).requires_grad_()

    def _loss(self, log_probs):
        output_lengths = torch.tensor([3, 3])
        enroll_targets = torch.tensor([[1], [1]])
        query_targets = torch.tensor([[1], [2]])
        lengths = torch.tensor([1, 1])
        hard_targets = torch.tensor([[2], [2], [1], [1]])
        hard_lengths = torch.ones(4, dtype=torch.long)
        labels = torch.tensor([1.0, 0.0])
        return pairwise_discriminative_objective(
            log_probs, output_lengths,
            enroll_targets, lengths,
            query_targets, lengths,
            hard_targets, hard_lengths, 2, labels, blank_id=0,
            ctc_weight=0.25, pair_weight=1.0,
            hard_negative_weight=0.25,
        )

    def test_correct_pair_order_has_lower_loss_and_backpropagates(self):
        good = self._log_probs(first_class=1, second_class=2)
        bad = self._log_probs(first_class=2, second_class=1)
        good_result = self._loss(good)
        bad_result = self._loss(bad)
        self.assertLess(good_result[0].item(), bad_result[0].item())
        self.assertLess(good_result[2].item(), bad_result[2].item())
        self.assertEqual(good_result[4], 0)
        good_result[0].backward()
        self.assertIsNotNone(good.grad)
        self.assertTrue(torch.isfinite(good.grad).all())


if __name__ == "__main__":
    unittest.main()
