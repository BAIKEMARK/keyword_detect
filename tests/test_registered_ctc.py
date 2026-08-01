from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))

from diagnose_registered_ctc import posterior_alignment  # noqa: E402


class RegisteredCTCTest(unittest.TestCase):
    def test_alignment_ignores_padding_and_is_symmetric(self):
        torch.manual_seed(53)
        log_probs = torch.randn(4, 6, 5).log_softmax(dim=-1)
        left_lengths = torch.tensor([4, 6])
        right_lengths = torch.tensor([5, 3])
        expected = posterior_alignment(
            log_probs, left_lengths, right_lengths)
        changed = log_probs.clone()
        changed[0, 4:] = -100.0
        changed[1, 6:] = -100.0
        changed[2, 5:] = -100.0
        changed[3, 3:] = -100.0
        actual = posterior_alignment(
            changed, left_lengths, right_lengths)
        torch.testing.assert_close(actual, expected)

        swapped = torch.cat([log_probs[2:], log_probs[:2]])
        torch.testing.assert_close(
            posterior_alignment(swapped, right_lengths, left_lengths), expected)


if __name__ == "__main__":
    unittest.main()
