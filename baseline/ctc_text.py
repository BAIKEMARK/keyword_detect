from __future__ import annotations

import string

import torch


class CharacterVocabulary:
    def __init__(self):
        self.symbols = ("<blank>", *string.ascii_lowercase, "'")
        self.blank_id = 0
        self._char_to_id = {
            symbol: index for index, symbol in enumerate(self.symbols)
            if index != self.blank_id
        }

    def __len__(self):
        return len(self.symbols)

    def normalize(self, text: str) -> str:
        normalized = text.strip().lower()
        if not normalized:
            raise ValueError("keyword text must not be empty")
        unsupported = sorted(set(normalized) - set(self._char_to_id))
        if unsupported:
            raise ValueError(
                f"unsupported keyword characters: {unsupported!r} in {text!r}")
        return normalized

    def encode(self, text: str) -> torch.Tensor:
        normalized = self.normalize(text)
        return torch.tensor(
            [self._char_to_id[char] for char in normalized],
            dtype=torch.long,
        )


def required_ctc_frames(targets: torch.Tensor,
                        target_lengths: torch.Tensor) -> torch.Tensor:
    if targets.ndim != 2:
        raise ValueError("targets must have shape (batch, max_target_length)")
    positions = torch.arange(targets.shape[1], device=targets.device)
    repeated = targets[:, 1:] == targets[:, :-1]
    valid_repeat = positions[1:].unsqueeze(0) < target_lengths.unsqueeze(1)
    return target_lengths + (repeated & valid_repeat).sum(dim=1)
