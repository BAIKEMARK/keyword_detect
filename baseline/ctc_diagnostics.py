from __future__ import annotations

from typing import Iterable, Sequence

import torch

from ctc_score import normalized_ctc_score
from ctc_text import required_ctc_frames


def greedy_ctc_decode(log_probs: torch.Tensor,
                      input_lengths: torch.Tensor,
                      blank_id: int) -> list[list[int]]:
    """Collapse frame-wise CTC argmax paths without looking at target text."""
    if log_probs.ndim != 3:
        raise ValueError("log_probs must have shape (batch, frames, classes)")
    if input_lengths.ndim != 1 or input_lengths.shape[0] != log_probs.shape[0]:
        raise ValueError("input_lengths must have one value per batch item")
    best = log_probs.argmax(dim=-1)
    decoded = []
    for row, length in zip(best, input_lengths.tolist()):
        previous = blank_id
        sequence = []
        for token in row[:int(length)].tolist():
            if token != blank_id and token != previous:
                sequence.append(int(token))
            previous = int(token)
        decoded.append(sequence)
    return decoded


def sequence_edit_distance(left: Sequence[int], right: Sequence[int]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, 1):
        current = [left_index]
        for right_index, right_token in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_token != right_token),
            ))
        previous = current
    return previous[-1]


def edit_similarity(decoded: Sequence[int], target: Sequence[int]) -> float:
    denominator = max(len(decoded), len(target), 1)
    return max(0.0, 1.0 - sequence_edit_distance(decoded, target) / denominator)


def _pad_sequences(sequences: Iterable[Sequence[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [list(sequence) for sequence in sequences]
    lengths = torch.tensor(
        [len(sequence) for sequence in sequences], dtype=torch.long,
        device=device,
    )
    if not sequences or int(lengths.max().item()) == 0:
        return torch.empty((len(sequences), 0), dtype=torch.long, device=device), lengths
    targets = torch.zeros(
        (len(sequences), int(lengths.max().item())),
        dtype=torch.long,
        device=device,
    )
    for index, sequence in enumerate(sequences):
        if sequence:
            targets[index, :len(sequence)] = torch.tensor(
                sequence, dtype=torch.long, device=device)
    return targets, lengths


def greedy_path_scores(log_probs: torch.Tensor,
                       input_lengths: torch.Tensor,
                       decoded: list[list[int]],
                       blank_id: int) -> torch.Tensor:
    """Score each collapsed greedy path with the exact CTC forward algorithm."""
    scores = log_probs.new_full((log_probs.shape[0],), -1e4)
    nonempty = torch.tensor(
        [bool(sequence) for sequence in decoded],
        dtype=torch.bool,
        device=log_probs.device,
    )
    if nonempty.any():
        targets, target_lengths = _pad_sequences(
            [sequence for sequence, keep in zip(decoded, nonempty.tolist())
             if keep], log_probs.device)
        scores[nonempty] = normalized_ctc_score(
            log_probs[nonempty], input_lengths[nonempty], targets,
            target_lengths, blank_id)
    empty = ~nonempty
    if empty.any():
        for index in empty.nonzero(as_tuple=False).flatten().tolist():
            length = int(input_lengths[index].item())
            scores[index] = log_probs[index, :length, blank_id].mean()
    return scores


def diagnostic_features(log_probs: torch.Tensor,
                         input_lengths: torch.Tensor,
                         targets: torch.Tensor,
                         target_lengths: torch.Tensor,
                         blank_id: int) -> dict[str, torch.Tensor | list[list[int]]]:
    """Return target, greedy-path, and confidence features for a CTC batch."""
    if targets.ndim != 2:
        raise ValueError("targets must have shape (batch, max_target_length)")
    decoded = greedy_ctc_decode(log_probs, input_lengths, blank_id)
    greedy_scores = greedy_path_scores(
        log_probs, input_lengths, decoded, blank_id)
    target_scores = log_probs.new_full((log_probs.shape[0],), -1e4)
    valid = input_lengths >= required_ctc_frames(targets, target_lengths)
    if valid.any():
        target_scores[valid] = normalized_ctc_score(
            log_probs[valid], input_lengths[valid], targets[valid],
            target_lengths[valid], blank_id)

    frame_confidence = log_probs.exp().amax(dim=-1)
    frame_confidence = torch.stack([
        row[:int(length)].mean()
        for row, length in zip(frame_confidence, input_lengths.tolist())
    ])
    blank_ratio = torch.stack([
        (row[:int(length)].argmax(dim=-1) == blank_id).float().mean()
        for row, length in zip(log_probs, input_lengths.tolist())
    ])
    target_lists = [
        row[:int(length)].tolist()
        for row, length in zip(targets, target_lengths.tolist())
    ]
    edit_scores = log_probs.new_tensor([
        edit_similarity(decoded_row, target_row)
        for decoded_row, target_row in zip(decoded, target_lists)
    ])
    target_len = target_lengths.to(log_probs.dtype)
    greedy_len = log_probs.new_tensor([len(row) for row in decoded])
    return {
        "target_score": target_scores,
        "greedy_score": greedy_scores,
        "likelihood_margin": target_scores - greedy_scores,
        "edit_similarity": edit_scores,
        "frame_confidence": frame_confidence,
        "blank_ratio": blank_ratio,
        "target_length": target_len,
        "greedy_length": greedy_len,
        "decoded": decoded,
    }
