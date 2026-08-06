from __future__ import annotations

from typing import Iterable, Sequence

import torch

from ctc_score import normalized_ctc_score
from ctc_text import required_ctc_frames


FORCED_ALIGNMENT_FEATURES = (
    "viterbi_score",
    "path_mass_margin",
    "aligned_token_logprob",
    "token_peak_mean",
    "token_peak_min",
    "boundary_peak_mean",
    "viterbi_blank_ratio",
    "token_span_ratio",
    "frames_per_token",
    "token_duration_cv",
)


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


def forced_alignment_features(
        log_probs: torch.Tensor,
        input_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        blank_id: int,
        target_scores: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Extract batched Viterbi CTC alignment and per-token confidence features."""
    if log_probs.ndim != 3:
        raise ValueError("log_probs must have shape (batch, frames, classes)")
    if targets.ndim != 2 or targets.shape[0] != log_probs.shape[0]:
        raise ValueError("targets must have shape (batch, max_target_length)")
    batch_size, max_frames, _ = log_probs.shape
    if input_lengths.shape != (batch_size,):
        raise ValueError("input_lengths must have one value per batch item")
    if target_lengths.shape != (batch_size,):
        raise ValueError("target_lengths must have one value per batch item")
    if torch.any(input_lengths <= 0) or torch.any(input_lengths > max_frames):
        raise ValueError("input lengths are outside the available CTC frames")
    if torch.any(target_lengths <= 0) or torch.any(
            target_lengths > targets.shape[1]):
        raise ValueError("target lengths are invalid")
    required = required_ctc_frames(targets, target_lengths)
    if torch.any(input_lengths < required):
        raise ValueError("forced alignment requires CTC-valid targets")

    max_target_length = targets.shape[1]
    max_states = 2 * max_target_length + 1
    state_lengths = 2 * target_lengths + 1
    state_positions = torch.arange(max_states, device=log_probs.device)
    state_mask = state_positions.unsqueeze(0) < state_lengths.unsqueeze(1)
    extended = torch.full(
        (batch_size, max_states), blank_id,
        dtype=torch.long, device=targets.device)
    extended[:, 1::2] = targets

    negative_infinity = torch.tensor(
        float("-inf"), dtype=log_probs.dtype, device=log_probs.device)
    delta = torch.full(
        (batch_size, max_states), negative_infinity,
        dtype=log_probs.dtype, device=log_probs.device)
    first_emission = log_probs[:, 0].gather(1, extended)
    delta[:, 0] = first_emission[:, 0]
    delta[:, 1] = first_emission[:, 1]
    delta = delta.masked_fill(~state_mask, negative_infinity)

    skip_allowed = torch.zeros_like(state_mask)
    skip_allowed[:, 2:] = (
        (extended[:, 2:] != blank_id)
        & (extended[:, 2:] != extended[:, :-2])
        & state_mask[:, 2:]
    )
    backpointers = torch.zeros(
        (max_frames, batch_size, max_states),
        dtype=torch.int8, device=log_probs.device)
    for frame in range(1, max_frames):
        stay = delta
        advance = torch.cat([
            delta.new_full((batch_size, 1), negative_infinity),
            delta[:, :-1],
        ], dim=1)
        skip = torch.cat([
            delta.new_full((batch_size, 2), negative_infinity),
            delta[:, :-2],
        ], dim=1).masked_fill(~skip_allowed, negative_infinity)
        best, pointer = torch.stack(
            (stay, advance, skip), dim=-1).max(dim=-1)
        emission = log_probs[:, frame].gather(1, extended)
        updated = (best + emission).masked_fill(
            ~state_mask, negative_infinity)
        active = frame < input_lengths
        delta = torch.where(active.unsqueeze(1), updated, delta)
        backpointers[frame] = torch.where(
            active.unsqueeze(1), pointer.to(torch.int8),
            backpointers[frame])

    last_blank = 2 * target_lengths
    last_token = last_blank - 1
    blank_score = delta.gather(1, last_blank.unsqueeze(1)).squeeze(1)
    token_score = delta.gather(1, last_token.unsqueeze(1)).squeeze(1)
    choose_blank = blank_score >= token_score
    final_state = torch.where(choose_blank, last_blank, last_token)
    best_path_score = torch.where(choose_blank, blank_score, token_score)

    current_state = final_state
    reversed_states = []
    batch_positions = torch.arange(batch_size, device=log_probs.device)
    for frame in range(max_frames - 1, -1, -1):
        reversed_states.append(current_state)
        if frame > 0:
            pointer = backpointers[frame, batch_positions, current_state]
            active = frame < input_lengths
            current_state = current_state - torch.where(
                active, pointer.to(torch.long), torch.zeros_like(current_state))
    state_path = torch.stack(reversed_states[::-1], dim=1)

    frame_positions = torch.arange(max_frames, device=log_probs.device)
    frame_mask = frame_positions.unsqueeze(0) < input_lengths.unsqueeze(1)
    path_tokens = extended.gather(1, state_path)
    path_log_probs = log_probs.gather(
        2, path_tokens.unsqueeze(-1)).squeeze(-1)
    token_frame_mask = frame_mask & state_path.remainder(2).eq(1)

    token_positions = torch.arange(
        max_target_length, device=log_probs.device)
    token_mask = token_positions.unsqueeze(0) < target_lengths.unsqueeze(1)
    token_states = 2 * token_positions + 1
    aligned_frames = (
        state_path.unsqueeze(2) == token_states.view(1, 1, -1)
    ) & frame_mask.unsqueeze(2) & token_mask.unsqueeze(1)
    target_frame_log_probs = log_probs.gather(
        2,
        targets.unsqueeze(1).expand(-1, max_frames, -1),
    )
    token_peaks = target_frame_log_probs.masked_fill(
        ~aligned_frames, negative_infinity).amax(dim=1)
    token_peaks_masked = token_peaks.masked_fill(~token_mask, 0.0)
    token_peak_mean = token_peaks_masked.sum(dim=1) / target_lengths
    token_peak_min = token_peaks.masked_fill(
        ~token_mask, float("inf")).amin(dim=1)
    first_peak = token_peaks[:, 0]
    last_peak = token_peaks.gather(
        1, (target_lengths - 1).unsqueeze(1)).squeeze(1)

    aligned_token_sum = path_log_probs.masked_fill(
        ~token_frame_mask, 0.0).sum(dim=1)
    aligned_token_count = token_frame_mask.sum(dim=1).clamp(min=1)
    durations = aligned_frames.sum(dim=1).to(log_probs.dtype)
    duration_mean = durations.sum(dim=1) / target_lengths
    centered = (durations - duration_mean.unsqueeze(1)).masked_fill(
        ~token_mask, 0.0)
    duration_variance = centered.square().sum(dim=1) / target_lengths

    first_token_frame = torch.where(
        token_frame_mask, frame_positions.unsqueeze(0), max_frames).amin(dim=1)
    last_token_frame = torch.where(
        token_frame_mask, frame_positions.unsqueeze(0), -1).amax(dim=1)
    span = (last_token_frame - first_token_frame + 1).to(log_probs.dtype)
    normalized_viterbi = best_path_score / target_lengths
    if target_scores is None:
        target_scores = normalized_ctc_score(
            log_probs, input_lengths, targets, target_lengths, blank_id)

    return {
        "viterbi_score": normalized_viterbi,
        "path_mass_margin": target_scores - normalized_viterbi,
        "aligned_token_logprob": aligned_token_sum / aligned_token_count,
        "token_peak_mean": token_peak_mean,
        "token_peak_min": token_peak_min,
        "boundary_peak_mean": 0.5 * (first_peak + last_peak),
        "viterbi_blank_ratio": (
            frame_mask & state_path.remainder(2).eq(0)
        ).sum(dim=1).to(log_probs.dtype) / input_lengths,
        "token_span_ratio": span / input_lengths,
        "frames_per_token": input_lengths.to(log_probs.dtype) / target_lengths,
        "token_duration_cv": duration_variance.sqrt() /
        duration_mean.clamp(min=1e-6),
    }


def alignment_diagnostic_features(
        log_probs: torch.Tensor,
        input_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        blank_id: int) -> dict[str, torch.Tensor | list[list[int]]]:
    """Combine existing CTC diagnostics with forced-alignment features."""
    features = diagnostic_features(
        log_probs, input_lengths, targets, target_lengths, blank_id)
    features.update(forced_alignment_features(
        log_probs, input_lengths, targets, target_lengths, blank_id,
        target_scores=features["target_score"],
    ))
    return features
