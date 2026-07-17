from __future__ import annotations

import torch


def ctc_log_probability(log_probs: torch.Tensor,
                        input_lengths: torch.Tensor,
                        targets: torch.Tensor,
                        target_lengths: torch.Tensor,
                        blank_id: int = 0) -> torch.Tensor:
    """Exact CTC log probability for each item in a padded batch."""
    if log_probs.ndim != 3:
        raise ValueError("log_probs must have shape (batch, frames, classes)")
    if targets.ndim != 2:
        raise ValueError("targets must have shape (batch, max_target_length)")
    batch_size, max_frames, _ = log_probs.shape
    if targets.shape[0] != batch_size:
        raise ValueError("target and log-probability batch sizes differ")
    if torch.any(input_lengths <= 0) or torch.any(input_lengths > max_frames):
        raise ValueError("input lengths are outside the available CTC frames")
    if torch.any(target_lengths <= 0) or torch.any(
            target_lengths > targets.shape[1]):
        raise ValueError("target lengths are invalid")

    max_target_length = targets.shape[1]
    max_states = 2 * max_target_length + 1
    extended = torch.full(
        (batch_size, max_states),
        blank_id,
        dtype=torch.long,
        device=targets.device,
    )
    extended[:, 1::2] = targets
    state_lengths = 2 * target_lengths + 1
    state_positions = torch.arange(max_states, device=targets.device)
    state_mask = state_positions.unsqueeze(0) < state_lengths.unsqueeze(1)

    negative_infinity = torch.tensor(
        float("-inf"), dtype=log_probs.dtype, device=log_probs.device)
    alpha = torch.full(
        (batch_size, max_states),
        negative_infinity,
        dtype=log_probs.dtype,
        device=log_probs.device,
    )
    alpha[:, 0] = log_probs[:, 0, blank_id]
    first_targets = extended[:, 1]
    alpha[:, 1] = log_probs[:, 0].gather(
        1, first_targets.unsqueeze(1)).squeeze(1)
    alpha = alpha.masked_fill(~state_mask, negative_infinity)

    skip_allowed = torch.zeros_like(state_mask)
    skip_allowed[:, 2:] = (
        (extended[:, 2:] != blank_id)
        & (extended[:, 2:] != extended[:, :-2])
        & state_mask[:, 2:]
    )

    for frame in range(1, max_frames):
        stay = alpha
        advance = torch.cat([
            alpha.new_full((batch_size, 1), negative_infinity),
            alpha[:, :-1],
        ], dim=1)
        skip = torch.cat([
            alpha.new_full((batch_size, 2), negative_infinity),
            alpha[:, :-2],
        ], dim=1)
        skip = skip.masked_fill(~skip_allowed, negative_infinity)
        transition = torch.logsumexp(
            torch.stack([stay, advance, skip], dim=0), dim=0)
        emission = log_probs[:, frame].gather(1, extended)
        updated = (transition + emission).masked_fill(
            ~state_mask, negative_infinity)
        active = frame < input_lengths
        alpha = torch.where(active.unsqueeze(1), updated, alpha)

    last_blank = (2 * target_lengths).unsqueeze(1)
    last_symbol = (2 * target_lengths - 1).unsqueeze(1)
    blank_score = alpha.gather(1, last_blank).squeeze(1)
    symbol_score = alpha.gather(1, last_symbol).squeeze(1)
    return torch.logaddexp(blank_score, symbol_score)


def normalized_ctc_score(log_probs: torch.Tensor,
                         input_lengths: torch.Tensor,
                         targets: torch.Tensor,
                         target_lengths: torch.Tensor,
                         blank_id: int = 0) -> torch.Tensor:
    score = ctc_log_probability(
        log_probs, input_lengths, targets, target_lengths, blank_id)
    return score / target_lengths.to(score.dtype)
