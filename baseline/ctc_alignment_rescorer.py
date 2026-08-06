from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ctc_diagnostics import (FORCED_ALIGNMENT_FEATURES,
                             alignment_diagnostic_features)


FEATURES = (
    "target_score",
    "greedy_score",
    "likelihood_margin",
    "edit_similarity",
    "frame_confidence",
    "blank_ratio",
    "target_length",
    "greedy_length",
    *FORCED_ALIGNMENT_FEATURES,
)


@torch.no_grad()
def collect_alignment_rows(model, loader, device, amp_enabled, blank_id):
    rows = []
    model.eval()
    for batch in loader:
        (waveforms, sample_lengths, targets, target_lengths,
         labels, pair_ids) = batch
        waveforms = waveforms.to(device, non_blocking=True)
        sample_lengths = sample_lengths.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=amp_enabled):
            log_probs, output_lengths = model.log_probs(
                waveforms, sample_lengths)
        features = alignment_diagnostic_features(
            log_probs.float(), output_lengths, targets, target_lengths,
            blank_id)
        for index, (pair_id, label) in enumerate(
                zip(pair_ids, labels.tolist())):
            rows.append({
                "id": pair_id,
                "label": int(label),
                **{
                    name: float(features[name][index].item())
                    for name in FEATURES
                },
            })
    return rows


def rows_to_matrix(rows: Sequence[Mapping]) -> np.ndarray:
    matrix = np.asarray(
        [[float(row[name]) for name in FEATURES] for row in rows],
        dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURES):
        raise ValueError("rows do not contain the CTC alignment features")
    if not np.isfinite(matrix).all():
        raise ValueError("CTC alignment features contain non-finite values")
    return matrix


def fit_rescorer(rows: Sequence[Mapping], c: float = 1.0) -> dict:
    if c <= 0:
        raise ValueError("c must be positive")
    features = rows_to_matrix(rows)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    if len(features) == 0:
        raise ValueError("cannot fit a rescorer with no rows")
    if not set(labels).issubset({0, 1}) or len(np.unique(labels)) < 2:
        raise ValueError("rescorer rows must contain both labels")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    classifier = LogisticRegression(
        C=c, class_weight="balanced", max_iter=1000, solver="lbfgs")
    classifier.fit((features - mean) / scale, labels)
    return {
        "feature_names": tuple(FEATURES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "c": c,
    }


def decision_scores(rows: Sequence[Mapping], model: Mapping) -> np.ndarray:
    if tuple(model.get("feature_names", ())) != tuple(FEATURES):
        raise ValueError("rescorer feature names do not match this code")
    features = rows_to_matrix(rows)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coefficients = np.asarray(model["coef"], dtype=np.float64)
    expected_shape = (len(FEATURES),)
    if mean.shape != expected_shape or scale.shape != expected_shape \
            or coefficients.shape != expected_shape:
        raise ValueError("invalid rescorer parameter shapes")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() \
            or np.any(scale <= 0):
        raise ValueError("invalid rescorer normalization parameters")
    return ((features - mean) / scale) @ coefficients + float(
        model["intercept"])


def cross_validated_scores(rows: Sequence[Mapping], folds: int = 5,
                           seed: int = 42, c: float = 1.0) -> np.ndarray:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    strata = np.asarray([
        f"{row.get('subset', 'all')}:{int(row['label'])}" for row in rows
    ])
    _, counts = np.unique(strata, return_counts=True)
    if len(counts) == 0 or int(counts.min()) < folds:
        raise ValueError("each subset/label stratum needs at least folds rows")
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=seed)
    scores = np.empty(len(rows), dtype=np.float64)
    indices = np.arange(len(rows))
    for train_indices, validation_indices in splitter.split(indices, strata):
        model = fit_rescorer([rows[index] for index in train_indices], c=c)
        scores[validation_indices] = decision_scores(
            [rows[index] for index in validation_indices], model)
    if not np.isfinite(scores).all() or len(scores) != len(labels):
        raise RuntimeError("cross validation produced invalid scores")
    return scores


def subset_auc(rows: Sequence[Mapping], scores: np.ndarray,
               subset: str) -> float:
    selected = [
        index for index, row in enumerate(rows)
        if subset == "all" or row.get("subset") == subset
    ]
    if not selected:
        raise ValueError(f"no rows found for subset {subset!r}")
    labels = np.asarray([int(rows[index]["label"]) for index in selected])
    selected_scores = np.asarray(scores, dtype=np.float64)[selected]
    if len(np.unique(labels)) < 2:
        raise ValueError(f"subset {subset!r} needs both labels")
    return float(roc_auc_score(labels, selected_scores))


def sigmoid_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -60.0, 60.0)))
