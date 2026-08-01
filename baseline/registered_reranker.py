from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:
    from diagnose_registered_ctc import FEATURES
except ModuleNotFoundError:  # pragma: no cover - supports package imports
    from baseline.diagnose_registered_ctc import FEATURES


def rows_to_matrix(rows: Sequence[Mapping]) -> np.ndarray:
    matrix = np.asarray(
        [[float(row[name]) for name in FEATURES] for row in rows],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURES):
        raise ValueError("rows do not contain the registered CTC features")
    if not np.isfinite(matrix).all():
        raise ValueError("registered CTC features contain non-finite values")
    return matrix


def fit_reranker(rows: Sequence[Mapping], c: float = 1.0) -> dict:
    if c <= 0:
        raise ValueError("c must be positive")
    features = rows_to_matrix(rows)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    if len(features) == 0:
        raise ValueError("cannot fit a reranker with no rows")
    if not set(labels).issubset({0, 1}) or len(np.unique(labels)) < 2:
        raise ValueError("reranker training rows must contain both labels")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (features - mean) / scale
    classifier = LogisticRegression(
        C=c, class_weight="balanced", max_iter=1000, solver="lbfgs")
    classifier.fit(normalized, labels)
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
        raise ValueError("reranker feature names do not match this code")
    features = rows_to_matrix(rows)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    if (mean.shape != (len(FEATURES),)
            or scale.shape != mean.shape
            or coef.shape != mean.shape):
        raise ValueError("invalid reranker parameter shapes")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("reranker normalization parameters are not finite")
    if np.any(scale <= 0):
        raise ValueError("reranker scales must be positive")
    normalized = (features - mean) / scale
    return normalized @ coef + float(model["intercept"])


def subset_auc(rows: Sequence[Mapping], scores: np.ndarray, subset: str) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(rows):
        raise ValueError("scores must have one value per row")
    selected = np.asarray([
        index for index, row in enumerate(rows)
        if subset == "all" or row.get("subset") == subset
    ], dtype=np.int64)
    if len(selected) == 0:
        raise ValueError(f"no rows found for subset {subset!r}")
    labels = np.asarray([int(rows[index]["label"]) for index in selected])
    if len(np.unique(labels)) < 2:
        raise ValueError(f"subset {subset!r} needs both labels")
    return float(roc_auc_score(labels, scores[selected]))
