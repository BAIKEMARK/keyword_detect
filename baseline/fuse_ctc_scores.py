from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class ScoreRow:
    pair_id: str
    subset: str
    score: float
    label: Optional[int]


@dataclass(frozen=True)
class AlignedRow:
    pair_id: str
    subset: str
    character_score: float
    phoneme_score: float
    label: Optional[int]


@dataclass(frozen=True)
class FusionResult:
    weight: float
    seen_auc: float
    unseen_auc: float
    mean_auc: float
    character_seen_auc: float
    character_unseen_auc: float
    character_mean_auc: float
    phoneme_seen_auc: float
    phoneme_unseen_auc: float
    phoneme_mean_auc: float


def _subset_from_id(pair_id: str) -> str:
    if pair_id.startswith("seen_"):
        return "seen"
    if pair_id.startswith("unseen_"):
        return "unseen"
    raise ValueError(f"unknown pair id prefix: {pair_id!r}")


def _official_key(pair_id: str):
    subset = _subset_from_id(pair_id)
    return (0 if subset == "seen" else 1, pair_id)


def _parse_score(value: str, pair_id: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid score for {pair_id!r}: {value!r}") from exc
    if not math.isfinite(score):
        raise ValueError(f"non-finite score for {pair_id!r}: {value!r}")
    return score


def _insert_unique(rows: Dict[str, ScoreRow], row: ScoreRow):
    if row.pair_id in rows:
        raise ValueError(f"duplicate pair id: {row.pair_id!r}")
    rows[row.pair_id] = row


def read_dev_scores(path: str) -> Dict[str, ScoreRow]:
    rows: Dict[str, ScoreRow] = {}
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"id", "subset", "score", "label"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"dev score CSV is missing columns: {path}")
        for item in reader:
            pair_id = item["id"]
            subset = item["subset"]
            if subset != _subset_from_id(pair_id):
                raise ValueError(f"subset does not match id: {pair_id!r}")
            try:
                label = int(item["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid label for {pair_id!r}: {item['label']!r}") \
                    from exc
            if label not in {0, 1}:
                raise ValueError(f"label must be 0 or 1 for {pair_id!r}")
            _insert_unique(rows, ScoreRow(
                pair_id, subset, _parse_score(item["score"], pair_id), label))
    if not rows:
        raise ValueError(f"dev score CSV is empty: {path}")
    return rows


def read_eval_scores(path: str) -> Dict[str, ScoreRow]:
    rows: Dict[str, ScoreRow] = {}
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or ())
        if "id" not in fields:
            raise ValueError(f"eval score CSV is missing id: {path}")
        if "posterior" in fields:
            value_field = "posterior"
        elif "score" in fields:
            value_field = "score"
        else:
            raise ValueError(
                f"eval score CSV needs posterior or score: {path}")
        for item in reader:
            pair_id = item["id"]
            subset = _subset_from_id(pair_id)
            _insert_unique(rows, ScoreRow(
                pair_id, subset,
                _parse_score(item[value_field], pair_id), None))
    if not rows:
        raise ValueError(f"eval score CSV is empty: {path}")
    return rows


def _check_same_ids(character, phoneme):
    character_ids = set(character)
    phoneme_ids = set(phoneme)
    if character_ids != phoneme_ids:
        missing = sorted(character_ids - phoneme_ids)[:3]
        extra = sorted(phoneme_ids - character_ids)[:3]
        raise ValueError(
            "character and phoneme ids differ: "
            f"missing_in_phoneme={missing}, extra_in_phoneme={extra}")


def align_dev_rows(character: Dict[str, ScoreRow],
                   phoneme: Dict[str, ScoreRow]) -> List[AlignedRow]:
    _check_same_ids(character, phoneme)
    output = []
    for pair_id in sorted(character, key=_official_key):
        char_row = character[pair_id]
        phone_row = phoneme[pair_id]
        if char_row.subset != phone_row.subset:
            raise ValueError(f"subset disagreement for {pair_id!r}")
        if char_row.label != phone_row.label:
            raise ValueError(f"label disagreement for {pair_id!r}")
        output.append(AlignedRow(
            pair_id, char_row.subset, char_row.score, phone_row.score,
            char_row.label))
    return output


def align_eval_rows(character: Dict[str, ScoreRow],
                    phoneme: Dict[str, ScoreRow]) -> List[AlignedRow]:
    _check_same_ids(character, phoneme)
    output = []
    for pair_id in sorted(character, key=_official_key):
        char_row = character[pair_id]
        phone_row = phoneme[pair_id]
        if char_row.subset != phone_row.subset:
            raise ValueError(f"subset disagreement for {pair_id!r}")
        output.append(AlignedRow(
            pair_id, char_row.subset, char_row.score, phone_row.score, None))
    return output


def average_percentile_ranks(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    if len(scores) == 1:
        return np.array([0.5], dtype=np.float64)

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks / (len(scores) - 1)


def _rank_within_subsets(scores: np.ndarray,
                         subsets: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    subsets = np.asarray(subsets)
    if scores.shape != subsets.shape:
        raise ValueError("scores and subsets must have the same shape")
    unknown = set(np.unique(subsets)) - {"seen", "unseen"}
    if unknown:
        raise ValueError(f"unknown subsets: {sorted(unknown)!r}")
    ranked = np.empty_like(scores)
    for subset in ("seen", "unseen"):
        mask = subsets == subset
        if not mask.any():
            raise ValueError(f"missing subset: {subset}")
        ranked[mask] = average_percentile_ranks(scores[mask])
    return ranked


def _subset_aucs(scores: np.ndarray, labels: np.ndarray,
                 subsets: np.ndarray) -> Tuple[float, float, float]:
    values = []
    for subset in ("seen", "unseen"):
        mask = subsets == subset
        if len(np.unique(labels[mask])) != 2:
            raise ValueError(f"subset {subset!r} needs both labels")
        values.append(float(roc_auc_score(labels[mask], scores[mask])))
    return values[0], values[1], 0.5 * (values[0] + values[1])


def search_phoneme_weight(character_scores: np.ndarray,
                          phoneme_scores: np.ndarray,
                          labels: np.ndarray,
                          subsets: np.ndarray,
                          step: float = 0.001) -> FusionResult:
    if not 0.0 < step <= 1.0:
        raise ValueError("step must be in (0, 1]")
    character = _rank_within_subsets(character_scores, subsets)
    phoneme = _rank_within_subsets(phoneme_scores, subsets)
    labels = np.asarray(labels, dtype=np.int64)
    subsets = np.asarray(subsets)
    if character.shape != phoneme.shape or character.shape != labels.shape:
        raise ValueError("score and label arrays must have the same shape")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be 0 or 1")

    char_aucs = _subset_aucs(character, labels, subsets)
    phone_aucs = _subset_aucs(phoneme, labels, subsets)
    steps = int(round(1.0 / step))
    weights = np.linspace(0.0, 1.0, steps + 1)
    best_weight = -1.0
    best_aucs = (-1.0, -1.0, -1.0)
    for weight in weights:
        fused = (1.0 - weight) * character + weight * phoneme
        aucs = _subset_aucs(fused, labels, subsets)
        if (aucs[2] > best_aucs[2] + 1e-12
                or (abs(aucs[2] - best_aucs[2]) <= 1e-12
                    and weight > best_weight)):
            best_weight = float(weight)
            best_aucs = aucs

    return FusionResult(
        best_weight,
        best_aucs[0], best_aucs[1], best_aucs[2],
        char_aucs[0], char_aucs[1], char_aucs[2],
        phone_aucs[0], phone_aucs[1], phone_aucs[2],
    )


def _arrays(rows: Sequence[AlignedRow]):
    character = np.array(
        [row.character_score for row in rows], dtype=np.float64)
    phoneme = np.array(
        [row.phoneme_score for row in rows], dtype=np.float64)
    subsets = np.array([row.subset for row in rows])
    return character, phoneme, subsets


def fuse_eval_rows(rows: Sequence[AlignedRow],
                   phoneme_weight: float) -> List[Tuple[str, float]]:
    if not 0.0 <= phoneme_weight <= 1.0:
        raise ValueError("phoneme weight must be in [0, 1]")
    character, phoneme, subsets = _arrays(rows)
    character = _rank_within_subsets(character, subsets)
    phoneme = _rank_within_subsets(phoneme, subsets)
    fused = (1.0 - phoneme_weight) * character + phoneme_weight * phoneme
    if not np.logical_and(fused >= 0.0, fused <= 1.0).all():
        raise RuntimeError("fused posteriors are outside [0, 1]")
    return [
        (row.pair_id, float(posterior))
        for row, posterior in zip(rows, fused)
    ]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char-dev", required=True)
    parser.add_argument("--phoneme-dev", required=True)
    parser.add_argument("--char-eval", required=True)
    parser.add_argument("--phoneme-eval", required=True)
    parser.add_argument("--out", default="submission_ctc_fusion.csv")
    parser.add_argument("--report", default=None)
    parser.add_argument("--weight-step", type=float, default=0.001)
    return parser.parse_args()


def main():
    args = _parse_args()
    dev_rows = align_dev_rows(
        read_dev_scores(args.char_dev), read_dev_scores(args.phoneme_dev))
    character, phoneme, subsets = _arrays(dev_rows)
    labels = np.array([row.label for row in dev_rows], dtype=np.int64)
    result = search_phoneme_weight(
        character, phoneme, labels, subsets, args.weight_step)

    print(f"character dev: seen={result.character_seen_auc:.4f} "
          f"unseen={result.character_unseen_auc:.4f} "
          f"mean={result.character_mean_auc:.4f}")
    print(f"phoneme dev:   seen={result.phoneme_seen_auc:.4f} "
          f"unseen={result.phoneme_unseen_auc:.4f} "
          f"mean={result.phoneme_mean_auc:.4f}")
    print(f"fusion dev:    seen={result.seen_auc:.4f} "
          f"unseen={result.unseen_auc:.4f} mean={result.mean_auc:.4f} "
          f"phoneme_weight={result.weight:.3f}")

    eval_rows = align_eval_rows(
        read_eval_scores(args.char_eval), read_eval_scores(args.phoneme_eval))
    fused_rows = fuse_eval_rows(eval_rows, result.weight)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "posterior"])
        writer.writerows(fused_rows)
    print(f"wrote {args.out} ({len(fused_rows)} rows)")

    report_path = args.report or f"{args.out}.json"
    report = {
        "fusion": asdict(result),
        "sources": {
            "character_dev": args.char_dev,
            "phoneme_dev": args.phoneme_dev,
            "character_eval": args.char_eval,
            "phoneme_eval": args.phoneme_eval,
        },
        "submission": args.out,
        "rows": len(fused_rows),
    }
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=True, indent=2, sort_keys=True)
        file.write("\n")
    print(f"wrote {report_path}")
    if result.mean_auc <= result.phoneme_mean_auc:
        print("warning: fusion does not improve phoneme-only dev AUC")


if __name__ == "__main__":
    main()
