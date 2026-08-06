from __future__ import annotations

import argparse
import csv
import os
import pickle

import numpy as np
import torch

from config import PATHS, TRAIN
from ctc_alignment_rescorer import (FEATURES, collect_alignment_rows,
                                    cross_validated_scores, decision_scores,
                                    fit_rescorer, subset_auc)
from ctc_data import load_ctc_score_pairs
from ctc_text import build_vocabulary, checkpoint_units, warm_vocabulary
from runtime import select_device
from train_wavlm_ctc import make_score_loader
from wavlm_ctc_model import (FrozenWavLMCTC, checkpoint_backbone_type,
                             checkpoint_head_config)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit a leakage-safe CTC forced-alignment rescorer on dev")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=TRAIN.seed)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dev", required=True)
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--out-features", default=None)
    return parser.parse_args(argv)


def _check_file(description, path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def _warm_dev_vocabulary(vocabulary):
    texts = []
    for path in (PATHS.dev_seen_csv, PATHS.dev_unseen_csv):
        texts.extend(
            pair["enroll_text"] for pair in load_ctc_score_pairs(
                path, with_label=True))
    return warm_vocabulary(vocabulary, texts)


def _report(prefix, rows, scores):
    seen = subset_auc(rows, scores, "seen")
    unseen = subset_auc(rows, scores, "unseen")
    print(
        f"{prefix}: seen={seen:.4f} unseen={unseen:.4f} "
        f"mean={0.5 * (seen + unseen):.4f}", flush=True)


def _write_dev_scores(path, rows, scores):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "subset", "score", "label"])
        writer.writerows(
            (f"{row['subset']}_{row['id']}", row["subset"],
             float(score), int(row["label"]))
            for row, score in zip(rows, scores)
        )


def _write_features(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        fields = ["id", "subset", "label", *FEATURES]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: row[name] for name in fields} for row in rows)


def main(argv=None):
    args = parse_args(argv)
    device = select_device(args.device)
    if args.workers is None:
        args.workers = TRAIN.num_workers if device.type == "cuda" else 0
    amp_enabled = args.amp and device.type == "cuda"
    _check_file("checkpoint", args.ckpt)
    _check_file("dev seen CSV", PATHS.dev_seen_csv)
    _check_file("dev unseen CSV", PATHS.dev_unseen_csv)

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vocabulary = build_vocabulary(checkpoint_units(checkpoint))
    if checkpoint_units(checkpoint) == "phoneme":
        count = _warm_dev_vocabulary(vocabulary)
        print(f"pronunciations: {count} unique", flush=True)
    if tuple(checkpoint["vocabulary"]) != vocabulary.symbols:
        raise ValueError("checkpoint CTC vocabulary does not match code")

    model_id = args.model_id or checkpoint["model_id"]
    model = FrozenWavLMCTC(
        len(vocabulary), model_id, checkpoint["dropout"],
        backbone_type=checkpoint_backbone_type(checkpoint),
        **checkpoint_head_config(checkpoint),
    ).to(device)
    model.load_head_state_dict(checkpoint["head"])
    model.eval()
    print(f"device: {device}", flush=True)
    print(f"workers: {args.workers}", flush=True)
    print(f"model: {model_id} (frozen)", flush=True)
    print(f"backbone: {model.backbone_type}", flush=True)
    print(f"folds: {args.folds} seed={args.seed} C={args.c}", flush=True)

    rows = []
    for subset, zip_path, csv_path in (
            ("seen", PATHS.dev_seen_zip, PATHS.dev_seen_csv),
            ("unseen", PATHS.dev_unseen_zip, PATHS.dev_unseen_csv)):
        loader = make_score_loader(
            zip_path, csv_path, checkpoint["max_samples"], args.bs,
            args.workers, device, vocabulary, with_label=True)
        subset_rows = collect_alignment_rows(
            model, loader, device, amp_enabled, vocabulary.blank_id)
        for row in subset_rows:
            row["subset"] = subset
        rows.extend(subset_rows)

    target_scores = np.asarray(
        [row["target_score"] for row in rows], dtype=np.float64)
    _report("target CTC", rows, target_scores)
    for name in FEATURES:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        _report(f"feature {name}", rows, values)

    oof_scores = cross_validated_scores(
        rows, folds=args.folds, seed=args.seed, c=args.c)
    _report("OOF alignment rescorer", rows, oof_scores)
    final_model = fit_rescorer(rows, c=args.c)
    final_scores = decision_scores(rows, final_model)
    _report("in-sample alignment rescorer", rows, final_scores)

    _write_dev_scores(args.out_dev, rows, oof_scores)
    if args.out_features:
        _write_features(args.out_features, rows)
        print(f"wrote {args.out_features} ({len(rows)} rows)", flush=True)
    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    final_model["metadata"] = {
        "checkpoint": args.ckpt,
        "model_id": model_id,
        "folds": args.folds,
        "seed": args.seed,
        "rows": len(rows),
        "features": list(FEATURES),
    }
    with open(args.out_model, "wb") as file:
        pickle.dump(final_model, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {args.out_dev} ({len(rows)} OOF rows)", flush=True)
    print(f"wrote {args.out_model}", flush=True)


if __name__ == "__main__":
    main()
