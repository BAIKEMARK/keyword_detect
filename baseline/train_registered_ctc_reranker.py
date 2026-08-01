from __future__ import annotations

import argparse
import csv
import os
import pickle

import numpy as np
import torch

from config import PATHS, TRAIN
from ctc_text import build_vocabulary, checkpoint_units, warm_vocabulary
from diagnose_registered_ctc import collect_features, make_loader
from registered_reranker import FEATURES, decision_scores, fit_reranker, subset_auc
from runtime import select_device
from wavlm_ctc_model import (FrozenWavLMCTC, checkpoint_backbone_type,
                             checkpoint_head_config)
from ctc_data import load_ctc_score_pairs


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--train-csv", default=PATHS.train_csv)
    parser.add_argument("--train-zip", default=PATHS.train_zip)
    parser.add_argument(
        "--subset", type=int, default=None,
        help="number of training pairs; omit to use all pairs")
    parser.add_argument("--seed", type=int, default=TRAIN.seed)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dev", required=True)
    parser.add_argument("--out-model", required=True)
    return parser.parse_args(argv)


def _check_file(description, path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def _load_pairs(path):
    return load_ctc_score_pairs(path, with_label=True)


def _select_pairs(pairs, subset, seed):
    if subset is None:
        return pairs
    if subset <= 0:
        raise ValueError("--subset must be positive when provided")
    if subset >= len(pairs):
        return pairs
    indices = np.random.default_rng(seed).permutation(len(pairs))[:subset]
    return [pairs[index] for index in indices]


def _warm_vocabulary(vocabulary, paths):
    texts = []
    for path in paths:
        texts.extend(
            pair["enroll_text"] for pair in load_ctc_score_pairs(
                path, with_label=True))
    return warm_vocabulary(vocabulary, texts)


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


def _write_model(path, model, metadata):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = dict(model)
    payload["metadata"] = metadata
    with open(path, "wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def _print_auc(prefix, rows, scores, subsets=("all",)):
    values = [
        (subset, subset_auc(rows, scores, subset))
        for subset in subsets
    ]
    print(
        f"{prefix}: "
        + " ".join(f"{subset}={value:.4f}" for subset, value in values),
        flush=True,
    )


def main(argv=None):
    args = parse_args(argv)
    torch_device = select_device(args.device)
    if args.workers is None:
        args.workers = TRAIN.num_workers if torch_device.type == "cuda" else 0
    amp_enabled = args.amp and torch_device.type == "cuda"

    _check_file("checkpoint", args.ckpt)
    _check_file("training CSV", args.train_csv)
    _check_file("training wav ZIP", args.train_zip)
    _check_file("dev seen CSV", PATHS.dev_seen_csv)
    _check_file("dev unseen CSV", PATHS.dev_unseen_csv)

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vocabulary = build_vocabulary(checkpoint_units(checkpoint))
    if checkpoint_units(checkpoint) == "phoneme":
        count = _warm_vocabulary(
            vocabulary,
            (args.train_csv, PATHS.dev_seen_csv, PATHS.dev_unseen_csv),
        )
        print(f"pronunciations: {count} unique", flush=True)
    if tuple(checkpoint["vocabulary"]) != vocabulary.symbols:
        raise ValueError("checkpoint CTC vocabulary does not match code")

    all_train_pairs = _load_pairs(args.train_csv)
    train_pairs = _select_pairs(all_train_pairs, args.subset, args.seed)
    model_id = args.model_id or checkpoint["model_id"]
    model = FrozenWavLMCTC(
        len(vocabulary), model_id, checkpoint["dropout"],
        backbone_type=checkpoint_backbone_type(checkpoint),
        **checkpoint_head_config(checkpoint),
    ).to(torch_device)
    model.load_head_state_dict(checkpoint["head"])
    model.eval()

    print(f"device: {torch_device}", flush=True)
    print(f"workers: {args.workers}", flush=True)
    print(f"model: {model_id} (frozen)", flush=True)
    print(f"backbone: {model.backbone_type}", flush=True)
    print(f"head: {checkpoint_head_config(checkpoint)['head_type']}", flush=True)
    print(f"train pairs: {len(train_pairs)} / {len(all_train_pairs)}", flush=True)
    print(f"amp: {amp_enabled}", flush=True)

    train_loader = make_loader(
        train_pairs, args.train_zip, checkpoint["max_samples"], args.bs,
        args.workers, torch_device, vocabulary)
    train_rows = collect_features(
        model, train_loader, torch_device, amp_enabled, vocabulary)
    for row in train_rows:
        row["subset"] = "train"

    reranker = fit_reranker(train_rows, c=args.c)
    train_scores = decision_scores(train_rows, reranker)
    _print_auc("train reranker", train_rows, train_scores)

    dev_rows = []
    for subset, zip_path, csv_path in (
            ("seen", PATHS.dev_seen_zip, PATHS.dev_seen_csv),
            ("unseen", PATHS.dev_unseen_zip, PATHS.dev_unseen_csv)):
        pairs = _load_pairs(csv_path)
        loader = make_loader(
            pairs, zip_path, checkpoint["max_samples"], args.bs, args.workers,
            torch_device, vocabulary)
        rows = collect_features(
            model, loader, torch_device, amp_enabled, vocabulary)
        for row in rows:
            row["subset"] = subset
        dev_rows.extend(rows)

    dev_scores = decision_scores(dev_rows, reranker)
    _print_auc("dev reranker", dev_rows, dev_scores)
    _write_dev_scores(args.out_dev, dev_rows, dev_scores)
    _write_model(
        args.out_model,
        reranker,
        {
            "checkpoint": args.ckpt,
            "model_id": model_id,
            "train_csv": args.train_csv,
            "train_zip": args.train_zip,
            "train_pairs": len(train_rows),
            "c": args.c,
            "features": list(FEATURES),
        },
    )
    print(f"wrote {args.out_dev} ({len(dev_rows)} rows)", flush=True)
    print(f"wrote {args.out_model}", flush=True)


if __name__ == "__main__":
    main()
