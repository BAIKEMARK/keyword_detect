from __future__ import annotations

import argparse
import csv
import os
import pickle

import numpy as np
import torch

from config import PATHS, TRAIN
from ctc_data import load_ctc_score_pairs
from ctc_text import build_vocabulary, checkpoint_units, warm_vocabulary
from diagnose_registered_ctc import collect_features, make_loader
from registered_reranker import FEATURES, decision_scores
from runtime import select_device
from wavlm_ctc_model import (FrozenWavLMCTC, checkpoint_backbone_type,
                             checkpoint_head_config)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--reranker", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def _check_file(description, path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def _load_reranker(path):
    with open(path, "rb") as file:
        model = pickle.load(file)
    if tuple(model.get("feature_names", ())) != tuple(FEATURES):
        raise ValueError("reranker feature names do not match this code")
    return model


def _warm_eval_vocabulary(vocabulary):
    texts = []
    for csv_path in (PATHS.eval_seen_csv, PATHS.eval_unseen_csv):
        texts.extend(
            pair["enroll_text"] for pair in load_ctc_score_pairs(
                csv_path, with_label=False))
    return warm_vocabulary(vocabulary, texts)


def _posterior(scores):
    scores = np.asarray(scores, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -60.0, 60.0)))


def main(argv=None):
    args = parse_args(argv)
    device = select_device(args.device)
    if args.workers is None:
        args.workers = TRAIN.num_workers if device.type == "cuda" else 0
    amp_enabled = args.amp and device.type == "cuda"

    _check_file("checkpoint", args.ckpt)
    _check_file("reranker", args.reranker)
    _check_file("eval seen CSV", PATHS.eval_seen_csv)
    _check_file("eval unseen CSV", PATHS.eval_unseen_csv)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    reranker = _load_reranker(args.reranker)

    vocabulary = build_vocabulary(checkpoint_units(checkpoint))
    if checkpoint_units(checkpoint) == "phoneme":
        count = _warm_eval_vocabulary(vocabulary)
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
    print(f"amp: {amp_enabled}", flush=True)

    output_rows = []
    for subset, zip_path, csv_path in (
            ("seen", PATHS.eval_seen_zip, PATHS.eval_seen_csv),
            ("unseen", PATHS.eval_unseen_zip, PATHS.eval_unseen_csv)):
        pairs = load_ctc_score_pairs(csv_path, with_label=False)
        loader = make_loader(
            pairs, zip_path, checkpoint["max_samples"], args.bs, args.workers,
            device, vocabulary)
        rows = collect_features(
            model, loader, device, amp_enabled, vocabulary)
        scores = decision_scores(rows, reranker)
        posteriors = _posterior(scores)
        output_rows.extend(
            (f"{subset}_{row['id']}", float(posterior))
            for row, posterior in zip(rows, posteriors)
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "posterior"])
        writer.writerows(output_rows)
    print(f"wrote {args.out} ({len(output_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
