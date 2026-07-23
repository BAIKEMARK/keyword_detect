from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from config import PATHS, TRAIN
from ctc_data import load_ctc_score_pairs
from ctc_text import build_vocabulary, checkpoint_units, warm_vocabulary
from infer_wavlm_ctc import collect_scores
from runtime import select_device
from train_wavlm_ctc import make_score_loader
from wavlm_ctc_model import FrozenWavLMCTC, checkpoint_head_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _warm_dev_phonemes(vocabulary):
    texts = []
    for csv_path in (PATHS.dev_seen_csv, PATHS.dev_unseen_csv):
        texts.extend(
            pair["enroll_text"]
            for pair in load_ctc_score_pairs(csv_path, with_label=True)
        )
    return warm_vocabulary(vocabulary, texts)


def _auc(rows):
    labels = np.array([row[2] for row in rows], dtype=np.int64)
    scores = np.array([row[1] for row in rows], dtype=np.float64)
    return float(roc_auc_score(labels, scores))


def main():
    args = parse_args()
    device = select_device(args.device)
    if args.workers is None:
        args.workers = TRAIN.num_workers if device.type == "cuda" else 0
    amp_enabled = args.amp and device.type == "cuda"
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    units = checkpoint_units(checkpoint)
    vocabulary = build_vocabulary(units)
    if units == "phoneme":
        count = _warm_dev_phonemes(vocabulary)
        print(f"pronunciations: {count} unique")
    if tuple(checkpoint["vocabulary"]) != vocabulary.symbols:
        raise ValueError("checkpoint CTC vocabulary does not match code")

    model_id = args.model_id or checkpoint["model_id"]
    head_config = checkpoint_head_config(checkpoint)
    model = FrozenWavLMCTC(
        len(vocabulary), model_id, checkpoint["dropout"],
        **head_config).to(device)
    model.load_head_state_dict(checkpoint["head"])
    print(f"device: {device}")
    print(f"workers: {args.workers}")
    print(f"model: {model_id} (frozen)")
    print(f"units: {units}")
    print(f"head: {head_config['head_type']}")
    print(f"loaded {args.ckpt} (dev mean AUC={checkpoint.get('auc')})")

    seen_loader = make_score_loader(
        PATHS.dev_seen_zip, PATHS.dev_seen_csv, checkpoint["max_samples"],
        args.bs, args.workers, device, vocabulary, with_label=True)
    unseen_loader = make_score_loader(
        PATHS.dev_unseen_zip, PATHS.dev_unseen_csv, checkpoint["max_samples"],
        args.bs, args.workers, device, vocabulary, with_label=True)
    seen = collect_scores(
        model, seen_loader, device, amp_enabled, vocabulary.blank_id)
    unseen = collect_scores(
        model, unseen_loader, device, amp_enabled, vocabulary.blank_id)
    seen_auc = _auc(seen)
    unseen_auc = _auc(unseen)
    mean_auc = 0.5 * (seen_auc + unseen_auc)
    print(f"dev: seen={seen_auc:.4f} unseen={unseen_auc:.4f} "
          f"mean={mean_auc:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "subset", "score", "label"])
        writer.writerows(
            (f"seen_{pair_id}", "seen", score, label)
            for pair_id, score, label in seen)
        writer.writerows(
            (f"unseen_{pair_id}", "unseen", score, label)
            for pair_id, score, label in unseen)
    print(f"wrote {args.out} ({len(seen) + len(unseen)} rows)")


if __name__ == "__main__":
    main()
