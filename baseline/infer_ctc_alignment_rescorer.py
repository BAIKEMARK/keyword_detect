from __future__ import annotations

import argparse
import csv
import os
import pickle

import torch

from config import PATHS, TRAIN
from ctc_alignment_rescorer import (FEATURES, collect_alignment_rows,
                                    decision_scores, sigmoid_scores)
from ctc_data import load_ctc_score_pairs
from ctc_text import build_vocabulary, checkpoint_units, warm_vocabulary
from runtime import select_device
from train_wavlm_ctc import make_score_loader
from wavlm_ctc_model import (FrozenWavLMCTC, checkpoint_backbone_type,
                             checkpoint_head_config)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Apply a CTC forced-alignment rescorer to eval")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--rescorer", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def _check_file(description, path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def _warm_eval_vocabulary(vocabulary):
    texts = []
    for path in (PATHS.eval_seen_csv, PATHS.eval_unseen_csv):
        texts.extend(
            pair["enroll_text"] for pair in load_ctc_score_pairs(
                path, with_label=False))
    return warm_vocabulary(vocabulary, texts)


def main(argv=None):
    args = parse_args(argv)
    device = select_device(args.device)
    if args.workers is None:
        args.workers = TRAIN.num_workers if device.type == "cuda" else 0
    amp_enabled = args.amp and device.type == "cuda"
    _check_file("checkpoint", args.ckpt)
    _check_file("rescorer", args.rescorer)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    with open(args.rescorer, "rb") as file:
        rescorer = pickle.load(file)
    if tuple(rescorer.get("feature_names", ())) != tuple(FEATURES):
        raise ValueError("rescorer feature names do not match this code")

    vocabulary = build_vocabulary(checkpoint_units(checkpoint))
    if checkpoint_units(checkpoint) == "phoneme":
        count = _warm_eval_vocabulary(vocabulary)
        print(f"pronunciations: {count} unique", flush=True)
    if tuple(checkpoint["vocabulary"]) != vocabulary.symbols:
        raise ValueError("checkpoint CTC vocabulary does not match code")

    model_id = args.model_id or checkpoint["model_id"]
    expected_model_id = rescorer.get("metadata", {}).get("model_id")
    if expected_model_id and os.path.abspath(str(expected_model_id)) != \
            os.path.abspath(str(model_id)):
        raise ValueError(
            f"rescorer model mismatch: expected {expected_model_id}, "
            f"got {model_id}")
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

    output_rows = []
    for subset, zip_path, csv_path in (
            ("seen", PATHS.eval_seen_zip, PATHS.eval_seen_csv),
            ("unseen", PATHS.eval_unseen_zip, PATHS.eval_unseen_csv)):
        loader = make_score_loader(
            zip_path, csv_path, checkpoint["max_samples"], args.bs,
            args.workers, device, vocabulary, with_label=False)
        rows = collect_alignment_rows(
            model, loader, device, amp_enabled, vocabulary.blank_id)
        posteriors = sigmoid_scores(decision_scores(rows, rescorer))
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
