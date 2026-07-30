from __future__ import annotations

import argparse
import csv
import os

import torch
from sklearn.metrics import roc_auc_score

from config import PATHS, TRAIN
from ctc_data import load_ctc_score_pairs
from ctc_diagnostics import diagnostic_features
from ctc_text import build_vocabulary, checkpoint_units, warm_vocabulary
from runtime import select_device
from train_wavlm_ctc import ctc_valid_mask, make_score_loader
from wavlm_ctc_model import FrozenWavLMCTC, checkpoint_head_config


FEATURES = (
    "target_score", "greedy_score", "likelihood_margin",
    "edit_similarity", "frame_confidence", "blank_ratio",
    "target_length", "greedy_length",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Diagnose CTC greedy and likelihood-ratio features without retraining")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


@torch.no_grad()
def collect_diagnostics(model, loader, device, amp_enabled, blank_id,
                        symbols):
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
        features = diagnostic_features(
            log_probs.float(), output_lengths, targets, target_lengths,
            blank_id)
        decoded_text = [
            " ".join(symbols[token] for token in sequence)
            for sequence in features["decoded"]
        ]
        for index, (pair_id, label) in enumerate(zip(pair_ids, labels.tolist())):
            rows.append({
                "id": pair_id,
                "label": int(label),
                **{name: float(features[name][index].item()) for name in FEATURES},
                "decoded": decoded_text[index],
            })
    return rows


def report(rows):
    for subset in ("all", "seen", "unseen"):
        selected = rows if subset == "all" else [
            row for row in rows if row["subset"] == subset]
        labels = [row["label"] for row in selected]
        print(f"[{subset}] rows={len(selected)}", flush=True)
        for feature in FEATURES:
            auc = roc_auc_score(labels, [row[feature] for row in selected])
            print(f"  {feature}: {auc:.4f}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    device = select_device(args.device)
    if args.workers is None:
        args.workers = TRAIN.num_workers if device.type == "cuda" else 0
    amp_enabled = args.amp and device.type == "cuda"
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    units = checkpoint_units(checkpoint)
    vocabulary = build_vocabulary(units)
    if units == "phoneme":
        texts = []
        for path in (PATHS.dev_seen_csv, PATHS.dev_unseen_csv):
            texts.extend(
                pair["enroll_text"]
                for pair in load_ctc_score_pairs(path, with_label=True)
            )
        warm_vocabulary(vocabulary, texts)
    if tuple(checkpoint["vocabulary"]) != vocabulary.symbols:
        raise ValueError("checkpoint CTC vocabulary does not match code")
    model_id = args.model_id or checkpoint["model_id"]
    head_config = checkpoint_head_config(checkpoint)
    model = FrozenWavLMCTC(
        len(vocabulary), model_id, checkpoint["dropout"],
        **head_config).to(device)
    model.load_head_state_dict(checkpoint["head"])
    print(f"device: {device}")
    print(f"model: {model_id} (frozen)")
    print(f"units: {units} head={head_config['head_type']}")
    print(f"loaded {args.ckpt} (dev mean AUC={checkpoint.get('auc')})")

    rows = []
    for subset, zip_path, csv_path in (
            ("seen", PATHS.dev_seen_zip, PATHS.dev_seen_csv),
            ("unseen", PATHS.dev_unseen_zip, PATHS.dev_unseen_csv)):
        loader = make_score_loader(
            zip_path, csv_path, checkpoint["max_samples"], args.bs,
            args.workers, device, vocabulary, with_label=True)
        subset_rows = collect_diagnostics(
            model, loader, device, amp_enabled, vocabulary.blank_id,
            vocabulary.symbols)
        for row in subset_rows:
            row["subset"] = subset
        rows.extend(subset_rows)

    report(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as file:
        fields = ["id", "subset", "label", *FEATURES, "decoded"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
