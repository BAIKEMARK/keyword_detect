from __future__ import annotations

import argparse
import csv
import os

import torch

from config import PATHS, TRAIN
from ctc_data import load_ctc_score_pairs
from ctc_score import normalized_ctc_score
from ctc_text import build_vocabulary, checkpoint_units, warm_vocabulary
from runtime import select_device
from train_wavlm_ctc import ctc_valid_mask, make_score_loader
from wavlm_ctc_model import FrozenWavLMCTC, checkpoint_head_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=os.path.join(PATHS.ckpt_dir, "wavlm_char_ctc_100k.pt"),
    )
    parser.add_argument("--out", default="submission_wavlm_char_ctc.csv")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


@torch.no_grad()
def collect_scores(model, loader, device, amp_enabled, blank_id):
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
        valid = ctc_valid_mask(output_lengths, targets, target_lengths)
        scores = log_probs.new_full((len(pair_ids),), -1e4)
        if valid.any():
            scores[valid] = normalized_ctc_score(
                log_probs[valid], output_lengths[valid], targets[valid],
                target_lengths[valid], blank_id)
        scores = scores.cpu()
        if not torch.isfinite(scores).all():
            raise RuntimeError("CTC inference produced non-finite scores")
        rows.extend(
            (pair_id, score, int(label) if label >= 0 else None)
            for pair_id, score, label in zip(
                pair_ids, scores.tolist(), labels.tolist())
        )
    return rows


def predict(model, loader, prefix, device, amp_enabled, blank_id):
    rows = collect_scores(model, loader, device, amp_enabled, blank_id)
    scores = torch.tensor([row[1] for row in rows], dtype=torch.float32)
    posteriors = torch.sigmoid(scores).tolist()
    return [
        (f"{prefix}_{pair_id}", posterior)
        for (pair_id, _, _), posterior in zip(rows, posteriors)
    ]


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
        texts = []
        for csv_path in (PATHS.eval_seen_csv, PATHS.eval_unseen_csv):
            texts.extend(
                pair["enroll_text"]
                for pair in load_ctc_score_pairs(csv_path, with_label=False)
            )
        pronunciation_count = warm_vocabulary(vocabulary, texts)
        print(f"pronunciations: {pronunciation_count} unique")
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
        PATHS.eval_seen_zip, PATHS.eval_seen_csv, checkpoint["max_samples"],
        args.bs, args.workers, device, vocabulary, with_label=False)
    unseen_loader = make_score_loader(
        PATHS.eval_unseen_zip, PATHS.eval_unseen_csv, checkpoint["max_samples"],
        args.bs, args.workers, device, vocabulary, with_label=False)
    rows = predict(
        model, seen_loader, "seen", device, amp_enabled, vocabulary.blank_id)
    rows += predict(
        model, unseen_loader, "unseen", device, amp_enabled,
        vocabulary.blank_id)
    print(f"total: {len(rows)} rows")

    with open(args.out, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "posterior"])
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
