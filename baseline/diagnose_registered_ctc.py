from __future__ import annotations

import argparse
import csv
import os
from functools import partial

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from config import AUDIO, PATHS, TRAIN
from ctc_data import (_encode_texts, load_ctc_score_pairs)
from ctc_score import normalized_ctc_score
from ctc_text import build_vocabulary, checkpoint_units, warm_vocabulary
from data import normalize_waveform, read_wav, truncate_waveform
from runtime import select_device, should_pin_memory
from train_wavlm_ctc import ctc_valid_mask
from wavlm_ctc_model import (FrozenWavLMCTC, checkpoint_backbone_type,
                             checkpoint_model_config,
                             load_ctc_checkpoint_state)


FEATURES = (
    "query_target_score", "audio_alignment", "target_minus_alignment",
    "query_minus_enroll_score", "target_alignment_mean",
)


class RegisteredCTCDataset(Dataset):
    def __init__(self, pairs, zip_path, max_samples):
        self.pairs = pairs
        self.zip_path = zip_path
        self.max_samples = max_samples

    def __len__(self):
        return len(self.pairs)

    def _waveform(self, name):
        waveform = read_wav(self.zip_path, name, AUDIO.sample_rate)
        return normalize_waveform(truncate_waveform(waveform, self.max_samples))

    def __getitem__(self, index):
        pair = self.pairs[index]
        pair_id = pair["id"]
        enroll = self._waveform(f"wav/{pair_id}_enroll.wav")
        query = self._waveform(f"wav/{pair_id}_query.wav")
        return (
            enroll, query, pair["enroll_text"], pair.get("label", -1), pair_id,
            len(enroll), len(query),
        )


def _pad(waveforms, width=None):
    if width is None:
        width = max(len(waveform) for waveform in waveforms)
    result = torch.zeros(len(waveforms), width, dtype=torch.float32)
    for index, waveform in enumerate(waveforms):
        result[index, :len(waveform)] = waveform
    return result


def collate_registered(batch, vocabulary):
    width = max(max(len(item[0]) for item in batch),
                max(len(item[1]) for item in batch))
    enroll = _pad([item[0] for item in batch], width)
    query = _pad([item[1] for item in batch], width)
    enroll_lengths = torch.tensor([item[5] for item in batch], dtype=torch.long)
    query_lengths = torch.tensor([item[6] for item in batch], dtype=torch.long)
    targets, target_lengths = _encode_texts(
        [item[2] for item in batch], vocabulary)
    labels = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    return (
        enroll, query, enroll_lengths, query_lengths, targets, target_lengths,
        labels, [item[4] for item in batch],
    )


def make_loader(pairs, zip_path, max_samples, batch_size, workers, device,
                vocabulary):
    return DataLoader(
        RegisteredCTCDataset(pairs, zip_path, max_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=partial(collate_registered, vocabulary=vocabulary),
        pin_memory=should_pin_memory(device),
        persistent_workers=workers > 0,
    )


def posterior_alignment(log_probs, left_lengths, right_lengths):
    """Symmetric frame alignment after removing the CTC blank channel."""
    posterior = log_probs.exp()[..., 1:]
    posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    similarity = torch.bmm(
        F.normalize(posterior[:len(left_lengths)], dim=-1),
        F.normalize(posterior[len(left_lengths):], dim=-1).transpose(1, 2),
    )
    left_steps = torch.arange(similarity.shape[1], device=similarity.device)
    right_steps = torch.arange(similarity.shape[2], device=similarity.device)
    left_mask = left_steps.unsqueeze(0) < left_lengths.unsqueeze(1)
    right_mask = right_steps.unsqueeze(0) < right_lengths.unsqueeze(1)
    valid = left_mask[:, :, None] & right_mask[:, None, :]
    fill = torch.finfo(similarity.dtype).min
    left_best = similarity.masked_fill(~valid, fill).max(dim=2).values
    right_best = similarity.masked_fill(~valid, fill).max(dim=1).values
    left_score = (left_best * left_mask).sum(dim=1) / left_mask.sum(dim=1)
    right_score = (right_best * right_mask).sum(dim=1) / right_mask.sum(dim=1)
    return 0.5 * (left_score + right_score)


def _target_scores(log_probs, output_lengths, targets, target_lengths,
                   blank_id):
    valid = ctc_valid_mask(output_lengths, targets, target_lengths)
    scores = log_probs.new_full((len(output_lengths),), -1e4)
    if valid.any():
        scores[valid] = normalized_ctc_score(
            log_probs[valid], output_lengths[valid], targets[valid],
            target_lengths[valid], blank_id)
    return scores


@torch.no_grad()
def collect_features(model, loader, device, amp_enabled, vocabulary):
    rows = []
    model.eval()
    for batch in loader:
        (enroll, query, enroll_sample_lengths, query_sample_lengths,
         targets, target_lengths, labels, pair_ids) = batch
        batch_size = len(pair_ids)
        waveforms = torch.cat([enroll, query], dim=0)
        sample_lengths = torch.cat([
            enroll_sample_lengths, query_sample_lengths,
        ])
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=amp_enabled):
            log_probs, output_lengths = model.log_probs(
                waveforms, sample_lengths)
        enroll_frames = output_lengths[:batch_size]
        query_frames = output_lengths[batch_size:]
        enroll_score = _target_scores(
            log_probs[:batch_size], enroll_frames, targets, target_lengths,
            vocabulary.blank_id)
        query_score = _target_scores(
            log_probs[batch_size:], query_frames, targets, target_lengths,
            vocabulary.blank_id)
        alignment = posterior_alignment(
            log_probs, enroll_frames, query_frames)
        values = {
            "query_target_score": query_score,
            "audio_alignment": alignment,
            "target_minus_alignment": query_score - alignment,
            "query_minus_enroll_score": query_score - enroll_score,
            "target_alignment_mean": 0.5 * (query_score + alignment),
        }
        for index, (pair_id, label) in enumerate(zip(pair_ids, labels.tolist())):
            rows.append({
                "id": pair_id,
                "label": int(label),
                **{name: float(value[index].item())
                   for name, value in values.items()},
            })
    return rows


def report(rows):
    for subset in ("all", "seen", "unseen"):
        selected = rows if subset == "all" else [
            row for row in rows if row["subset"] == subset]
        labels = [row["label"] for row in selected]
        print(f"[{subset}] rows={len(selected)}")
        for feature in FEATURES:
            print(f"  {feature}: "
                  f"{roc_auc_score(labels, [row[feature] for row in selected]):.4f}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = select_device(args.device)
    if args.workers is None:
        args.workers = TRAIN.num_workers if device.type == "cuda" else 0
    amp_enabled = args.amp and device.type == "cuda"
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vocabulary = build_vocabulary(checkpoint_units(checkpoint))
    texts = []
    for csv_path in (PATHS.dev_seen_csv, PATHS.dev_unseen_csv):
        texts.extend(
            pair["enroll_text"]
            for pair in load_ctc_score_pairs(csv_path, with_label=True)
        )
    warm_vocabulary(vocabulary, texts)
    if tuple(checkpoint["vocabulary"]) != vocabulary.symbols:
        raise ValueError("checkpoint CTC vocabulary does not match code")
    model_id = args.model_id or checkpoint["model_id"]
    model = FrozenWavLMCTC(
        len(vocabulary), model_id, checkpoint["dropout"],
        backbone_type=checkpoint_backbone_type(checkpoint),
        **checkpoint_model_config(checkpoint),
    ).to(device)
    load_ctc_checkpoint_state(model, checkpoint)
    rows = []
    for subset, zip_path, csv_path in (
            ("seen", PATHS.dev_seen_zip, PATHS.dev_seen_csv),
            ("unseen", PATHS.dev_unseen_zip, PATHS.dev_unseen_csv)):
        loader = make_loader(
            load_ctc_score_pairs(csv_path, with_label=True), zip_path,
            checkpoint["max_samples"], args.bs, args.workers, device,
            vocabulary)
        subset_rows = collect_features(
            model, loader, device, amp_enabled, vocabulary)
        for row in subset_rows:
            row["subset"] = subset
        rows.extend(subset_rows)
    report(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as file:
        fields = ["id", "subset", "label", *FEATURES]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
