from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from config import AUDIO, PATHS, TRAIN
from ctc_data import (CTCScoreDataset, CTCUtteranceDataset,
                      ctc_score_collate, ctc_utterance_collate,
                      load_ctc_score_pairs, load_ctc_training_examples)
from ctc_score import normalized_ctc_score
from ctc_text import CharacterVocabulary, required_ctc_frames
from data import NoiseAugmenter
from runtime import select_device, should_pin_memory
from wavlm_ctc_model import FrozenWavLMCTC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="microsoft/wavlm-base-plus")
    parser.add_argument("--max-seconds", type=float, default=2.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subset", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise-prob", type=float, default=0.5)
    parser.add_argument("--noise-snr-min", type=float, default=-10.0)
    parser.add_argument("--noise-snr-max", type=float, default=5.0)
    parser.add_argument(
        "--noise-dir",
        default=os.path.join(PATHS.root, "noise", "DEMAND_16k", "wav"),
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--out",
        default=os.path.join(PATHS.ckpt_dir, "wavlm_char_ctc_100k.pt"),
    )
    return parser.parse_args()


def make_train_loader(examples, max_samples, batch_size, workers, device,
                      vocabulary, augment):
    dataset = CTCUtteranceDataset(
        examples, PATHS.train_zip, AUDIO, max_samples, augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=ctc_utterance_collate(vocabulary),
        pin_memory=should_pin_memory(device),
        drop_last=True,
        persistent_workers=workers > 0,
    )


def make_score_loader(zip_path, csv_path, max_samples, batch_size, workers,
                      device, vocabulary, with_label=True):
    dataset = CTCScoreDataset(
        load_ctc_score_pairs(csv_path, with_label),
        zip_path,
        AUDIO,
        max_samples,
        inference=not with_label,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=ctc_score_collate(vocabulary),
        pin_memory=should_pin_memory(device),
        persistent_workers=workers > 0,
    )


def _move(tensor, device):
    return tensor.to(device, non_blocking=True)


def ctc_valid_mask(output_lengths, targets, target_lengths):
    return output_lengths >= required_ctc_frames(targets, target_lengths)


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled, blank_id):
    model.eval()
    scores, labels = [], []
    for batch in loader:
        waveforms, sample_lengths, targets, target_lengths, target, pair_ids = batch
        waveforms = _move(waveforms, device)
        sample_lengths = _move(sample_lengths, device)
        targets = _move(targets, device)
        target_lengths = _move(target_lengths, device)
        with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=amp_enabled):
            log_probs, output_lengths = model.log_probs(
                waveforms, sample_lengths)
        valid = ctc_valid_mask(output_lengths, targets, target_lengths)
        batch_scores = log_probs.new_full((len(pair_ids),), -1e4)
        if valid.any():
            batch_scores[valid] = normalized_ctc_score(
                log_probs[valid], output_lengths[valid], targets[valid],
                target_lengths[valid], blank_id)
        if not torch.isfinite(batch_scores).all():
            raise RuntimeError("CTC evaluation produced non-finite scores")
        scores.append(batch_scores.cpu().numpy())
        labels.append(target.numpy())
    return roc_auc_score(np.concatenate(labels), np.concatenate(scores))


def main():
    args = parse_args()
    torch.manual_seed(TRAIN.seed)
    np.random.seed(TRAIN.seed)
    device = select_device(args.device)
    if args.workers is None:
        args.workers = TRAIN.num_workers if device.type == "cuda" else 0
    amp_enabled = args.amp and device.type == "cuda"
    max_samples = int(round(args.max_seconds * AUDIO.sample_rate))
    if max_samples <= 0:
        raise ValueError("--max-seconds must be positive")

    vocabulary = CharacterVocabulary()
    examples = load_ctc_training_examples(PATHS.train_csv)
    count = min(args.subset, len(examples))
    indices = np.random.default_rng(TRAIN.seed).permutation(len(examples))[:count]
    train_examples = [examples[index] for index in indices]

    print(f"device: {device}", flush=True)
    print(f"workers: {args.workers}", flush=True)
    print(f"model: {args.model_id} (frozen)", flush=True)
    print(f"vocabulary: {len(vocabulary)} classes", flush=True)
    print(f"max audio: {max_samples} samples ({args.max_seconds:.2f}s)",
          flush=True)
    print(f"amp: {amp_enabled}", flush=True)
    print(f"train utterances: {count} / {len(examples)}", flush=True)

    augment = None
    if args.noise_prob > 0:
        augment = NoiseAugmenter(
            AUDIO.sample_rate,
            args.noise_prob,
            args.noise_snr_min,
            args.noise_snr_max,
            args.noise_dir,
            TRAIN.seed,
        )
        if not augment.noise_paths:
            raise FileNotFoundError(
                f"no real noise files found under: {args.noise_dir}")
        print(f"real noise files: {len(augment.noise_paths)}", flush=True)
    print(f"audio noise: prob={args.noise_prob} "
          f"snr=[{args.noise_snr_min}, {args.noise_snr_max}]", flush=True)

    train_loader = make_train_loader(
        train_examples, max_samples, args.bs, args.workers, device,
        vocabulary, augment)
    dev_seen = make_score_loader(
        PATHS.dev_seen_zip, PATHS.dev_seen_csv, max_samples, args.bs,
        args.workers, device, vocabulary)
    dev_unseen = make_score_loader(
        PATHS.dev_unseen_zip, PATHS.dev_unseen_csv, max_samples, args.bs,
        args.workers, device, vocabulary)

    model = FrozenWavLMCTC(
        len(vocabulary), args.model_id, args.dropout).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"params: trainable={trainable:,} frozen={frozen:,}", flush=True)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr)
    criterion = torch.nn.CTCLoss(
        blank=vocabulary.blank_id, reduction="mean", zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        loss_sum = 0.0
        skipped_epoch = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for iteration, batch in enumerate(train_loader, 1):
            waveforms, sample_lengths, targets, target_lengths, wav_names = batch
            waveforms = _move(waveforms, device)
            sample_lengths = _move(sample_lengths, device)
            targets = _move(targets, device)
            target_lengths = _move(target_lengths, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=amp_enabled):
                log_probs, output_lengths = model.log_probs(
                    waveforms, sample_lengths)
                valid = ctc_valid_mask(
                    output_lengths, targets, target_lengths)
                skipped = int((~valid).sum().item())
                skipped_epoch += skipped
                if not valid.any():
                    raise RuntimeError("training batch has no CTC-valid targets")
                loss = criterion(
                    log_probs[valid].transpose(0, 1),
                    targets[valid],
                    output_lengths[valid],
                    target_lengths[valid],
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
            if iteration % args.log_every == 0:
                print(f"  ep{epoch} {iteration}/{len(train_loader)} "
                      f"ctc_loss={loss_sum/iteration:.4f}", flush=True)

        seen_auc = evaluate(
            model, dev_seen, device, amp_enabled, vocabulary.blank_id)
        unseen_auc = evaluate(
            model, dev_unseen, device, amp_enabled, vocabulary.blank_id)
        mean_auc = 0.5 * (seen_auc + unseen_auc)
        elapsed = time.time() - started
        peak_gb = 0.0
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"[epoch {epoch}] seen={seen_auc:.4f} unseen={unseen_auc:.4f} "
              f"mean={mean_auc:.4f} time={elapsed:.0f}s "
              f"peak_cuda={peak_gb:.2f}GB skipped_ctc={skipped_epoch}",
              flush=True)

        if mean_auc > best:
            best = mean_auc
            torch.save({
                "head": model.head_state_dict(),
                "model_id": args.model_id,
                "vocabulary": vocabulary.symbols,
                "dropout": args.dropout,
                "max_samples": max_samples,
                "noise_prob": args.noise_prob,
                "noise_snr_min": args.noise_snr_min,
                "noise_snr_max": args.noise_snr_max,
                "noise_dir": args.noise_dir,
                "auc": mean_auc,
                "seen_auc": seen_auc,
                "unseen_auc": unseen_auc,
                "epoch": epoch,
            }, args.out)
            print(f"  saved -> {args.out}", flush=True)

    print(f"done. best dev mean AUC = {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
