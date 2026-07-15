from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from config import AUDIO, PATHS, TRAIN
from data import (NoiseAugmenter, WavePairDataset, collate_wave_pairs,
                  load_pairs)
from runtime import select_device, should_pin_memory
from wavlm_model import FrozenWavLMMatcher


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="microsoft/wavlm-base-plus")
    ap.add_argument("--projection-dim", type=int, default=128)
    ap.add_argument("--max-seconds", type=float, default=2.5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=TRAIN.pos_weight)
    ap.add_argument("--subset", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--noise-prob", type=float, default=0.5)
    ap.add_argument("--noise-snr-min", type=float, default=-10.0)
    ap.add_argument("--noise-snr-max", type=float, default=5.0)
    ap.add_argument(
        "--noise-dir",
        default=os.path.join(PATHS.root, "noise", "DEMAND_16k", "wav"),
    )
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument(
        "--out",
        default=os.path.join(PATHS.ckpt_dir, "wavlm_base_plus_50k.pt"),
    )
    return ap.parse_args()


def make_loader(pairs, zip_path, max_samples, batch_size, workers, device,
                shuffle=False, augment=None, inference=False):
    dataset = WavePairDataset(
        pairs,
        zip_path,
        AUDIO,
        max_samples=max_samples,
        inference=inference,
        query_augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_wave_pairs,
        pin_memory=should_pin_memory(device),
        drop_last=shuffle,
        persistent_workers=workers > 0,
    )


def move_batch(batch, device):
    enroll, query, labels, ids, e_lens, q_lens = batch
    return (
        enroll.to(device, non_blocking=True),
        query.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        ids,
        e_lens.to(device, non_blocking=True),
        q_lens.to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled):
    model.eval()
    probabilities, labels = [], []
    for batch in loader:
        enroll, query, target, _, e_lens, q_lens = move_batch(batch, device)
        with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=amp_enabled):
            logit = model(enroll, query, e_lens, q_lens)
        probabilities.append(torch.sigmoid(logit).float().cpu().numpy())
        labels.append(target.cpu().numpy())
    return roc_auc_score(np.concatenate(labels), np.concatenate(probabilities))


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

    print(f"device: {device}", flush=True)
    print(f"workers: {args.workers}", flush=True)
    print(f"model: {args.model_id} (frozen)", flush=True)
    print(f"max audio: {max_samples} samples ({args.max_seconds:.2f}s)",
          flush=True)
    print(f"amp: {amp_enabled}", flush=True)

    all_pairs = load_pairs(PATHS.train_csv, with_label=True)
    n = min(args.subset, len(all_pairs))
    indices = np.random.default_rng(TRAIN.seed).permutation(len(all_pairs))[:n]
    train_pairs = [all_pairs[i] for i in indices]
    print(f"train: {n} / {len(all_pairs)} pairs", flush=True)

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
    print(f"query noise: prob={args.noise_prob} "
          f"snr=[{args.noise_snr_min}, {args.noise_snr_max}]",
          flush=True)

    train_loader = make_loader(
        train_pairs, PATHS.train_zip, max_samples, args.bs, args.workers,
        device, shuffle=True, augment=augment)
    dev_seen = make_loader(
        load_pairs(PATHS.dev_seen_csv, True), PATHS.dev_seen_zip, max_samples,
        args.bs, args.workers, device)
    dev_unseen = make_loader(
        load_pairs(PATHS.dev_unseen_csv, True), PATHS.dev_unseen_zip,
        max_samples, args.bs, args.workers, device)

    model = FrozenWavLMMatcher(
        args.model_id, projection_dim=args.projection_dim).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"params: trainable={trainable:,} frozen={frozen:,}", flush=True)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, device=device))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        loss_sum = 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for iteration, batch in enumerate(train_loader, 1):
            enroll, query, target, _, e_lens, q_lens = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=amp_enabled):
                logit = model(enroll, query, e_lens, q_lens)
                loss = criterion(logit, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
            if iteration % args.log_every == 0:
                print(f"  ep{epoch} {iteration}/{len(train_loader)} "
                      f"loss={loss_sum/iteration:.4f}", flush=True)

        seen_auc = evaluate(model, dev_seen, device, amp_enabled)
        unseen_auc = evaluate(model, dev_unseen, device, amp_enabled)
        mean_auc = 0.5 * (seen_auc + unseen_auc)
        elapsed = time.time() - started
        peak_gb = 0.0
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"[epoch {epoch}] seen={seen_auc:.4f} unseen={unseen_auc:.4f} "
              f"mean={mean_auc:.4f} time={elapsed:.0f}s "
              f"peak_cuda={peak_gb:.2f}GB", flush=True)

        if mean_auc > best:
            best = mean_auc
            torch.save({
                "head": model.head_state_dict(),
                "model_id": args.model_id,
                "projection_dim": args.projection_dim,
                "max_samples": max_samples,
                "pos_weight": args.pos_weight,
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
