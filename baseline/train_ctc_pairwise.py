from __future__ import annotations

import argparse
import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import AUDIO, PATHS, TRAIN
from ctc_data import (CTCPairTrainingDataset, ctc_training_pair_collate,
                      load_ctc_score_pairs, load_ctc_training_pairs)
from ctc_hard_negative import build_phoneme_hard_negative_candidates
from ctc_text import build_vocabulary, checkpoint_units, warm_vocabulary
from data import NoiseAugmenter
from runtime import select_device, should_pin_memory
from train_wavlm_ctc import (atomic_torch_save, capture_rng_state,
                             ctc_valid_mask, default_last_checkpoint_path,
                             evaluate, make_score_loader, restore_rng_state)
from wavlm_ctc_model import (FrozenWavLMCTC, checkpoint_backbone_type,
                             checkpoint_model_config,
                             load_ctc_checkpoint_state)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Pair-label discriminative fine-tuning for a CTC KWS model")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--train-csv", default=PATHS.train_csv)
    parser.add_argument("--train-zip", default=PATHS.train_zip)
    parser.add_argument(
        "--subset", type=int, default=50000,
        help="number of official pairs; 50000 pairs correspond to 100K audio sides")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--ctc-weight", type=float, default=0.25)
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--pair-margin", type=float, default=0.1)
    parser.add_argument("--hard-negative-weight", type=float, default=0.25)
    parser.add_argument("--hard-negative-margin", type=float, default=0.1)
    parser.add_argument("--hard-negative-k", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=TRAIN.seed)
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
    parser.add_argument("--out", required=True)
    parser.add_argument("--last-out", default=None)
    return parser.parse_args(argv)


def normalized_training_scores(log_probs, output_lengths, targets,
                               target_lengths, blank_id):
    valid = ctc_valid_mask(output_lengths, targets, target_lengths)
    scores = log_probs.new_full((len(output_lengths),), -1e4)
    if valid.any():
        nll = F.ctc_loss(
            log_probs[valid].transpose(0, 1), targets[valid],
            output_lengths[valid], target_lengths[valid], blank=blank_id,
            reduction="none", zero_infinity=True)
        scores[valid] = -nll / target_lengths[valid].to(nll.dtype)
    return scores, valid


def pairwise_discriminative_objective(
        log_probs, output_lengths,
        enroll_targets, enroll_target_lengths,
        query_targets, query_target_lengths,
        hard_targets, hard_target_lengths, hard_negative_count,
        labels, blank_id,
        ctc_weight=0.25, pair_weight=1.0, pair_margin=0.1,
        hard_negative_weight=0.25, hard_negative_margin=0.1):
    batch_size = len(output_lengths)
    if hard_negative_count <= 0:
        raise ValueError("hard_negative_count must be positive")
    if len(hard_targets) != batch_size * hard_negative_count:
        raise ValueError("hard-negative target count does not match the batch")
    if labels.shape != (batch_size,):
        raise ValueError("labels must have one value per pair")

    query_scores, query_valid = normalized_training_scores(
        log_probs, output_lengths, query_targets, query_target_lengths,
        blank_id)
    enroll_scores, enroll_valid = normalized_training_scores(
        log_probs, output_lengths, enroll_targets, enroll_target_lengths,
        blank_id)

    expanded_log_probs = log_probs[:, None].expand(
        -1, hard_negative_count, -1, -1).reshape(
            batch_size * hard_negative_count,
            log_probs.shape[1], log_probs.shape[2])
    expanded_lengths = output_lengths[:, None].expand(
        -1, hard_negative_count).reshape(-1)
    hard_scores, hard_valid = normalized_training_scores(
        expanded_log_probs, expanded_lengths, hard_targets,
        hard_target_lengths, blank_id)
    hard_scores = hard_scores.reshape(batch_size, hard_negative_count)
    hard_valid = hard_valid.reshape(batch_size, hard_negative_count)
    hardest_scores = hard_scores.masked_fill(~hard_valid, -1e4).max(dim=1).values
    has_hard_negative = hard_valid.any(dim=1)

    ctc_loss = -query_scores[query_valid].mean()
    pair_valid = query_valid & enroll_valid & has_hard_negative
    if not pair_valid.any():
        raise RuntimeError("training batch has no valid discriminative pairs")
    labels = labels.to(dtype=log_probs.dtype)
    positive_gap = enroll_scores - hardest_scores
    negative_gap = query_scores - enroll_scores
    official_gap = labels * positive_gap + (1.0 - labels) * negative_gap
    pair_loss = F.softplus(
        pair_margin - official_gap[pair_valid]).mean()

    hard_valid_rows = query_valid & has_hard_negative
    hard_loss = F.softplus(
        hard_negative_margin
        - (query_scores[hard_valid_rows] - hardest_scores[hard_valid_rows])
    ).mean()
    total = (
        ctc_weight * ctc_loss
        + pair_weight * pair_loss
        + hard_negative_weight * hard_loss
    )
    skipped = int((~pair_valid).sum().item())
    return total, ctc_loss, pair_loss, hard_loss, skipped


def _select_pairs(pairs, subset, seed):
    if subset is None or subset >= len(pairs):
        return list(pairs)
    if subset <= 0:
        raise ValueError("--subset must be positive")
    indices = np.random.default_rng(seed).permutation(len(pairs))[:subset]
    return [pairs[index] for index in indices]


def _pad_candidates(candidates, count):
    result = {}
    for anchor, values in candidates.items():
        values = tuple(values)
        if not values:
            raise ValueError(f"no hard-negative candidates for {anchor!r}")
        result[anchor] = (values + (values[-1],) * count)[:count]
    return result


def make_pair_loader(pairs, zip_path, max_samples, batch_size, workers,
                     device, vocabulary, augment, hard_candidates):
    dataset = CTCPairTrainingDataset(
        pairs, zip_path, AUDIO, max_samples, augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=ctc_training_pair_collate(vocabulary, hard_candidates),
        pin_memory=should_pin_memory(device),
        drop_last=True,
        persistent_workers=workers > 0,
    )


def _checkpoint_state(base, model, optimizer, scaler, args, device,
                      epoch, seen_auc, unseen_auc, best_auc, best_epoch,
                      pair_count, max_samples):
    state = dict(base)
    training_config = dict(base.get("training_config", {}))
    training_config.update({
        "objective": "pairwise_discriminative_ctc",
        "pairwise_train_csv": args.train_csv,
        "pairwise_train_zip": args.train_zip,
        "pairwise_train_pairs": pair_count,
        "pairwise_batch_size": args.bs,
        "pairwise_learning_rate": args.lr,
        "pairwise_backbone_learning_rate": args.backbone_lr,
        "pairwise_ctc_weight": args.ctc_weight,
        "pairwise_pair_weight": args.pair_weight,
        "pairwise_pair_margin": args.pair_margin,
        "pairwise_hard_negative_weight": args.hard_negative_weight,
        "pairwise_hard_negative_margin": args.hard_negative_margin,
        "pairwise_hard_negative_k": args.hard_negative_k,
        "pairwise_seed": args.seed,
        "pairwise_target_epochs": args.epochs,
    })
    mean_auc = 0.5 * (seen_auc + unseen_auc)
    state.update({
        "format_version": max(int(base.get("format_version", 1)), 2),
        "head": model.head_state_dict(),
        "backbone_trainable": model.trainable_backbone_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": capture_rng_state(device),
        "training_config": training_config,
        "pairwise_base_checkpoint": args.ckpt,
        "pairwise_epoch": epoch,
        "pairwise_best_auc": best_auc,
        "pairwise_best_epoch": best_epoch,
        "pairwise_train_pairs": pair_count,
        "max_samples": max_samples,
        "auc": mean_auc,
        "seen_auc": seen_auc,
        "unseen_auc": unseen_auc,
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "epoch": epoch,
    })
    return state


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    if args.workers is None:
        args.workers = TRAIN.num_workers if device.type == "cuda" else 0
    amp_enabled = args.amp and device.type == "cuda"
    if args.epochs <= 0 or args.bs <= 0 or args.log_every <= 0:
        raise ValueError("epochs, batch size and log interval must be positive")
    for name in ("lr", "backbone_lr", "ctc_weight", "pair_weight",
                 "hard_negative_weight", "grad_clip"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.lr == 0 or args.hard_negative_k <= 0:
        raise ValueError("learning rate and hard-negative count must be positive")
    if args.pair_margin < 0 or args.hard_negative_margin < 0:
        raise ValueError("pair and hard-negative margins must be non-negative")
    if args.ctc_weight + args.pair_weight + args.hard_negative_weight <= 0:
        raise ValueError("at least one loss weight must be positive")
    for description, path in (
            ("checkpoint", args.ckpt), ("training CSV", args.train_csv),
            ("training wav ZIP", args.train_zip)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{description} not found: {path}")

    source_path = args.resume or args.ckpt
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"resume checkpoint not found: {source_path}")
    checkpoint = torch.load(
        source_path, map_location="cpu", weights_only=False)
    if checkpoint_units(checkpoint) != "phoneme":
        raise ValueError("pairwise discriminative training requires phoneme CTC")
    vocabulary = build_vocabulary("phoneme")
    if tuple(checkpoint["vocabulary"]) != vocabulary.symbols:
        raise ValueError("checkpoint CTC vocabulary does not match code")

    all_pairs = load_ctc_training_pairs(args.train_csv)
    train_pairs = _select_pairs(all_pairs, args.subset, args.seed)
    if len(train_pairs) < args.bs:
        raise ValueError("training subset must contain at least one full batch")
    texts = [
        text for pair in all_pairs
        for text in (pair["enroll_text"], pair["query_text"])
    ]
    for csv_path in (PATHS.dev_seen_csv, PATHS.dev_unseen_csv):
        texts.extend(
            pair["enroll_text"]
            for pair in load_ctc_score_pairs(csv_path, with_label=True)
        )
    count = warm_vocabulary(vocabulary, texts)
    print(f"pronunciations: {count} unique", flush=True)

    print("hard-negative mining: building Top-K phoneme candidates...", flush=True)
    hard_candidates = build_phoneme_hard_negative_candidates(
        vocabulary,
        [pair["query_text"] for pair in train_pairs],
        texts,
        neighbors_per_anchor=args.hard_negative_k,
    )
    hard_candidates = _pad_candidates(
        hard_candidates, args.hard_negative_k)
    print(f"hard-negative mining: ready ({len(hard_candidates)} anchors, "
          f"K={args.hard_negative_k})", flush=True)

    max_samples = int(checkpoint["max_samples"])
    if args.max_seconds is not None:
        if args.max_seconds <= 0:
            raise ValueError("--max-seconds must be positive")
        requested = int(round(args.max_seconds * AUDIO.sample_rate))
        if requested != max_samples:
            raise ValueError(
                "--max-seconds must match the initialization checkpoint")
    augment = None
    if args.noise_prob > 0:
        augment = NoiseAugmenter(
            AUDIO.sample_rate, args.noise_prob, args.noise_snr_min,
            args.noise_snr_max, args.noise_dir, args.seed)
        if not augment.noise_paths:
            raise FileNotFoundError(
                f"no real noise files found under: {args.noise_dir}")

    model_id = args.model_id or checkpoint["model_id"]
    model = FrozenWavLMCTC(
        len(vocabulary), model_id, checkpoint["dropout"],
        backbone_type=checkpoint_backbone_type(checkpoint),
        **checkpoint_model_config(checkpoint),
    ).to(device)
    load_ctc_checkpoint_state(model, checkpoint)
    head_parameters = list(model.head.parameters())
    backbone_parameters = [
        parameter for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    parameter_groups = [{"params": head_parameters, "lr": args.lr}]
    if backbone_parameters:
        parameter_groups.append({
            "params": backbone_parameters, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(parameter_groups)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    best = -1.0
    best_epoch = 0
    if args.resume:
        start_epoch = int(checkpoint.get("pairwise_epoch", 0)) + 1
        best = float(checkpoint.get(
            "pairwise_best_auc", checkpoint.get("auc", -1.0)))
        best_epoch = int(checkpoint.get("pairwise_best_epoch", 0))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint.get("rng_state"), device)
    if args.epochs < start_epoch:
        raise ValueError("--epochs is smaller than the next resume epoch")

    if args.last_out is None:
        args.last_out = default_last_checkpoint_path(args.out)
    if os.path.abspath(args.out) == os.path.abspath(args.last_out):
        raise ValueError("--out and --last-out must be different files")
    for path in (args.out, args.last_out):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    train_loader = make_pair_loader(
        train_pairs, args.train_zip, max_samples, args.bs, args.workers,
        device, vocabulary, augment, hard_candidates)
    dev_seen = make_score_loader(
        PATHS.dev_seen_zip, PATHS.dev_seen_csv, max_samples, args.bs,
        args.workers, device, vocabulary)
    dev_unseen = make_score_loader(
        PATHS.dev_unseen_zip, PATHS.dev_unseen_csv, max_samples, args.bs,
        args.workers, device, vocabulary)

    positives = sum(pair["label"] for pair in train_pairs)
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad)
    print(f"device: {device}", flush=True)
    print(f"model: {model_id}", flush=True)
    print(f"backbone: {model.backbone_type}", flush=True)
    print(f"initial checkpoint: {source_path}", flush=True)
    print(f"train pairs: {len(train_pairs)} / {len(all_pairs)} "
          f"positive={positives} negative={len(train_pairs) - positives}",
          flush=True)
    print(f"batch size: {args.bs} epochs: target={args.epochs}", flush=True)
    print(f"loss: ctc={args.ctc_weight} pair={args.pair_weight} "
          f"hard={args.hard_negative_weight} K={args.hard_negative_k}",
          flush=True)
    print(f"learning rate: head={args.lr} backbone={args.backbone_lr}",
          flush=True)
    print(f"trainable parameters: {trainable:,}", flush=True)
    print(f"best checkpoint: {args.out}", flush=True)
    print(f"latest checkpoint: {args.last_out}", flush=True)

    objective = partial(
        pairwise_discriminative_objective,
        blank_id=vocabulary.blank_id,
        ctc_weight=args.ctc_weight,
        pair_weight=args.pair_weight,
        pair_margin=args.pair_margin,
        hard_negative_weight=args.hard_negative_weight,
        hard_negative_margin=args.hard_negative_margin,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        started = time.time()
        totals = np.zeros(4, dtype=np.float64)
        skipped_epoch = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for iteration, batch in enumerate(train_loader, 1):
            (waveforms, sample_lengths,
             enroll_targets, enroll_target_lengths,
             query_targets, query_target_lengths,
             hard_targets, hard_target_lengths, hard_count,
             labels, pair_ids) = batch
            tensors = [
                enroll_targets, enroll_target_lengths,
                query_targets, query_target_lengths,
                hard_targets, hard_target_lengths, labels,
            ]
            (enroll_targets, enroll_target_lengths,
             query_targets, query_target_lengths,
             hard_targets, hard_target_lengths, labels) = [
                tensor.to(device, non_blocking=True) for tensor in tensors
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=amp_enabled):
                log_probs, output_lengths = model.log_probs(
                    waveforms, sample_lengths)
                loss, ctc_loss, pair_loss, hard_loss, skipped = objective(
                    log_probs, output_lengths,
                    enroll_targets, enroll_target_lengths,
                    query_targets, query_target_lengths,
                    hard_targets, hard_target_lengths, hard_count, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite pairwise loss at epoch {epoch} "
                    f"iteration {iteration}")
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for group in optimizer.param_groups
                     for parameter in group["params"]], args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            totals += [
                loss.item(), ctc_loss.item(), pair_loss.item(), hard_loss.item()]
            skipped_epoch += skipped
            if iteration % args.log_every == 0:
                means = totals / iteration
                print(f"  ep{epoch} {iteration}/{len(train_loader)} "
                      f"loss={means[0]:.4f} ctc={means[1]:.4f} "
                      f"pair={means[2]:.4f} hard={means[3]:.4f}",
                      flush=True)

        seen_auc = evaluate(
            model, dev_seen, device, amp_enabled, vocabulary.blank_id)
        unseen_auc = evaluate(
            model, dev_unseen, device, amp_enabled, vocabulary.blank_id)
        mean_auc = 0.5 * (seen_auc + unseen_auc)
        improved = mean_auc > best
        if improved:
            best = mean_auc
            best_epoch = epoch
        state = _checkpoint_state(
            checkpoint, model, optimizer, scaler, args, device, epoch,
            seen_auc, unseen_auc, best, best_epoch, len(train_pairs),
            max_samples)
        atomic_torch_save(state, args.last_out)
        if improved:
            atomic_torch_save(state, args.out)
        elapsed = time.time() - started
        peak_gb = 0.0
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"[epoch {epoch}] seen={seen_auc:.4f} "
              f"unseen={unseen_auc:.4f} mean={mean_auc:.4f} "
              f"time={elapsed:.0f}s peak_cuda={peak_gb:.2f}GB "
              f"skipped_pairs={skipped_epoch}", flush=True)
        print(f"  saved latest -> {args.last_out}", flush=True)
        if improved:
            print(f"  saved best -> {args.out}", flush=True)

    print(f"done. best dev mean AUC = {best:.4f} "
          f"at pairwise epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
