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
from ctc_text import build_vocabulary, required_ctc_frames, warm_vocabulary
from data import NoiseAugmenter
from runtime import select_device, should_pin_memory
from wavlm_ctc_model import FrozenWavLMCTC


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="microsoft/wavlm-base-plus")
    parser.add_argument("--units", choices=("char", "phoneme"), default="char")
    parser.add_argument(
        "--head", choices=("linear", "temporal"), default="linear")
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--adapter-layers", type=int, default=2)
    parser.add_argument("--train-zip", default=PATHS.train_zip)
    parser.add_argument("--train-csv", default=PATHS.train_csv)
    parser.add_argument("--max-seconds", type=float, default=2.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--subset", type=int, default=None,
        help="number of utterances; omit to use all training audio")
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
        default=None,
    )
    parser.add_argument(
        "--last-out",
        default=None,
        help="latest epoch checkpoint; defaults to <out stem>.last.pt",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="resume from the next epoch of this checkpoint",
    )
    return parser.parse_args(argv)


def make_train_loader(examples, train_zip, max_samples, batch_size, workers,
                      device, vocabulary, augment):
    dataset = CTCUtteranceDataset(
        examples, train_zip, AUDIO, max_samples, augment)
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


def default_last_checkpoint_path(out_path):
    stem, extension = os.path.splitext(out_path)
    if not extension:
        extension = ".pt"
    return f"{stem}.last{extension}"


def training_config(args, max_samples, train_utterances, amp_enabled, device):
    return {
        "model_id": args.model_id,
        "units": args.units,
        "head_type": args.head,
        "adapter_dim": args.adapter_dim,
        "adapter_layers": args.adapter_layers,
        "train_csv": args.train_csv,
        "train_zip": args.train_zip,
        "train_utterances": train_utterances,
        "max_samples": max_samples,
        "dropout": args.dropout,
        "batch_size": args.bs,
        "learning_rate": args.lr,
        "noise_prob": args.noise_prob,
        "noise_snr_min": args.noise_snr_min,
        "noise_snr_max": args.noise_snr_max,
        "noise_dir": args.noise_dir,
        "amp": amp_enabled,
        "workers": args.workers,
        "device": str(device),
        "seed": TRAIN.seed,
        "target_epochs": args.epochs,
    }


def _checkpoint_value(checkpoint, key):
    config = checkpoint.get("training_config", {})
    return config.get(key, checkpoint.get(key))


def validate_resume_checkpoint(checkpoint, config, vocabulary):
    expected = dict(config)
    expected["vocabulary"] = tuple(vocabulary.symbols)
    path_keys = {"train_csv", "train_zip", "noise_dir"}
    checked_keys = (
        "model_id", "units", "head_type", "adapter_dim", "adapter_layers",
        "vocabulary", "train_csv", "train_zip",
        "train_utterances", "max_samples", "dropout", "batch_size",
        "learning_rate", "noise_prob", "noise_snr_min", "noise_snr_max",
        "noise_dir", "seed",
    )
    mismatches = []
    for key in checked_keys:
        actual = _checkpoint_value(checkpoint, key)
        if actual is None:
            continue
        wanted = expected[key]
        if key == "vocabulary":
            actual = tuple(actual)
        elif key in path_keys:
            actual = os.path.abspath(os.path.normpath(actual))
            wanted = os.path.abspath(os.path.normpath(wanted))
        if actual != wanted:
            mismatches.append(f"{key}: checkpoint={actual!r}, current={wanted!r}")
    if mismatches:
        raise ValueError(
            "resume checkpoint is incompatible:\n  " + "\n  ".join(mismatches))


def capture_rng_state(device):
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state, device):
    if not state:
        return
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(state, path):
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def checkpoint_state(model, optimizer, scaler, config, vocabulary, device,
                     epoch, seen_auc, unseen_auc, mean_auc, best_auc,
                     best_epoch):
    return {
        "format_version": 2,
        "head": model.head_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": capture_rng_state(device),
        "training_config": config,
        "model_id": config["model_id"],
        "units": config["units"],
        "head_type": config["head_type"],
        "adapter_dim": config["adapter_dim"],
        "adapter_layers": config["adapter_layers"],
        "vocabulary": vocabulary.symbols,
        "train_csv": config["train_csv"],
        "train_zip": config["train_zip"],
        "train_utterances": config["train_utterances"],
        "dropout": config["dropout"],
        "max_samples": config["max_samples"],
        "noise_prob": config["noise_prob"],
        "noise_snr_min": config["noise_snr_min"],
        "noise_snr_max": config["noise_snr_max"],
        "noise_dir": config["noise_dir"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "auc": mean_auc,
        "seen_auc": seen_auc,
        "unseen_auc": unseen_auc,
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "epoch": epoch,
    }


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
    if args.adapter_dim <= 0 or args.adapter_layers <= 0:
        raise ValueError("--adapter-dim and --adapter-layers must be positive")

    for description, path in (
            ("training CSV", args.train_csv),
            ("training wav ZIP", args.train_zip)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{description} not found: {path}")

    vocabulary = build_vocabulary(args.units)
    examples = load_ctc_training_examples(args.train_csv)
    if args.units == "phoneme":
        texts = [example["text"] for example in examples]
        for csv_path in (PATHS.dev_seen_csv, PATHS.dev_unseen_csv):
            texts.extend(
                pair["enroll_text"]
                for pair in load_ctc_score_pairs(csv_path, with_label=True)
            )
        pronunciation_count = warm_vocabulary(vocabulary, texts)
        print(f"pronunciations: {pronunciation_count} unique", flush=True)
    if args.subset is not None and args.subset <= 0:
        raise ValueError("--subset must be positive when provided")
    count = len(examples) if args.subset is None else min(
        args.subset, len(examples))
    indices = np.random.default_rng(TRAIN.seed).permutation(len(examples))[:count]
    train_examples = [examples[index] for index in indices]

    print(f"device: {device}", flush=True)
    print(f"workers: {args.workers}", flush=True)
    print(f"model: {args.model_id} (frozen)", flush=True)
    print(f"units: {args.units}", flush=True)
    print(f"head: {args.head}", flush=True)
    if args.head == "temporal":
        print(f"adapter: dim={args.adapter_dim} layers={args.adapter_layers}",
              flush=True)
    print(f"train csv: {args.train_csv}", flush=True)
    print(f"train wav: {args.train_zip}", flush=True)
    print(f"vocabulary: {len(vocabulary)} classes", flush=True)
    print(f"max audio: {max_samples} samples ({args.max_seconds:.2f}s)",
          flush=True)
    print(f"amp: {amp_enabled}", flush=True)
    print(f"train utterances: {count} / {len(examples)}", flush=True)
    print(f"epochs: target={args.epochs}", flush=True)
    print(f"batch size: {args.bs}", flush=True)
    print(f"learning rate: {args.lr}", flush=True)
    print(f"process id: {os.getpid()}", flush=True)

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

    if args.out is None:
        args.out = os.path.join(
            PATHS.ckpt_dir, f"wavlm_{args.units}_ctc_100k.pt")
    if args.last_out is None:
        args.last_out = default_last_checkpoint_path(args.out)
    if os.path.abspath(args.out) == os.path.abspath(args.last_out):
        raise ValueError("--out and --last-out must be different files")
    for checkpoint_path in (args.out, args.last_out):
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    config = training_config(
        args, max_samples, count, amp_enabled, device)
    resume_checkpoint = None
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(
                f"resume checkpoint not found: {args.resume}")
        resume_checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(resume_checkpoint, config, vocabulary)

    train_loader = make_train_loader(
        train_examples, args.train_zip, max_samples, args.bs, args.workers, device,
        vocabulary, augment)
    dev_seen = make_score_loader(
        PATHS.dev_seen_zip, PATHS.dev_seen_csv, max_samples, args.bs,
        args.workers, device, vocabulary)
    dev_unseen = make_score_loader(
        PATHS.dev_unseen_zip, PATHS.dev_unseen_csv, max_samples, args.bs,
        args.workers, device, vocabulary)

    model = FrozenWavLMCTC(
        len(vocabulary), args.model_id, args.dropout,
        head_type=args.head,
        adapter_dim=args.adapter_dim,
        adapter_layers=args.adapter_layers,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"params: trainable={trainable:,} frozen={frozen:,}", flush=True)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr)
    criterion = torch.nn.CTCLoss(
        blank=vocabulary.blank_id, reduction="mean", zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best = -1.0
    best_epoch = 0
    start_epoch = 1
    if resume_checkpoint is not None:
        model.load_head_state_dict(resume_checkpoint["head"])
        completed_epoch = int(resume_checkpoint.get("epoch", 0))
        start_epoch = completed_epoch + 1
        best = float(resume_checkpoint.get(
            "best_auc", resume_checkpoint.get("auc", -1.0)))
        best_epoch = int(resume_checkpoint.get(
            "best_epoch", completed_epoch if best >= 0 else 0))
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        else:
            print("resume compatibility mode: optimizer state missing; "
                  "using a new optimizer", flush=True)
        if "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler"])
        elif amp_enabled:
            print("resume compatibility mode: AMP scaler state missing; "
                  "using a new scaler", flush=True)
        restore_rng_state(resume_checkpoint.get("rng_state"), device)
        print(f"resumed {args.resume}: completed_epoch={completed_epoch} "
              f"best={best:.4f} best_epoch={best_epoch}", flush=True)
    if args.epochs < start_epoch:
        raise ValueError(
            f"--epochs={args.epochs} is smaller than the next resume epoch "
            f"{start_epoch}; --epochs is the target total epoch count")

    print(f"best checkpoint: {args.out}", flush=True)
    print(f"latest checkpoint: {args.last_out}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
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

        improved = mean_auc > best
        if improved:
            best = mean_auc
            best_epoch = epoch
        state = checkpoint_state(
            model, optimizer, scaler, config, vocabulary, device,
            epoch, seen_auc, unseen_auc, mean_auc, best, best_epoch)
        atomic_torch_save(state, args.last_out)
        print(f"  saved latest -> {args.last_out}", flush=True)
        if improved:
            atomic_torch_save(state, args.out)
            print(f"  saved best -> {args.out}", flush=True)

    print(f"done. best dev mean AUC = {best:.4f} "
          f"at epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
