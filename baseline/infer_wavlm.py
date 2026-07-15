from __future__ import annotations

import argparse
import csv
import os

import torch

from config import PATHS, TRAIN
from data import load_pairs
from runtime import select_device
from train_wavlm import make_loader, move_batch
from wavlm_model import FrozenWavLMMatcher


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default=os.path.join(PATHS.ckpt_dir, "wavlm_base_plus_50k.pt"),
    )
    ap.add_argument("--out", default="submission_wavlm_base_plus_50k.csv")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


@torch.no_grad()
def predict(model, zip_path, csv_path, prefix, max_samples, device, args,
            amp_enabled):
    loader = make_loader(
        load_pairs(csv_path, False),
        zip_path,
        max_samples,
        args.bs,
        args.workers,
        device,
        inference=True,
    )
    rows = []
    model.eval()
    for batch in loader:
        enroll, query, _, ids, e_lens, q_lens = move_batch(batch, device)
        with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=amp_enabled):
            logits = model(enroll, query, e_lens, q_lens)
        probabilities = torch.sigmoid(logits).float().cpu()
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("inference produced non-finite posterior values")
        for pair_id, posterior in zip(ids, probabilities.tolist()):
            rows.append((f"{prefix}_{pair_id}", posterior))
    return rows


def main():
    args = parse_args()
    device = select_device(args.device)
    if args.workers is None:
        args.workers = TRAIN.num_workers if device.type == "cuda" else 0
    amp_enabled = args.amp and device.type == "cuda"

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_id = args.model_id or checkpoint["model_id"]
    projection_dim = checkpoint["projection_dim"]
    max_samples = checkpoint["max_samples"]
    print(f"device: {device}")
    print(f"workers: {args.workers}")
    print(f"model: {model_id} (frozen)")
    print(f"loaded {args.ckpt} (dev mean AUC={checkpoint.get('auc')})")

    model = FrozenWavLMMatcher(
        model_id, projection_dim=projection_dim).to(device)
    model.load_head_state_dict(checkpoint["head"])

    rows = predict(
        model, PATHS.eval_seen_zip, PATHS.eval_seen_csv, "seen",
        max_samples, device, args, amp_enabled)
    rows += predict(
        model, PATHS.eval_unseen_zip, PATHS.eval_unseen_csv, "unseen",
        max_samples, device, args, amp_enabled)
    print(f"total: {len(rows)} rows")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "posterior"])
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
