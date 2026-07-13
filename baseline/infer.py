"""推理脚本：生成提交 CSV。

用法：
    python infer.py --ckpt checkpoints/best.pt --out submission.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader

from config import AUDIO, PATHS, TRAIN
from data import PairDataset, collate, load_pairs
from model import build_model
from runtime import select_device, should_pin_memory


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(PATHS.ckpt_dir, "best.pt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "submission.csv"))
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader workers. Default: 8 on CUDA, 0 otherwise")
    ap.add_argument("--device", type=str, default="auto",
                    help="auto, cuda, mps, or cpu")
    ap.add_argument("--model", choices=["global", "frame_maxmean"], default=None,
                    help="Override checkpoint model type")
    return ap.parse_args()


@torch.no_grad()
def predict(model, zip_path, csv_path, prefix, device, args):
    ds = PairDataset(load_pairs(csv_path, False), zip_path, AUDIO, inference=True)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False,
                        num_workers=args.workers, collate_fn=collate,
                        pin_memory=should_pin_memory(device))
    rows = []
    for e, q, _, ids in loader:
        e, q = e.to(device), q.to(device)
        prob = torch.sigmoid(model(e, q)).cpu().numpy()
        for pid, p in zip(ids, prob):
            rows.append((f"{prefix}_{pid}", float(p)))
    return rows


def main():
    args = parse_args()
    device = select_device(args.device)
    print(f"device: {device}")
    if args.workers is None:
        args.workers = TRAIN.num_workers if device.type == "cuda" else 0
    print(f"workers: {args.workers}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_name = args.model or ckpt.get("model_name", "global")
    print(f"model: {model_name}")
    model = build_model(
        model_name,
        AUDIO.n_mels,
        ckpt.get("embed_dim", TRAIN.embed_dim),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.ckpt} (dev mean AUC={ckpt.get('auc')})")

    rows = predict(model, PATHS.eval_seen_zip, PATHS.eval_seen_csv, "seen", device, args)
    rows += predict(model, PATHS.eval_unseen_zip, PATHS.eval_unseen_csv, "unseen", device, args)
    print(f"total: {len(rows)} rows")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "posterior"])
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
