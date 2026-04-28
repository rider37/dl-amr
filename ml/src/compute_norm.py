#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.dataloaders import AMRDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-channel mean/std for input normalization")
    parser.add_argument("--dataset_path", default="ml/data/processed/cylinder_demo", help="Dataset path")
    parser.add_argument("--split", default="train", help="Split name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for stats")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--target", action="store_true", help="Compute stats for target (y) instead of input (x)")
    parser.add_argument("--task", default="seg", help="Dataset task (seg|reg)")
    args = parser.parse_args()

    ds = AMRDataset(args.dataset_path, split=args.split, task=args.task)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sum_c = None
    sumsq_c = None
    count = 0

    for batch in loader:
        x = batch["y"].float() if args.target else batch["x"].float()  # (B, C, H, W)
        b, c, h, w = x.shape
        x_reshaped = x.view(b, c, -1)
        if sum_c is None:
            sum_c = torch.zeros(c, dtype=torch.float64)
            sumsq_c = torch.zeros(c, dtype=torch.float64)
        sum_c += x_reshaped.sum(dim=(0, 2)).double()
        sumsq_c += (x_reshaped.double() ** 2).sum(dim=(0, 2))
        count += b * h * w

    if sum_c is None or count == 0:
        raise RuntimeError("No samples found to compute stats.")

    mean = (sum_c / count).tolist()
    var = (sumsq_c / count - torch.tensor(mean, dtype=torch.float64) ** 2).clamp(min=0.0)
    std = torch.sqrt(var).tolist()

    dataset_path = Path(args.dataset_path)
    if args.out:
        out_path = Path(args.out)
    else:
        if dataset_path.is_dir():
            out_path = dataset_path / "norm_stats.json"
        else:
            out_path = dataset_path.with_suffix(".norm.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"mean": mean, "std": std, "count": count}, f, indent=2)

    print(f"Saved normalization stats to {out_path}")


if __name__ == "__main__":
    main()
