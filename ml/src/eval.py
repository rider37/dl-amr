#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import random
from pathlib import Path

import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_build.src.utils import read_yaml
from ml.src.dataloaders import AMRDataset, NormalizeByStats
from ml.src.metrics import dice_score, iou_score, mae, mse, nrmse
from ml.src.utils import load_norm_stats, setup_logger
from registry.registry import register_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--config", default="ml/configs/eval_default.yaml", help="Eval config")
    parser.add_argument("--viz_samples", type=int, default=0, help="Random samples to visualize (seg only)")
    parser.add_argument("--viz_out_dir", default=None, help="Output directory for overlays")
    parser.add_argument("--viz_seed", type=int, default=42, help="Random seed for visualization")
    parser.add_argument(
        "--viz_with_flow",
        action="store_true",
        help="Visualize coarse/fine flow magnitude alongside predicted mask when fields are available",
    )
    parser.add_argument("--report_csv", default=None, help="Optional CSV path for metrics")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    logger = setup_logger("eval")

    dataset_path = cfg.get("dataset_path", "ml/data/processed/cylinder_demo")
    split = cfg.get("split", "test")
    task = cfg.get("task", "seg")
    preds_dir = Path(cfg.get("preds_dir", "ml/runs/latest/preds"))

    transform = None
    norm_stats_path = cfg.get("norm_stats")
    if norm_stats_path:
        stats = load_norm_stats(norm_stats_path)
        transform = NormalizeByStats(stats["mean"], stats["std"])
    ds = AMRDataset(dataset_path, split=split, task=task, transform=transform)

    metrics = {"dice": [], "iou": [], "mae": [], "mse": [], "nrmse": []}
    available_indices = []
    for i in range(len(ds)):
        sample = ds[i]
        pred_path = preds_dir / f"{i:05d}.npz"
        if not pred_path.exists():
            logger.warning("Missing pred %s", pred_path)
            continue
        available_indices.append(i)
        pred_data = np.load(pred_path, allow_pickle=True)
        pred_np = pred_data["pred"]
        if pred_np.ndim == 2:
            pred = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0)
        else:
            pred = torch.from_numpy(pred_np).unsqueeze(0)
        target = sample["y"]
        if target.ndim == 2:
            target = target.unsqueeze(0)
        target = target.unsqueeze(0)

        if task == "seg":
            pred_mask = (pred > 0.5).float()
            metrics["dice"].append(dice_score(pred_mask, target).item())
            metrics["iou"].append(iou_score(pred_mask, target).item())
        else:
            metrics["mae"].append(mae(pred, target).item())
            metrics["mse"].append(mse(pred, target).item())
            metrics["nrmse"].append(nrmse(pred, target).item())

    report_path = Path(cfg.get("report_path", "ml/runs/latest/eval_report.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: float(np.mean(v)) if v else None for k, v in metrics.items()}
    with open(report_path, "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2)

    report_csv = args.report_csv or cfg.get("report_csv")
    if report_csv:
        csv_path = Path(report_csv)
    else:
        csv_path = report_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    # Visualization
    if args.viz_samples > 0 and available_indices:
        out_dir = Path(args.viz_out_dir) if args.viz_out_dir else report_path.parent / "viz_samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(args.viz_seed)
        k = min(args.viz_samples, len(available_indices))
        chosen = rng.sample(available_indices, k)

        fields_root = None
        if args.viz_with_flow:
            ds_path = Path(dataset_path)
            dataset_name = ds_path.name if ds_path.is_dir() else ds_path.stem
            candidate = Path("dataset_build/outputs/fields") / dataset_name
            if candidate.exists():
                fields_root = candidate

        for idx in chosen:
            pred_path = preds_dir / f"{idx:05d}.npz"
            pred_data = np.load(pred_path, allow_pickle=True)
            sample = ds[idx]
            meta = ds.meta[idx] if hasattr(ds, "meta") and ds.meta else {}
            case_key = meta.get("case_key", meta.get("case", f"{idx:05d}"))
            time = meta.get("time", idx)
            out_path = out_dir / f"{case_key}_{time}.png"

            if task == "seg":
                pred_mask = pred_data.get("mask")
                if pred_mask is None:
                    pred_mask = (pred_data["pred"] > 0.5).astype(np.float32)
                label = sample["y"].numpy()[0]

                if args.viz_with_flow and fields_root is not None:
                    field_path = fields_root / case_key / f"{time}.npz"
                    u_coarse = None
                    u_proj = None
                    if field_path.exists():
                        fdata = np.load(field_path, allow_pickle=True)
                        u_coarse = fdata.get("U_coarse")
                        u_proj = fdata.get("U_proj")
                    if u_coarse is not None and u_proj is not None:
                        coarse_mag = np.sqrt(u_coarse[0] ** 2 + u_coarse[1] ** 2)
                        fine_mag = np.sqrt(u_proj[0] ** 2 + u_proj[1] ** 2)
                        vmin = float(min(coarse_mag.min(), fine_mag.min()))
                        vmax = float(max(coarse_mag.max(), fine_mag.max()))

                        fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=150)
                        im0 = axes[0].imshow(coarse_mag, cmap="viridis", vmin=vmin, vmax=vmax)
                        axes[0].set_title("coarse |U|", fontsize=8)
                        im1 = axes[1].imshow(fine_mag, cmap="viridis", vmin=vmin, vmax=vmax)
                        axes[1].set_title("fine |U|", fontsize=8)
                        im2 = axes[2].imshow(pred_mask, cmap="Blues", vmin=0, vmax=1)
                        axes[2].set_title("pred mask", fontsize=8)
                        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                        for ax in axes:
                            ax.set_axis_off()
                        fig.tight_layout(pad=0.2)
                    else:
                        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
                        ax.imshow(label, cmap="gray", vmin=0, vmax=1)
                        ax.imshow(label, cmap="Reds", alpha=0.4)
                        ax.imshow(pred_mask, cmap="Blues", alpha=0.4)
                        ax.set_axis_off()
                        fig.tight_layout(pad=0)
                else:
                    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
                    ax.imshow(label, cmap="gray", vmin=0, vmax=1)
                    ax.imshow(label, cmap="Reds", alpha=0.4)
                    ax.imshow(pred_mask, cmap="Blues", alpha=0.4)
                    ax.set_axis_off()
                    fig.tight_layout(pad=0)
            else:
                # regression visualization: GT, pred, abs error
                gt = sample["y"].numpy()
                pred = pred_data["pred"]
                if gt.ndim == 3 and pred.ndim == 3:
                    gt_vis = np.linalg.norm(gt, axis=0)
                    pred_vis = np.linalg.norm(pred, axis=0)
                    err = np.abs(pred_vis - gt_vis)
                    title = "|Δ|"
                else:
                    gt_vis = gt[0] if gt.ndim == 3 else gt
                    pred_vis = pred[0] if pred.ndim == 3 else pred
                    err = np.abs(pred_vis - gt_vis)
                    title = "e_U"
                aux = pred_data.get("aux")
                aux_vis = None
                aux_title = None
                if aux is not None:
                    if aux.ndim == 3:
                        aux_vis = aux[0]
                    elif aux.ndim == 2:
                        aux_vis = aux
                    # Try to infer aux meaning from file path name (attn/hetero)
                    if "attn" in str(preds_dir):
                        aux_title = "attn"
                    elif "hetero" in str(preds_dir):
                        aux_title = "unc"
                ncols = 4 if aux_vis is not None else 3
                fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3), dpi=150)
                vmin = float(min(gt_vis.min(), pred_vis.min()))
                vmax = float(max(gt_vis.max(), pred_vis.max()))
                im0 = axes[0].imshow(gt_vis, cmap="viridis", vmin=vmin, vmax=vmax)
                axes[0].set_title(f"gt {title}", fontsize=8)
                im1 = axes[1].imshow(pred_vis, cmap="viridis", vmin=vmin, vmax=vmax)
                axes[1].set_title(f"pred {title}", fontsize=8)
                im2 = axes[2].imshow(err, cmap="magma")
                axes[2].set_title("|error|", fontsize=8)
                if aux_vis is not None:
                    im3 = axes[3].imshow(aux_vis, cmap="plasma")
                    axes[3].set_title(aux_title or "aux", fontsize=8)
                    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
                fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                for ax in axes:
                    ax.set_axis_off()
                fig.tight_layout(pad=0.2)
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        logger.info("Saved %d overlay images to %s", k, out_dir)

    run_name = cfg.get("run_name", f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    register_experiment(name=run_name, config=cfg, metrics=summary, output_dir=str(report_path.parent))
    logger.info("Eval summary: %s", summary)


if __name__ == "__main__":
    main()
