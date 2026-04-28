#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import random
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_build.src.utils import read_yaml
from ml.src.dataloaders import AMRDataset, NormalizeByStats
from ml.src.models import AttnDeltaFullRes, HeteroDeltaFullRes, UNet
from ml.src.utils import load_checkpoint, load_norm_stats, setup_logger


def overlay(label: np.ndarray, pred: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.imshow(label, cmap="gray", vmin=0, vmax=1)
    ax.imshow(label, cmap="Reds", alpha=0.4)
    ax.imshow(pred, cmap="Blues", alpha=0.4)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_reg_viz(gt: np.ndarray, pred: np.ndarray, aux: np.ndarray | None, out_path: Path) -> None:
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

    aux_vis = None
    aux_title = None
    if aux is not None:
        if aux.ndim == 3:
            aux_vis = aux[0]
        elif aux.ndim == 2:
            aux_vis = aux
        aux_title = "aux"

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


def build_model(model_cfg: dict, in_channels: int, model_type: str) -> torch.nn.Module:
    model_type = model_type.lower()
    if model_type in {"unet", "unet2d"}:
        return UNet(
            in_channels=in_channels,
            out_channels=int(model_cfg.get("out_channels", 1)),
            depth=int(model_cfg.get("depth", 4)),
            base_channels=int(model_cfg.get("base_channels", 32)),
        )
    if model_type in {"attn_delta", "attn"}:
        return AttnDeltaFullRes(
            in_ch=in_channels,
            out_ch=int(model_cfg.get("out_channels", 3)),
            base=int(model_cfg.get("base_channels", 32)),
        )
    if model_type in {"hetero_delta", "hetero"}:
        return HeteroDeltaFullRes(
            in_ch=in_channels,
            out_ch_mean=int(model_cfg.get("out_channels", 3)),
            out_ch_logvar=int(model_cfg.get("out_ch_logvar", 1)),
            base=int(model_cfg.get("base_channels", 32)),
            logvar_min=float(model_cfg.get("logvar_min", -12.0)),
            logvar_max=float(model_cfg.get("logvar_max", 6.0)),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _invert_target_transform(
    y: np.ndarray,
    scale,
    mean,
    std,
    eps: float = 1e-8,
) -> np.ndarray:
    if mean is not None and std is not None:
        mean = np.asarray(mean).reshape(-1, 1, 1)
        std = np.asarray(std).reshape(-1, 1, 1)
        y = y * (std + eps) + mean
    if scale is not None:
        scale = np.asarray(scale)
        if scale.size == 1:
            y = y / scale
        else:
            y = y / scale.reshape(-1, 1, 1)
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with trained model")
    parser.add_argument("--config", default="ml/configs/infer_default.yaml", help="Infer config")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    logger = setup_logger("infer")

    checkpoint_path = Path(cfg.get("checkpoint", "ml/runs/latest/best.ckpt"))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model_cfg = ckpt.get("model_cfg", {})

    dataset_path = cfg.get("dataset_path", "ml/data/processed/cylinder_demo")
    split = cfg.get("split", "test")
    task = cfg.get("task", "seg")
    model_type = cfg.get("model_type") or ckpt.get("model_type", "unet")

    transform = None
    norm_stats_path = cfg.get("norm_stats")
    if norm_stats_path:
        stats = load_norm_stats(norm_stats_path)
        transform = NormalizeByStats(stats["mean"], stats["std"])
    ds = AMRDataset(dataset_path, split=split, task=task, transform=transform)
    in_channels = ds[0]["x"].shape[0]
    model = build_model(model_cfg, in_channels=in_channels, model_type=model_type)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    out_dir = Path(cfg.get("out_dir", f"ml/runs/infer_{checkpoint_path.parent.name}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_dir = out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    viz_samples = int(cfg.get("viz_samples", 0))
    viz_seed = int(cfg.get("viz_seed", 42))
    viz_out_dir = Path(cfg.get("viz_out_dir", str(out_dir / "viz")))
    viz_set = set()
    if viz_samples > 0:
        rng = random.Random(viz_seed)
        k = min(viz_samples, len(ds))
        viz_set = set(rng.sample(range(len(ds)), k))
        viz_out_dir.mkdir(parents=True, exist_ok=True)

    target_scale = cfg.get("target_scale", cfg.get("delta_target_scale"))
    target_norm_stats = cfg.get("target_norm_stats", cfg.get("delta_norm_stats"))
    target_mean = None
    target_std = None
    if target_norm_stats:
        tstats = load_norm_stats(target_norm_stats)
        target_mean = tstats["mean"]
        target_std = tstats["std"]

    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            x = sample["x"].unsqueeze(0).to(device)
            logits = model(x)
            aux = None
            if isinstance(logits, (tuple, list)):
                aux = logits[1]
                logits = logits[0]
            if task == "seg":
                pred = torch.sigmoid(logits).cpu().numpy()[0, 0]
                pred_mask = (pred > 0.5).astype(np.float32)
            else:
                pred = logits.cpu().numpy()[0]
                pred = _invert_target_transform(pred, target_scale, target_mean, target_std)
                pred_mask = None

            if aux is not None:
                aux_np = aux.cpu().numpy()[0]
            else:
                aux_np = None
            np.savez_compressed(preds_dir / f"{i:05d}.npz", pred=pred, mask=pred_mask, aux=aux_np)

            if task == "seg":
                label = sample["y"].numpy()[0]
                overlay(label, pred_mask, overlay_dir / f"{i:05d}.png")
            else:
                if viz_set and i in viz_set:
                    gt = sample["y"].numpy()
                    save_reg_viz(gt, pred, aux_np, viz_out_dir / f"{i:05d}.png")

    logger.info("Inference complete. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
