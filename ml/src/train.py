#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.dataloaders import AMRDataset, NormalizeByStats
from ml.src.losses import BCEDiceLoss, GeneralizedDiceLoss, ssim_loss, sliced_wasserstein_distance
from ml.src.metrics import dice_score, iou_score, mae, mse
from ml.src.models import AttnDeltaFullRes, HeteroDeltaFullRes, UNet
from ml.src.utils import load_norm_stats, save_checkpoint, seed_everything, setup_logger
from dataset_build.src.utils import read_yaml
from registry.registry import register_experiment


def build_model(model_cfg: dict, in_channels: int, model_type: str) -> nn.Module:
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


def export_torchscript_subprocess(
    ckpt_path: Path,
    model_cfg_path: str | None,
    model_type: str,
    in_channels: int,
    example_input: torch.Tensor,
    out_path: Path,
) -> None:
    import subprocess

    h = int(example_input.shape[-2])
    w = int(example_input.shape[-1])
    cmd = [
        sys.executable,
        str(ROOT / "ml" / "src" / "export_torchscript.py"),
        "--checkpoint",
        str(ckpt_path),
        "--out",
        str(out_path),
        "--in_channels",
        str(in_channels),
        "--height",
        str(h),
        "--width",
        str(w),
    ]
    if model_type:
        cmd += ["--model_type", model_type]
    if model_cfg_path:
        cmd += ["--model_config", model_cfg_path]
    subprocess.run(cmd, check=True)


def export_state_dict(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, str(path))


def _apply_target_transform(
    y: torch.Tensor,
    scale,
    mean,
    std,
    eps: float = 1e-8,
) -> torch.Tensor:
    if scale is not None:
        s = torch.as_tensor(scale, device=y.device, dtype=y.dtype)
        if s.numel() == 1:
            y = y * s
        else:
            y = y * s.view(-1, 1, 1)
    if mean is not None and std is not None:
        m = torch.as_tensor(mean, device=y.device, dtype=y.dtype).view(-1, 1, 1)
        s = torch.as_tensor(std, device=y.device, dtype=y.dtype).view(-1, 1, 1)
        y = (y - m) / (s + eps)
    return y


def _invert_target_transform(
    y: torch.Tensor,
    scale,
    mean,
    std,
    eps: float = 1e-8,
) -> torch.Tensor:
    if mean is not None and std is not None:
        m = torch.as_tensor(mean, device=y.device, dtype=y.dtype).view(-1, 1, 1)
        s = torch.as_tensor(std, device=y.device, dtype=y.dtype).view(-1, 1, 1)
        y = y * (s + eps) + m
    if scale is not None:
        sc = torch.as_tensor(scale, device=y.device, dtype=y.dtype)
        if sc.numel() == 1:
            y = y / sc
        else:
            y = y / sc.view(-1, 1, 1)
    return y


def gaussian_nll_loss(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL without the constant 0.5*log(2*pi)."""
    if logvar.shape != mean.shape:
        if logvar.ndim == mean.ndim and logvar.shape[1] == 1:
            logvar = logvar.expand_as(mean)
        else:
            raise ValueError(f"logvar shape {tuple(logvar.shape)} is not broadcastable to mean shape {tuple(mean.shape)}")
    return 0.5 * (torch.exp(-logvar) * (target - mean) ** 2 + logvar).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UNet for AMR error segmentation/regression")
    parser.add_argument("--config", default="ml/configs/train_default.yaml", help="Training config")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    model_cfg_path = cfg.get("model_config", "ml/configs/model_unet.yaml")
    model_cfg = read_yaml(model_cfg_path)

    seed_everything(int(cfg.get("seed", 42)))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.get("run_dir", f"ml/runs/{run_id}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("train", log_file=str(run_dir / "train.log"))

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    use_wandb = bool(cfg.get("use_wandb", False))
    if use_wandb and not os.environ.get("WANDB_API_KEY"):
        use_wandb = False
    if use_wandb:
        try:
            import wandb
        except Exception:
            use_wandb = False
        if use_wandb:
            wandb.init(project=cfg.get("wandb_project", "dl-amr"), config=cfg, name=run_id)

    task = cfg.get("task", "seg")
    model_type = cfg.get("model_type", "unet")
    dataset_path = cfg.get("dataset_path", "ml/data/processed/cylinder_demo")
    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 2))

    norm_stats_path = cfg.get("norm_stats")
    transform = None
    if norm_stats_path:
        stats = load_norm_stats(norm_stats_path)
        transform = NormalizeByStats(stats["mean"], stats["std"])

    target_scale = cfg.get("target_scale", cfg.get("delta_target_scale"))
    target_norm_stats = cfg.get("target_norm_stats", cfg.get("delta_norm_stats"))
    target_mean = None
    target_std = None
    if task == "reg" and target_norm_stats:
        tstats = load_norm_stats(target_norm_stats)
        target_mean = tstats["mean"]
        target_std = tstats["std"]

    train_ds = AMRDataset(dataset_path, split="train", task=task, transform=transform)
    val_ds = AMRDataset(dataset_path, split="val", task=task, transform=transform)

    sample_x = train_ds[0]["x"]
    in_channels = sample_x.shape[0]
    example_input = sample_x.unsqueeze(0)
    model = build_model(model_cfg, in_channels=in_channels, model_type=model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if task == "seg":
        loss_type = ""
        loss_fn = BCEDiceLoss(bce_weight=float(cfg.get("bce_weight", 0.5)))
    else:
        loss_type = str(cfg.get("reg_loss", "l1")).lower()
        if loss_type == "l1":
            loss_fn = nn.L1Loss()
        elif loss_type == "mse":
            loss_fn = nn.MSELoss()
        elif loss_type in {"nll", "gaussian_nll", "hetero_nll"}:
            loss_fn = None
        else:
            raise ValueError(f"Unsupported reg_loss: {loss_type}")

    gdl_weight = float(cfg.get("gdl_weight", 0.0))
    ssim_weight = float(cfg.get("ssim_weight", 0.0))
    swd_weight = float(cfg.get("swd_weight", 0.0))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    amp_enabled = bool(cfg.get("amp", False))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    epochs = int(cfg.get("epochs", 10))
    default_best_metric = "val_dice" if task == "seg" else ("val_loss" if loss_type in {"nll", "gaussian_nll", "hetero_nll"} else "val_mae")
    best_metric_name = str(cfg.get("best_metric", default_best_metric)).lower()
    higher_is_better = best_metric_name in {"val_dice", "dice", "val_iou", "iou"}
    best_metric = -1e9 if task == "seg" else 1e9
    export_ts = bool(cfg.get("export_torchscript", True))
    ts_path_cfg = cfg.get("torchscript_path")
    export_pt = bool(cfg.get("export_state_dict", True))
    pt_path_cfg = cfg.get("state_dict_path")

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_mae", "val_mse"])

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            if task == "reg" and (target_scale is not None or target_mean is not None):
                y = _apply_target_transform(y, target_scale, target_mean, target_std)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(x)
                if task == "seg":
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    loss = loss_fn(logits, y)
                    if gdl_weight > 0:
                        loss = loss + gdl_weight * GeneralizedDiceLoss()(logits, y)
                else:
                    if loss_type in {"nll", "gaussian_nll", "hetero_nll"}:
                        if not isinstance(logits, (tuple, list)) or len(logits) < 2:
                            raise ValueError("reg_loss=nll requires a heteroscedastic model returning (mean, logvar)")
                        logits, logvar = logits[0], logits[1]
                        loss = gaussian_nll_loss(logits, logvar, y)
                    else:
                        if isinstance(logits, (tuple, list)):
                            logits = logits[0]
                        loss = loss_fn(logits, y)
                    if ssim_weight > 0:
                        loss = loss + ssim_weight * ssim_loss(logits, y)
                    if swd_weight > 0:
                        loss = loss + swd_weight * sliced_wasserstein_distance(logits, y)
            scaler.scale(loss).backward()
            if cfg.get("grad_clip", 0.0):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        val_mae = 0.0
        val_mse = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                y_raw = y
                if task == "reg" and (target_scale is not None or target_mean is not None):
                    y = _apply_target_transform(y, target_scale, target_mean, target_std)
                logits = model(x)
                if task == "seg":
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    loss = loss_fn(logits, y)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_dice += dice_score(preds, y).item()
                    val_iou += iou_score(preds, y).item()
                else:
                    if loss_type in {"nll", "gaussian_nll", "hetero_nll"}:
                        if not isinstance(logits, (tuple, list)) or len(logits) < 2:
                            raise ValueError("reg_loss=nll requires a heteroscedastic model returning (mean, logvar)")
                        logits, logvar = logits[0], logits[1]
                        loss = gaussian_nll_loss(logits, logvar, y)
                    else:
                        if isinstance(logits, (tuple, list)):
                            logits = logits[0]
                        loss = loss_fn(logits, y)
                    pred_eval = logits
                    if task == "reg" and (target_scale is not None or target_mean is not None):
                        pred_eval = _invert_target_transform(pred_eval, target_scale, target_mean, target_std)
                    val_mae += mae(pred_eval, y_raw).item()
                    val_mse += mse(pred_eval, y_raw).item()
                val_loss += loss.item()

        val_loss /= max(len(val_loader), 1)
        if task == "seg":
            val_dice /= max(len(val_loader), 1)
            val_iou /= max(len(val_loader), 1)
        else:
            val_mae /= max(len(val_loader), 1)
            val_mse /= max(len(val_loader), 1)

        metric_map = {
            "val_loss": val_loss,
            "loss": val_loss,
            "nll": val_loss,
            "val_mae": val_mae,
            "mae": val_mae,
            "val_mse": val_mse,
            "mse": val_mse,
            "val_dice": val_dice,
            "dice": val_dice,
            "val_iou": val_iou,
            "iou": val_iou,
        }
        if best_metric_name not in metric_map:
            raise ValueError(f"Unsupported best_metric: {best_metric_name}")
        metric = metric_map[best_metric_name]
        is_best = metric > best_metric if higher_is_better else metric < best_metric

        if is_best:
            best_metric = metric
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": model_cfg,
                    "model_type": model_type,
                    "epoch": epoch,
                    "metric": metric,
                },
                run_dir / "best.ckpt",
            )
            if export_pt:
                pt_path = Path(pt_path_cfg) if pt_path_cfg else Path("best.pt")
                if not pt_path.is_absolute():
                    pt_path = run_dir / pt_path
                export_state_dict(model, pt_path)
                logger.info("Saved state_dict model: %s", pt_path)

        save_checkpoint(
            {
                "model_state": model.state_dict(),
                "model_cfg": model_cfg,
                "model_type": model_type,
                "epoch": epoch,
                "metric": metric,
            },
            run_dir / "last.ckpt",
        )

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_dice, val_iou, val_mae, val_mse])

        logger.info(
            "Epoch %d | train %.4f | val %.4f | dice %.4f | iou %.4f | mae %.4f",
            epoch,
            train_loss,
            val_loss,
            val_dice,
            val_iou,
            val_mae,
        )
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "val_mae": val_mae,
                    "val_mse": val_mse,
                }
            )

    # Plot curves
    data = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    if data:
        epochs_list = [int(r["epoch"]) for r in data]
        train_losses = [float(r["train_loss"]) for r in data]
        val_losses = [float(r["val_loss"]) for r in data]
        plt.figure()
        plt.plot(epochs_list, train_losses, label="train")
        plt.plot(epochs_list, val_losses, label="val")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig(run_dir / "learning_curve.png")
        plt.close()

    register_experiment(
        name=run_id,
        config=cfg,
        metrics={"best_metric": best_metric},
        output_dir=str(run_dir),
    )
    latest = Path("ml/runs/latest")
    if latest.is_symlink() or not latest.exists():
        if latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_dir.resolve())
    elif latest.exists():
        logger.warning("ml/runs/latest exists and is not a symlink. Skipping update.")
    if use_wandb:
        wandb.finish()

    if export_ts:
        best_ckpt = run_dir / "best.ckpt"
        if best_ckpt.exists():
            ts_path = Path(ts_path_cfg) if ts_path_cfg else Path("best.ts")
            if not ts_path.is_absolute():
                ts_path = run_dir / ts_path
            export_torchscript_subprocess(
                best_ckpt,
                model_cfg_path,
                model_type,
                in_channels,
                example_input,
                ts_path,
            )
            logger.info("Saved TorchScript model: %s", ts_path)
        else:
            logger.warning("best.ckpt not found; skipping TorchScript export")

    logger.info("Training complete. Outputs in %s", run_dir)


if __name__ == "__main__":
    main()
