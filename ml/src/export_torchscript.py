#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.models import AttnDeltaFullRes, HeteroDeltaFullRes, UNet
from ml.src.utils import load_checkpoint
from dataset_build.src.utils import read_yaml


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TorchScript from checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to best.ckpt")
    parser.add_argument("--model_type", default=None, help="Model type override")
    parser.add_argument("--model_config", default=None, help="Model config YAML override")
    parser.add_argument("--out", required=True, help="Output TorchScript path")
    parser.add_argument("--in_channels", type=int, required=True, help="Input channels")
    parser.add_argument("--height", type=int, required=True, help="Input height")
    parser.add_argument("--width", type=int, required=True, help="Input width")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    model_cfg = ckpt.get("model_cfg", {})
    if args.model_config:
        model_cfg = read_yaml(args.model_config)
    model_type = args.model_type or ckpt.get("model_type", "unet")

    model = build_model(model_cfg, in_channels=args.in_channels, model_type=model_type)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    example = torch.zeros((1, args.in_channels, args.height, args.width), dtype=torch.float32)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        try:
            scripted = torch.jit.script(model)
            scripted.save(str(out_path))
            return
        except Exception:
            traced = torch.jit.trace(model, example, strict=False)
            traced.save(str(out_path))


if __name__ == "__main__":
    main()
