# `ml/` — Heteroscedastic U-Net

PyTorch implementation of the deep-learning AMR refinement indicator.

## Layout

```
ml/
├── src/
│   ├── train.py                   # training entry point
│   ├── eval.py                    # evaluation on test set
│   ├── infer.py                   # batched inference, writes predictions to disk
│   ├── export_torchscript.py      # export checkpoint to TorchScript (.ts)
│   ├── compute_norm.py            # compute per-channel normalisation stats
│   ├── models/                    # HeteroDeltaFullRes (paper), AttnDeltaFullRes, UNet
│   ├── losses/                    # NLL, BCE-Dice, GDL, SSIM, sliced Wasserstein
│   ├── metrics/                   # regression and segmentation metrics
│   ├── dataloaders/               # AMRDataset, normalisation transforms
│   └── utils/                     # checkpoint, logging, normalisation, seeding
├── configs/                       # YAML configs used in the paper
└── pretrained/                    # download instructions + model card
```

## Run as a package, not a script

The modules use intra-package imports (`from ml.src.X import ...`). Run from
the **repo root** as Python modules:

```bash
python -m ml.src.dataloaders.build_nc_delta --cases cases --output ml/data/processed/nc_delta_uv_uni
python -m ml.src.train --config ml/configs/train_nc_delta_hetero.yaml     # run used in the paper (seed 517)
python -m ml.src.infer --config ml/configs/infer_nc_delta_hetero.yaml     # writes preds/*.npz for the test block
python -m ml.src.eval  --config ml/configs/eval_nc_delta_hetero.yaml
```

## Dataset format

Training and evaluation expect a preprocessed dataset directory containing:

| File                  | Format         | Contents                                              |
|-----------------------|----------------|-------------------------------------------------------|
| `train.pt`            | `torch.save`   | dict with `X: (N, 2, 160, 432)`, `y: (N, 2, 160, 432)`, `mask: (N, 1, 160, 432)` |
| `val.pt`              | `torch.save`   | same schema                                            |
| `test.pt`             | `torch.save`   | same schema                                            |
| `norm_stats.json`     | JSON           | per-channel mean/std for input $(u^\ast, v^\ast)$      |
| `target_norm.json`    | JSON           | per-channel mean/std for the target residual            |
| `blocked_split_ids.json` | JSON        | snapshot indices of the chronological train/val/test blocks |

- `x` channels: $(u^\ast, v^\ast) = (U_x/U_\infty, U_y/U_\infty)$. Pressure is excluded
  because the remapping transients of dynamic refinement contaminate it.
- `y` channels: mesh-induced residual
  $\mathbf{q}^{\mathrm{fine}} - \mathbf{q}^{\mathrm{coarse}}$, with snapshots paired by
  instantaneous shedding phase.
- Spatial grid: uniform Cartesian $432 \times 160$ covering
  $x/D \in [-2, 25]$, $y/D \in [-5, 5]$.
- The dataset is built by `ml/src/dataloaders/build_nc_delta.py` from the
  coarse and fine training cases `cases/circular_Re{100,150}/{coarse,fine}`:
  both are sampled onto the grid, coarse and fine snapshots are paired by the
  Hilbert phase of the lift signal, and the pairs are split into contiguous
  chronological blocks (80/10/10 %). The "fine" training case is the medium
  mesh of the grid-convergence study (Section 2.2 of the paper).

The `dataset_path` field in each YAML config points to the directory
holding these files; override per run with command-line YAML override.

## Pretrained model

See [`pretrained/README.md`](pretrained/README.md) for download instructions
and [`pretrained/model_card.md`](pretrained/model_card.md) for training data,
architecture, and performance.
