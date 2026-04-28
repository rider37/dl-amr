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
python -m ml.src.train --config ml/configs/train_delta_hetero_star_uvp_wake_re100_150_nll_lr3e4.yaml
python -m ml.src.eval  --config ml/configs/eval_delta_hetero_star_uvp_wake_re100_150_nll_lr3e4.yaml
python -m ml.src.infer --config ml/configs/infer_delta_hetero_star_uvp_wake_re100_150_nll_lr3e4.yaml
```

## Dataset format

Training and evaluation expect a preprocessed dataset directory containing:

| File                  | Format         | Contents                                              |
|-----------------------|----------------|-------------------------------------------------------|
| `train.pt`            | `torch.save`   | dict with `X: (N, 3, 64, 224)`, `y: (N, 3, 64, 224)`, `mask: (N, 64, 224)` |
| `val.pt`              | `torch.save`   | same schema                                            |
| `test.pt`             | `torch.save`   | same schema                                            |
| `norm_stats.json`     | JSON           | per-channel mean/std for input $(u^\ast, v^\ast, p^\ast)$ |
| `target_norm.json`    | JSON           | per-channel mean/std for target $\Delta\mathbf{q}_t$    |

- `x` channels: $(u^\ast, v^\ast, p^\ast) = (U_x/U_\infty, U_y/U_\infty, p/(\rho U_\infty^2))$
- `y` channels: $\Delta\mathbf{q}_t = \mathbf{q}_{t+1} - \mathbf{q}_t$
- Spatial grid: uniform Cartesian $64 \times 224$ covering the wake region
  $x/D \in [2, 39]$, $y/D \in [-5, 5]$.
- The grid is obtained by resampling instantaneous OpenFOAM snapshots from
  `cases/circular_Re100/fine` and `cases/circular_Re150/fine` (the latter
  is not in the paper case set; generate at $Re = 100, 150$ using the
  same case template as `circular_Re200/fine` with the `nu` value in
  `constant/transportProperties` adjusted accordingly).

The `dataset_path` field in each YAML config points to the directory
holding these files; override per run with command-line YAML override.

## Pretrained model

See [`pretrained/README.md`](pretrained/README.md) for download instructions
and [`pretrained/model_card.md`](pretrained/model_card.md) for training data,
architecture, and performance.
