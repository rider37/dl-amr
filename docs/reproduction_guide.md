# Reproduction guide

This document describes how to reproduce the results of:

> Kang et al., *Heteroscedastic Temporal-Residual Learning for Dynamic
> Adaptive Mesh Refinement in Bluff-Body Wake Flows*, JCP 2026.

There are two reproduction levels:

1. **Quick** — verify environment, regenerate paper figures from cached data.
2. **Full** — run all OpenFOAM simulations end-to-end (long-running).

---

## 1. Prerequisites

| Component         | Tested version | Notes                                   |
|-------------------|----------------|-----------------------------------------|
| Python            | 3.10+          | PyTorch, NumPy, SciPy, matplotlib, PyVista, pandas, scikit-learn |
| OpenFOAM          | v2312          | Other recent versions may work          |
| LibTorch (C++)    | matched to PyTorch export version | Required for `amrPimpleFoam` runtime ML inference |
| GCC / clang       | C++17 compatible | For OpenFOAM solver build              |
| GPU (NVIDIA)      | CUDA 11.8+ (optional) | Speeds up ML inference; CPU fallback supported |

Install Python deps:
```bash
make install                          # uses env.yml (conda) or requirements.txt
```

Build the OpenFOAM solvers (`amrPimpleFoam` and `protectedPimpleFoam`):
```bash
source $WM_PROJECT_DIR/etc/bashrc     # ensure OpenFOAM env is sourced
export LIBTORCH_DIR=/path/to/libtorch # required for amrPimpleFoam
make solver
```

---

## 2. Quick reproduction (paper figures)

Each figure falls into one of three reproducibility tiers (see
[`../reference_data/README.md`](../reference_data/README.md) for details):

| Tier | Figures            | Data source                                                       |
|------|--------------------|--------------------------------------------------------------------|
| 1    | Fig 6              | none — values inlined in the script (works out of the box)         |
| 2    | Figs 9, 10, 11, 12 | `make download-reference` (ML test set + per-sample predictions)   |
| 3    | Figs 1, 3, 4, 5, 7, 8 | full OpenFOAM case results in `cases/<geom>/<method>/`         |

```bash
make download-models                  # pretrained model
make download-reference               # Tier-2 data: test.pt + preds/
make smoke-test                       # verify environment + cases
make figs                             # regenerate paper figures
```

`make figs` runs every figure script. Tier-1 always succeeds; Tier-2/3
scripts whose data is absent print a clear *[SKIP]* line and exit with
code 2 (counted as **skipped**, not failed). Outputs go to
`analysis/output/` (configurable via `DL_AMR_OUTDIR`).

Each figure script can also be run individually from the repo root:

```bash
python analysis/generate_fig1_7_overview.py                # Figs 1, 7 (composite + line sampling)
python analysis/generate_fig3_phase_averaged.py            # Fig 3
python analysis/generate_fig4_5_instantaneous_and_umean.py # Figs 4, 5
python analysis/generate_fig6_error_vs_dof.py              # Fig 6
python analysis/generate_fig8_anchor_variants.py           # Fig 8
python analysis/generate_fig9_10_11_uncertainty.py         # Figs 9, 10, 11
python analysis/generate_fig12_threshold_sensitivity.py    # Fig 12
```

If you have already run all OpenFOAM simulations and have predictions on
disk, you can skip `make download-reference` and point environment variables
at your local copies:

```bash
export DL_AMR_CASES=/path/to/cases       # OpenFOAM cases base
export DL_AMR_DATA=/path/to/csv/data     # CSVs (sweep summaries, calibration)
export DL_AMR_OUTDIR=/path/to/output     # figure output directory
```

---

## 3. Full reproduction (OpenFOAM simulations)

> ⚠ Full simulations of all cases take many CPU-hours. Plan accordingly.

### 3.1 Single case (example: circular DL-AMR)

```bash
cd cases/circular_Re200/dl_amr
ln -sf $(pwd)/../../../ml/pretrained/heteroscedastic_unet.pt constant/model.ts
./Allrun
```

The `Allrun` script runs `blockMesh`, sets up parallel decomposition (if
configured), and calls `amrPimpleFoam`.

### 3.2 All baseline cases for a geometry

Targets follow the pattern
`run-{circular,square,diamond}-{fine,coarse,dl-amr,grad-amr}`:

```bash
# Circular cylinder, Re=200
make run-circular-fine
make run-circular-coarse
make run-circular-dl-amr
make run-circular-grad-amr

# Square cylinder, Re=150
make run-square-fine
make run-square-coarse
make run-square-dl-amr
make run-square-grad-amr

# Diamond cylinder, Re=150
make run-diamond-fine
make run-diamond-coarse
make run-diamond-dl-amr
make run-diamond-grad-amr
```

Each target is equivalent to `cd cases/<geom>/<method> && ./Allrun`.

### 3.3 Post-processing

After simulations, generate paper figures from the new results:

```bash
make figs
```

---

## 4. Training the model from scratch (optional)

The pretrained model used in the paper is provided. To retrain:

1. **Prepare the training dataset.** The training pipeline expects a
   preprocessed dataset under `ml/data/processed/<dataset_name>/` with
   files `train.pt`, `val.pt`, `test.pt` (PyTorch tensors saved with
   `torch.save`) plus `norm_stats.json` and `target_norm.json`. Each `.pt`
   contains a dict with keys `X` (input, shape `(N, 3, 64, 224)`,
   channels = $u^\ast, v^\ast, p^\ast$), `y` (target temporal residual,
   same shape), and `mask` (wake-region mask, shape `(N, 64, 224)`).

   The dataset used in the paper was generated from circular-cylinder
   reference simulations at $Re = 100$ and $Re = 150$ (run the
   corresponding `cases/circular_Re*/fine/` cases, then resample
   snapshots to a $64 \times 224$ uniform grid covering the wake region
   $x/D \in [2, 39]$, $y/D \in [-5, 5]$, and form
   $\Delta\mathbf{q}_t = \mathbf{q}_{t+1} - \mathbf{q}_t$).

2. **Train:**
   ```bash
   python -m ml.src.train --config ml/configs/train_delta_hetero_star_uvp_wake_re100_150_nll_lr3e4.yaml
   ```
   Override the dataset location with the `dataset_path` field in the
   YAML or set `DL_AMR_DATA` if your scripts honour it.

3. **Export to TorchScript** for OpenFOAM consumption:
   ```bash
   python -m ml.src.export_torchscript --checkpoint <path-to-best.ckpt> \
                                       --output ml/pretrained/heteroscedastic_unet.pt
   ```

> The full preprocessed training dataset (~40 GB) is not included in this
> repository. See [`data_availability.md`](data_availability.md) for
> licensing and request information.

---

## 5. Expected runtimes (hardware reference)

Reference machine: AMD Ryzen 9 9900X (single-core), NVIDIA RTX 3090.

| Task                                  | Wall time |
|---------------------------------------|-----------|
| Smoke test                            | <1 min    |
| Figure regeneration (all)             | ~5 min    |
| Single OpenFOAM case (fine)           | ~hours    |
| Single OpenFOAM case (DL-AMR)         | ~hours (60–70% of fine) |
| Model training (100 epochs)           | ~2–4 h    |

---

## 6. Troubleshooting

| Symptom                                   | Cause / fix                                           |
|-------------------------------------------|--------------------------------------------------------|
| `model.ts not found` in `dl_amr/Allrun`  | Run `make download-models`, then symlink (see §3.1)    |
| `amrPimpleFoam: command not found`       | Source OpenFOAM env (`source $WM_PROJECT_DIR/etc/bashrc`) and `make solver` |
| LibTorch link errors during solver build | Set `LIBTORCH_DIR` env var to your LibTorch install path |
| Float arithmetic warnings in OpenFOAM    | Use `gcc 9+` and double precision build (DPInt32Opt)   |

For other issues, please open a GitHub issue.
