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

Build the OpenFOAM solver:
```bash
source $WM_PROJECT_DIR/etc/bashrc     # ensure OpenFOAM env is sourced
make solver
```

---

## 2. Quick reproduction (paper figures)

```bash
make download-models                  # pretrained model + reference data
make download-reference
make smoke-test                       # verify everything is in place
make figs                             # regenerate all paper figures
```

The figure scripts live in `analysis/`. Each script can also be run
individually:

```bash
python analysis/generate_fig3_phase_averaged_v2.py
python analysis/generate_all_new_figures.py
python analysis/generate_uncertainty_figures.py
# ...
```

Outputs are written to `analysis/output/` (or the paths configured in
each script).

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

```bash
make run-circular-fine
make run-circular-coarse
make run-circular-grad-amr
make run-circular-dl-amr
```

Repeat for `square_Re150` and `diamond_Re150` (define analogous Makefile
targets, or run each case's `Allrun` directly).

### 3.3 Post-processing

After simulations, generate paper figures from the new results:

```bash
make figs
```

---

## 4. Training the model from scratch (optional)

The pretrained model used in the paper is provided. To retrain:

1. Generate the training dataset (preprocessed snapshots) from the
   circular-cylinder reference simulations:
   ```bash
   # Reference cases must be run first:
   #   cases/circular_Re200/fine, plus circular_Re100/fine, circular_Re150/fine
   python ml/src/dataloaders/build_dataset.py --config ml/configs/...
   ```
2. Train:
   ```bash
   python ml/src/train.py --config ml/configs/train_delta_hetero_star_uvp_wake_re100_150_nll_lr3e4.yaml
   ```
3. Export to TorchScript for OpenFOAM consumption:
   ```bash
   python ml/src/export_torchscript.py --checkpoint <path-to-best.ckpt> \
                                       --output ml/pretrained/heteroscedastic_unet.pt
   ```

> The full preprocessed training dataset (~40 GB) is not included in this
> repository. See [`data_availability.md`](data_availability.md) for
> instructions and licensing.

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
