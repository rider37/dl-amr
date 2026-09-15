# Reproduction guide

This document describes how to reproduce the results of:

> Kang et al., *Heteroscedastic Residual Learning for Dynamic
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

Build the `hexRef4` library and the OpenFOAM solver `amrPimpleFoam`:
```bash
source $WM_PROJECT_DIR/etc/bashrc     # ensure OpenFOAM env is sourced
export LIBTORCH_DIR=/path/to/libtorch # required for amrPimpleFoam
make solver
```

---

## 2. Quick reproduction (paper figures)

Each figure falls into one of three reproducibility tiers (see
[`../reference_data/README.md`](../reference_data/README.md) for details):

| Tier | Figures                 | Data source                                                      |
|------|-------------------------|-------------------------------------------------------------------|
| 2    | Figs 8, 9, E.1         | `make download-reference` (ML test split + per-sample predictions) |
| 2    | Figs 4, 5, 7, D.1, F.1  | `make download-fields` (cached time-averaged fields and probes)    |
| 3    | Figs 3, 6, all tables | full OpenFOAM case results in `cases/<geom>/<variant>/`         |

```bash
make download-artifacts               # pretrained model + both reference-data archives
make smoke-test                       # verify environment + cases
make figs                             # regenerate paper figures
```

(`make download-artifacts` runs `download-models`, `download-reference` and
`download-fields`; invoke them separately if you only need one.)

`make figs` runs every figure script. Scripts whose data is absent print a
clear *[SKIP]* line and exit with code 2 (counted as **skipped**, not failed). Outputs go to
`analysis/output/` (configurable via `DL_AMR_OUTDIR`).

Each figure script can also be run individually from the repo root:

```bash
python analysis/uncertainty_figures.py          # Figs 8, 9, E.1
python analysis/figscripts/make_fig4_D1.py      # Figs 4, D.1
python analysis/figscripts/make_fig5.py         # Fig 5
python analysis/figscripts/make_fig7_F1.py      # Figs 7, F.1
python analysis/figscripts/make_fig3_6.py       # Figs 3, 6   (raw case results)
```

The tables are produced by the post-processing pipeline in
`analysis/postprocessing/`; see [`../analysis/README.md`](../analysis/README.md)
for the figure-and-table manifest and the mapping between the authors' run
names and the public case directories.

If you have already run all OpenFOAM simulations and have predictions on
disk, you can skip `make download-reference` and point environment variables
at your local copies:

```bash
export DLAMR_CASES=/path/to/cases        # OpenFOAM case results
export DLAMR_REFDATA=/path/to/reference_data   # test.pt + preds/
export DLAMR_CACHE=/path/to/reference_data/fields
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

`fine` and `coarse` run from `t = 0` (`Allrun` calls `blockMesh` first). The
adaptive and static variants restart from a common coarse-mesh state; see
[`../cases/README.md`](../cases/README.md) for the restart protocol and the
per-variant settings.

### 3.2 All variants of a geometry

Targets follow the pattern `run-<geometry>-<variant>` with geometry in
`circular`, `square`, `diamond` and variant in `fine`, `coarse`, `grad_amr`,
`vort_amr`, `q_amr`, `dl_amr`, `static`, `dl_amr_meanhead` (all geometries)
and `vort_wrapper`, `kelly_wake` (circular only):

```bash
make run-circular-coarse       # produces the restart state first
make run-circular-dl_amr
make run-circular-static
make run-square-dl_amr_meanhead
```

Each target is equivalent to `cd cases/<geom>/<variant> && ./Allrun`.

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
   contains a dict with keys `X` (input, shape `(N, 2, 160, 432)`,
   channels = $u^\ast, v^\ast$), `y` (target mesh-induced residual
   $\mathbf{q}^{\mathrm{fine}} - \mathbf{q}^{\mathrm{coarse}}$, same shape),
   and `mask` (wake-region mask). The split is over contiguous chronological
   blocks, not random, so that no test snapshot is adjacent in time to a
   training snapshot.

   The dataset used in the paper was generated from circular-cylinder
   reference simulations at $Re = 100$ and $Re = 150$ (run the
   corresponding `cases/circular_Re*/fine/` cases and the coarse mesh of
   the same geometry, resample both onto the $432 \times 160$ uniform grid
   covering $x/D \in [-2, 25]$, $y/D \in [-5, 5]$, pair coarse and fine
   snapshots by shedding phase, and form the two-grid discrepancy
   $\Delta\mathbf{q} = \mathbf{q}^{\mathrm{fine}} - \mathbf{q}^{\mathrm{coarse}}$;
   the "fine" solution of the training pairs is the medium mesh of the
   grid-convergence study, Section 2.2 of the paper).

2. **Train:**
   ```bash
   python -m ml.src.train --config ml/configs/train_nc_delta_hetero.yaml
   ```
   Override the dataset location with the `dataset_path` field in the
   YAML.

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
| Single OpenFOAM case (DL-AMR)         | ~23% of the fine-mesh time (circular case, Table 5) |
| Model training (100 epochs)           | ~20 min (RTX 3090) |

---

## 6. Troubleshooting

| Symptom                                   | Cause / fix                                           |
|-------------------------------------------|--------------------------------------------------------|
| `model.ts not found` in `dl_amr/Allrun`  | Run `make download-models`, then symlink (see §3.1)    |
| `amrPimpleFoam: command not found`       | Source OpenFOAM env (`source $WM_PROJECT_DIR/etc/bashrc`) and `make solver` |
| LibTorch link errors during solver build | Set `LIBTORCH_DIR` env var to your LibTorch install path |
| Float arithmetic warnings in OpenFOAM    | Use `gcc 9+` and double precision build (DPInt32Opt)   |

For other issues, please open a GitHub issue.
