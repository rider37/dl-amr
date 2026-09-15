# Reference data

The figure scripts read their data from this directory. Three archives are distributed with the
Zenodo deposit and the GitHub Release (see [`../docs/data_availability.md`](../docs/data_availability.md)):

| Archive | Size | Extracted to | Reproduces |
|---|---:|---|---|
| `reference_data_minimal.tar.gz` | ~410 MB | `reference_data/test.pt`, `reference_data/preds/` | Figs. 8, 9, E.1 (`analysis/uncertainty_figures.py`) |
| `reference_data_fields.tar.gz` | ~65 MB | `reference_data/fields/` | Figs. 4, 5, 7, D.1, F.1 (`analysis/figscripts/`) |
| `pretrained_models.tar.gz` | ~55 MB | `ml/pretrained/` | closed-loop cases (`constant/model.ts`) |

Figures 3 and 6 and the tables additionally need the raw OpenFOAM case results, obtained by
running `cases/<geometry>/<variant>/Allrun` or on request from the corresponding author.
Scripts whose data is not present exit with code **2** ("skipped") and are counted separately
from real failures by `make figs`.

```bash
make download-reference     # test.pt + preds/
make download-fields        # fields/
make download-models        # ml/pretrained/
make download-artifacts     # all three
```

## Layout after download

```
reference_data/
├── README.md                   (this file)
├── calibration_bins.csv        IN-REPO   calibration bin statistics (Fig. 8a), for reference
├── roc_auc.csv                 IN-REPO   AUC against the positive-class quantile (Fig. 9b)
├── uncertainty_report.json     IN-REPO   summary metrics (rho, AUC)
├── test.pt                     ARCHIVE   torch.save dict {X, y, mask, meta}
├── preds/NNNNN.npz             ARCHIVE   per-sample predictions (pred, mask, aux)
└── fields/                     ARCHIVE   cached wake-field data, see fields/README.md
    ├── nc4x_fig_cache/<geometry>_<variant>.npz   time-averaged fields on the analysis grid
    ├── c7/*.json                                  St(t), U_c(x), offline error-capture curves
    ├── finest_probe.json, probe6_all.json         wake-probe time series
    ├── ssot.json                                  values behind Tables 1, 5, 6, 8
    └── tables/*.tex                               the LaTeX tables of the paper as generated
```

`test.pt` (held-out blocked test split of the circular-cylinder training data, Re = 100 and 150):

| Key    | Shape                 | Content                                                     |
|--------|-----------------------|-------------------------------------------------------------|
| `X`    | `(252, 2, 160, 432)`  | inputs $(u^\ast, v^\ast)$ in physical units on the $432 \times 160$ grid covering $x/D \in [-2, 25]$, $y/D \in [-5, 5]$ |
| `y`    | `(252, 2, 160, 432)`  | two-grid discrepancy $\mathbf{q}^{\mathrm{fine}} - \mathbf{q}^{\mathrm{coarse}}$, physical units |
| `mask` | `(252, 1, 160, 432)`  | wake cells (body excluded)                                   |

`preds/<NNNNN>.npz` (one file per test sample, produced with `heteroscedastic_unet.pt`):

| Key    | Shape           | Content                                                                    |
|--------|-----------------|----------------------------------------------------------------------------|
| `pred` | `(2, 160, 432)` | mean-head prediction $\widehat{\Delta\mathbf{q}}$, physical units          |
| `aux`  | `(1, 160, 432)` | log-variance in physical units; $\hat{\sigma} = \exp(\mathrm{aux}[0]/2)$ with the single-scalar scale convention of Appendix B |
| `mask` | scalar (`None`) | unused                                                                     |

Running `python analysis/uncertainty_figures.py` on these files reproduces the headline values of
the paper: Spearman $\rho = 0.809$ against $|\Delta\mathbf{q}|$ and $0.845$ against the mean-head
residual, calibration slope $1.10$ with a median RMSE/$\hat{\sigma}$ ratio of $3.3$, AUC $= 0.848$ at
$q_{95}$ ($0.845$–$0.876$ from $q_{80}$ to $q_{99}$), and a $20\%$-area error capture of $45\%$
($\hat{\sigma}$), $65\%$ (oracle), $63\%$ (mean head), $33\%$, $47\%$ and $48\%$ ($|\nabla\mathbf{U}|$,
$|\omega|$, $Q^{+}$).

## Environment overrides

| Variable         | Default                    | Used by                              |
|------------------|----------------------------|--------------------------------------|
| `DLAMR_REFDATA`  | `reference_data/`          | `analysis/uncertainty_figures.py`    |
| `DLAMR_CACHE`    | `reference_data/fields/`   | `analysis/figscripts/*`              |
| `DLAMR_CASES`    | `cases/`                   | Figs. 3, 6 and the post-processing pipeline |
| `DL_AMR_OUTDIR`  | `analysis/output/`         | all figure scripts                   |

## License

CC-BY 4.0; see [`../LICENSE-DATA`](../LICENSE-DATA). Citation: [`../CITATION.cff`](../CITATION.cff).
