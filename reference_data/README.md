# Reference data

Each paper figure falls into one of three reproducibility tiers:

| Tier | Figures            | What is needed                                                | How to obtain                                       |
|------|--------------------|---------------------------------------------------------------|-----------------------------------------------------|
| 1    | Fig 6              | Nothing — values are inlined in the script                    | works out of the box                                |
| 2    | Figs 9, 10, 11, 12 | ML test set + per-sample predictions + uncertainty CSVs       | `make download-reference` (this directory)          |
| 3    | Figs 1, 3, 4, 5, 7, 8 | Full OpenFOAM case results (postProcessing/, time dirs)     | run `cases/<geom>/<method>/Allrun`, or pull a case bundle from Zenodo into `cases/` |

`make figs` runs every figure script. Tier-1 always succeeds; Tier-2/3
scripts whose data is not present exit with code **2** ("skipped") and are
counted separately from real failures.

---

## What lives in this directory

```
reference_data/
├── README.md                   (this file)
├── calibration_bins.csv        IN-REPO   uncertainty calibration bin stats   (Fig 9)
├── roc_auc.csv                 IN-REPO   ROC AUC values per channel/τ        (Fig 10)
├── uncertainty_report.json     IN-REPO   summary metrics (ρ, NLL, etc.)      (Fig 11)
├── test.pt                     ZENODO    torch.save dict {X, y, mask}        (Figs 9–12)
└── preds/                      ZENODO    per-sample predictions
    ├── 00000.npz                          arrays: pred, mask, aux
    ├── 00001.npz
    └── ...
```

`test.pt` schema:

| Key   | Shape              | Type                                 |
|-------|--------------------|--------------------------------------|
| `X`   | `(N, 3, 64, 224)`  | normalized inputs $(u^\ast, v^\ast, p^\ast)$ |
| `y`   | `(N, 3, 64, 224)`  | normalized target $\Delta\mathbf{q}_t$       |
| `mask`| `(N, 64, 224)`     | wake-region mask                     |

`preds/<NNNNN>.npz` (one file per test sample, zero-padded index;
emitted by `ml/src/infer.py`):

| Key    | Shape           | Description                                                         |
|--------|-----------------|---------------------------------------------------------------------|
| `pred` | `(3, 64, 224)`  | predicted Δq mean (denormalized to physical units)                  |
| `mask` | scalar (`None`) | per-sample mask, used only for the segmentation task (else `None`)  |
| `aux`  | `(1, 64, 224)`  | auxiliary head output. The heteroscedastic U-Net uses a single shared log-variance channel (`out_ch_logvar=1`); figures read it as `logvar = aux[0]` (shape `(64, 224)`) and derive σ via `np.exp(logvar / 2)`. |

## Download

```bash
make download-reference
# or
bash scripts/download_reference_data.sh
```

The script tries Zenodo first then falls back to GitHub Release.
See [`../docs/data_availability.md`](../docs/data_availability.md) for DOI.

## Environment overrides

Each script honours these environment variables (with the defaults above):

| Variable           | Default                                | Used by                           |
|--------------------|----------------------------------------|-----------------------------------|
| `DL_AMR_DATA`      | `reference_data/`                      | uncertainty CSV/JSON loader       |
| `DL_AMR_TESTPT`    | `reference_data/test.pt`               | Figs 9–12                         |
| `DL_AMR_PREDS`     | `reference_data/preds/`                | Figs 9–12                         |
| `DL_AMR_CASES`     | `cases/`                               | Figs 1, 3, 4, 5, 7, 8             |
| `DL_AMR_OUTDIR`    | `analysis/output/`                     | all figure scripts (output dir)   |

Set these to point at your own copies of the data.

## License

CC-BY 4.0; see [`../LICENSE-DATA`](../LICENSE-DATA).
Citation: see [`../CITATION.cff`](../CITATION.cff).
