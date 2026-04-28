# Reference data

Minimal processed data sufficient to reproduce the paper figures and
tables **without** running the full OpenFOAM simulations.

This directory holds **download instructions only**. Actual data is
distributed via Zenodo (DOI archive) and GitHub Release (mirror).

## Contents (after `make download-reference`)

```
reference_data/
├── circular_Re200/
│   ├── forceCoeffs_fine.csv
│   ├── forceCoeffs_coarse.csv
│   ├── forceCoeffs_dl_amr.csv
│   ├── forceCoeffs_grad_amr.csv
│   ├── UMean_fine.npz             # time-averaged U field on uniform grid
│   ├── UMean_coarse.npz
│   ├── UMean_dl_amr.npz
│   └── UMean_grad_amr.npz
├── square_Re150/    (same files)
├── diamond_Re150/   (same files)
├── grad_sweep_summary.csv         # cell count, L2, runtime per (l, u) combo
└── metrics_table.csv               # paper Table values
```

## Download

```bash
make download-reference
# or directly:
bash scripts/download_reference_data.sh
```

The script tries Zenodo first, then falls back to GitHub Release. See
[`../docs/data_availability.md`](../docs/data_availability.md) for DOI
and licensing.

## File formats

- **CSVs**: comma-separated, ASCII, with header row.
- **NPZs**: NumPy archive with three arrays per file:
  - `Xi` (ny, nx) — x-coordinate grid in $D$ units
  - `Yi` (ny, nx) — y-coordinate grid in $D$ units
  - `Ux` (ny, nx) — time-averaged streamwise velocity normalised by $U_\infty$

Loading example:
```python
import numpy as np
data = np.load('reference_data/circular_Re200/UMean_fine.npz')
Xi, Yi, Ux = data['Xi'], data['Yi'], data['Ux']
```

## License

Reference data is distributed under CC-BY 4.0; see [`../LICENSE-DATA`](../LICENSE-DATA).
Citation: see [`../CITATION.cff`](../CITATION.cff).
