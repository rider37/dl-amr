# Data availability

## What is included in this repository

- **OpenFOAM case templates** (`cases/`): initial conditions (`0/`),
  physical properties (`constant/`, excluding generated `polyMesh/`), and
  solver settings (`system/`). Mesh is generated at run time by
  `blockMesh`.
- **Modified OpenFOAM solver** (`solver/amrPimpleFoam/`): C++ source code.
- **Training and analysis code** (`ml/`, `analysis/`): Python sources.
- **Configuration files** (`ml/configs/`): YAMLs used for the paper.
- **Small in-repo reference data** (`reference_data/`):
  uncertainty-calibration summary CSVs/JSON
  (`calibration_bins.csv`, `roc_auc.csv`, `uncertainty_report.json`)
  used by Figs. 9-11. The larger `test.pt` and `preds/*.npz` they need
  are **not** in the repository — see *external artefacts* below.

## What is distributed externally (Zenodo / GitHub Release)

The following large artefacts are distributed via Zenodo (DOI-archived)
and GitHub Release (mirror) rather than this Git repository:

- **Pretrained TorchScript model** (`heteroscedastic_unet.pt`, ~31 MB).
  See [`ml/pretrained/README.md`](../ml/pretrained/README.md) for download
  instructions and the [model card](../ml/pretrained/model_card.md).
- **Minimal reference dataset** for Tier-2 figures (Figs. 9-12), bundled
  as `reference_data_minimal.tar.gz` (~250 MB compressed). It contains
  exactly: `reference_data/test.pt` (the held-out ML test split with keys
  `X`, `y`, `mask`) and `reference_data/preds/<NNNNN>.npz`, one
  per-sample prediction file emitted by `ml/src/infer.py`. It does **not**
  include OpenFOAM raw outputs. See
  [`reference_data/README.md`](../reference_data/README.md) for the schema.

Both can be fetched with:
```bash
make download-models
make download-reference
# or, in one shot:
make download-artifacts
```

## Training dataset

The full preprocessed training dataset (≈40 GB; circular cylinder at
$Re = 100, 150$, snapshots resampled to $64 \times 224$ Cartesian grids
with input channels $(u^\ast, v^\ast, p^\ast)$ and target $\Delta\mathbf{q}_t$)
is **not included** in this repository because of its size.

The dataset can be regenerated from the OpenFOAM reference simulations
using the scripts under `ml/src/dataloaders/`. The required reference
simulations are defined in the `cases/` templates; see the
[reproduction guide](reproduction_guide.md#4-training-the-model-from-scratch-optional)
for the end-to-end training pipeline.

A copy of the preprocessed dataset is available from the corresponding
authors upon reasonable request.

## What is **not** distributed

- **Full OpenFOAM raw outputs** (time directories, processor
  decompositions, VTK files) for Tier-3 figures (Figs. 1, 3, 4, 5, 7, 8).
  These are not bundled in any release artefact. They are regenerated
  deterministically by running the provided case templates
  (`make run-{circular,square,diamond}-{fine,coarse,dl-amr,grad-amr}`,
  or directly `cd cases/<geom>/<method> && ./Allrun`).
- Intermediate analysis artefacts that the published figure scripts
  produce on the fly.

## Licensing of distributed artefacts

| Artefact                              | License        |
|---------------------------------------|----------------|
| Source code (Python)                  | MIT            |
| OpenFOAM solver modifications         | GNU GPL v3.0   |
| Pretrained model weights              | CC-BY 4.0      |
| Reference data, configuration, docs   | CC-BY 4.0      |

See [`../LICENSE`](../LICENSE) for details.

## Suggested citation

```
Kang, J., Oh, M., Son, J., Jeon, J., Lee, S. (2026).
Heteroscedastic Temporal-Residual Learning for Dynamic Adaptive Mesh
Refinement in Bluff-Body Wake Flows. Journal of Computational Physics.
DOI: 10.xxxx/yyyy   # TODO: replace after JCP publication

Software DOI (Zenodo, concept — always resolves to latest): 10.5281/zenodo.19870610
Current version (v1.0.3):                                    10.5281/zenodo.19874536
```
