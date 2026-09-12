# DL-AMR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-CODE)
[![License: GPL v3](https://img.shields.io/badge/Solver-GPL_v3-blue.svg)](solver/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19870610.svg)](https://doi.org/10.5281/zenodo.19870610)

Companion code for:

> **Heteroscedastic Residual Learning for Dynamic Adaptive Mesh
> Refinement in Bluff-Body Wake Flows**
> Kang, Oh, Son, Jeon, Lee — *Journal of Computational Physics* (2026)
> DOI: `10.xxxx/yyyy`

A deep-learning adaptive mesh refinement (DL-AMR) framework for two-dimensional
laminar bluff-body wake flows. A heteroscedastic U-Net is trained to predict the
two-grid discrepancy, the difference between the solution on a refined reference
mesh and the solution on the mesh that will be refined, together with an
input-dependent scale of that discrepancy. The variance-derived score drives
runtime mesh refinement in OpenFOAM through a modified solver `amrPimpleFoam`,
with cells marked by rank against a fixed cell budget.

## Quick reproduction

```bash
make install              # Python environment
make download-models      # Pretrained TorchScript model only
make download-reference   # Minimal reference dataset (test.pt + preds/)
make download-fields      # Cached wake-field data (time-averaged fields, probes, tables)
make smoke-test           # Verify environment + case templates
make figs                 # Regenerate cached-data paper figures
```

`make download-artifacts` is a one-shot shortcut for the three
`download-*` targets above.

**What `make figs` produces** depends on which data you have. Each figure
script falls into one of three reproducibility tiers:

| Tier | Figures                 | Data needed                                                    |
|------|-------------------------|-----------------------------------------------------------------|
| 2    | Figs 9, 10, E.1         | `make download-reference` (test split + per-sample predictions) |
| 2    | Figs 4, 5, 8, D.1, F.1  | `make download-fields` (cached time-averaged fields and probes)  |
| 3    | Figs 3, 6, 7 and all tables | full OpenFOAM case results under `cases/<geom>/<variant>/` |

Scripts whose data is absent print a clear `[SKIP]` line and exit with
code 2 (counted as **skipped**, not failed). See
[`analysis/README.md`](analysis/README.md) for the figure-and-table
manifest, [`reference_data/README.md`](reference_data/README.md) for the
data layout and [`docs/reproduction_guide.md`](docs/reproduction_guide.md)
for the full reproduction recipe.

## Repository layout

```
.
├── ml/                # Heteroscedastic U-Net training & inference (Python)
│   ├── src/           # Models, losses, training loop
│   ├── configs/       # Training/eval YAMLs used in the paper
│   └── pretrained/    # Pretrained model download instructions (Zenodo/Release)
│
├── solver/            # OpenFOAM solver modifications (GPL v3)
│   ├── amrPimpleFoam/ # PIMPLE solver with runtime ML inference + rank-budget AMR
│   └── hexRef4/       # in-plane 1->4 refinement library (2-D port of hexRef8)
│
├── cases/             # OpenFOAM case templates (system/ and constant/ as run)
│   ├── circular_Re200/{fine,coarse,grad_amr,vort_amr,q_amr,dl_amr,static,
│   │                   dl_amr_meanhead,vort_wrapper,kelly_wake}/
│   ├── square_Re150/{fine,coarse,grad_amr,vort_amr,q_amr,dl_amr,static,dl_amr_meanhead}/
│   └── diamond_Re150/{... same as square ...}/
│
├── analysis/          # Figure scripts, post-processing pipeline, figure/table manifest
├── reference_data/    # Downloaded reference data (test split, predictions, cached fields)
├── scripts/           # Helper scripts (downloads, smoke test, figure pipeline)
└── docs/              # Reproduction guide, data availability statement
```

## Requirements

- Python 3.10+ (PyTorch, NumPy, SciPy, matplotlib, PyVista)
- OpenFOAM v2312 (other recent versions may work)
- LibTorch C++ runtime (matched to the PyTorch version used for export);
  set `LIBTORCH_DIR` before `make solver`:
  ```bash
  export LIBTORCH_DIR=/path/to/libtorch
  ```
- (Recommended) NVIDIA GPU with CUDA for ML inference

## Full reproduction

For complete reproduction of the simulations and figures, see
[`docs/reproduction_guide.md`](docs/reproduction_guide.md).

## Citation

If you use this code, model, or data, please cite the paper above and the
software DOI from Zenodo. See [`CITATION.cff`](CITATION.cff).

## License

This repository uses multiple licenses depending on the component
(see [`LICENSE`](LICENSE)):

| Component                               | License        |
|-----------------------------------------|----------------|
| Python training and analysis code       | MIT            |
| OpenFOAM solver modifications           | GNU GPL v3.0   |
| Pretrained models, reference data, docs | CC-BY 4.0      |

## Data availability

The full preprocessed training dataset is not included in this repository
because of its size. The dataset can be regenerated using the provided case
templates and preprocessing scripts; see
[`docs/data_availability.md`](docs/data_availability.md).

## Contact

For questions about the code or paper, open a GitHub issue or contact the
corresponding authors:

- Joongoo Jeon — <jgjeon41@postech.ac.kr>
- Sangseung Lee — <sangseunglee@inha.ac.kr>
