# DL-AMR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-CODE)
[![License: GPL v3](https://img.shields.io/badge/Solver-GPL_v3-blue.svg)](solver/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Companion code for:

> **Heteroscedastic Temporal-Residual Learning for Dynamic Adaptive Mesh
> Refinement in Bluff-Body Wake Flows**
> Kang, Oh, Son, Jeon, Lee — *Journal of Computational Physics* (2026)
> DOI: `10.xxxx/yyyy`

A deep-learning adaptive mesh refinement (DL-AMR) framework for two-dimensional
bluff-body wake flows. A heteroscedastic U-Net learns per-grid-point aleatoric
uncertainty from the temporal residual of velocity and pressure fields; this
uncertainty drives runtime mesh refinement in OpenFOAM through a modified
solver `amrPimpleFoam`.

## Quick reproduction (3 commands)

```bash
make install              # Python environment
make download-models      # Pretrained TorchScript model + reference data
make figs                 # Regenerate paper figures from cached data
```

## Repository layout

```
.
├── ml/                # Heteroscedastic U-Net training & inference (Python)
│   ├── src/           # Models, losses, training loop
│   ├── configs/       # Training/eval YAMLs used in the paper
│   └── pretrained/    # Pretrained model download instructions (Zenodo/Release)
│
├── solver/            # OpenFOAM solver modifications (GPL v3)
│   └── amrPimpleFoam/ # PIMPLE solver with runtime ML inference + AMR
│
├── cases/             # OpenFOAM case templates (initial conditions only)
│   ├── circular_Re200/{fine,coarse,dl_amr,grad_amr}/
│   ├── square_Re150/{fine,coarse,dl_amr,grad_amr}/
│   └── diamond_Re150/{fine,coarse,dl_amr,grad_amr}/
│
├── analysis/          # Figure generation scripts
├── reference_data/    # Minimal processed data (CSVs, time-averaged fields)
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
