# OpenFOAM case templates

Each subdirectory is a self-contained OpenFOAM case template with the
**initial conditions and configuration only** — no time directories,
processor decompositions, or post-processing artefacts. The mesh itself
is generated at run time by `blockMesh` from `system/blockMeshDict`.

## Layout

```
cases/
├── circular_Re200/
│   ├── fine/         # high-resolution reference
│   ├── coarse/       # baseline coarse mesh (AMR starting point)
│   ├── dl_amr/       # DL-AMR (heteroscedastic uncertainty indicator)
│   └── grad_amr/     # gradient-based AMR (|∇U| indicator) — best L2 config
├── square_Re150/     # same four methods
└── diamond_Re150/    # same four methods
```

## Method-specific notes

### `fine/` and `coarse/`
Standard PIMPLE simulations with `pimpleFoam`. Differ only in the
`blockMeshDict` resolution.

### `dl_amr/`
Uses the modified `amrPimpleFoam` solver from `../../solver/`. Requires
the pretrained TorchScript model linked at `constant/model.ts`:

```bash
ln -sf $(pwd)/../../../ml/pretrained/heteroscedastic_unet.pt \
       constant/model.ts
```

Refinement is triggered every `refineInterval` time steps (see
`constant/dynamicMeshDict`). The threshold $\sigma_{\mathrm{thr}}$ is
configured in `constant/mlInferDict`.

### `grad_amr/`
Uses `amrPimpleFoam` with the gradient-norm indicator instead of the ML
model (configured in `constant/dynamicMeshDict`). Threshold lower/upper
bounds are tuned per geometry; see paper for selected values.

## Running a case

```bash
cd cases/<geometry>/<method>
./Allrun
```

The provided `Allrun` invokes `blockMesh` and the appropriate solver,
and writes timing information to `log.*` files. Runs are
single-process by default; for parallel runs, edit
`system/decomposeParDict` and use `decomposePar`/`reconstructPar`.

## Cleaning up

```bash
./Allclean   # remove time directories, processor*, postProcessing, logs
```

Or, repository-wide:
```bash
make clean-cases
```

## Reproducibility notes

- Time stepping is adaptive (`adjustTimeStep on`, max `Co = 0.5`).
  Wall-clock numbers in the paper are reported on a single CPU core.
- All cases use the same near-body O-grid topology; only the wake
  resolution differs across `fine`/`coarse`/`*_amr`.
- Reference fine-mesh time-averaged fields are available via
  `make download-reference` for figure verification without re-running
  the simulations.
