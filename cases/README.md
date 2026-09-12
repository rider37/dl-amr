# Simulation cases

Case set-ups for every variant reported in the paper. Each directory contains the `system/`
and `constant/` dictionaries exactly as run; mesh and field data are not included (see
*Restart protocol* below).

## Variants

| Directory | Refinement indicator | Solver |
|---|---|---|
| `coarse` | none (base mesh) | `pimpleFoam` |
| `fine` | none (uniformly refined reference) | `pimpleFoam` |
| `grad_amr` | velocity-gradient magnitude | `pimpleFoam` + coded function object |
| `vort_amr` | vorticity magnitude | `pimpleFoam` + coded function object |
| `q_amr` | Q-criterion | `pimpleFoam` + coded function object |
| `dl_amr` | learned uncertainty (this work) | `amrPimpleFoam` |
| `static` | wake window refined once, then frozen | `pimpleFoam` |
| `dl_amr_meanhead` | mean head of the same network (score ablation, Table 7) | `amrPimpleFoam`, `refineMode delta_norm` |
| `vort_wrapper` (circular only) | vorticity magnitude passed through the learned policy's wrapper (Table G.1) | `amrPimpleFoam` with `constant/vort_grid.ts` |
| `kelly_wake` (circular only) | Kelly gradient-jump estimator, wake candidate set (Table G.1) | `pimpleFoam` + coded function object |

All adaptive variants use the same marking rule: cells are sorted by decreasing indicator
value and marked until a cell-count budget is reached (Eq. (11) of the paper). For the classical
indicators the budget is the `Nbudget` constant inside the coded function object in
`system/controlDict`; for `dl_amr` it is `attnThr` in `constant/mlInferDict`. The budgets are
calibrated per indicator and geometry so that the realised mean cell count stays at or below the
49,884 cells of the `static` mesh.

Shared refinement settings (Section 3 of the paper): refinement interval 40 time steps, maximum
refinement level 2, unrefinement buffer 2 layers, in-plane 1→4 subdivision (`hexRef4`).

## Restart protocol

Every variant is restarted from a common state so that the comparison is not contaminated by
different transients. The start and end times are:

| Geometry | Restart time | End time | Evaluation window |
|---|---|---|---|
| `circular_Re200` | 95 | 245 | [165, 245] |
| `square_Re150` | 142.5 | 347 | [237.5, 346.5] |
| `diamond_Re150` | 95 | 256 | [170, 256] |

To reproduce a case:

1. Build the solver and the `hexRef4` library (`solver/`), and set `FOAM_USER_SRC` to the
   directory containing `hexRef4` so that the coded function objects can find its headers.
2. Run `coarse` from `t = 0` with `blockMesh` until the restart time above.
3. Copy the resulting time directory into the target case directory.
4. For `dl_amr` and `dl_amr_meanhead`, place the TorchScript model at `constant/model.ts`
   (`make download-models` from the repo root). `vort_wrapper` ships its own
   `constant/vort_grid.ts` (no trained weights; built by `ml/src/tools/make_vort_wrapper.py`).
5. Run `./Allrun`.

`fine` and `coarse` are run from `t = 0` and need no restart state.

## Reynolds numbers

Circular cylinder at Re = 200; square and diamond cylinders at Re = 150. The training reference
cases (`circular_Re100`, `circular_Re150`) contain only the `fine` set-up used to generate the
learning target.
