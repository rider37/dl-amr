# Analysis scripts: which figure and table comes from where

Every figure and table of the paper is produced by a script in this directory. The table below
lists the script, the data it needs and the download target that provides that data.

| Paper item | Script | Data needed | Provided by |
|---|---|---|---|
| Fig. 9 (calibration, ROC) | `uncertainty_figures.py` | `reference_data/test.pt`, `reference_data/preds/` | `make download-reference` |
| Fig. 10 (error capture, AUC vs quantile) | `uncertainty_figures.py` | same | `make download-reference` |
| Fig. E.1 (spatial maps) | `uncertainty_figures.py` | same | `make download-reference` |
| Fig. 4 (error maps), Fig. D.1 (profiles) | `figscripts/make_fig4_D1.py` | `reference_data/fields/nc4x_fig_cache/` | `make download-fields` |
| Fig. 5 (error vs cells, vs cost) | `figscripts/make_fig5.py` | `nc4x_fig_cache/`, `fields/tables/runtime.tex` | `make download-fields` |
| Fig. 8 (convection velocity), Fig. F.1 (St(t)) | `figscripts/make_fig8_F1.py` | `fields/c7/*.json` | `make download-fields` |
| Fig. 3 (phase-averaged fields), Fig. 6 (vorticity + mesh) | `figscripts/make_fig3_6.py` | raw case results (`cases/<geom>/<variant>/`), `fields/finest_probe.json` | run the cases, or request the results |
| Fig. 7 (force spectra) | `figscripts/make_fig7.py` | raw `postProcessing/forceCoeffs` | run the cases |
| Table 1 (wake accuracy) | `postprocessing/nc2_metrics*.py` → `build_ssot.py` → `make_tables.py` | raw case results | run the cases |
| Table 2 (wake dynamics) | `postprocessing/wake_dynamics_table2.py` → `make_tables.py` | raw results, probes, force histories | run the cases |
| Table 3 (cross-geometry transfer) | `postprocessing/cross_geometry_table3.py` (square/diamond) with `g6_uncertainty.py` (circular test block) | test split + square/diamond fine and coarse fields | run the cases |
| Table 4 (grid convergence) | `postprocessing/grid_convergence_table4.py` | three circular meshes | run the cases |
| Tables 5, 6 (cost, overhead) | `postprocessing/bench_runtime_table5.py`, `bench_parse.py` → `build_ssot.py` | dedicated single-core benchmark runs | run the benchmark |
| Table 7, Table G.1 (score ablation, wrapper, Kelly) | `postprocessing/score_ablation_eval.sh`, `score_ablation_table7.py` | ablation case results | run the cases |
| Table 8 (error coverage) | `postprocessing/error_coverage_table8.py` | raw results | run the cases |
| Table A.1 (single-application test) | `postprocessing/single_application_*.{py,sh}` | circular coarse checkpoint + solver | run the test |
| Table F.1 (raw vs coherent statistics) | `postprocessing/raw_vs_coherent_moments.py` → `raw_vs_coherent_tableF1.py` | raw results | run the cases |
| Budgets of Appendix B | `postprocessing/budget_calibrate_and_run.py` | solver | run the cases |

`make figs` runs the figure scripts in the order above; scripts whose data are missing print
`[SKIP]` and exit with code 2.

## Data locations

Set through environment variables (defaults in parentheses):

| Variable | Meaning |
|---|---|
| `DLAMR_REFDATA` | `reference_data/` with `test.pt` and `preds/` |
| `DLAMR_CACHE` | `reference_data/fields/` (cached wake-field data) |
| `DLAMR_CASES` | `cases/` (raw OpenFOAM results) |
| `DLAMR_ROOT`, `DLAMR_EVAL` | repository root and scratch directory for the post-processing scripts |
| `DL_AMR_OUTDIR` | output directory (`analysis/output/`) |

## Post-processing scripts

`postprocessing/` contains the authors' pipeline as run for the paper, with only the data roots
replaced by the variables above (`_roots.py`). These scripts refer to the runs by the authors'
case names. The correspondence to the public case directories is:

| Authors' name | Public directory | Authors' name | Public directory |
|---|---|---|---|
| `grid_convergence/circular_Re200_finer` | `circular_Re200/fine` | `grid_convergence/square_Re150_finer` | `square_Re150/fine` |
| `NC2_coarse` | `circular_Re200/coarse` | `NCsq_coarse_full` | `square_Re150/coarse` |
| `NC49k_sigma` | `circular_Re200/dl_amr` | `SQ49k_sigma` | `square_Re150/dl_amr` |
| `NC49k_gradU`, `NC49k_vort`, `NC49k_Q` | `circular_Re200/{grad_amr,vort_amr,q_amr}` | `SQ49k_gradU`, `SQ49k_vort`, `SQ49k_Q` | `square_Re150/{grad_amr,vort_amr,q_amr}` |
| `ST_static1_circ` | `circular_Re200/static` | `ST_static1_sq` | `square_Re150/static` |
| `NC49k_mu`, `SQ49k_mu`, `DIA49k_mu` | `<geometry>/dl_amr_meanhead` | `grid_convergence/diamond_Re150_finer` | `diamond_Re150/fine` |
| `NC49k_vortgrid` | `circular_Re200/vort_wrapper` | `NCdia_coarse_full` | `diamond_Re150/coarse` |
| `KLW1_circular_kelly` | `circular_Re200/kelly_wake` | `DIA49k_*`, `ST_static1_dia` | `diamond_Re150/*` |

`figscripts/fig_common.py` already maps the cache keys (`fine`, `coarse`, `sigma`, `gradU`,
`vort`, `Q`, `static`) to the public directories, so the figure scripts need no renaming.
