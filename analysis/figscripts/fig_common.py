"""Shared settings for the paper figures: journal style, method palette, data locations.

Data locations are taken from environment variables so the scripts run unchanged on
the authors' machine and on a fresh clone:

``DLAMR_CACHE``   directory with the cached post-processed data (``make download-fields``
                  extracts it to ``reference_data/fields``): ``nc4x_fig_cache/*.npz``
                  (time-averaged fields), ``c7/*.json`` (shedding frequency, convection
                  velocity), ``finest_probe.json``, ``tables/runtime.tex``.
``DLAMR_CASES``   root of the OpenFOAM case results (default ``cases/``); only needed by the
                  scripts that read raw fields or force histories (Figs. 3, 6, 7).
``DL_AMR_OUTDIR`` output directory (default ``analysis/output``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import journal_figure as jf  # noqa: E402

REPO = HERE.parents[1]
ROOT = REPO
CACHE = Path(os.environ.get('DLAMR_CACHE', REPO / 'reference_data' / 'fields'))
WM = CACHE / 'nc4x_fig_cache'                       # time-averaged fields per (geometry, variant)
CR = Path(os.environ.get('DLAMR_CASES', REPO / 'cases'))
OUT = Path(os.environ.get('DL_AMR_OUTDIR', REPO / 'analysis' / 'output'))
OUT.mkdir(parents=True, exist_ok=True)

# analysis grid of the wake evaluation region (same as the metric scripts)
xi = np.arange(-1.5, 20.0001, 0.05)
yi = np.arange(-5.0, 5.0001, 0.05)
Xi, Yi = np.meshgrid(xi, yi)

GEOMS = [
    ('circular', 'circular', 'Circular'),
    ('square', 'square', 'Square'),
    ('diamond', 'diamond', 'Diamond'),
]

# cache key -> case directory (relative to DLAMR_CASES), public layout
CASE = {
 'circular': dict(fine='circular_Re200/fine', coarse='circular_Re200/coarse',
                  sigma='circular_Re200/dl_amr', gradU='circular_Re200/grad_amr',
                  vort='circular_Re200/vort_amr', Q='circular_Re200/q_amr',
                  static='circular_Re200/static'),
 'square':   dict(fine='square_Re150/fine', coarse='square_Re150/coarse',
                  sigma='square_Re150/dl_amr', gradU='square_Re150/grad_amr',
                  vort='square_Re150/vort_amr', Q='square_Re150/q_amr',
                  static='square_Re150/static'),
 'diamond':  dict(fine='diamond_Re150/fine', coarse='diamond_Re150/coarse',
                  sigma='diamond_Re150/dl_amr', gradU='diamond_Re150/grad_amr',
                  vort='diamond_Re150/vort_amr', Q='diamond_Re150/q_amr',
                  static='diamond_Re150/static'),
}
WINDOW = {'circular': (165.0, 245.0), 'square': (237.5, 346.5), 'diamond': (170.0, 256.0)}
KEYS = {'Fine': 'fine', 'Coarse': 'coarse', 'DL-AMR': 'sigma',
        r'$|\nabla\mathbf{U}|$ AMR': 'gradU', r'$|\omega|$ AMR': 'vort',
        r'$Q$ AMR': 'Q', 'static': 'static'}


def cases_for(g: str) -> dict:
    """g in {circular, square, diamond}: label -> dict(key=cache key, case=case directory)."""
    return {m: dict(key=k, case=CASE[g][k]) for m, k in KEYS.items()}


# Okabe-Ito based method styles (line plots)
STYLE = {
    'Fine':      dict(color='#000000', ls='-',  lw=1.4, zorder=5),
    'Coarse':    dict(color='#8a8a8a', ls=(0, (4, 2)), lw=1.0, zorder=2),
    'DL-AMR':    dict(color='#D55E00', ls='-',  lw=1.2, zorder=4),
    r'$|\nabla\mathbf{U}|$ AMR': dict(color='#009E73', ls='-.', lw=1.0, zorder=3),
    r'$|\omega|$ AMR':           dict(color='#CC79A7', ls=':',  lw=1.2, zorder=3),
    r'$Q$ AMR':              dict(color='#0072B2', ls=(0, (6, 2)), lw=1.0, zorder=3),
    'static':                    dict(color='#56B4E9', ls=(0, (1, 1)), lw=1.1, zorder=3),
}

SHORT = {'Fine': 'Fine', 'Coarse': 'Coarse', 'DL-AMR': 'DL-AMR',
         r'$|\nabla\mathbf{U}|$ AMR': r'$|\nabla\mathbf{U}|$ AMR',
         r'$|\omega|$ AMR': r'$|\omega|$ AMR', r'$Q$ AMR': r'$Q$ AMR'}


def body_mask(short: str):
    if short == 'square':
        return (np.abs(Xi) < 0.55) & (np.abs(Yi) < 0.55)
    if short == 'diamond':
        return np.abs(Xi) + np.abs(Yi) < 0.55
    return Xi**2 + Yi**2 < 0.55**2


def body_patch(short: str):
    import matplotlib.patches as mp
    if short == 'square':
        return mp.Rectangle((-.55, -.55), 1.1, 1.1, color='k', zorder=6)
    if short == 'diamond':
        return mp.Polygon([(.55, 0), (0, .55), (-.55, 0), (0, -.55)], color='k', zorder=6)
    return mp.Circle((0, 0), 0.55, color='k', zorder=6)


def mean_ux(g: str, key: str) -> np.ndarray:
    return np.load(WM / f'{g}_{key}.npz')['mean_ux']


def require(path: Path, what: str) -> None:
    """Exit with the conventional [SKIP] code 2 when a data file is missing."""
    if not Path(path).exists():
        print(f'[SKIP] {what} not found: {path}')
        sys.exit(2)
