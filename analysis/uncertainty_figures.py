#!/usr/bin/env python3
"""Uncertainty-validation figures from the minimal reference dataset (Tier 2).

Reproduces, from ``reference_data/test.pt`` and ``reference_data/preds/*.npz`` alone:

* Fig. 8  -- calibration curve (a) and ROC curve for the top 5 % of |dq| (b)
* Fig. 9  -- offline error capture against selected area (a) and AUC against the
             positive-class quantile (b)
* Fig. E.1 -- spatial maps of |dq| and sigma-hat for three held-out snapshots

and prints the headline numbers quoted in the paper (Spearman rho, AUC, capture
fractions, calibration slope and offset).  No trained model is needed: the
per-sample predictions shipped with the dataset are used directly.

Usage (from the repository root, after ``make download-reference``)::

    python analysis/uncertainty_figures.py            # writes analysis/output/fig{8,9,E1}.pdf

Environment variables: ``DLAMR_REFDATA`` (default ``reference_data``),
``DL_AMR_OUTDIR`` (default ``analysis/output``).
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'figscripts'))
import journal_figure as jf  # noqa: E402

REPO = HERE.parent
REF = Path(os.environ.get('DLAMR_REFDATA', REPO / 'reference_data'))
OUT = Path(os.environ.get('DL_AMR_OUTDIR', HERE / 'output'))
OUT.mkdir(parents=True, exist_ok=True)

# Sampling grid of the wake window W1 (432 x 160 points, spacing 0.063 D)
NX, NY = 432, 160
XG = np.linspace(-2.0, 25.0, NX)
YG = np.linspace(-5.0, 5.0, NY)
DX = float(XG[1] - XG[0])

C_DATA, C_REF = '#D55E00', '#000000'


def skip(msg: str) -> None:
    print(f'[SKIP] {msg}')
    sys.exit(2)


# ------------------------------------------------------------------ load data
test_pt = REF / 'test.pt'
pred_files = sorted(glob.glob(str(REF / 'preds' / '*.npz')))
if not test_pt.exists() or not pred_files:
    skip(f'reference data not found under {REF} (run "make download-reference")')

import torch  # noqa: E402  (only used to read test.pt)

d = torch.load(test_pt, map_location='cpu', weights_only=False)
X = d['X'].numpy().astype(np.float64)          # (N, 2, NY, NX) inputs (u*, v*), physical units
Y = d['y'].numpy().astype(np.float64)          # (N, 2, NY, NX) two-grid discrepancy, physical units
MK = d['mask'][:, 0].numpy() > 0               # (N, NY, NX) wake cells (body excluded)
N = X.shape[0]
if len(pred_files) != N:
    skip(f'{len(pred_files)} prediction files for {N} test samples')

MU = np.empty_like(Y)
SIG = np.empty((N, NY, NX))
for i, f in enumerate(pred_files):
    z = np.load(f, allow_pickle=True)
    MU[i] = z['pred']
    SIG[i] = np.exp(0.5 * z['aux'][0])         # sigma-hat in physical units (Appendix B convention)

absd = np.linalg.norm(Y, axis=1)               # |dq|
rms = np.sqrt(np.mean((MU - Y) ** 2, axis=1))  # RMS of the mean-head residual per point
mu_n = np.linalg.norm(MU, axis=1)              # mean-head score |dq-hat|

# classical cell-local scores on the same grid (central differences; axis 1 = y, axis 2 = x)
u, v = X[:, 0], X[:, 1]
du_dy, du_dx = np.gradient(u, DX, axis=(1, 2))
dv_dy, dv_dx = np.gradient(v, DX, axis=(1, 2))
gradU = np.sqrt(du_dx ** 2 + du_dy ** 2 + dv_dx ** 2 + dv_dy ** 2)
vort = np.abs(dv_dx - du_dy)
S2 = du_dx ** 2 + dv_dy ** 2 + 0.5 * (du_dy + dv_dx) ** 2
O2 = 0.5 * (dv_dx - du_dy) ** 2
Qp = np.maximum(0.5 * (O2 - S2), 0.0)
del u, v, du_dy, du_dx, dv_dy, dv_dx, S2, O2

s, a, r, m = SIG[MK], absd[MK], rms[MK], mu_n[MK]
g, w, qp = gradU[MK], vort[MK], Qp[MK]
print(f'{N} test pairs, {s.size:,} wake grid points')

# ------------------------------------------------------------------ statistics
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

rho_absd = float(spearmanr(s, a).statistic)
rho_rms = float(spearmanr(s, r).statistic)

# 20 equal-count bins in predicted sigma
qs = np.quantile(s, np.linspace(0, 1, 21)); qs[-1] += 1e-12
bi = np.clip(np.searchsorted(qs, s, side='right') - 1, 0, 19)
bin_sig = np.array([s[bi == k].mean() for k in range(20)])
bin_rms = np.array([np.sqrt((r[bi == k] ** 2).mean()) for k in range(20)])
slope = float(np.polyfit(np.log(bin_sig), np.log(bin_rms), 1)[0])
ratio = bin_rms / bin_sig
print(f'Spearman rho: {rho_absd:.3f} (vs |dq|), {rho_rms:.3f} (vs mean-head residual)')
print(f'calibration: log-log slope {slope:.2f}, RMSE/sigma median {np.median(ratio):.1f} '
      f'(range {ratio.min():.1f}-{ratio.max():.1f})')

pos95 = (a >= np.quantile(a, 0.95)).astype(int)
fpr, tpr, _ = roc_curve(pos95, s)
auc95 = float(roc_auc_score(pos95, s))
QS = np.array([80, 85, 90, 95, 98, 99])
auc_q = {int(q): float(roc_auc_score((a >= np.quantile(a, q / 100)).astype(int), s)) for q in QS}
print('AUC(q95) sigma-hat:', f'{auc95:.3f}', '| by quantile:', {k: round(v, 3) for k, v in auc_q.items()})

qgrid = np.linspace(0.005, 1.0, 200)


def captured(score: np.ndarray) -> np.ndarray:
    e = a[np.argsort(-score)]
    c = np.cumsum(e); c /= c[-1]
    idx = np.clip((qgrid * len(e)).astype(int) - 1, 0, len(e) - 1)
    return c[idx]


i20 = int(np.argmin(np.abs(qgrid - 0.2)))
SCORES = {'sigma': s, 'ideal': a, 'mean head': m, '|gradU|': g, '|omega|': w, 'Q+': qp}
CAP = {k: captured(v) for k, v in SCORES.items()}
AUC95 = {k: float(roc_auc_score(pos95, v)) for k, v in SCORES.items() if k != 'ideal'}
print('top-20% error capture:', {k: f'{100 * c[i20]:.1f}%' for k, c in CAP.items()})
print('AUC(q95) by score:', {k: round(v, 3) for k, v in AUC95.items()})

summary = dict(n_pairs=N, n_points=int(s.size), spearman_absd=rho_absd, spearman_rms=rho_rms,
               calib_slope=slope, calib_ratio_median=float(np.median(ratio)),
               calib_ratio_range=[float(ratio.min()), float(ratio.max())],
               auc_q95=auc95, auc_by_quantile=auc_q,
               capture_20pct={k: float(c[i20]) for k, c in CAP.items()}, auc95_by_score=AUC95)
(OUT / 'uncertainty_summary.json').write_text(json.dumps(summary, indent=1))

# ------------------------------------------------------------------ Fig. 8
jf.apply_style('JCP', font_pt=12)
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = jf.new_figure_grid('JCP', 1, 2, width='onehalf', aspect=0.52)
ax = axes[0]
lo, hi = min(bin_sig.min(), bin_rms.min()), max(bin_sig.max(), bin_rms.max())
ax.plot([lo, hi], [lo, hi], '--', color=C_REF, lw=1.0, label='ideal')
ax.plot(bin_sig, bin_rms, 'o-', color=C_DATA, ms=4.5, lw=1.4, mec='k', mew=0.4, label='observed')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'mean predicted $\hat{\sigma}$', fontsize=11.2)
ax.set_ylabel(r'observed RMSE / $U_\infty$', fontsize=11.2)
ax.set_title('(a) calibration', fontsize=11.2)
ax.tick_params(which='both', labelsize=10.2)
ax.grid(alpha=0.25, which='both', lw=0.4)
ax.legend(frameon=False, loc='lower right', fontsize=10.2, handlelength=1.5, borderaxespad=0.6)
ax = axes[1]
ax.plot([0, 1], [0, 1], '--', color=C_REF, lw=1.0, label='random')
ax.plot(fpr, tpr, '-', color=C_DATA, lw=1.6, label=f'AUC $=$ {auc95:.3f}')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect('equal')
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticklabels(['', '0.2', '0.4', '0.6', '0.8', '1.0'])
ax.set_xlabel('false positive rate', fontsize=11.2); ax.set_ylabel('true positive rate', fontsize=11.2)
ax.set_title(r'(b) ROC, top $5\%$ of $|\Delta\mathbf{q}|$', fontsize=11.2)
ax.tick_params(which='both', labelsize=10.2)
ax.grid(alpha=0.25, lw=0.4)
ax.legend(frameon=False, loc='lower right', fontsize=10.2, handlelength=1.5, borderaxespad=0.6)
fig.savefig(OUT / 'fig8.pdf', dpi=500); plt.close(fig)
print('fig8.pdf saved')

# ------------------------------------------------------------------ Fig. 9
jf.apply_style('JCP', font_pt=10.5)
fig, axes = jf.new_figure_grid('JCP', 1, 2, width='onehalf', aspect=0.48, font_pt=10.5)
ax = axes[0]
ax.plot(100 * qgrid, 100 * CAP['ideal'], color='k', ls=':', lw=1.0, label='ideal ranking')
ax.plot(100 * qgrid, 100 * CAP['sigma'], color=C_DATA, lw=1.4, label=r'$\hat{\sigma}$')
ax.plot(100 * qgrid, 100 * CAP['mean head'], color='0.35', ls=(0, (3, 1, 1, 1)), lw=1.0,
        label=r'$|\widehat{\Delta\mathbf{q}}|$ (mean head)')
ax.plot(100 * qgrid, 100 * qgrid, color='0.6', ls='--', lw=0.9, label='random')
ax.axvline(20, color='k', lw=0.6, alpha=0.5)
ax.set_xlabel('selected area (%)', fontsize=10); ax.set_ylabel('captured error mass (%)', fontsize=10)
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.legend(fontsize=8.5, frameon=False, loc='lower right', handlelength=2.2)
ax = axes[1]
ax.plot(QS, [auc_q[int(q)] for q in QS], 'o-', color='#0072B2', ms=4, lw=1.2)
ax.set_xlabel('high-error quantile (%)', fontsize=10); ax.set_xticks([80, 85, 90, 95, 99])
ax.set_ylabel('AUC', fontsize=10); ax.set_ylim(0.5, 1.0); ax.axhline(0.5, color='0.6', ls='--', lw=0.8)
for a_, t in zip(axes, ('(a)', '(b)')):
    a_.text(0.02, 0.97, t, transform=a_.transAxes, ha='left', va='top', fontsize=10)
fig.savefig(OUT / 'fig9.pdf', dpi=500); plt.close(fig)
print('fig9.pdf saved')

# ------------------------------------------------------------------ Fig. E.1
en = (Y ** 2).sum(axis=(1, 2, 3))
idxs = [int(np.argsort(en)[int(q * (N - 1))]) for q in (0.25, 0.5, 0.9)]
Xg, Yg = np.meshgrid(XG, YG)
TGT = {k: absd[i] for k, i in enumerate(idxs)}
SG = {k: SIG[i] for k, i in enumerate(idxs)}
vmax_t = float(np.percentile(np.concatenate([t.ravel() for t in TGT.values()]), 99.5))
vmax_s = float(np.percentile(np.concatenate([t.ravel() for t in SG.values()]), 99.5))
VM = (vmax_t, vmax_s)
fig, axes = jf.new_figure_grid('JCP', 3, 2, width='onehalf', aspect=0.55)
ims = [None, None]
for rr in range(3):
    for c, (fld, lab) in enumerate([(TGT[rr], r'$|\Delta\mathbf{q}|$ (target)'),
                                    (SG[rr], r'$\hat{\sigma}$ (predicted)')]):
        ax = axes[rr, c]
        ims[c] = ax.pcolormesh(Xg, Yg, fld, cmap='viridis', vmin=0, vmax=VM[c], shading='auto', rasterized=True)
        ax.set_xlim(XG.min(), XG.max()); ax.set_ylim(YG.min(), YG.max()); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        if rr == 0:
            ax.set_title(lab, fontsize=10)
        if c == 0:
            ax.set_ylabel(f'snapshot {rr + 1}', fontsize=9.5)
for c, lab in enumerate([r'$|\Delta\mathbf{q}|$', r'$\hat{\sigma}$']):
    cb = fig.colorbar(ims[c], ax=list(axes[:, c]), shrink=0.80, pad=0.02, location='bottom', aspect=28,
                      ticks=[0, VM[c] / 2, VM[c]])
    cb.ax.set_xticklabels([f'{v_:.3f}' for v_ in (0, VM[c] / 2, VM[c])])
    cb.set_label(lab, fontsize=9.5); cb.ax.tick_params(labelsize=10)
fig.savefig(OUT / 'figE1.pdf', dpi=450); plt.close(fig)
print('figE1.pdf saved')
