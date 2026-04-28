"""
σ_thr sensitivity analysis for DL-AMR.
Sweep threshold values and compute:
- Fraction of pixels flagged for refinement
- Estimated cell count (proportional)
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _data_check import require_or_skip

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

ROOT = os.environ.get('DL_AMR_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
PRED_DIR = os.environ.get('DL_AMR_PREDS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reference_data', 'preds'))
DATA_PT = os.environ.get('DL_AMR_TESTPT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reference_data', 'test.pt'))
OUT_DIR = os.environ.get('DL_AMR_OUTDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
os.makedirs(OUT_DIR, exist_ok=True)

require_or_skip(
    'Fig 12 (threshold sensitivity)',
    'Run `make download-reference` to fetch test.pt and per-sample predictions, '
    'or set DL_AMR_TESTPT / DL_AMR_PREDS to point at your own copies.',
    DATA_PT, PRED_DIR,
)

C_BLUE = '#2166ac'
C_RED = '#b2182b'
C_GRAY = '#888888'

# Load data
print('Loading data...')
test_data = torch.load(DATA_PT, map_location='cpu', weights_only=False)
y_all = test_data['y'].numpy()  # (801, 3, 64, 224)

# Collect all sigma and |delta_q|
all_sigma = []
all_abs_delta = []
for i in range(y_all.shape[0]):
    p = np.load(f'{PRED_DIR}/{i:05d}.npz', allow_pickle=True)
    logvar = p['aux'][0]
    sigma = np.sqrt(np.exp(logvar))
    abs_delta = np.linalg.norm(y_all[i], axis=0)
    all_sigma.append(sigma.ravel())
    all_abs_delta.append(abs_delta.ravel())

all_sigma = np.concatenate(all_sigma)
all_abs_delta = np.concatenate(all_abs_delta)
n_total = len(all_sigma)
print(f'Total pixels: {n_total}')

# Sweep thresholds
thresholds = np.linspace(0.05, 2.0, 50)
frac_refined = []
mean_delta_in_refined = []
mean_delta_in_unrefined = []

for thr in thresholds:
    mask = all_sigma >= thr
    frac = mask.sum() / n_total
    frac_refined.append(frac)
    if mask.sum() > 0:
        mean_delta_in_refined.append(all_abs_delta[mask].mean())
    else:
        mean_delta_in_refined.append(0.0)
    if (~mask).sum() > 0:
        mean_delta_in_unrefined.append(all_abs_delta[~mask].mean())
    else:
        mean_delta_in_unrefined.append(0.0)

frac_refined = np.array(frac_refined)
mean_delta_in_refined = np.array(mean_delta_in_refined)
mean_delta_in_unrefined = np.array(mean_delta_in_unrefined)

# Contrast ratio: how well does the threshold separate high vs low error
contrast = mean_delta_in_refined / (mean_delta_in_unrefined + 1e-10)

# =====================================================================
# Figure: Two-panel plot
# (a) Fraction of pixels refined vs threshold
# (b) Mean |Δq| in refined vs unrefined regions
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

# (a) Refinement fraction
ax1.plot(thresholds, frac_refined * 100, '-', color=C_BLUE, linewidth=1.5)
ax1.axvline(x=0.6, color=C_RED, linestyle='--', linewidth=0.8, alpha=0.7, label=r'$\sigma_{\mathrm{thr}} = 0.6$')
ax1.set_xlabel(r'Threshold $\sigma_{\mathrm{thr}}$')
ax1.set_ylabel('Pixels flagged for refinement (%)')
ax1.set_xlim(thresholds[0], thresholds[-1])
ax1.set_ylim(bottom=0)
ax1.legend(loc='upper right', framealpha=0.9)
ax1.grid(True, alpha=0.25, linewidth=0.5)
ax1.set_title(r'(a) Refinement fraction vs.\ $\sigma_{\mathrm{thr}}$')

# (b) Mean |Δq| separation
ax2.plot(thresholds, mean_delta_in_refined, '-', color=C_RED, linewidth=1.5, label='Refined region')
ax2.plot(thresholds, mean_delta_in_unrefined, '-', color=C_BLUE, linewidth=1.5, label='Unrefined region')
ax2.axvline(x=0.6, color=C_GRAY, linestyle='--', linewidth=0.8, alpha=0.7)
ax2.set_xlabel(r'Threshold $\sigma_{\mathrm{thr}}$')
ax2.set_ylabel(r'Mean $|\Delta \mathbf{q}|$')
ax2.set_xlim(thresholds[0], thresholds[-1])
ax2.set_ylim(bottom=0)
ax2.legend(loc='center right', framealpha=0.9)
ax2.grid(True, alpha=0.25, linewidth=0.5)
ax2.set_title(r'(b) Error separation vs.\ $\sigma_{\mathrm{thr}}$')

fig.subplots_adjust(wspace=0.35)
fig.savefig(f'{OUT_DIR}/dl_threshold_sensitivity.png')
plt.close(fig)
print('Saved dl_threshold_sensitivity.png')

# Print key values
idx_05 = np.argmin(np.abs(thresholds - 0.5))
print(f'\nAt σ_thr = 0.5:')
print(f'  Refined fraction: {frac_refined[idx_05]*100:.1f}%')
print(f'  Mean |Δq| in refined: {mean_delta_in_refined[idx_05]:.5f}')
print(f'  Mean |Δq| in unrefined: {mean_delta_in_unrefined[idx_05]:.5f}')
print(f'  Contrast ratio: {contrast[idx_05]:.1f}x')

# Also print for a range of thresholds
print('\nThreshold sweep summary:')
for thr_val in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
    idx = np.argmin(np.abs(thresholds - thr_val))
    print(f'  σ_thr={thr_val:.1f}: refined={frac_refined[idx]*100:.1f}%, '
          f'contrast={contrast[idx]:.1f}x')
