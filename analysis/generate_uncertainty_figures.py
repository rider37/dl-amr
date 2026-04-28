"""
Regenerate uncertainty validation figures for the paper.
- Publication-quality design
- Serif fonts, 300 DPI
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Publication style — moderate font sizes
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
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

ROOT = os.environ.get('DL_AMR_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR = f'{ROOT}/eval/figures/uncertainty_nll_uvp_wake'
OUT_DIR = f'{ROOT}/paper/fig'
PRED_DIR = f'{ROOT}/ml/runs/infer_delta_hetero_star_uvp_wake_re100_150_nll_lr3e4_test/preds'
DATA_PT = f'{ROOT}/ml/data/processed/cylinder_delta_star_uvp_wake_re100_150/test.pt'

# Load CSV data
cal = pd.read_csv(f'{DATA_DIR}/calibration_bins.csv')

# Colors
C_BLUE = '#2166ac'
C_GRAY = '#888888'

# =====================================================================
# Figure 6: Monotonicity (sigma_bin_vs_abs_delta)
# =====================================================================
fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.scatter(cal['mean_sigma'], cal['mean_abs_delta'], s=20, color=C_BLUE,
           zorder=3, edgecolors='white', linewidths=0.3)
ax.plot(cal['mean_sigma'], cal['mean_abs_delta'], '-', color=C_BLUE,
        linewidth=1.2, alpha=0.7)
ax.set_xlabel(r'Mean predicted $\hat{\sigma}$ per bin')
ax.set_ylabel(r'Mean $|\Delta \mathbf{q}|$ per bin')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.25, linewidth=0.5)
fig.savefig(f'{OUT_DIR}/sigma_bin_vs_abs_delta.png')
plt.close(fig)
print('Saved sigma_bin_vs_abs_delta.png')

# =====================================================================
# Load raw data for ROC + spatial overlay
# =====================================================================
print('Loading test data...')
test_data = torch.load(DATA_PT, map_location='cpu', weights_only=False)
y_all = test_data['y'].numpy()  # (801, 3, 64, 224)

all_sigma = []
all_abs_delta = []
all_pred_arrays = []

for i in range(y_all.shape[0]):
    p = np.load(f'{PRED_DIR}/{i:05d}.npz', allow_pickle=True)
    logvar = p['aux'][0]  # (64, 224)
    sigma = np.sqrt(np.exp(logvar))
    abs_delta = np.linalg.norm(y_all[i], axis=0)  # (64, 224)
    all_sigma.append(sigma.ravel())
    all_abs_delta.append(abs_delta.ravel())
    all_pred_arrays.append(p['pred'])

all_sigma = np.concatenate(all_sigma)
all_abs_delta = np.concatenate(all_abs_delta)
print(f'Total pixels: {len(all_sigma)}')

# ROC computation
tau_q95 = np.quantile(all_abs_delta, 0.95)
labels = (all_abs_delta > tau_q95).astype(int)
fpr, tpr, _ = roc_curve(labels, all_sigma)
auc_val = roc_auc_score(labels, all_sigma)
print(f'ROC AUC (q95): {auc_val:.4f}')

# =====================================================================
# Figure 7: Calibration + ROC side-by-side (matched height)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

# 7a: Calibration
max_val = max(cal['mean_sigma'].max(), cal['observed_rmse_norm'].max()) * 1.05
ax1.plot([0, max_val], [0, max_val], '--', color=C_GRAY, linewidth=0.8,
         label='Ideal', zorder=1)
ax1.scatter(cal['mean_sigma'], cal['observed_rmse_norm'], s=20, color=C_BLUE,
            zorder=3, edgecolors='white', linewidths=0.3)
ax1.plot(cal['mean_sigma'], cal['observed_rmse_norm'], '-', color=C_BLUE,
         linewidth=1.2, alpha=0.7, zorder=2)
ax1.set_xlabel(r'Mean predicted $\hat{\sigma}$ per bin')
ax1.set_ylabel('Observed normalised RMSE per bin')
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.25, linewidth=0.5)
ax1.set_title('(a) Calibration curve')

# 7b: ROC
ax2.fill_between(fpr, tpr, alpha=0.08, color=C_BLUE)
ax2.plot(fpr, tpr, '-', color=C_BLUE, linewidth=1.5,
         label=f'AUC = {auc_val:.3f}')
ax2.plot([0, 1], [0, 1], '--', color=C_GRAY, linewidth=0.8)
ax2.set_xlabel('False positive rate')
ax2.set_ylabel('True positive rate')
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.set_aspect('equal')
ax2.legend(loc='lower right', framealpha=0.9)
ax2.grid(True, alpha=0.25, linewidth=0.5)
ax2.set_title(r'(b) ROC for $|\Delta \mathbf{q}| > q_{95}$')

fig.subplots_adjust(wspace=0.35)
fig.savefig(f'{OUT_DIR}/calibration_roc.png')
plt.close(fig)
print('Saved calibration_roc.png')

# =====================================================================
# Figure 8: Spatial overlay (3 representative samples)
# Layout: 4 rows (one per field) x 3 cols (one per sample)
# This gives each panel more horizontal space and larger visuals
# =====================================================================
sample_indices = [719, 610, 110]

row_titles = [
    r'$|\Delta \mathbf{q}|$ (target)',
    r'$\hat{\sigma}$ (uncertainty)',
    r'$|\widehat{\Delta \mathbf{q}}|$ (prediction)',
    r'$|\widehat{\Delta \mathbf{q}} - \Delta \mathbf{q}|$ (error)',
]
col_labels = ['Snapshot 1', 'Snapshot 2', 'Snapshot 3']

fig, axes = plt.subplots(4, 3, figsize=(12, 10))

# First pass: collect vmax per row (shared colorbar range across samples)
row_vmax = [0.0] * 4
all_fields = []
for col, idx in enumerate(sample_indices):
    p = np.load(f'{PRED_DIR}/{idx:05d}.npz', allow_pickle=True)
    pred = p['pred']
    logvar = p['aux'][0]
    sigma = np.sqrt(np.exp(logvar))
    target = y_all[idx]

    abs_target = np.linalg.norm(target, axis=0)
    abs_pred = np.linalg.norm(pred, axis=0)
    abs_err = np.linalg.norm(pred - target, axis=0)

    fields = [abs_target, sigma, abs_pred, abs_err]
    all_fields.append(fields)
    for row, field in enumerate(fields):
        row_vmax[row] = max(row_vmax[row], np.percentile(field, 99))

# Second pass: plot with shared vmax per row, single colorbar per row
row_ims = [None] * 4
for col, fields in enumerate(all_fields):
    for row, field in enumerate(fields):
        ax = axes[row, col]
        im = ax.imshow(field, aspect='auto', cmap='inferno', vmin=0,
                       vmax=row_vmax[row], origin='lower',
                       interpolation='bilinear')
        row_ims[row] = im
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(col_labels[col], fontsize=13, pad=8)
        if col == 0:
            ax.set_ylabel(row_titles[row], fontsize=12, labelpad=8)

# Add one colorbar per row, positioned outside the plot area
fig.subplots_adjust(wspace=0.06, hspace=0.18, right=0.85)
fig.canvas.draw()
for row in range(4):
    bbox = axes[row, 2].get_position()
    cax = fig.add_axes([0.87, bbox.y0, 0.012, bbox.y1 - bbox.y0])
    cb = fig.colorbar(row_ims[row], cax=cax)
    cb.ax.tick_params(labelsize=10)
fig.savefig(f'{OUT_DIR}/spatial_overlay_top3.png')
plt.close(fig)
print('Saved spatial_overlay_top3.png')

print('All figures generated.')
