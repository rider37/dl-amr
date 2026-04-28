"""Generate Error vs DOF figure — the AMR gold-standard comparison."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(os.environ.get('DL_AMR_OUTDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')))

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
})

# Data from wake_accuracy table
data = {
    'Circular ($Re=200$)': {
        'Coarse':    (57798,  0.0243),
        'DL-AMR':    (90734,  0.0145),
        r'$|\nabla \mathbf{U}|$ AMR': (84809, 0.0239),
        'Fine':      (135840, 0.0),
    },
    'Square ($Re=150$)': {
        'Coarse':    (57798,  0.0232),
        'DL-AMR':    (84105,  0.0135),
        r'$|\nabla \mathbf{U}|$ AMR': (85162, 0.0262),
        'Fine':      (135840, 0.0),
    },
    'Diamond ($Re=150$)': {
        'Coarse':    (57798,  0.1303),
        'DL-AMR':    (106855, 0.0811),
        r'$|\nabla \mathbf{U}|$ AMR': (99115, 0.1310),
        'Fine':      (135840, 0.0),
    },
}

method_styles = {
    'Coarse':    {'color': '#0072B2', 'marker': 'v', 'ms': 8},
    'DL-AMR':    {'color': '#D55E00', 'marker': 'o', 'ms': 9},
    r'$|\nabla \mathbf{U}|$ AMR': {'color': '#009E73', 'marker': 's', 'ms': 8},
    'Fine':      {'color': '#000000', 'marker': '*', 'ms': 10},
}

fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=False)

for ax, (geom_label, variants) in zip(axes, data.items()):
    for method, (cells, l2) in variants.items():
        if method == 'Fine':
            continue  # L2=0, skip on log scale
        st = method_styles[method]
        ax.plot(cells / 1000, l2, marker=st['marker'], color=st['color'],
                markersize=st['ms'], linestyle='none', label=method,
                markeredgecolor='black', markeredgewidth=0.5, zorder=5)

    # Fine reference line at bottom
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle=':', alpha=0.3)

    ax.set_xlabel('Mean cells ($\\times 10^3$)')
    ax.set_title(geom_label, fontsize=10)
    ax.set_xlim(40, 150)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=9)

    # Add arrows showing DL-AMR advantage
    dl_cells, dl_l2 = variants['DL-AMR']
    grad_cells, grad_l2 = variants[r'$|\nabla \mathbf{U}|$ AMR']
    coarse_cells, coarse_l2 = variants['Coarse']

axes[0].set_ylabel(r'$L_2(\overline{U}_x)$')

# Single legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, 1.08), frameon=False)

fig.tight_layout(rect=[0, 0, 1, 0.95])
outpath = OUT_DIR / 'error_vs_dof.png'
fig.savefig(outpath)
plt.close(fig)
print(f"Saved {outpath}")

# Print key comparisons
for geom, variants in data.items():
    dl = variants['DL-AMR']
    grad = variants[r'$|\nabla \mathbf{U}|$ AMR']
    coarse = variants['Coarse']
    print(f"\n{geom}:")
    print(f"  DL-AMR:  {dl[0]:,} cells, L2={dl[1]:.4f}")
    print(f"  grad:    {grad[0]:,} cells, L2={grad[1]:.4f}")
    print(f"  At similar cell count, DL-AMR L2 is {(1-dl[1]/grad[1])*100:.0f}% lower than grad-AMR")
