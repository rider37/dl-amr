#!/usr/bin/env python3
"""논문 Figure 9: 사각주(Re=150) 양력·항력 계수 파워 스펙트럼 — 6방법, 대칭 런."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from fig_common import CR, OUT, STYLE, WINDOW, cases_for, jf
from scipy.signal import periodogram

jf.apply_style('JCP', font_pt=12)
import matplotlib.pyplot as plt  # noqa: E402

g = 'square'
C = cases_for(g)
ORDER = ['Fine', 'Coarse', r'$|\nabla\mathbf{U}|$ AMR', r'$|\omega|$ AMR', r'$Q$ AMR', 'DL-AMR',
         'static']


def coef(case):
    fs = sorted((CR / case / 'postProcessing/forceCoeffs').glob('*/coefficient*.dat'))
    best = max(fs, key=lambda f: f.stat().st_size)
    d = np.loadtxt(best, comments='#')
    d = d[np.argsort(d[:, 0])]
    t0, t1 = WINDOW[g]
    tg = np.arange(t0, t1 + 1e-9, 0.005)   # 표 2 잡음바닥과 동일 정의(보충 §5): Δt=0.005 보간, Hann, |F|^2/N
    return np.interp(tg, d[:, 0], d[:, 4]), np.interp(tg, d[:, 0], d[:, 1])


def spec(x, dt=0.005):
    y = x - x.mean(); F = np.fft.rfft(y * np.hanning(len(y)))
    return np.fft.rfftfreq(len(y), dt), np.abs(F) ** 2 / len(y)


fig, axes = jf.new_figure_grid('JCP', 1, 2, width='onehalf', aspect=0.52)
for m in ORDER:
    cl, cd = coef(C[m]['case'])
    st = STYLE[m]
    for ax, sig in zip(axes, (cl, cd)):
        f, P = spec(sig)
        ax.semilogy(f, P, color=st['color'], ls=st['ls'], lw=st['lw'],
                    zorder=st['zorder'], label=m, rasterized=True)
axes[0].set_ylabel(r'spectral power of $C_L$', fontsize=10.5)
axes[1].set_ylabel(r'spectral power of $C_D$', fontsize=10.5)
axes[0].set_ylim(1e-12, 1e3); axes[1].set_ylim(1e-13, 1e0)
for ax in axes:
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r'$St$', fontsize=10.5)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(which='both', labelsize=9.5)
fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=4, fontsize=9.6, frameon=False)
jf.add_panel_labels(axes)
fig.savefig(OUT / 'fig7.pdf', dpi=500)
plt.close(fig)
print('fig7.pdf saved', flush=True)
