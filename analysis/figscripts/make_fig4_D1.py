#!/usr/bin/env python3
"""fig4(오차맵)·fig5(산점)·fig7(프로파일) — 배치폭 140mm(onehalf), 방법=행/형상=열, 폰트>=7.5pt."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from fig_common import (GEOMS, OUT, STYLE, Xi, Yi, body_mask, body_patch,
                        cases_for, jf, mean_ux, xi, yi)
from matplotlib.ticker import MaxNLocator

jf.apply_style('JCP', font_pt=10.5)
import matplotlib.pyplot as plt  # noqa: E402

W = 'onehalf'          # 140 mm ~= 실제 textwidth 137 mm (스케일 0.98)
FT, FL, FA = 10.0, 10.0, 9.5   # 제목/축라벨/주석
METHODS5 = ['Coarse', r'$|\nabla\mathbf{U}|$ AMR', r'$|\omega|$ AMR', r'$Q$ AMR', 'DL-AMR',
            'static']
# 7열 배치에서는 열 제목 폭이 부족하므로 축약형을 쓴다(캡션에 전개)
SHORT_T = {'Coarse': 'Coarse', 'DL-AMR': 'DL-AMR', r'$|\nabla\mathbf{U}|$ AMR': r'$|\nabla\mathbf{U}|$',
           r'$|\omega|$ AMR': r'$|\omega|$', r'$Q$ AMR': r'$Q$',
           'static': 'static'}

def tidy(ax, nx=4, ny=3):
    ax.xaxis.set_major_locator(MaxNLocator(nx))
    ax.yaxis.set_major_locator(MaxNLocator(ny))


# ---------------- fig4: |error| maps (방법=행, 형상=열, 열별 스케일+하단 컬러바) ----------------
fig, axes = jf.new_figure_grid('JCP', len(METHODS5), 3, width='onehalf', aspect=1.0,
                               font_pt=10.5)
for c, (short, g, glab) in enumerate(GEOMS):
    C = cases_for(g)
    ref = mean_ux(g, C['Fine']['key'])
    body = body_mask(short)
    errs = [np.where(body, np.nan, np.abs(mean_ux(g, C[m]['key']) - ref)) for m in METHODS5]
    vmax = np.nanpercentile(np.concatenate([e.ravel() for e in errs]), 99)
    for r, (m, e) in enumerate(zip(METHODS5, errs)):
        ax = axes[r, c]
        im = ax.pcolormesh(Xi, Yi, e, cmap='viridis', vmin=0, vmax=vmax,
                           shading='auto', rasterized=True)
        ax.add_patch(body_patch(short))
        ax.set_xlim(-1.5, 20); ax.set_ylim(-5, 5); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(glab, fontsize=10)
        if c == 0:
            ax.set_ylabel(m, fontsize=9.5, rotation=0, ha='right', va='center',
                          labelpad=4)
    cb = fig.colorbar(im, ax=list(axes[:, c]), shrink=0.80, pad=0.02,
                      location='bottom', aspect=28,
                      ticks=[0.0, vmax / 2, vmax])
    nd = 2 if vmax >= 0.1 else 3
    cb.ax.set_xticklabels([f'{v:.{nd}f}' for v in (0.0, vmax / 2, vmax)])
    cb.set_label(r'$|\overline{U}_x-\overline{U}_x^{\,\mathrm{ref}}|$', fontsize=9.5)
    cb.ax.tick_params(labelsize=10)
fig.savefig(OUT / 'fig4.pdf', dpi=500)
plt.close(fig)
print('fig4.pdf saved', flush=True)

plt.close(fig)

# ---------------- fig7: profiles (3행 station x 3열 형상) ----------------
STATIONS = [5.0, 10.0, 15.0]
ORDER = ['Fine', 'Coarse', r'$|\nabla\mathbf{U}|$ AMR', r'$|\omega|$ AMR', r'$Q$ AMR', 'DL-AMR',
         'static']
fig, axes = jf.new_figure_grid('JCP', 3, 3, width=W, aspect=1.05,
                               font_pt=10.5)
for r, xs in enumerate(STATIONS):
    ic = int(np.argmin(np.abs(xi - xs)))
    for c, (short, g, glab) in enumerate(GEOMS):
        ax = axes[r, c]
        C = cases_for(g)
        for m in ORDER:
            v = mean_ux(g, C[m]['key'])[:, ic]
            st = STYLE[m]
            ax.plot(v, yi, color=st['color'], ls=st['ls'], lw=st['lw'],
                    zorder=st['zorder'], label=m)
        ax.set_ylim(-3.5, 3.5)
        tidy(ax, 3, 4)
        if r == 0:
            ax.set_title(glab, fontsize=10)
        ax.text(0.05, 0.04, f'$x/D={xs:.0f}$', transform=ax.transAxes, fontsize=9.5)
        if c == 0:
            ax.set_ylabel('$y/D$', fontsize=10)
        else:
            ax.set_yticklabels([])
        if r == 2:
            ax.set_xlabel(r'$\overline{U}_x/U_\infty$', fontsize=10)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside upper center', ncol=4, fontsize=9.5,
           frameon=False)
fig.savefig(OUT / 'figD1.pdf', dpi=500)
plt.close(fig)
print('figD1.pdf saved', flush=True)
