#!/usr/bin/env python3
"""fig6 재생성: 통일 프로토콜 정확도-vs-셀수 (형상별 1패널, 7방법 + fine 기준선)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from fig_common import CACHE, GEOMS, OUT, STYLE, WM, body_mask, cases_for, jf, mean_ux

FONT = 10.5
jf.apply_style('JCP', font_pt=FONT)
import matplotlib.pyplot as plt  # noqa: E402

# 개정본 추가 기준선 (Table 5와 동일한 캐시 키)
EXTRA = {'static': dict(key='static', color='#56B4E9', zorder=3)}

MARK = {'Coarse': 'o',
        r'$|\nabla\mathbf{U}|$ AMR': 's', r'$|\omega|$ AMR': '^', r'$Q$ AMR': 'D',
        'DL-AMR': '*', 'static': 'P'}
MS = {'Coarse': 5, 'DL-AMR': 10,
      r'$|\nabla\mathbf{U}|$ AMR': 5, r'$|\omega|$ AMR': 6, r'$Q$ AMR': 5,
      'static': 6}
METHODS = list(MARK)

fig, axes = jf.new_figure_grid('JCP', 2, 2, width='onehalf', aspect=0.92,
                               font_pt=FONT)
axes = axes.ravel()
for c, (short, g, glab) in enumerate(GEOMS):
    ax = axes[c]
    C = cases_for(g)
    ref = mean_ux(g, C['Fine']['key'])
    w = ~body_mask(short)
    l2s = []
    for m in METHODS:
        key = C[m]['key'] if m in C else EXTRA[m]['key']
        st = STYLE[m] if m in STYLE else EXTRA[m]
        z = np.load(WM / f'{g}_{key}.npz')
        cells = float(z['ncell']) / 1e3
        l2 = float(np.sqrt(np.nanmean(((z['mean_ux'] - ref)[w]) ** 2)))
        l2s.append(l2)
        ax.plot(cells, l2, MARK[m], color=st['color'], ms=MS[m],
                mec='k', mew=0.4, label=m, zorder=st['zorder'])
    ax.set_xlim(14, 52)
    ax.set_xticks([20, 30, 40, 50])
    ax.set_ylim(0, max(l2s) * 1.15)
    ax.set_title(f'({"abc"[c]}) {glab}', fontsize=10)
    ax.set_xlabel(r'mean cells ($\times 10^3$)', fontsize=10)
    if c in (0, 2):
        ax.set_ylabel(r'$L_2(\overline{U}_x)$', fontsize=10)
# (d) circular: accuracy against measured solver cost (Table 5, dedicated single core)
import re as _re
RT = {}
for ln in (CACHE / 'tables/runtime.tex').read_text().splitlines():
    mm = _re.match(r'\\?(?:rev\{)?(Coarse|\$\\gradU\$ AMR|\$\|\\omega\|\$ AMR|\$Q\$ AMR|DL-AMR|static|Fine)\}? & \d+ & ([\d.]+) &', ln)
    if mm: RT[mm.group(1)] = float(mm.group(2))
KEYMAP = {'Coarse': 'Coarse', r'$|\nabla\mathbf{U}|$ AMR': r'$\gradU$ AMR', r'$|\omega|$ AMR': r'$|\omega|$ AMR', r'$Q$ AMR': r'$Q$ AMR', 'DL-AMR': 'DL-AMR', 'static': 'static'}
ax = axes[3]; C = cases_for('circular'); ref = mean_ux('circular', C['Fine']['key']); w = ~body_mask('circular'); l2s = []
for m in METHODS:
    key = C[m]['key'] if m in C else EXTRA[m]['key']; st = STYLE[m] if m in STYLE else EXTRA[m]
    z = np.load(WM / f'circular_{key}.npz'); l2 = float(np.sqrt(np.nanmean(((z['mean_ux'] - ref)[w]) ** 2))); l2s.append(l2)
    ax.plot(RT[KEYMAP[m]], l2, MARK[m], color=st['color'], ms=MS[m], mec='k', mew=0.4, zorder=st['zorder'])
ax.set_xscale('log'); ax.set_xlim(1.5, 300); ax.axvline(RT['Fine'], color='k', lw=0.8, ls='--'); ax.text(RT['Fine'], max(l2s) * 1.05, 'fine', ha='right', va='top', fontsize=8.5)
ax.set_ylim(0, max(l2s) * 1.15); ax.set_title('(d) Circular', fontsize=10); ax.set_xlabel(r'solver cost (s per $t^{*}$)', fontsize=10)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside upper center', ncol=4, fontsize=9.5,
           frameon=False, columnspacing=1.4, handletextpad=0.4)
fig.savefig(OUT / 'fig5.pdf', dpi=500)
plt.close(fig)
print('fig5.pdf saved', flush=True)
