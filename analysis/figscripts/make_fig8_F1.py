#!/usr/bin/env python3
"""논문 Figure 8 (와류 대류 속도 U_c(x), 1×3) · 보충 Fig S5 (St(t), 1×3). 와도장은 본문 Fig 6 배경으로 사용(make_fig3_6.py).
입력: eval/figures/c7/c7_uc.json, c7_dynamics.json (eval/scripts/c7_*.py 로 생성)."""
import sys, json, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from fig_common import CACHE, STYLE, jf, ROOT, OUT
jf.apply_style('JCP', font_pt=10.5)
import matplotlib.pyplot as plt
D=json.load(open(CACHE/'c7/c7_dynamics.json')); UC=json.load(open(CACHE/'c7/c7_uc.json'))
LAB={'fine':'Fine','gradU':r'$|\nabla\mathbf{U}|$ AMR','vort':r'$|\omega|$ AMR','Q':r'$Q$ AMR','sigma':'DL-AMR','static':'static'}
SH={'fine':'Fine','gradU':r'$|\nabla\mathbf{U}|$','vort':r'$|\omega|$','Q':r'$Q$','sigma':'DL-AMR','static':'static'}
GE=(('circ','Circular'),('sq','Square'),('dia','Diamond'))
# Figure 8: U_c(x)
fig,axes=jf.new_figure_grid('JCP',1,3,width='onehalf',aspect=0.40,font_pt=10.5)
for c,(g,gl) in enumerate(GE):
    ax=axes[c]
    for k in LAB:
        r=UC[g][k]; st=STYLE[LAB[k]]; ax.plot(r['xmid'],r['Uc'],marker='o',ms=3,color=st['color'],ls=st['ls'],lw=st['lw'],zorder=st['zorder'],label=SH[k])
    ax.set_xticks([3,4,5,6,7,8]); ax.set_xlim(3,8); ax.set_xlabel(r'$x/D$'); ax.set_title(f'({"abc"[c]}) {gl}',fontsize=10.5); ax.grid(True,lw=0.4,alpha=0.5)
    if c==0: ax.set_ylabel(r'$U_c/U_\infty$')
h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,frameon=False,ncol=6,loc='upper center',bbox_to_anchor=(0.5,1.12),handlelength=2.0,columnspacing=1.0,fontsize=9)
fig.savefig(OUT/'fig8.pdf',dpi=500,bbox_inches='tight'); plt.close(fig); print('fig8.pdf saved',flush=True)
# Fig S5: St(t)
fig,axes=jf.new_figure_grid('JCP',1,3,width='onehalf',aspect=0.40,font_pt=10.5)
for c,(g,gl) in enumerate(GE):
    ax=axes[c]; tmin=min(D[g][k]['Stp_t']['t'][0] for k in LAB); tmax=max(D[g][k]['Stp_t']['t'][-1] for k in LAB)
    for k in LAB:
        r=D[g][k]; st=STYLE[LAB[k]]; ax.plot(r['Stp_t']['t'],r['Stp_t']['St'],color=st['color'],ls=st['ls'],lw=st['lw'],zorder=st['zorder'],label=SH[k])
    allv=np.concatenate([np.array(D[g][k]['Stp_t']['St']) for k in LAB]); ax.set_ylim(allv.min()-0.001,allv.max()+0.001)
    ax.ticklabel_format(useOffset=False); ax.set_xlim(tmin,tmax); ax.set_xlabel(r'$t$'); ax.set_title(f'({"abc"[c]}) {gl}',fontsize=10.5); ax.grid(True,lw=0.4,alpha=0.5)
    if c==0: ax.set_ylabel(r'$St(t)$')
h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,frameon=False,ncol=6,loc='upper center',bbox_to_anchor=(0.5,1.12),handlelength=2.0,columnspacing=1.0,fontsize=9)
fig.savefig(OUT/'figF1.pdf',dpi=500,bbox_inches='tight'); plt.close(fig); print('figF1.pdf saved',flush=True)
