#!/usr/bin/env python3
"""Table F.1 (coherent vs raw fluctuation-statistic errors) from the body-unmasked stage-2 recomputation (same grid/mask as Table 1)."""
import json
from pathlib import Path
R=_roots.ROOT; LAB={'coarse':'Coarse','gradU':r'$\gradU$ AMR','vort':r'$|\omega|$ AMR','Q':r'$Q$ AMR','sigma':'DL-AMR','static':'static'}
L=[r'\begin{table}[!htbp]',r'\centering',r'\small',
   r'\caption{\rev{Errors against the fine reference of the coherent (harmonic-fit) and raw finite-window fluctuation statistics on the same snapshots, and the fraction of $k$ outside the mean and first two harmonics (residual/$k$).}}',
   r'\label{tab:supp-rawcoh}',r'\setlength{\tabcolsep}{4pt}',
   r'\rev{\begin{tabular}{ll S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=1.2]}',r'\toprule',
   r' & & \multicolumn{2}{c}{$L_2(k)$} & \multicolumn{2}{c}{$L_2(\overline{u^\prime v^\prime})$} & \\',
   r'Case & Method & {coherent} & {raw} & {coherent} & {raw} & {residual/$k$ (\%)} \\',r'\midrule']
stats={}
for g,gl in (('circ','Circular'),('sq','Square'),('dia','Diamond')):
    d=json.load(open(R/f'analysis/output/eval/stage2/moments_halfwin_{g}_nobody.json'))
    L.append(f"{gl} & Fine & {{---}} & {{---}} & {{---}} & {{---}} & {100*d['fine']['kres_over_k']:.2f} \\\\")
    for k in ('coarse','gradU','vort','Q','sigma','static'):
        v=d[k]; L.append(f" & {LAB[k]} & {v['L2k_h']:.4f} & {v['L2k_r']:.4f} & {v['L2uv_h']:.4f} & {v['L2uv_r']:.4f} & {100*v['kres_over_k']:.2f} \\\\")
        stats[(g,k)]=(100*(v['L2k_r']/v['L2k_h']-1),100*(v['L2uv_r']/v['L2uv_h']-1),100*v['kres_over_k'],v['L2k_h'],v['L2uv_h'])
    if g!='dia': L.append(r'\midrule')
L+=[r'\bottomrule',r'\end{tabular}}',r'\end{table}']
(R/'analysis/output/tables/appF1_rawcoh.tex').write_text('\n'.join(L)+'\n')
for g in ('circ','sq','dia'):
    print(g,'max |Δ%| k/uv:',max(abs(stats[(g,k)][0]) for k in LAB),'/',max(abs(stats[(g,k)][1]) for k in LAB),' max residual/k',max(stats[(g,k)][2] for k in LAB))
# Table 1 consistency check on coherent values
import re
t1=(R/'analysis/output/tables/wake_accuracy.tex').read_text()
NAME={'coarse':'Coarse','gradU':r'$\gradU$ AMR','vort':r'$|\omega|$ AMR','Q':r'$Q$ AMR','sigma':'DL-AMR','static':r'\rev{static}'}
for g,gl in (('circ','Circular'),('sq','Square'),('dia','Diamond')):
    blk=t1[t1.index(r'\emph{'+gl+'}'):]; blk=blk[:blk.find(r'\midrule',10) if r'\midrule' in blk[10:] else None]
    for k in LAB:
        m=re.search(re.escape(NAME[k])+r' & \d+ & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+)',blk)
        if m:
            k1,uv1=float(m.group(4)),float(m.group(5)); kh,uvh=stats[(g,k)][3],stats[(g,k)][4]
            if f"{kh:.4f}"!=f"{k1:.4f}" or f"{uvh:.4f}"!=f"{uv1:.4f}": print('MISMATCH',g,k,'T1',k1,uv1,'S7',round(kh,4),round(uvh,4))
print('consistency check done')
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
