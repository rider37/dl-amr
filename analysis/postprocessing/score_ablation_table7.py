#!/usr/bin/env python3
"""3단계 ablation 표: analysis/output/eval/stage3/{circ,sq,dia}.log (nc2_metrics 출력) → tables/ablation_policy.tex
행: σ̂ (DL-AMR) / mean head |Δq̂| / |ω| through the learned wrapper / |ω| standard / Kelly+W1 (원형만)."""
import re, json
from pathlib import Path
R=_roots.ROOT; D=R/'analysis/output/eval/stage3'
LAB={'NC49k_sigma':r'$\hat{\sigma}$','SQ49k_sigma':r'$\hat{\sigma}$','DIA49k_sigma':r'$\hat{\sigma}$',
     'NC49k_mu':r'mean head','SQ49k_mu':r'mean head','DIA49k_mu':r'mean head',
     'NC49k_vortgrid':r'$|\omega|$ via wrapper','NC49k_vort':r'$|\omega|$ cell-local','KLW1_circular_kelly':r'Kelly, wake set'}
ORDER=['NC49k_sigma','NC49k_mu','NC49k_vortgrid','NC49k_vort','KLW1_circular_kelly','SQ49k_sigma','SQ49k_mu','DIA49k_sigma','DIA49k_mu']
rows={}
for g in ('circ','sq','dia','kelly'):
    p=D/f'{g}.log'
    if not p.exists(): continue
    for ln in p.read_text().split('\n'):
        q=ln.split()
        if len(q)==10 and q[0] in LAB:
            v=[float(x.replace(',','').replace('+','')) for x in q[1:]]; rows[q[0]]=dict(cells=int(v[0]),Ux=v[1],Uy=v[2],wz=v[3],k=v[4],uv=v[5],dSt=v[6],Cdr=v[7],Clr=v[8])
json.dump(rows,open(D/'stage3_rows.json','w'),indent=1)
L=[r'\begin{table}[!t]',r'\centering',r'\small',
   r'\caption{\rev{Score ablation. The learned policy\textquotesingle s candidate set, sampling path, level weight and settings are held fixed and only the score is exchanged; every row is calibrated to the common cell count. The Kelly row is a residual-based baseline run with the same wrapper.}}',
   r'\label{tab:ablation}',r'\small\setlength{\tabcolsep}{2.5pt}\renewcommand{\arraystretch}{0.85}',
   r'\begin{tabular}{l S[table-format=5.0] S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=+1.2] S[table-format=1.4]}',r'\toprule',
   r'Score & {Cells} & {$\overline{U}_x$} & {$\overline{U}_y$} & {$\overline{\omega}_z$} & {$k$} & {$\overline{u\textquotesingle v\textquotesingle}$} & {$\Delta St$ (\%)} & {$C_{D,\mathrm{rms}}$} \\',r'\midrule']
cur=None
for c in ORDER:
    if c not in rows: continue
    g={'NC':'Circular','SQ':'Square','DI':'Diamond','KL':'Circular'}[c[:2]]
    if g!=cur:
        if cur is not None: L.append(r'\midrule')
        L.append(r'\multicolumn{9}{l}{\emph{'+g+'}} \\\\')
    r=rows[c]; L.append(f"\\quad {LAB[c]} & {r['cells']} & {r['Ux']:.4f} & {r['Uy']:.4f} & {r['wz']:.4f} & {r['k']:.4f} & {r['uv']:.4f} & {r['dSt']:+.2f} & {r['Cdr']:.4f} \\\\")
    cur=g
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
(R/'analysis/output/tables/ablation_policy.tex').write_text('\n'.join(L)+'\n'); print('rows:',list(rows)); print('\n'.join(L[8:]))
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
