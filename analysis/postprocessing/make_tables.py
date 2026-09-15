#!/usr/bin/env python3
"""SSOT → LaTeX 표. 손으로 옮겨 적는 수치 0개."""
import json
from pathlib import Path
R=_roots.ROOT; S=json.load(open(R/'analysis/output/eval/ssot.json'))
OUT=R/'analysis/output/tables'; OUT.mkdir(exist_ok=True)
LBL={'coarse':r'Coarse','gradU':r'$\gradU$ AMR','vort':r'$|\omega|$ AMR',
     'Q':r'$Q$ AMR','sigma':r'DL-AMR','static':r'\rev{static}'}
ORD=['coarse','gradU','vort','Q','sigma','static']
SH=[('circ','Circular'),('sq','Square'),('dia','Diamond')]
COLS=[('L2_Ux',r'$\overline{U}_x$'),('L2_Uy',r'$\overline{U}_y$'),('L2_wz',r'$\overline{\omega}_z$'),
      ('L2_k',r'$k$'),('L2_uv',r"$\overline{u'v'}$")]

def best(runs,key,lower=True):
    v={k:runs[k][key] for k in ORD if k in runs}
    return min(v,key=lambda k:abs(v[k])) if lower else max(v,key=lambda k:v[k])

# ── 표 1: 후류 정확도 ──────────────────────────────────────────
L=[r'\begin{table}[!htbp]',r'\centering',r'\small',
   r"\caption{\rev{Wake-field accuracy. $L_2$ errors of the time-averaged fields against the fine reference over "
   r"$x/D\in[-1.5,20]$, $|y|/D\le5$. The statistics $k$ and $\overline{u'v'}$ are the coherent fluctuation statistics of \cref{sec:setup}. $\Delta St$ is the relative "
   r"shedding-frequency error. Within each geometry, bold marks the lowest error in each error column, the smallest $|\Delta St|$, and the $C_{D,\mathrm{rms}}$ closest to the fine value. Ties at the displayed precision are all marked.}}",
   r'\label{tab:wake-accuracy}',
   r'\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{0.80}',
   r'\begin{tabular}{l S[table-format=6.0] S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=+1.2] S[table-format=1.4]}',
   r'\toprule',
   r"{Variant} & {Cells} & {$\overline{U}_x$} & {$\overline{U}_y$} & {$\overline{\omega}_z$} & {$k$} & {$\overline{u'v'}$} & {$\Delta St$ (\%)} & {$C_{D,\mathrm{rms}}$} \\",
   r'\midrule']
for sh,ttl in SH:
    A=S['accuracy'][sh]; runs=A['runs']; f=runs['finest']
    L.append(rf'\multicolumn{{9}}{{l}}{{\emph{{{ttl}}}}} \\')
    L.append(rf'\quad Fine & {f["cells"]} & {{---}} & {{---}} & {{---}} & {{---}} & {{---}} & {{---}} & {f["Cd_rms"]:.4f} \\')
    bst={k:best(runs,k) for k,_ in COLS}; bst['dSt_pct']=best(runs,'dSt_pct'); bst['Cd_rms']=None
    cdb=min((k for k in ORD if k!='coarse'), key=lambda k:abs(runs[k]['Cd_rms']-f['Cd_rms']))
    for k in ORD:
        r=runs[k]; row=[rf'\quad {LBL[k]}', f'{r["cells"]}']
        for key,_ in COLS:
                        v=f'{r[key]:.4f}'; tie=abs(r[key]-runs[bst[key]][key])<5e-5; row.append(rf'\bfseries {v}' if tie else v)
        v=f'{r["dSt_pct"]:+.2f}'; row.append(rf'\bfseries {v}' if bst['dSt_pct']==k else v)
        v=f'{r["Cd_rms"]:.4f}'; row.append(rf'\bfseries {v}' if cdb==k else v)
        L.append(' & '.join(row)+r' \\')
    L.append(r'\midrule' if sh!='dia' else r'\bottomrule')
L+= [r'\end{tabular}',r'\end{table}']
(OUT/'wake_accuracy.tex').write_text('\n'.join(L))

# ── 표 2: 런타임 (원형, 단독 코어) ─────────────────────────────
rt={r['case']:r for r in S['runtime']}
c=S['cost_breakdown']; tot=sum(v for k,v in c.items() if not k.startswith('_'))
_k=lambda d,*n: next((d[x] for x in n if x in d), None)
ROWS=[('c_coarse_v2','c_coarse',r'Coarse'),
      ('c_gradU',None,r'$\gradU$ AMR'),('c_vort',None,r'$|\omega|$ AMR'),
      ('c_Q',None,r'$Q$ AMR'),('c_sigma',None,r'DL-AMR'),
      ('c_static',None,r'\rev{static}'),
      ('c_finest_v2','c_finest',r'Fine')]
def rate(r): return r.get('rate', r.get('rate_exec'))
fine=_k(rt,'c_finest_v2','c_finest'); fr=rate(fine)
L=[r'\begin{table}[!htbp]',r'\centering',r'\small',
   r'\caption{\rev{Solver cost on the circular case: dedicated single core, field output disabled, first two '
   r'convective time units excluded. $\beta$ is the cost per cell relative to the fine mesh.}}',
   r'\label{tab:runtime}',
   r'\begin{tabular}{l S[table-format=6.0] S[table-format=3.2] S[table-format=1.3] S[table-format=1.2]}',
   r'\toprule',
   r'{Configuration} & {Cells} & {s per $t^{*}$} & {$\times$ fine} & {$\beta$} \\',r'\midrule']
for k1,k2,lab in ROWS:
    r=_k(rt,k1,k2) if k2 else rt.get(k1)
    if r is None: continue
    n=r['cells']; rr=rate(r)
    beta=(rr/n)/(fr/fine['cells'])
    L.append(rf'{lab} & {n:.0f} & {rr:.2f} & {rr/fr:.3f} & {beta:.2f} \\')
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
(OUT/'runtime.tex').write_text('\n'.join(L))

# ── 표 3: 비용 성분분해 ────────────────────────────────────────
L=[r'\begin{table}[!htbp]',r'\centering',r'\small',
   rf'\caption{{\rev{{Cost decomposition of one refinement step for the circular DL-AMR case, averaged '
   rf'over {c["_n"]} events of the dedicated single-core run.}}}}',
   r'\label{tab:overhead}',r'\setlength{\tabcolsep}{2.5pt}',
   r'\begin{tabular}{l S[table-format=1.4] S[table-format=3.1]}',r'\toprule',
   r'{Component} & {Time (s)} & {Share (\%)} \\',r'\midrule']
NM={'meshUpdate':r'Mesh refinement and remapping','fwd':r'Network forward pass (GPU)',
    'flag':r'Flag generation and marking','pre':r'Pre-processing','idxRebuild':r'Index-cache rebuild',
    'prep':r'Input assembly','sample':r'Mesh-to-grid sampling'}
for k,v in sorted(((k,v) for k,v in c.items() if not k.startswith('_')),key=lambda x:-x[1]):
    L.append(rf'{NM[k]} & {v:.4f} & {100*v/tot:.1f} \\')
L+=[r'\midrule',rf'Total & {tot:.4f} & 100.0 \\',r'\bottomrule',r'\end{tabular}',r'\end{table}']
(OUT/'overhead.tex').write_text('\n'.join(L))

# ── 표 4: 오차질량 피복률 ──────────────────────────────────────
cv=S['coverage']
# 최상위레벨 footprint 면적비 (analysis/output/eval/nc49k_alloc.pkl, f2>0.5 의 영역 내 비율, 2026-09-05 실측)
AREA={'gradU':(1.1,1.9,0.7),'vort':(1.4,2.3,0.7),'Q':(10.4,3.1,5.9),'sigma':(29.1,30.1,14.5)}
L=[r'\begin{table}[!htbp]',r'\centering',r'\small',
   r'\caption{\rev{Overlap between the mesh-induced error and the persistent finest-level footprint of each indicator, at the common cell budget. Cov.: fraction of the error mass $\lVert\overline{\mathbf{u}}_{\mathrm{coarse}}-\overline{\mathbf{u}}_{\mathrm{fine}}\rVert$, integrated over $x/D \in [-2, 20]$, $|y|/D \leq 4$ (body excluded), that lies in cells held at the finest level for more than half of the evaluation window. Area: fraction of that region occupied by those cells.}}',
   r'\label{tab:coverage}',r'\setlength{\tabcolsep}{4pt}',
   r'\begin{tabular}{l S[table-format=2.1] S[table-format=2.1] S[table-format=2.1] S[table-format=2.1] S[table-format=2.1] S[table-format=2.1]}',
   r'\toprule',r' & \multicolumn{2}{c}{Circular (\%)} & \multicolumn{2}{c}{Square (\%)} & \multicolumn{2}{c}{Diamond (\%)} \\',
   r'{Indicator} & {Cov.} & {Area} & {Cov.} & {Area} & {Cov.} & {Area} \\',r'\midrule']
for k,lab in [('gradU',r'$\gradU$'),('vort',r'$|\omega|$'),('Q',r'$Q$'),('sigma',r'$\hat{\sigma}$')]:
    a=AREA[k]
    L.append(rf'{lab} & {100*cv["circ"][k]:.1f} & {a[0]:.1f} & {100*cv["sq"][k]:.1f} & {a[1]:.1f} & {100*cv["dia"][k]:.1f} & {a[2]:.1f} \\')
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
(OUT/'coverage.tex').write_text('\n'.join(L))


# ══════════════ 개정본 추가 표 (rev2) ══════════════
FT = json.load(open(R/'analysis/output/eval/force_table.json'))
WD = json.load(open(R/'analysis/output/eval/wake_dynamics.json'))
FP = json.load(open(R/'analysis/output/eval/finest_probe.json'))
LBL2 = {'fine':'Fine','coarse':'Coarse','gradU':r'$\gradU$ AMR','vort':r'$|\omega|$ AMR',
        'Q':r'$Q$ AMR','sigma':r'DL-AMR','static':r'\rev{static}'}
ORD2 = ['fine','coarse','gradU','vort','Q','sigma','static']

# ── 표 5: 문헌 검증 ────────────────────────────────────────────
LIT = {'circ': [(r'Henderson (1997)', '1.341', '{---}', '0.197'),
                (r'Posdziech \& Grundmann (2007)', '1.3412', '{---}', '0.1971'),
                (r'Qu et al.\ (2013)', '1.3370', '0.4840', '0.1961')],
       'sq':   [(r'Sohankar et al.\ (1998), square at incidence, $\beta{=}2.5\%$', '1.400', '0.166', '0.160'),
                (r'Sohankar et al.\ (1999), square cylinder, $\beta{=}5.6\%$', '1.44', '0.23', '0.165')],
       'dia':  [(r'Sohankar et al.\ (1998), square at $45^\circ$ incidence', r'{$\sim$1.6}', '{---}', r'{$\sim$0.17}'),
                (r'Yoon et al.\ (2010), square at $45^\circ$ incidence', r'{$\sim$1.8}', '{---}', r'{$\sim$0.18}')]}
L=[r'\begin{table}[!htbp]',r'\centering',r'\small',
   r'\caption{Validation of the fine-mesh reference ($272{,}248$ cells) against published numerical '
   r'benchmarks. All coefficients use the projected frontal width $D$ as the reference length and are '
   r"evaluated over each geometry's evaluation window (\cref{tab:wake-accuracy}). Literature values for "
   r'the diamond case are read from figures and should be regarded as approximate.}',
   r'\label{tab:literature}',
   r'\begin{tabularx}{\textwidth}{@{}l>{\raggedright\arraybackslash}XSSS@{}}',r'\toprule',
   r'{Case} & {Source} & {$\overline{C}_D$} & {$C_{L,\mathrm{rms}}$} & {$St$} \\',r'\midrule']
for sh,ttl in SH:
    f = FT[sh]['fine']; st = WD[sh]['fine']['St_f']
    L.append(rf'{ttl}')
    L.append(rf'    & Present (fine) & {f["Cd_bar"]:.4f} & {f["Cl_rms"]:.4f} & {st:.4f} \\')
    for src,cd,cl,stv in LIT[sh]:
        L.append(rf'    & {src} & {cd} & {cl} & {stv} \\')
    L.append(r'\midrule' if sh!='dia' else r'\bottomrule')
L+=[r'\end{tabularx}',r'\end{table}']
(OUT/'literature.tex').write_text('\n'.join(L))

# ── 표 6: 힘계수 ───────────────────────────────────────────────
L=[r'\begin{table}[!htbp]',r'\centering',r'\small',
   r"\caption{\rev{Force coefficients for all variants, evaluated over each geometry's evaluation "
   r'window (\cref{tab:wake-accuracy}). The near-body mesh topology is shared across all cases; '
   r'shedding frequencies are reported in \cref{tab:literature,tab:wake-dynamics}.}}',
   r'\label{tab:force-metrics}',
   r'\sisetup{round-mode=places, round-precision=4, table-format=1.4}',
   r'\renewcommand{\arraystretch}{0.9}\begin{tabular}{ll SS}',r'\toprule',
   r'Case & Method & {$\overline{C}_D$} & {$C_{L,\mathrm{rms}}$} \\',r'\midrule']
for sh,ttl in SH:
    for i,k in enumerate(ORD2):
        d = FT[sh][k]
        head = ttl if i==0 else ''
        L.append(rf'{head} & {LBL2[k]} & {d["Cd_bar"]:.4f} & {d["Cl_rms"]:.4f} \\')
    L.append(r'\midrule' if sh!='dia' else r'\bottomrule')
L+=[r'\end{tabular}',r'\end{table}']
(OUT/'force_metrics.tex').write_text('\n'.join(L))

# ── 표 7: 비정상 후류 동역학 ───────────────────────────────────
L=[r'\begin{table}[!htbp]',r'\centering',r'\small',
   r"\caption{\rev{Unsteady wake diagnostics. $St_f$ and $St_p$ are the shedding frequencies from the lift spectrum and from a wake probe at $(5,0)$, "
   r"$\mathrm{std}\,U_y'$ the probe cross-stream fluctuation amplitude, "
   r'and $\Delta N/N$ the fraction of cells changed per mesh update.}}',
   r'\label{tab:wake-dynamics}',
   r'\rev{\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{0.85}\begin{tabular}{llcccc}',r'\toprule',
   r"Case & Method & $St_f$ & $St_p$ & $\mathrm{std}\,U_y'$ & $\Delta N/N$ (\%) \\",
   r'\midrule']
for sh,ttl in SH:
    short = ttl.split()[0]
    for i,k in enumerate(ORD2):
        d = WD[sh][k]
        stp = f'{d["St_p"]:.4f}' if 'St_p' in d else '---'
        sv  = FP[sh]['stdV'] if k=='fine' else d.get('stdV1')
        svs = f'{sv:.4f}' if sv is not None else '---'
        tn  = f'{d["turn"]:.2f}' if 'turn' in d and k not in ('fine','coarse','static') else '0'
        head = short if i==0 else ''
        L.append(rf'{head} & {LBL2[k]} & {d["St_f"]:.4f} & {stp} & {svs} & {tn} \\')
    L.append(r'\midrule' if sh!='dia' else r'\bottomrule')
L+=[r'\end{tabular}}',r'\end{table}']
(OUT/'wake_dynamics.tex').write_text('\n'.join(L))

# ── 표 8: ROC AUC ─────────────────────────────────────────────
U = S['uncertainty']
L=[r'\begin{table}[!htbp]',r'\centering',
   r'\caption{\rev{AUC for uncertainty-based detection of high mesh-residual regions on the held-out '
   r'test split. Positives are grid points whose $|\Delta\mathbf{q}|$ exceeds the quantile threshold $\tau$.}}',
   r'\label{tab:roc-auc}',r'\small',
   r'\rev{\begin{tabular}{l S[table-format=1.4] S[table-format=1.3] S[table-format=2.1]}',r'\toprule',
   r'{Quantile} & {$\tau$} & {AUC} & {Positive (\%)} \\',r'\midrule']
for q,pc in (('q90',10.0),('q95',5.0),('q98',2.0),('q99',1.0)):
    L.append(rf'$q_{{{q[1:]}}}$ & {U["tau"][q]:.4f} & {U["auc"][q]:.3f} & {pc:.1f} \\')
L+=[r'\bottomrule',r'\end{tabular}}',r'\end{table}']
(OUT/'roc_auc.tex').write_text('\n'.join(L))

# ── 표 9: 형상 간 전이 ─────────────────────────────────────────
TR = S['transfer']
RW = [('circular Re200 (test)', 'Circular (training geometry)'),
      ('square Re150 (zero-shot)', 'Square (unseen)'),
      ('diamond Re150 (zero-shot)', 'Diamond (unseen)')]
L=[r'\begin{table}[!htbp]',r'\centering',
   r'\caption{\rev{Cross-geometry transfer of the uncertainty indicator ($q_{95}$ positives defined per geometry). '
   r'The circular row is the held-out test split of the training data ($Re = 100$ and $150$), and the square and diamond rows are the $Re = 150$ runs of \cref{sec:setup}. '
   r'The near wake is $x/D\in[0,12]$, $|y|/D<2.5$, and $\rho$ is the Spearman rank correlation.}}',
   r'\label{tab:cross-geometry-uncertainty}',r'\small',
   r'\rev{\begin{tabular}{l S[table-format=3.0] S[table-format=1.3] S[table-format=1.3] S[table-format=1.3]}',
   r'\toprule',
   r'{Geometry} & {Pairs} & {AUC($q_{95}$)} & {AUC($q_{95}$), near wake} & {$\rho$} \\',
   r'\midrule']
for key,lab in RW:
    t = TR[key]
    rho = S['uncertainty']['spearman'] if key.startswith('circular') else t["spearman"]  # 원형 행은 §4.4 와 같은 전체점 계산값 사용 (0.809)
    L.append(rf'{lab} & {t["n"]} & {t["auc"]["95"]:.3f} & {t["near_auc95"]:.3f} & {rho:.3f} \\')
L+=[r'\bottomrule',r'\end{tabular}}',r'\end{table}']
(OUT/'cross_geometry.tex').write_text('\n'.join(L))

print('생성:', *[p.name for p in sorted(OUT.glob('*.tex'))])
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
