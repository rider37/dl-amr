#!/usr/bin/env python3
"""49k 예산 SSOT. 67k 판(build_ssot.py)과 같은 구조·같은 키를 낸다.

정확도  : 49k 평가 로그 3개
런타임  : bench_49k (static1/gradU/vort/sigma/Q 신규 측정)
          + fine·coarse 는 예산 무관 고정 격자라 67k 측정 재사용
피복률  : nc49k_alloc.pkl (Q 포함)
"""
import json
import re
from datetime import datetime
from pathlib import Path

R = _roots.ROOT
SP = Path('/tmp/claude-1000/-home-jiwon/73436739-5be8-42d4-893c-49134a687457/scratchpad')

NAME = {'NC2_coarse': 'coarse', 'NCsq_coarse_full': 'coarse', 'NCdia_coarse_full': 'coarse',
        'NC49k_gradU': 'gradU', 'SQ49k_gradU': 'gradU', 'DIA49k_gradU': 'gradU',
        'NC49k_vort': 'vort', 'SQ49k_vort': 'vort', 'DIA49k_vort': 'vort',
        'NC49k_Q': 'Q', 'SQ49k_Q': 'Q', 'DIA49k_Q': 'Q',
        'NC49k_sigma': 'sigma', 'SQ49k_sigma': 'sigma', 'DIA49k_sigma': 'sigma',
        'ST_static1_circ': 'static', 'ST_static1_sq': 'static', 'ST_static1_dia': 'static',
        'finest': 'finest'}
KEYS = ['cells', 'L2_Ux', 'L2_Uy', 'L2_wz', 'L2_k', 'L2_uv', 'dSt_pct', 'Cd_rms', 'Cl_rms']


def parse(log, window):
    out = {}
    for ln in Path(log).read_text().split('\n'):
        p = ln.split()
        if len(p) != 10:
            continue
        nm = NAME.get(p[0])
        if nm is None:
            continue
        v = [float(x.replace(',', '').replace('+', '')) for x in p[1:]]
        out[nm] = dict(zip(KEYS, [int(v[0])] + v[1:]))
        out[nm]['case'] = p[0]
    return dict(window=window, ref='grid_convergence/*_finer (272,248 cells)', runs=out)


S = {'meta': dict(
    built=datetime.now().isoformat(timespec='seconds'),
    budget_target=48761, base_cells=16944, static1_cells=49884,
    note='budget set so that the Q-criterion does not spill into the free stream; '
         'all variants land below the 49,884-cell uniformly refined static mesh',
    static_def='W1 (x in [-2,25], |y|<=5) refined once; no second-level cells')}

S['accuracy'] = {
    'circ': parse(SP / 'nc49k_final.log', r'$t^{*}\in[165,245]$, 15.7 shedding periods'),
    'sq':   parse(SP / 'sq49k_final.log', r'$t^{*}\in[237.5,346.5]$, 16.6 shedding periods'),
    'dia':  parse(SP / 'dia49k_final2.log', r'$t^{*}\in[170,256]$, 16.4 shedding periods')}

# ── 런타임: bench_49k + 고정격자 재사용 ──
b49 = json.loads((R / 'sim/runs/bench_49k/bench49k_rates.json').read_text())
MAP = [('b49_static1', 'c_static', r'\rev{static}'),
       ('b49_gradU', 'c_gradU', r'$\gradU$ AMR'),
       ('b49_vort', 'c_vort', r'$|\omega|$ AMR'),
       ('b49_sigma', 'c_sigma', r'DL-AMR ($\hat{\sigma}$)'),
       ('b49_Q', 'c_Q', r'$Q$ AMR'),
       ('c_coarse_v2', 'c_coarse_v2', r'Coarse (no refinement)'),
       ('c_finest_v2', 'c_finest_v2', r'Fine reference')]
# 셀 수는 평가창 실측값을 쓴다 (벤치 로그의 nCells 는 20 t.u. 평균)
CELLS = {'c_static': 49884,
         'c_gradU': S['accuracy']['circ']['runs']['gradU']['cells'],
         'c_vort': S['accuracy']['circ']['runs']['vort']['cells'],
         'c_sigma': S['accuracy']['circ']['runs']['sigma']['cells'],
         'c_Q': S['accuracy']['circ']['runs']['Q']['cells'],
         'c_coarse_v2': 16944, 'c_finest_v2': 272248}
S['runtime'] = [dict(case=dst, lab=lab, rate=b49[src]['rate'], cells=CELLS[dst],
                     source=b49[src].get('src', src), done=True)
                for src, dst, lab in MAP if src in b49]

# ── 비용 성분분해 (49k 벤치의 DL 케이스) ──
t = (R / 'sim/runs/bench_49k/b49_sigma/log.run').read_text(errors='ignore')
comp = {}
for k in ('pre', 'idxRebuild', 'sample', 'prep', 'fwd', 'flag'):
    v = [float(x) for x in re.findall(rf'{k}=([0-9.eE+-]+)', t)]
    if v:
        comp[k] = sum(v) / len(v)
mu = [float(x) for x in re.findall(r'meshUpdate=([0-9.eE+-]+)', t)]
if mu:
    comp['meshUpdate'] = sum(mu) / len(mu)
    comp['_n'] = len(mu)
S['cost_breakdown'] = comp

# ── 불확실도·전이 (예산 무관) ──
for k, f in (('uncertainty', 'g6_uncertainty_uni.json'), ('transfer', 'g7_transfer_uni.json')):
    p = R / 'eval/scripts' / f
    if p.exists():
        S[k] = json.loads(p.read_text())

# ── 오차질량 피복률 ──
import pickle
al = pickle.load(open(R / 'analysis/output/eval/nc49k_alloc.pkl', 'rb'))
S['coverage'] = {sh: {k: round(v['cov'], 4) for k, v in al['R'][sh].items()
                      if isinstance(v, dict)} for sh in al['R']}

out = R / 'analysis/output/eval/ssot.json'
out.write_text(json.dumps(S, indent=1, ensure_ascii=False))
print('저장:', out)
for sh in ('circ', 'sq', 'dia'):
    r = S['accuracy'][sh]['runs']
    print(f"  {sh}: {list(r.keys())}  셀 "
          f"{min(v['cells'] for k, v in r.items() if k not in ('finest','coarse')):,}"
          f"~{max(v['cells'] for k, v in r.items() if k not in ('finest','coarse')):,}")
print(f"  runtime: {[(r['case'], round(r['rate'],2)) for r in S['runtime']]}")
print(f"  coverage: {S['coverage']}")
print(f"  cost_breakdown n={comp.get('_n')}")
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
