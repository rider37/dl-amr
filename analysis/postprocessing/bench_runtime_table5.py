#!/usr/bin/env python3
"""49k 예산 런타임 벤치. 기존 프로토콜과 동일:
   taskset -c 4 단독 코어, 쓰기 없음, t=200 시드 -> 220, 202~220 구간 ClockTime.
   반드시 직렬 실행 (동시 실행 시 +6.6% 오염 실측).
fine / coarse 는 AMR 예산과 무관한 고정 격자라 기존 측정을 재사용한다.
"""
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

B = _roots.CR
BD = (_roots.ROOT/'sim/runs/bench_49k')
BD.mkdir(parents=True, exist_ok=True)
T0, T1, TMEAS = 200.0, 220.0, 202.0
CORE = '4'

CASES = [  # (bench, src, solver)
    ('b49_static1', 'ST_static1_circ', 'pimpleFoam'),
    ('b49_gradU',   'NC49k_gradU',     'pimpleFoam'),
    ('b49_vort',    'NC49k_vort',      'pimpleFoam'),
    ('b49_sigma',   'NC49k_sigma',     'amrPimpleFoam'),
    ('b49_Q',       'NC49k_Q',         'pimpleFoam'),
]

RUNSH = """#!/usr/bin/env bash
source /usr/lib/openfoam/openfoam2312/etc/bashrc
export FOAM_USER_APPBIN=$FOAM_USER_APPBIN
export PATH=$FOAM_USER_APPBIN:$PATH
export OMP_NUM_THREADS=1
cd "$(dirname "$0")"
taskset -c {core} {solver} > log.run 2>&1
"""


def busy():
    out = subprocess.run(['ps', '-eo', 'comm'], capture_output=True, text=True).stdout
    return sum(1 for l in out.splitlines()
               if l.strip() in ('pimpleFoam', 'amrPimpleFoamTo', 'amrPimpleFoam'))


def make(bench, src, solver):
    S, D = B / src, BD / bench
    if D.exists():
        shutil.rmtree(D)
    D.mkdir()
    for d in ('constant', 'system'):
        shutil.copytree(S / d, D / d)
    sd = S / '200'
    shutil.copytree(sd, D / '200')
    (D / 'open.foam').touch()
    (D / 'run.sh').write_text(RUNSH.format(core=CORE, solver=solver))
    p = D / 'system' / 'controlDict'
    s = p.read_text()
    s = re.sub(r'^startTime\s+[0-9.]+;', f'startTime       {T0:g};', s, flags=re.M)
    s = re.sub(r'^endTime\s+[0-9.]+;', f'endTime         {T1:g};', s, flags=re.M)
    s = re.sub(r'^writeInterval\s+[0-9.]+;', 'writeInterval   100000;', s, flags=re.M)
    p.write_text(s)


def rate(bench):
    """202~220 구간의 ClockTime 기울기 [s / t.u.]"""
    txt = (BD / bench / 'log.run').read_text(errors='ignore')
    ev = re.findall(r'^Time = ([0-9.eE+-]+)|ClockTime = (\d+)', txt, re.M)
    t, pairs = None, []
    for a, b in ev:
        if a:
            t = float(a)
        elif b and t is not None:
            pairs.append((t, int(b)))
    lo = [c for tt, c in pairs if tt >= TMEAS]
    if len(lo) < 2:
        return None
    t_lo = next(tt for tt, c in pairs if tt >= TMEAS)
    t_hi, c_hi = pairs[-1]
    c_lo = next(c for tt, c in pairs if tt >= TMEAS)
    return (c_hi - c_lo) / (t_hi - t_lo), t_lo, t_hi


assert busy() == 0, f'다른 솔버 {busy()}개 실행 중 — 벤치 중단'
print(f'프로토콜: taskset -c {CORE}, t={T0:g}->{T1:g}, 측정 {TMEAS:g}->{T1:g}, 직렬\n', flush=True)

res = {}
for bench, src, solver in CASES:
    assert busy() == 0, '다른 솔버 감지 — 중단'
    make(bench, src, solver)
    print(f'  {bench:12s} 시작 ({src}, {solver}) ...', end='', flush=True)
    t0 = time.time()
    subprocess.run(['bash', 'run.sh'], cwd=BD / bench,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    r = rate(bench)
    n = re.findall(r'nCells=(\d+)', (BD / bench / 'log.run').read_text(errors='ignore'))
    cells = sum(map(int, n[-40:])) / len(n[-40:]) if n else None
    if r:
        res[bench] = dict(src=src, rate=r[0], t=[r[1], r[2]], cells=cells,
                          wall=time.time() - t0)
        print(f'  {r[0]:7.2f} s/t.u.  (벽시계 {(time.time()-t0)/60:.1f}분)', flush=True)
    else:
        print('  실패', flush=True)

# 예산 무관 고정 격자: 기존 측정 재사용
old = json.loads((_roots.ROOT/'analysis/output/eval/ssot.json').read_text())['runtime']
for r in old:
    if r['case'] in ('c_coarse_v2', 'c_finest_v2'):
        res[r['case']] = dict(src='reused', rate=r['rate'], cells=r['cells'],
                              note='fixed mesh, budget-independent')

P = (_roots.ROOT/'sim/runs/bench_49k/bench49k_rates.json')
P.write_text(json.dumps(res, indent=1))
print(f'\n저장: {P}')
fine = res['c_finest_v2']['rate']
print(f'\n{"case":14s}{"s/t.u.":>10s}{"cells":>10s}{"beta":>8s}{"R*":>8s}')
R = 272248 / 16944
for k, v in res.items():
    b = v['rate'] / fine
    print(f'{k:14s}{v["rate"]:10.2f}{(v["cells"] or 0):10,.0f}{b:8.3f}{b*R:8.2f}')
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
