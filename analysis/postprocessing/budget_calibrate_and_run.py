#!/usr/bin/env python3
"""원형, 4개 AMR 지표를 static1 급 셀 수(47,638~49,884)로 재실행.

목표 48,761 (구간 중앙), 허용 [47,638, 49,884].
각 케이스마다 파일럿 6 t.u. -> 정착 셀수 -> 예산 수정 을 허용 구간에 들 때까지
최대 4회 반복하고, 그다음 본런(95 -> 245)을 건다. 네 케이스 동시 진행.
"""
import re
import shutil
import subprocess
import time
from pathlib import Path

B = _roots.CR
TARGET, LO, HI, MAXIT, PILOT = 48761, 47638, 49884, 4, 6.0
SEED, END = 95.0, 245.0

# (새 케이스, 원본, 초기예산, 예산위치)
CASES = [
    ('NC49k_gradU', 'NC4x_gradU',  36400, 'fo'),
    ('NC49k_vort',  'NC4x_vort',   34500, 'fo'),
    ('NC49k_Q',     'NC4x_Q',      28763, 'fo'),
    ('NC49k_sigma', 'NC4x_sigma',  35900, 'ml'),
]

LOG = open(str(_roots.ROOT/'sim/tools/nc49k_calib.log'), 'w', buffering=1)


def say(m):
    print(m, flush=True)
    LOG.write(m + '\n')


def make_case(dst, src, nb, where):
    S, D = B / src, B / dst
    if D.exists():
        shutil.rmtree(D)
    D.mkdir()
    for d in ('constant', 'system'):
        shutil.copytree(S / d, D / d)
    shutil.copy(S / 'run.sh', D / 'run.sh')
    (D / '95').mkdir()
    for f in S.joinpath('95').iterdir():
        if f.is_file():
            shutil.copy(f, D / '95' / f.name)
    if (S / '95' / 'uniform').exists():
        shutil.copytree(S / '95' / 'uniform', D / '95' / 'uniform')
    (D / 'open.foam').touch()
    p = D / 'system' / 'controlDict'
    s = p.read_text()
    s = re.sub(r'^startTime\s+[0-9.]+;', f'startTime       {SEED:g};', s, flags=re.M)
    m = re.search(r'name\s+(budgetFlag_\w+);', s)
    if m:
        s = s.replace(m.group(1), f'budgetFlag_{dst}')
        s = re.sub(r'budgetFlag\[pctl_\w+\]', f'budgetFlag[{dst}]', s)
    p.write_text(s)
    set_nb(dst, nb, where)


def set_nb(case, nb, where):
    if where == 'fo':
        p = B / case / 'system' / 'controlDict'
        s = p.read_text()
        assert re.search(r'const label Nbudget = \d+;', s), case
        p.write_text(re.sub(r'const label Nbudget = \d+;',
                            f'const label Nbudget = {nb};', s))
    else:
        p = B / case / 'constant' / 'mlInferDict'
        s = p.read_text()
        assert re.search(r'attnThr\s+[0-9.]+;', s), case
        p.write_text(re.sub(r'attnThr\s+[0-9.]+;', f'attnThr         {nb};', s))


def get_nb(case, where):
    if where == 'fo':
        s = (B / case / 'system' / 'controlDict').read_text()
        return int(re.search(r'const label Nbudget = (\d+);', s).group(1))
    s = (B / case / 'constant' / 'mlInferDict').read_text()
    return int(float(re.search(r'attnThr\s+([0-9.]+);', s).group(1)))


def set_end(case, end):
    p = B / case / 'system' / 'controlDict'
    s = p.read_text()
    p.write_text(re.sub(r'^endTime\s+[0-9.]+;', f'endTime         {end:g};', s, flags=re.M))


def reset(case):
    d = B / case
    for t in d.iterdir():
        if re.fullmatch(r'[0-9]+(\.[0-9]+)?', t.name) and abs(float(t.name) - SEED) > 1e-9:
            shutil.rmtree(t)
    (d / 'log.pimpleFoam').unlink(missing_ok=True)
    if (d / 'postProcessing').exists():
        shutil.rmtree(d / 'postProcessing')


def launch(case):
    return subprocess.Popen(['bash', 'run.sh'], cwd=B / case,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cells(case, tmin):
    txt = (B / case / 'log.pimpleFoam').read_text(errors='ignore')
    ev = re.findall(r'^Time = ([0-9.eE+-]+)|nCells=(\d+)', txt, re.M)
    t, v = None, []
    for a, b in ev:
        if a:
            t = float(a)
        elif b and t is not None and t >= tmin:
            v.append(int(b))
    return v


def nfatal(case):
    return (B / case / 'log.pimpleFoam').read_text(errors='ignore').count('FOAM FATAL')


say(f'목표 {TARGET:,}  허용 [{LO:,}, {HI:,}]  파일럿 {PILOT:g} t.u.\n')
for dst, src, nb, where in CASES:
    make_case(dst, src, nb, where)
    say(f'  생성 {dst}  <- {src}  초기예산 {nb:,} ({where})')

state = {c: dict(where=w, done=False) for c, _, _, w in CASES}
for it in range(1, MAXIT + 1):
    todo = [c for c in state if state[c]['done'] is False]
    if not todo:
        break
    say(f'\n=== 보정 {it}회차 : ' +
        ', '.join(f'{c}(nb={get_nb(c, state[c]["where"])})' for c in todo) + ' ===')
    procs = {}
    for c in todo:
        reset(c)
        set_end(c, SEED + PILOT)
        procs[c] = launch(c)
    for p in procs.values():
        p.wait()
    for c in todo:
        w = state[c]['where']
        v = cells(c, SEED + 2.0)
        if not v:
            say(f'  {c}: 정제 기록 없음 (FATAL={nfatal(c)}) -> 제외')
            state[c]['done'] = None
            continue
        m = sum(v) / len(v)
        nb = get_nb(c, w)
        say(f'  {c}: 셀수 {m:,.0f}  목표대비 {(m-TARGET)/TARGET*100:+.1f}%  nb={nb:,}')
        if LO <= m <= HI:
            state[c]['done'] = True
            say(f'      -> 허용 구간. 예산 {nb:,} 확정')
        else:
            new = max(1000, int(round(nb * TARGET / m)))
            set_nb(c, new, w)
            say(f'      -> 예산 {nb:,} -> {new:,}')

for c in state:
    if state[c]['done'] is False:
        state[c]['done'] = True
        say(f'  {c}: {MAXIT}회 내 미수렴, 마지막 예산으로 본런')

run = [c for c in state if state[c]['done'] is True]
say('\n=== 본런 시작 (95 -> 245) : ' + ', '.join(run) + ' ===')
procs = {}
for c in run:
    reset(c)
    set_end(c, END)
    say(f'  {c}: nb={get_nb(c, state[c]["where"]):,}')
    procs[c] = launch(c)
t0 = time.time()
for p in procs.values():
    p.wait()
say(f'\n=== 본런 완주 ({(time.time()-t0)/3600:.1f} h) ===')
for c in run:
    v = cells(c, 165.0)
    m = sum(v) / len(v) if v else float('nan')
    say(f'  {c}: 평가창 셀수 {m:,.0f}  FATAL={nfatal(c)}')
say('\nDONE_NC49K')
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
