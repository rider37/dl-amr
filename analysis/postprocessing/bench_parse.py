#!/usr/bin/env python3
"""4x 속도 벤치 파서: ClockTime 기준 s/t.u. (시동 2 t.u. 제외) + 평균 셀 수."""
import re,json
from pathlib import Path
B=(_roots.ROOT/'sim/runs/bench_4x')
SKIP=2.0
ORDER=[('circular','c'),('square','s'),('diamond','d')]
KIND=[('coarse','static coarse 17k'),('gradU','AMR |∇u|'),('vort','AMR |ω|'),
      ('Q','AMR Q⁺'),('sigma','AMR σ̂ (DL)'),('finest','static finest 272k')]
rows=[]
for gname,gk in ORDER:
    for kk,klab in KIND:
        d=B/f'{gk}_{kk}'; L=d/'log.run'
        if not L.exists(): continue
        txt=L.read_text(errors='ignore')
        T=[float(x) for x in re.findall(r'^Time = ([0-9.eE+-]+)',txt,re.M)]
        CK=[float(x) for x in re.findall(r'ClockTime = ([0-9]+) s',txt)]
        NC=[int(x) for x in re.findall(r'^Cells:\s+(\d+)',txt,re.M)] or \
           [int(x) for x in re.findall(r'nCells:\s*(\d+)',txt)]
        n=min(len(T),len(CK))
        if n<10: continue
        T,CK=T[:n],CK[:n]
        t0=T[0]+SKIP
        i0=next(i for i,t in enumerate(T) if t>=t0)
        dt=T[-1]-T[i0]; dc=CK[-1]-CK[i0]
        if dt<=0: continue
        cells=(sum(NC)/len(NC)) if NC else None
        rows.append(dict(case=f'{gk}_{kk}',shape=gname,kind=klab,
                         t0=T[i0],t1=T[-1],span=dt,clock=dc,rate=dc/dt,
                         nsteps=n-i0,cells=cells,done=('End' in txt)))
json.dump(rows,open(B/'bench_rates.json','w'),indent=1)
print(f"{'케이스':<12}{'형상':<10}{'구성':<20}{'창(t.u.)':>10}{'s/t.u.':>10}{'셀':>9}{'완료':>6}")
for r in rows:
    c=f"{r['cells']:.0f}" if r['cells'] else '-'
    print(f"{r['case']:<12}{r['shape']:<10}{r['kind']:<20}{r['span']:>10.1f}{r['rate']:>10.2f}{c:>9}{'O' if r['done'] else 'X':>6}")
# 형상별 상대비
print('\n=== 형상별 static coarse 대비 배수 ===')
for gname,gk in ORDER:
    base=[r for r in rows if r['case']==f'{gk}_coarse']
    if not base: continue
    b=base[0]['rate']
    line=f'{gname:<10}'
    for kk,_ in KIND:
        rr=[r for r in rows if r['case']==f'{gk}_{kk}']
        line+=f"  {kk}={rr[0]['rate']/b:.2f}x" if rr else f'  {kk}=-'
    print(line)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
