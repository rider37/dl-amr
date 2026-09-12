#!/usr/bin/env python3
"""tab:wake-dynamics 재생성용 — 전 변종(coarse/gradU/vort/Q/sigma/static) + finest.
정의는 nc4x_dynamics.py 와 동일. 결과를 analysis/output/eval/wake_dynamics_49k.json 으로 저장."""
import glob, re, json
from pathlib import Path
import numpy as np

CR = _roots.CR

def forces(case):
    fs = sorted(glob.glob(str(CR/case/'postProcessing/forceCoeffs/*/coefficient*.dat')))
    a = np.vstack([np.loadtxt(x, comments='#') for x in fs]); a = a[np.argsort(a[:, 0])]
    _, i = np.unique(a[:, 0], return_index=True); a = a[i]
    return a[:, 0], a[:, 1], a[:, 4]

def probe(case, idx=1):
    """probesW 의 idx 번째 점 (0:(3,0) 1:(5,0) 2:(8,0)) 의 (t, Ux, Uy)."""
    fs = sorted(glob.glob(str(CR/case/'postProcessing/probesW/*/U')))
    rows = []
    for f in fs:
        for ln in open(f):
            if ln.startswith('#'):
                continue
            s = ln.replace('(', ' ').replace(')', ' ').split()
            try: rows.append([float(x) for x in s])
            except ValueError: pass
    if not rows:
        return None, None, None
    n = min(len(r) for r in rows)
    a = np.array([r[:n] for r in rows]); a = a[np.argsort(a[:, 0])]
    _, i = np.unique(a[:, 0], return_index=True); a = a[i]
    return a[:, 0], a[:, 1+3*idx], a[:, 1+3*idx+1]

def spec(t, x, dt=0.005):
    g = np.arange(t[0], t[-1], dt); y = np.interp(g, t, x - x.mean())
    F = np.fft.rfft(y*np.hanning(len(y))); fr = np.fft.rfftfreq(len(g), dt)
    return fr, np.abs(F)**2/len(g)

def peak(f, P, lo=0.05, hi=1.0):
    b = np.where((f > lo) & (f < hi))[0]; k = b[np.argmax(P[b])]
    a_, b_, c_ = np.log(P[k-1]), np.log(P[k]), np.log(P[k+1])
    return float(f[k] + 0.5*(a_-c_)/(a_-2*b_+c_)*(f[1]-f[0]))

def dyn(case, lo, hi):
    t, cd, cl = forces(case); m = (t >= lo) & (t <= hi)
    t, cd, cl = t[m], cd[m], cl[m]
    f, P = spec(t, cl); St = peak(f, P)
    excl = np.zeros_like(f, bool)
    for n in range(1, 7): excl |= np.abs(f - n*St) < 0.35*St
    o = dict(case=case, St_f=St, Cd=float(cd.mean()), Cd_rms=float(cd.std()),
             Cl_rms=float(cl.std()),
             floor=float(np.median(P[(f > 0.05) & (f < 3.0) & ~excl])))
    tp, ux, uy = probe(case)
    if tp is not None:
        mm = (tp >= lo) & (tp <= hi)
        if mm.sum() > 100:
            o['stdV'] = float(uy[mm].std())
            # finest 는 필드 출력이 Δt=1.0 뿐이므로 표에는 공통 간격 값을 쓴다
            gg = np.arange(lo, hi + 1e-9, 1.0)
            o['stdV1'] = float(np.interp(gg, tp[mm], uy[mm]).std())
            fp, Pp = spec(tp[mm], uy[mm]); o['St_p'] = peak(fp, Pp)
    lg = CR/case/'log.pimpleFoam'
    if lg.exists():
        pr = re.findall(r'from (\d+) to (\d+) cells', lg.read_text(errors='ignore'))
        if pr:
            pr = pr[-400:]
            o['turn'] = 100*float(np.mean([abs(int(b)-int(a))/int(a) for a, b in pr]))
    return o

GROUPS = [
 ('circ', 165, 245, dict(fine='grid_convergence/circular_Re200_finer', coarse='NC2_coarse',
   gradU='NC49k_gradU', vort='NC49k_vort', Q='NC49k_Q', sigma='NC49k_sigma', static='ST_static1_circ')),
 ('sq', 237.5, 346.5, dict(fine='grid_convergence/square_Re150_finer', coarse='NCsq_coarse_full',
   gradU='SQ49k_gradU', vort='SQ49k_vort', Q='SQ49k_Q', sigma='SQ49k_sigma',
   static='ST_static1_sq')),
 ('dia', 170, 256, dict(fine='grid_convergence/diamond_Re150_finer', coarse='NCdia_coarse_full',
   gradU='DIA49k_gradU', vort='DIA49k_vort', Q='DIA49k_Q', sigma='DIA49k_sigma',
   static='ST_static1_dia')),
]
OUT = {}
for g, lo, hi, runs in GROUPS:
    OUT[g] = {'window': [lo, hi]}
    print(f'\n=== {g} [{lo:g},{hi:g}] ===')
    print(f'{"방법":>8s}{"St_f":>8s}{"St_p":>8s}{"stdV":>8s}{"floor":>11s}{"floor비":>9s}{"turn%":>8s}')
    ref = None
    for lab, case in runs.items():
        d = dyn(case, lo, hi); OUT[g][lab] = d
        if lab == 'fine': ref = d
        print(f'{lab:>8s}{d["St_f"]:8.4f}{d.get("St_p",float("nan")):8.4f}'
              f'{d.get("stdV",float("nan")):8.4f}{d["floor"]:11.3e}'
              f'{d["floor"]/ref["floor"]:9.1f}{d.get("turn",float("nan")):8.2f}', flush=True)
json.dump(OUT, open(str(_roots.ROOT/'analysis/output/eval/wake_dynamics_49k.json'), 'w'), indent=1)
print('\nWROTE analysis/output/eval/wake_dynamics_49k.json')
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
