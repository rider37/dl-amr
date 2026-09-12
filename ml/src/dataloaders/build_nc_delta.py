#!/usr/bin/env python3
"""Build the two-grid discrepancy dataset used to train the heteroscedastic U-Net.

Inputs are the coarse-mesh velocity fields (u*, v*) of the circular-cylinder training cases at
Re = 100 and 150, sampled onto the uniform 432 x 160 grid covering x/D in [-2, 25],
y/D in [-5, 5]; the target is the two-grid discrepancy (fine - coarse) on the same grid,
with coarse and fine snapshots paired by the Hilbert phase of the lift signal
(search window +-3.2 convective time units); the mask excludes body cells.
Pairs are formed at 0.2-unit spacing over t in [150, 400] and split into contiguous
chronological blocks (80/10/10 %) per Reynolds number (Section 2.2 of the paper).

Usage (from the repository root, after running the four training cases)::

    python -m ml.src.dataloaders.build_nc_delta \
        --cases cases --output ml/data/processed/nc_delta_uv_uni

``--cases`` must contain ``circular_Re{100,150}/coarse`` and ``circular_Re{100,150}/fine``
(the "fine" case is the medium mesh of the grid-convergence study, Section 2.2), each with
``postProcessing/forceCoeffs`` and stored fields every 0.1 time units.

Outputs: train.pt / val.pt / test.pt (dict with X, y, mask, meta), blocked_split_ids.json,
norm_stats.json, target_norm.json, grid_x.npy, grid_y.npy.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from scipy.signal import hilbert

ap = argparse.ArgumentParser()
ap.add_argument('--cases', default='cases', help='directory holding circular_Re100/ and circular_Re150/')
ap.add_argument('--output', default='ml/data/processed/nc_delta_uv_uni')
ap.add_argument('--re', type=int, nargs='+', default=[100, 150])
args = ap.parse_args()
CR = Path(args.cases)
OUT = Path(args.output); OUT.mkdir(parents=True, exist_ok=True)

NX, NY = 432, 160
X0, X1, Y0, Y1 = -2.0, 25.0, -5.0, 5.0
T0, T1, DT = 150.0, 400.0, 0.2
SEARCH = 3.2

mx = np.linspace(X0, X1, NX); my = np.linspace(Y0, Y1, NY)   # uniform grid (solver makePlaneGrid convention)
Mx, My = np.meshgrid(mx, my); GM = pv.StructuredGrid(Mx, My, np.zeros_like(Mx))
np.save(OUT / 'grid_x.npy', mx); np.save(OUT / 'grid_y.npy', my)
print(f'grid: x {mx[0]:.2f}..{mx[-1]:.2f} ({NX}), y {my[0]:.2f}..{my[-1]:.2f} ({NY})', flush=True)


def cl_phase(case: Path):
    """Unwrapped Hilbert phase of the lift coefficient, on a 0.01-unit time grid."""
    fs = sorted(glob.glob(str(case / 'postProcessing/forceCoeffs/*/coefficient.dat')))
    a = np.vstack([np.loadtxt(x, comments='#') for x in fs]); a = a[np.argsort(a[:, 0])]
    _, i = np.unique(a[:, 0], return_index=True); a = a[i]
    s = (a[:, 0] >= T0 - 10) & (a[:, 0] <= T1 + 1)
    t = a[s, 0]; y = a[s, 4] - a[s, 4].mean()
    g = np.arange(t[0], t[-1], 0.01); yg = np.interp(g, t, y)
    return g, np.unwrap(np.angle(hilbert(yg)))


def sample(rd, av, t):
    rd.set_active_time_value(float(av[np.abs(av - t).argmin()]))
    mm = rd.read()[0]
    mm = mm.cell_data_to_point_data() if 'U' in mm.cell_data else mm
    s = GM.sample(mm)
    ok = np.asarray(s.point_data['vtkValidPointMask']).reshape(NX, NY).T > 0.5
    U = np.asarray(s.point_data['U'])
    return (U[:, 0].reshape(NX, NY).T.astype(np.float32),
            U[:, 1].reshape(NX, NY).T.astype(np.float32), ok)


X, Y, MK, META = [], [], [], []
for re_ in args.re:
    dc, df = CR / f'circular_Re{re_}' / 'coarse', CR / f'circular_Re{re_}' / 'fine'
    for d in (dc, df):
        (d / 'open.foam').touch(exist_ok=True)
    rc = pv.OpenFOAMReader(str(dc / 'open.foam')); rf = pv.OpenFOAMReader(str(df / 'open.foam'))
    avc = np.array(rc.time_values); avf = np.array(rf.time_values)
    gc, pc = cl_phase(dc); gf, pf = cl_phase(df)
    ts = np.arange(T0, T1 + 1e-9, DT)
    ts = ts[(ts >= avc.min()) & (ts <= avc.max())]
    print(f'Re{re_}: {len(avc)} coarse and {len(avf)} fine snapshots, {len(ts)} samples', flush=True)
    for j, t in enumerate(ts):
        ph = np.interp(t, gc, pc)
        w = (gf >= t - SEARCH) & (gf <= t + SEARCH)
        if not w.any():
            continue
        d = np.abs(((pf[w] - ph + np.pi) % (2 * np.pi)) - np.pi)
        tf = gf[w][np.argmin(d)]
        uc, vc, okc = sample(rc, avc, t); uf, vf, okf = sample(rf, avf, tf)
        ok = okc & okf
        X.append(np.stack([uc, vc])); Y.append(np.stack([uf - uc, vf - vc]))
        MK.append(ok[None].astype(np.uint8))
        META.append(dict(Re=re_, t=float(t), t_fine=float(tf), dphi=float(d.min())))
        if j % 200 == 0:
            print(f'  Re{re_} {j}/{len(ts)}  t={t:.1f} -> {tf:.1f} (dphi={np.degrees(d.min()):.1f} deg)', flush=True)

X = torch.from_numpy(np.asarray(X, dtype=np.float32))
Y = torch.from_numpy(np.asarray(Y, dtype=np.float32))
MK = torch.from_numpy(np.asarray(MK, dtype=np.uint8))
print(f'\n{len(X)} pairs  X{tuple(X.shape)}  Y{tuple(Y.shape)}', flush=True)

# blocked split: contiguous chronological 80/10/10 blocks per Reynolds number
idx = np.arange(len(META)); tt = np.array([m['t'] for m in META]); rr = np.array([m['Re'] for m in META])
tr, va, te = [], [], []
for re_ in args.re:
    s = idx[rr == re_]; s = s[np.argsort(tt[s])]; n = len(s)
    tr += list(s[:int(0.8 * n)]); va += list(s[int(0.8 * n):int(0.9 * n)]); te += list(s[int(0.9 * n):])
sp = dict(train=[int(i) for i in tr], val=[int(i) for i in va], test=[int(i) for i in te])
json.dump(sp, open(OUT / 'blocked_split_ids.json', 'w'))
for k, ids in sp.items():
    ids = torch.tensor(ids)
    torch.save({'X': X[ids], 'y': Y[ids], 'mask': MK[ids], 'meta': [META[i] for i in ids.tolist()]}, OUT / f'{k}.pt')
tr_t = torch.tensor(sp['train'])
json.dump(dict(mean=X[tr_t].mean(dim=(0, 2, 3)).tolist(), std=X[tr_t].std(dim=(0, 2, 3)).tolist()),
          open(OUT / 'norm_stats.json', 'w'), indent=1)
json.dump(dict(mean=Y[tr_t].mean(dim=(0, 2, 3)).tolist(), std=Y[tr_t].std(dim=(0, 2, 3)).tolist()),
          open(OUT / 'target_norm.json', 'w'), indent=1)
print('split sizes', {k: len(v) for k, v in sp.items()})
print('input statistics', json.load(open(OUT / 'norm_stats.json')))
print('target statistics', json.load(open(OUT / 'target_norm.json')))
print('written to', OUT)
