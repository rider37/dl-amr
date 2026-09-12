#!/usr/bin/env python3
"""논문 Figure 3(중간후류 확대 위상평균 Ux, CL-max)·Figure 8(위상정합 순간장+메시).

6방법, 대칭 런. 전영역 위상평균(구 fig3a)은 미수록이라 제거함.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pyvista as pv
from fig_common import CACHE, CR, ROOT, GEOMS, OUT, STYLE, WINDOW, Xi, Yi, body_mask, body_patch, cases_for, jf, xi, yi
from scipy.spatial import cKDTree

jf.apply_style('JCP', font_pt=10.5)
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ['Fine', 'Coarse', r'$|\nabla\mathbf{U}|$ AMR', r'$|\omega|$ AMR', r'$Q$ AMR', 'DL-AMR',
         'static']
MESH_ROWS = ['Fine', 'Coarse', r'$|\nabla\mathbf{U}|$ AMR', r'$|\omega|$ AMR', r'$Q$ AMR',
         'DL-AMR', 'static']
# 8열 배치에서는 열 제목 폭이 부족하므로 축약형(캡션에 전개)
SHORT_T = {'Fine': 'Fine', 'Coarse': 'Coarse', 'DL-AMR': 'DL-AMR',
           r'$|\nabla\mathbf{U}|$ AMR': r'$|\nabla\mathbf{U}|$', r'$|\omega|$ AMR': r'$|\omega|$',
           r'$Q$ AMR': r'$Q$', 'static': 'static'}
NPEAK = 15
def ts_for(short):
    t0, t1 = WINDOW[short]
    return np.arange(t0, t1 + 1e-9, 1.0)
# 신체계 후보영역 x/D in [-2,25] 을 담는 창
xiw = np.linspace(-2.0, 25.0, 810)
yiw = np.linspace(-5.0, 5.0, 200)
XiW, YiW = np.meshgrid(xiw, yiw)
grid = pv.StructuredGrid(XiW, YiW, np.zeros_like(XiW))


def cl_series(case, TS):
    fs = sorted((CR / case / 'postProcessing/forceCoeffs').glob('*/coefficient*.dat'))
    best = max(fs, key=lambda f: f.stat().st_size)
    d = np.loadtxt(best, comments='#')
    d = d[np.argsort(d[:, 0])]
    return np.interp(TS, d[:, 0], d[:, 4])


def reader_for(case):
    r = pv.OpenFOAMReader(str(CR / case / 'open.foam'))
    try:
        r.enable_all_cell_arrays()
    except Exception:
        pass
    return r


def sample_ux(r, t):
    tv = np.array(r.time_values)
    r.set_active_time_value(float(tv[np.abs(tv - t).argmin()]))
    m = r.read()[0]
    mm = m.cell_data_to_point_data() if 'U' in m.cell_data else m
    s = grid.sample(mm)
    return np.asarray(s.point_data['U'])[:, 0].reshape(810, 200).T, m


_FP = None


def probe_uy(case, TS):
    """후류 프로브 (5,0) 의 U_y 를 TS 위에서 반환.
    위상 기준으로 C_L 을 쓰면 재격자화 잡음이 큰 변종에서 양력과 후류 구조의
    위상 결합이 느슨해져 평균이 뭉개진다. 후류 그림이므로 후류 신호를 쓴다."""
    global _FP
    fs = sorted((CR / case / 'postProcessing/probesW').glob('*/U'))
    if fs:
        rows = []
        for f in fs:
            for ln in f.read_text().split('\n'):
                if not ln or ln.startswith('#'):
                    continue
                q = ln.replace('(', ' ').replace(')', ' ').split()
                try:
                    rows.append([float(x) for x in q])
                except ValueError:
                    pass
        n = min(len(r) for r in rows)
        a = np.array([r[:n] for r in rows]); a = a[np.argsort(a[:, 0])]
        _, i = np.unique(a[:, 0], return_index=True); a = a[i]
        return np.interp(TS, a[:, 0], a[:, 1 + 3 * 1 + 1])   # probe idx 1 = (5,0)
    # finest 는 probesW 가 없어 필드에서 뽑아 둔 값을 쓴다
    if _FP is None:
        import json
        _FP = json.loads((CACHE / 'finest_probe.json').read_text())
    key = next(k for k in ('circ', 'sq', 'dia') if
               {'circ': 'circular', 'sq': 'square', 'dia': 'diamond'}[k] in case)
    d = _FP[key]
    return np.interp(TS, np.array(d['t']), np.array(d['uy']))


def peak_times(case, TS):
    """후류 프로브 U_y 의 위상이 0 에 가장 가까운 NPEAK 개 시각.
    방출 기본파 대역만 통과시켜 재격자화 잡음의 영향을 제거한다."""
    from scipy.signal import hilbert
    c = probe_uy(case, TS)
    c = c - c.mean()
    F = np.fft.rfft(c)
    k = int(np.argmax(np.abs(F[1:]))) + 1              # 방출 기본파
    band = np.zeros_like(F); lo, hi = max(1, k - 2), min(len(F), k + 3)
    band[lo:hi] = F[lo:hi]
    ph = np.angle(hilbert(np.fft.irfft(band, len(c))))
    return TS[np.argsort(np.abs(np.angle(np.exp(1j * ph))))[:NPEAK]]


def phase_avg_ux(case, TS):
    peaks = peak_times(case, TS)
    r = reader_for(case)
    acc = None
    for t in peaks:
        ux, _ = sample_ux(r, t)
        acc = ux if acc is None else acc + ux
    return acc / len(peaks)


# ---------------- 위상평균 필드 수집 (pickle 캐시) ----------------
import pickle
_CACHE = Path(__file__).parent / '.pa_cache_nc4x_v3.pkl'
PA = pickle.loads(_CACHE.read_bytes()) if _CACHE.exists() else {}
_new = 0
for short, g, glab in GEOMS:
    C = cases_for(g)
    for m in ORDER:
        if (short, m) in PA:
            continue
        print(f'phase-avg {short}/{m}', flush=True)
        PA[(short, m)] = phase_avg_ux(C[m]['case'], ts_for(short))
        _new += 1
if _new:
    _CACHE.write_bytes(pickle.dumps(PA))
print(f'PA cache: {len(PA)} entries ({_new} new)', flush=True)

# ---------------- Figure 3: zoom + fine contour (형상=행, 방법=열) ----------------
XZ = (5.0, 13.0); YZ = (-2.25, 2.25)  # 중간후류: 방법 간 격차가 가장 큰 구간
LEV = np.linspace(0.2, 1.1, 10)
# 방법=행, 형상=열 (fig4/fig8 과 동일 배치 -> 패널이 3배 커진다)
ORDER3 = [m for m in ORDER if m != 'Fine']   # fine 은 등고선으로 겹쳐 있어 행 제외
fig, axes = jf.new_figure_grid('JCP', len(ORDER3), 3, width='onehalf', aspect=1.02,
                               font_pt=10.5)
for c_i, (short, g, glab) in enumerate(GEOMS):
    fine = PA[(short, 'Fine')]
    for r_i, m in enumerate(ORDER3):
        ax = axes[r_i, c_i]
        im = ax.pcolormesh(XiW, YiW, PA[(short, m)], cmap='RdBu_r',
                           vmin=-0.4, vmax=1.4, shading='auto', rasterized=True)
        ax.contour(XiW, YiW, fine, levels=LEV[::2], colors='k', linewidths=0.5,
                   linestyles='--')
        ax.contour(XiW, YiW, PA[(short, m)], levels=LEV[::2], colors='k',
                   linewidths=0.8)
        ax.add_patch(body_patch(short))          # 물체 내부 보간값 가림 (fig8 과 동일 처리)
        ax.set_xlim(*XZ); ax.set_ylim(*YZ); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        if r_i == 0:
            ax.set_title(glab, fontsize=10.5)
        if c_i == 0:
            ax.set_ylabel(SHORT_T[m], fontsize=10.5, rotation=0,
                          ha='right', va='center', labelpad=4)
cb = fig.colorbar(im, ax=[ax for row in axes for ax in row], shrink=0.42,
                  pad=0.02, location='bottom', aspect=40)
cb.set_label(r'$U_x/U_\infty$', fontsize=11)
cb.ax.tick_params(labelsize=10)
fig.savefig(OUT / 'fig3.pdf', dpi=400)
plt.close(fig)
print('fig3.pdf saved', flush=True)

import os
if os.environ.get('ONLY') == '3':
    raise SystemExit(0)
# ---------------- Figure 6: 위상정합 순간 와도장(솔버 vorticity) + 실제 격자선 (방법=행, 형상=열) ----------------
from matplotlib.collections import LineCollection  # noqa: E402

XW = (-1.5, 18); YW = (-3, 3)
xw = np.linspace(*XW, 900); yw = np.linspace(*YW, 280)
Xw, Yw = np.meshgrid(xw, yw)
gridw = pv.StructuredGrid(Xw, Yw, np.zeros_like(Xw))

fig, axes = jf.new_figure_grid('JCP', len(MESH_ROWS), 3, width='onehalf', aspect=0.86,
                               font_pt=10.5)
for c_i, (short, g, glab) in enumerate(GEOMS):
    C = cases_for(g)
    for r_i, m in enumerate(MESH_ROWS):
        print(f'fig8 {short}/{m}', flush=True)
        case = C[m]['case']
        TSg = ts_for(short)
        tpk = float(peak_times(case, TSg)[0])
        r = reader_for(case)
        tv = np.array(r.time_values)
        r.set_active_time_value(float(tv[np.abs(tv - tpk).argmin()]))
        msh = r.read()[0]
        # 배경 = 솔버 와도 ω_z (postProcess -func vorticity 로 해당 시각에 미리 계산; VTK 점 기울기는 레벨 경계 얼룩)
        assert 'vorticity' in msh.cell_data, f'{case}: run postProcess -func vorticity -time {tpk} first'
        mm = msh.cell_data_to_point_data()
        s2 = gridw.sample(mm)
        # StructuredGrid 는 F-order 로 저장된다 -> (nx, ny) 로 편 뒤 전치
        wz = np.asarray(s2.point_data['vorticity'])[:, 2].reshape(len(xw), len(yw)).T
        # z 중앙면의 실제 격자 모서리
        zc = float(np.asarray(msh.points)[:, 2].mean())
        ed = msh.slice(normal='z', origin=(0.0, 0.0, zc)).extract_all_edges()
        ep = np.asarray(ed.points)[:, :2]
        el = np.asarray(ed.lines).reshape(-1, 3)[:, 1:]
        segs = ep[el]
        segs = segs[((segs[:, :, 0] >= XW[0] - 0.1).all(1)
                     & (segs[:, :, 0] <= XW[1] + 0.1).all(1)
                     & (np.abs(segs[:, :, 1]) <= YW[1] + 0.1).all(1))]
        ax = axes[r_i, c_i]
        im = ax.pcolormesh(Xw, Yw, wz, cmap='RdBu_r', vmin=-3, vmax=3,
                           shading='auto', rasterized=True)
        ax.add_collection(LineCollection(segs, colors='0.25', linewidths=0.07, alpha=0.40, rasterized=True))
        ax.add_patch(body_patch(short))
        ax.set_xlim(*XW); ax.set_ylim(*YW); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        if r_i == 0:
            ax.set_title(glab, fontsize=10)
        if c_i == 0:
            ax.set_ylabel(SHORT_T[m], fontsize=9.5, rotation=0, ha='right', va='center',
                          labelpad=4)
cb = fig.colorbar(im, ax=[ax for row in axes for ax in row], shrink=0.45,
                  pad=0.02, location='bottom', aspect=45)
cb.set_label(r'$\omega_z D/U_\infty$', fontsize=9.5)
cb.ax.tick_params(labelsize=9.5)
fig.savefig(OUT / 'fig6.pdf', dpi=600)
plt.close(fig)
print('fig6.pdf saved', flush=True)
