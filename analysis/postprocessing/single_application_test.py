#!/usr/bin/env python3
"""WP1-B: 리뷰어 1의 동일예산 정제유용성 시험을 **현재 49k 설정**으로 재수행.

원판(wp1_mask_experiment.py) 대비 바뀐 것
  기저격자   circular_Re200_base78k (78,200, hex8)  ->  NC2_coarse (16,944, hexRef4)
  모델       3채널 uvp, 64x224, x in [2,39]          ->  2채널 uv, 432x160, x in [-2,25]
  예산       78200 + 7m                              ->  16944 + 3m,  목표 49,884 -> m=10,980
  마스크     sigma gradU vort dt wake oracle rand     ->  + Q (논문 기준선)
  체크포인트 t=300 부근                                ->  평가창 안 (원형 t=200)

사용:
  wp1b_mask_experiment.py --stage prep   --geom circular --t0 200
  wp1b_mask_experiment.py --stage cases  --geom circular --t0 200
"""
from __future__ import annotations
import argparse, json, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = _roots.ROOT
CR = ROOT / 'sim/runs/blockMesh_cases'
RUN = ROOT / 'ml/runs/nc_delta_hetero_uni'
DAT = ROOT / 'ml/data/processed/nc_delta_uv_uni'
OFSRC = 'set +u; source /usr/lib/openfoam/openfoam2312/etc/bashrc; set -eo pipefail'

GEOM = {
    'circular': dict(base='NC2_coarse', fine='grid_convergence/circular_Re200_finer'),
    'square':   dict(base='NCsq_coarse_full', fine='grid_convergence/square_Re150_finer'),
    'diamond':  dict(base='NCdia_coarse_full', fine='grid_convergence/diamond_Re150_finer'),
}
BASE_CELLS, TARGET_CELLS = 16944, 49884
# 논문과 동일하게 최대 레벨 2까지 정제한다. hexRef4 를 두 번 적용하면 부모 1개가 16개가
# 되므로 부모당 +15 셀. 창 안 후보가 9,236개뿐이라 1레벨(부모당 +3)로는 목표에 못 미친다.
M_PARENTS = (TARGET_CELLS - BASE_CELLS) // 15
K_ADVANCE = 10.0
MASKS = ('sigma', 'gradU', 'vort', 'Q', 'dt', 'wake', 'oracle', 'rand')
NX, NY = 432, 160
XMIN, XMAX, YMIN, YMAX = -2.0, 25.0, -5.0, 5.0        # mlInferDict 의 region 과 동일


def of_read_scalar(p: Path) -> np.ndarray:
    t = p.read_text()
    m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(', t)
    s = t.index('(', m.end() - 1) + 1
    v = np.fromstring(t[s:t.index(')', s)], sep='\n')
    assert len(v) == int(m.group(1))
    return v


def of_read_vector(p: Path, ncomp: int = 3) -> np.ndarray:
    t = p.read_text()
    m = re.search(r'internalField\s+nonuniform\s+List<(?:vector|tensor)>\s*\n(\d+)\s*\n\(\n', t)
    n = int(m.group(1))
    body = ' '.join(t[m.end():].splitlines()[:n])
    v = np.fromstring(body.replace('(', ' ').replace(')', ' '), sep=' ')
    assert len(v) == n * ncomp, (len(v), n, ncomp)
    return v.reshape(n, ncomp)


def write_flag(p: Path, flags: np.ndarray) -> None:
    vals = '\n'.join('1' if f else '0' for f in flags)
    p.write_text('FoamFile\n{\n    version 2.0;\n    format ascii;\n'
                 '    class volScalarField;\n    location "%s";\n    object refineFlag;\n}\n\n'
                 'dimensions      [0 0 0 0 0 0 0];\n\ninternalField   nonuniform List<scalar>\n'
                 '%d\n(\n%s\n)\n;\n\nboundaryField\n{\n    ".*"\n    {\n        type zeroGradient;\n'
                 '    }\n    #includeEtc "caseDicts/setConstraintTypes"\n}\n'
                 % (p.parent.name, len(flags), vals))


def prep(geom: str, t0: str) -> None:
    base = CR / GEOM[geom]['base']
    cmd = (f'{OFSRC}; cd {base} && postProcess '
           f'-funcs "(writeCellCentres grad(U) vorticity Q)" -time {t0} > log.wp1bPost 2>&1')
    subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
    print('postProcess 완료:', base.name, t0)


def top_m(ind: np.ndarray, elig: np.ndarray, m: int) -> np.ndarray:
    ind = np.where(elig, ind, -np.inf)
    f = np.zeros(len(ind), dtype=bool)
    f[np.argsort(-ind)[:m]] = True
    return f


def sigma_indicator(base: Path, t0: str, cx, cy) -> np.ndarray:
    import torch, pyvista as pv
    sys.path.insert(0, str(ROOT))
    from ml.src.models.hetero_model import HeteroDeltaFullRes
    ns = json.loads((DAT / 'norm_stats.json').read_text())
    xg = np.linspace(XMIN, XMAX, NX); yg = np.linspace(YMIN, YMAX, NY)
    Xg, Yg = np.meshgrid(xg, yg)
    r = pv.OpenFOAMReader(str(base / 'open.foam')); r.set_active_time_value(float(t0))
    msh = r.read()[0]
    msh = msh.cell_data_to_point_data() if 'U' in msh.cell_data else msh
    s = pv.StructuredGrid(Xg, Yg, np.zeros_like(Xg)).sample(msh)
    U = np.asarray(s.point_data['U'])
    x = np.stack([U[:, 0].reshape(NX, NY).T, U[:, 1].reshape(NX, NY).T]).astype(np.float32)
    for c in range(2):
        x[c] = (x[c] - ns['mean'][c]) / (ns['std'][c] + 1e-12)
    mdl = HeteroDeltaFullRes(in_ch=2, out_ch_mean=2, out_ch_logvar=1, base=32,
                             logvar_min=-12., logvar_max=6.)
    mdl.load_state_dict(torch.load(RUN / 'best.pt', map_location='cpu', weights_only=True))
    mdl.eval()
    with torch.no_grad():
        _, lv = mdl(torch.from_numpy(x).unsqueeze(0))
    lv = lv.numpy()[0, 0]
    dx = (XMAX - XMIN) / (NX - 1); dy = (YMAX - YMIN) / (NY - 1)
    fu = np.clip((cx - XMIN) / dx, 0, NX - 1); fv = np.clip((cy - YMIN) / dy, 0, NY - 1)
    i0 = np.clip(fu.astype(int), 0, NX - 2); j0 = np.clip(fv.astype(int), 0, NY - 2)
    a, b = fu - i0, fv - j0
    lvc = ((1 - a) * (1 - b) * lv[j0, i0] + a * (1 - b) * lv[j0, i0 + 1]
           + (1 - a) * b * lv[j0 + 1, i0] + a * b * lv[j0 + 1, i0 + 1])
    return np.sqrt(np.exp(lvc))


def oracle_indicator(geom: str, base: Path, t0: str, cx, cy) -> np.ndarray:
    import pyvista as pv
    r = pv.OpenFOAMReader(str(CR / GEOM[geom]['fine'] / 'open.foam'))
    r.set_active_time_value(float(t0))
    msh = r.read()[0]
    msh = msh.cell_data_to_point_data() if 'U' in msh.cell_data else msh
    cloud = pv.PolyData(np.column_stack([cx, cy, np.zeros_like(cx)]))
    uf = np.asarray(cloud.sample(msh).point_data['U'])[:, :2]
    ub = of_read_vector(base / t0 / 'U')[:, :2]
    return np.linalg.norm(uf - ub, axis=1)          # 학습 타깃과 같은 2채널 정의


def build_masks(geom: str, t0: str, mo: dict | None = None) -> dict:
    base = CR / GEOM[geom]['base']
    cx = of_read_scalar(base / t0 / 'Cx'); cy = of_read_scalar(base / t0 / 'Cy')
    elig = (cx >= XMIN) & (cx <= XMAX) & (np.abs(cy) <= abs(YMIN)) & (np.hypot(cx, cy) > 0.75)
    mo = mo or {}
    m = M_PARENTS
    print(f'후보 셀 {elig.sum():,} / 전체 {len(cx):,},  기본 m={m:,}, 보정 {mo}')
    M = {}
    M['sigma'] = top_m(sigma_indicator(base, t0, cx, cy), elig, m)
    g = of_read_vector(base / t0 / 'grad(U)', ncomp=9)
    M['gradU'] = top_m(np.sqrt((g ** 2).sum(1)), elig, m)
    w = of_read_vector(base / t0 / 'vorticity')
    M['vort'] = top_m(np.abs(w[:, 2]), elig, m)
    M['Q'] = top_m(of_read_scalar(base / t0 / 'Q'), elig, m)
    u0 = of_read_vector(base / t0 / 'U')[:, :2]
    u1 = of_read_vector(base / str(int(float(t0)) - 1) / 'U')[:, :2]
    M['dt'] = top_m(np.linalg.norm(u0 - u1, axis=1), elig, m)
    for yb in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0):      # m 을 담을 수 있을 때까지 폭을 넓힌다
        inbox = (cx >= 2) & (np.abs(cy) <= yb) & elig
        if inbox.sum() >= m: break
    pool = np.nonzero(inbox)[0]; f = np.zeros(len(cx), dtype=bool)
    f[pool[np.argsort(cx[pool])][:m]] = True
    M['wake'] = f
    M['oracle'] = top_m(oracle_indicator(geom, base, t0, cx, cy), elig, m)
    from scipy.spatial import cKDTree
    rng = np.random.RandomState(42); pool = np.nonzero(elig)[0]
    tree = cKDTree(np.column_stack([cx, cy])); f = np.zeros(len(cx), dtype=bool); n = 0
    guard = 0
    while n < m and guard < 200000:
        guard += 1
        s = pool[rng.randint(len(pool))]
        _, idx = tree.query([cx[s], cy[s]], k=12)
        for i in np.atleast_1d(idx):
            if n >= m: break
            if elig[i] and not f[i]: f[i] = True; n += 1
    assert n == m, f'rand 마스크가 {n}/{m} 에서 멈춤 (후보 {int(elig.sum())})'
    M['rand'] = f
    return M


def build_masks_with(geom: str, t0: str, mo: dict) -> dict:
    """마스크별 m 을 달리 적용해야 하므로, 필요한 마스크마다 build_masks 를 다시 부른다."""
    out = {}
    for name, mm in mo.items():
        global M_PARENTS
        keep, M_PARENTS = M_PARENTS, mm
        try:
            out[name] = build_masks(geom, t0)[name]
        finally:
            M_PARENTS = keep
    return out


DYN = """FoamFile
{ version 2.0; format ascii; class dictionary; location "constant"; object dynamicMeshDict; }

dynamicFvMesh   dynamicRefine2DFvMesh;

dynamicRefine2DFvMeshCoeffs
{
    refineInterval  10;
    field           refineFlag;
    lowerRefineLevel 0.9;
    upperRefineLevel 1.1;
    unrefineLevel   0.8;
    nBufferLayers   0;
    maxRefinement   2;
    maxCells        200000;
    correctFluxes ( (phi none) (phi_0 none) (phi_0_0 none) (ghf none) );
    dumpLevel       true;
    protectMode     none;
}
"""
READ_FO = """
    readFlagFO
    {
        type            readFields;
        libs            (fieldFunctionObjects);
        fields          (refineFlag);
        readOnStart     true;
        executeControl  none;
        writeControl    none;
    }
"""


def make_case(geom: str, t0: str, name: str, flags: np.ndarray) -> Path:
    base = CR / GEOM[geom]['base']
    dst = CR / f'wp1b_{geom}_t{t0}_{name}'
    if dst.exists(): shutil.rmtree(dst)
    dst.mkdir()
    shutil.copytree(base / 'system', dst / 'system')
    # 동적격자는 위상 변경 뒤 플럭스 보정에 pcorr 해법이 필요하다.
    # 정적 기저 케이스의 fvSolution 에는 없으므로 논문 AMR 케이스에서 그 항목만 가져온다.
    fvs = dst / 'system' / 'fvSolution'; t = fvs.read_text()
    if 'pcorr' not in t:
        blk = ('\n    pcorr\n    {\n        solver          PCG;\n'
               '        preconditioner  DIC;\n        tolerance       1e-06;\n'
               '        relTol          0;\n    }\n'
               '    pcorrFinal\n    {\n        $pcorr;\n        relTol          0;\n    }\n')
        i = t.index('{', t.index('solvers')) + 1
        fvs.write_text(t[:i] + blk + t[i:])
    (dst / 'constant').mkdir()
    for it in (base / 'constant').iterdir():
        if it.name == 'polyMesh': continue
        (shutil.copytree if it.is_dir() else shutil.copy2)(it, dst / 'constant' / it.name)
    (dst / 'constant' / 'dynamicMeshDict').write_text(DYN)
    shutil.copytree(base / t0, dst / t0)
    for f in (dst / t0).iterdir():
        if 'Mean' in f.name or 'Prime' in f.name or f.name in ('Cx', 'Cy', 'Cz', 'C',
                'grad(U)', 'vorticity', 'Q'): f.unlink()
    write_flag(dst / t0 / 'refineFlag', flags)
    cd = dst / 'system' / 'controlDict'; t = cd.read_text()
    if 'libhexRef4' not in t:                      # dynamicRefine2DFvMesh 를 쓰려면 필요
        t = re.sub(r'(^application\s+\S+;)', r'\1\n\nlibs            ("libhexRef4.so");',
                   t, count=1, flags=re.M)
    t = re.sub(r'^startTime\s+[0-9.]+;', f'startTime       {t0};', t, flags=re.M)
    t = re.sub(r'^endTime\s+[0-9.]+;', f'endTime         {float(t0)+K_ADVANCE:g};', t, flags=re.M)
    t = re.sub(r'^writeInterval\s+[0-9.]+;', 'writeInterval   1;', t, flags=re.M)
    i = t.rfind('\n}\n'); t = t[:i] + READ_FO + t[i:]
    cd.write_text(t)
    (dst / 'Allrun').write_text('#!/usr/bin/env bash\ncd "$(dirname "$0")"\n' + OFSRC +
        '\nexport FOAM_USER_APPBIN=$FOAM_USER_APPBIN\n'
        'export PATH=$FOAM_USER_APPBIN:$PATH\nblockMesh > log.blockMesh 2>&1\n'
        'pimpleFoam > log.pimpleFoam 2>&1\n')
    (dst / 'Allrun').chmod(0o755)
    (dst / 'open.foam').touch()
    return dst


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['prep', 'cases'])
    ap.add_argument('--geom', default='circular'); ap.add_argument('--t0', default='200')
    ap.add_argument('--masks', nargs='+', default=list(MASKS))
    ap.add_argument('--m', type=int, default=0, help='이 마스크들에 쓸 부모 셀 수(0=기본)')
    ap.add_argument('--endtime', type=float, default=0.0, help='0 이면 t0+K')
    a = ap.parse_args()
    if a.m: M_PARENTS = a.m
    if a.endtime: K_ADVANCE = a.endtime
    if a.stage == 'prep':
        prep(a.geom, a.t0)
    else:
        M = build_masks(a.geom, a.t0)
        for n in a.masks:
            d = make_case(a.geom, a.t0, n, M[n])
            print(f'{d.name}: m={int(M[n].sum()):,}  예상 셀 {BASE_CELLS + 15*int(M[n].sum()):,}')
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
