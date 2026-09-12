#!/usr/bin/env python3
"""gap 모델 폐루프 런의 오차질량 피복률 — 실제 관측 cellLevel 기준.
nc4x_alloc_analysis.py 와 동일 정의: Σ E·[f2>0.5] / Σ E,  f2 = 레벨2 시간분율."""
import pickle,json
import numpy as np, pyvista as pv
from pathlib import Path
from scipy.spatial import cKDTree
R=_roots.ROOT; CR=R/'sim/runs/blockMesh_cases'
al=pickle.load(open(R/'analysis/output/eval/nc4x_alloc.pkl','rb')); xl,yl=al['xl'],al['yl']
Ax,Ay=np.meshgrid(xl,yl); AP=np.stack([Ax.ravel(),Ay.ravel()],1); H,W=Ax.shape
import sys
ONLY=sys.argv[1:] or ['circ','sq','dia']
JOBS={k:v for k,v in {
      'circ':(['NC4x_sigma','NC4xG_sigma'],np.arange(165,245.1,4.0)),
      'sq'  :(['NC4x_sq_sigma','NC4xG_sq_sigma'],np.arange(237.5,346.1,4.0)),
      'dia' :(['NC4x_dia_sigma','NC4xG_dia_sigma'],np.arange(170,256.1,4.0))}.items() if k in ONLY}
out={}
for sh,(cases,TS) in JOBS.items():
    E=al['R'][sh]['E']; out[sh]={}
    for c in cases:
        f=CR/c/'open.foam'; f.touch(); r=pv.OpenFOAMReader(str(f)); av=np.array(r.time_values)
        acc=np.zeros((H,W)); n=0
        for t in TS:
            r.set_active_time_value(float(av[np.abs(av-t).argmin()])); m=r.read()[0]
            cc=np.asarray(m.cell_centers().points)[:,:2]
            lv=np.asarray(m.cell_data['cellLevel'])
            _,i=cKDTree(cc).query(AP)
            acc+=(lv[i].reshape(H,W)>=1.5); n+=1
        f2=acc/n
        out[sh][c]=dict(cov=float((E*(f2>0.5)).sum()/E.sum()), covw=float((E*f2).sum()/E.sum()))
        print(f'  {sh:4s} {c:18s} 피복 {100*out[sh][c]["cov"]:5.1f}%  (시간가중 {100*out[sh][c]["covw"]:5.1f}%)',flush=True)
json.dump(out,open(R/('analysis/output/eval/gap_coverage_%s.json'%'_'.join(ONLY)),'w'),indent=1)
print('저장 완료')
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
