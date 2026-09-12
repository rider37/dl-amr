#!/usr/bin/env python3
"""전 변종(21 케이스) 중심선 x/D=3..8 (1D 간격) U_y 시계열을 저장 필드에서 추출 (Δt=1.0, 평가창).
-> analysis/output/eval/probe6_all.json  { geom: { key: {case,t,x,uy[nt][6]} } }"""
import json, sys, numpy as np, pyvista as pv
sys.path.insert(0, str(_roots.ROOT/'analysis'/'figscripts'))
from fig_common import CR, CASE, WINDOW
XS=[3.,4.,5.,6.,7.,8.]; pts=pv.PolyData(np.array([[x,0,0] for x in XS]))
out={}
for g,short in (('circ','circular'),('sq','square'),('dia','diamond')):
    lo,hi=WINDOW[short]; out[g]={}
    for k in ('fine','coarse','gradU','vort','Q','sigma','static'):
        case=CASE[short][k]; r=pv.OpenFOAMReader(str(CR/case/'open.foam'))
        try: r.enable_all_cell_arrays()
        except Exception: pass
        tv=np.array(r.time_values); ts=tv[(tv>=lo-1e-6)&(tv<=hi+1e-6)]
        U=[]
        for t in ts:
            r.set_active_time_value(float(t)); m=r.read()[0]; mm=m.cell_data_to_point_data() if 'U' in m.cell_data else m
            U.append(np.asarray(pts.sample(mm).point_data['U'])[:,1])
        U=np.array(U); out[g][k]=dict(case=case,t=ts.tolist(),x=XS,uy=U.tolist()); print(g,k,'n',len(ts),flush=True)
json.dump(out,open(str(_roots.ROOT/'analysis/output/eval/probe6_all.json'),'w')); print('WROTE probe6_all.json')
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
