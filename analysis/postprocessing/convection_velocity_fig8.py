#!/usr/bin/env python3
"""와류 대류 속도 U_c(x): 중심선 인접 지점(1D 간격)의 U_y 기본파 위상차 → 시간 지연 → U_c = Δx/τ (Zhou & Antonia 1992 방식의 위상 기반 판).
전 변종 동일 프로토콜(저장 필드, Δt=1.0, 자기 St_f 조화적합). -> analysis/output/eval/c7/c7_uc.json"""
import json, sys, numpy as np
sys.path.insert(0, str(_roots.ROOT/'analysis'/'figscripts'))
from fig_common import ROOT
P=json.load(open(ROOT/'analysis/output/eval/probe6_all.json')); WD=json.load(open(ROOT/'analysis/output/eval/wake_dynamics.json'))
def harm(TS,x,Stf):
    w=2*np.pi*Stf*TS; A=np.stack([np.ones_like(TS),np.cos(w),np.sin(w),np.cos(2*w),np.sin(2*w)],1); c=np.linalg.lstsq(A,x,rcond=None)[0]
    return float(np.arctan2(c[2],c[1])), float(np.hypot(c[1],c[2]))
out={}
for g in P:
    out[g]={}
    for k,d in P[g].items():
        TS=np.array(d['t']); uy=np.array(d['uy']); xs=d['x']; Stf=WD[g][k]['St_f']
        ph=[];amp=[]
        for j in range(len(xs)): p,a=harm(TS,uy[:,j],Stf); ph.append(p); amp.append(a)
        lag=[]; uc=[]
        for j in range(len(xs)-1):
            dphi=(ph[j+1]-ph[j])%(2*np.pi)          # x≈A cos(wt−φ): 하류 지점 φ_down=φ_up+wτ → 지연 위상 (0,2π); 1D 간격은 반파장(≈2.2D) 안이라 유일
            tau=dphi/(2*np.pi*Stf); lag.append(float(tau)); uc.append(float((xs[j+1]-xs[j])/tau))
        out[g][k]=dict(case=d['case'],St_f=Stf,x=xs,xmid=[(xs[j]+xs[j+1])/2 for j in range(len(xs)-1)],phase=ph,amp=amp,tau=lag,Uc=uc)
        print(g,k,'U_c/U∞ at x=3.5..7.5:',np.round(uc,3),flush=True)
json.dump(out,open(ROOT/'analysis/output/eval/c7/c7_uc.json','w'),indent=1); print('WROTE c7_uc.json')
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
