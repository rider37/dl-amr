#!/usr/bin/env python3
"""R2 C7 보완 분석: (1) 시간에 따른 방출 주파수 St(t) — 양력 신호의 복소 복조(2주기 이동평균) 순시 주파수,
(2) 후류 위상 오차 — 중심선 프로브 (3,0),(5,0),(8,0) 의 U_y 기본파 위상을 양력 기본파 기준으로 잰 위상 지연 Δφ(x),
fine 대비 오차와 그로부터 얻는 와류가로 파장 λ/D. 전 변종 동일 프로토콜(단위 간격 표본, 자기 St_f 조화적합).
-> analysis/output/eval/c7/c7_dynamics.json"""
import sys, json, glob
from pathlib import Path
sys.path.insert(0, str(_roots.ROOT/'analysis'/'figscripts'))
import numpy as np
from fig_common import CR, ROOT, CASE, WINDOW
WD=json.load(open(ROOT/'analysis/output/eval/wake_dynamics.json'))
KEYS=['fine','coarse','gradU','vort','Q','sigma','static']; XS=[3.,5.,8.]
FP3=json.load(open(ROOT/'analysis/output/eval/finest_probe3.json')) if (ROOT/'analysis/output/eval/finest_probe3.json').exists() else None
FP1=json.load(open(ROOT/'analysis/output/eval/finest_probe.json'))
def load_cl(case):
    rows=[]
    for f in sorted(glob.glob(str(CR/case/'postProcessing/forceCoeffs/*/coefficient*.dat'))):
        d=np.loadtxt(f,comments='#'); rows.append(d[:,[0,4]])
    a=np.vstack(rows); a=a[np.argsort(a[:,0])]; _,i=np.unique(a[:,0],return_index=True); return a[i]
def load_probes(case):
    fs=sorted((CR/case/'postProcessing/probesW').glob('*/U')); rows=[]
    for f in fs:
        for ln in f.read_text().split('\n'):
            if not ln or ln.startswith('#'): continue
            q=ln.replace('(',' ').replace(')',' ').split()
            try: rows.append([float(x) for x in q])
            except ValueError: pass
    n=min(len(r) for r in rows); a=np.array([r[:n] for r in rows]); a=a[np.argsort(a[:,0])]; _,i=np.unique(a[:,0],return_index=True); a=a[i]
    return a[:,0], np.stack([a[:,1+3*p+1] for p in range(3)],1)   # U_y at probes 0,1,2
NPER=4.0  # 이동 창 길이(주기 수)
def st_series(t,x,Stf,lo,hi):
    """이동 창 조화적합(평균+2고조파, 자기 St_f) 기본파 위상의 시간 기울기로 얻는 순시 St(t). 각 신호의 원 표본 간격 사용."""
    dt=float(np.median(np.diff(t))); dtu=max(min(dt,0.1),0.005) if dt<0.5 else dt
    g=np.arange(lo,hi+1e-9,dtu); y=np.interp(g,t,x); Tw=NPER/Stf; step=0.25
    cen=np.arange(lo+Tw/2,hi-Tw/2+1e-9,step); ph=[]
    for c in cen:
        m=(g>=c-Tw/2)&(g<=c+Tw/2); tt=g[m]-c; w=2*np.pi*Stf*g[m]   # 절대 시간: 정확히 St_f 이면 위상 일정
        A=np.stack([np.ones_like(tt),np.cos(w),np.sin(w),np.cos(2*w),np.sin(2*w)],1); cf=np.linalg.lstsq(A,y[m],rcond=None)[0]; ph.append(np.arctan2(cf[2],cf[1]))
    ph=np.unwrap(np.array(ph)); st=Stf+np.gradient(ph,cen)/(2*np.pi)
    n=max(1,int(round((1.0/Stf)/step))); st=np.convolve(st,np.ones(n)/n,mode='same')   # 1주기 평활
    k=slice(n//2,len(cen)-n//2); return cen[k],st[k]
def harm_phase(TS,x,Stf):
    w=2*np.pi*Stf*TS; A=np.stack([np.ones_like(TS),np.cos(w),np.sin(w),np.cos(2*w),np.sin(2*w)],1)
    c=np.linalg.lstsq(A,x,rcond=None)[0]; return float(np.arctan2(c[2],c[1])), float(np.hypot(c[1],c[2]))  # x≈A cos(wt-φ)
wrap=lambda d: (d+180.)%360.-180.
out={}
for g,short in (('circ','circular'),('sq','square'),('dia','diamond')):
    lo,hi=WINDOW[short]; TS=np.arange(lo,hi+1e-9,1.0); res={}
    for k in KEYS:
        case=CASE[short][k]; Stf=WD[g][k]['St_f']; cl=load_cl(case)
        gt,st=st_series(cl[:,0],cl[:,1],Stf,lo,hi)
        clu=np.interp(TS,cl[:,0],cl[:,1]); phc,_=harm_phase(TS,clu,Stf)
        if k=='fine':
            uy=np.array(FP3[g]['uy']) if FP3 else None; tt=np.array(FP3[g]['t']) if FP3 else None
            tp5,up5=np.array(FP1[g]['t']),np.array(FP1[g]['uy'])
        else:
            tt,uy=load_probes(case); tp5,up5=tt,uy[:,1]
        gp,stp=st_series(tp5,up5,Stf,lo,hi)
        lag=[];amp=[]
        for p in range(3):
            if uy is None: lag.append(np.nan); amp.append(np.nan); continue
            up=np.interp(TS,tt,uy[:,p]); php,ap=harm_phase(TS,up,Stf); lag.append(float(wrap(np.degrees(phc-php)))); amp.append(ap)  # 양력 대비 프로브 지연(deg, +=lag)
        res[k]=dict(case=case,St_f=Stf,dt_cl=float(np.median(np.diff(cl[:,0]))),dt_probe=float(np.median(np.diff(tp5))),St_t=dict(t=gt[::max(1,int(0.5/np.median(np.diff(gt))))].round(3).tolist(),St=st[::max(1,int(0.5/np.median(np.diff(gt))))].round(5).tolist()),
                    St_mean=float(st.mean()),St_std=float(st.std()),St_min=float(st.min()),St_max=float(st.max()),
                    Stp_t=dict(t=gp[::max(1,int(0.5/np.median(np.diff(gp))))].round(3).tolist(),St=stp[::max(1,int(0.5/np.median(np.diff(gp))))].round(5).tolist()),Stp_mean=float(stp.mean()),Stp_std=float(stp.std()),Stp_min=float(stp.min()),Stp_max=float(stp.max()),lag_deg=lag,amp=amp)
        print(g,k,f'St_f {Stf:.4f} | lift St(t) std {st.std():.4f} [{st.min():.4f},{st.max():.4f}] | probe St(t) std {stp.std():.4f} [{stp.min():.4f},{stp.max():.4f}] lag {np.round(lag,1)}',flush=True)
    # fine 대비 위상 오차와 파장
    f=res['fine']
    for k in KEYS:
        r=res[k]; r['lag_err_deg']=[float(wrap(a-b)) if np.isfinite(a) and np.isfinite(b) else None for a,b in zip(r['lag_deg'],f['lag_deg'])]
    out[g]=dict(window=[lo,hi],**res)
json.dump(out,open(ROOT/'analysis/output/eval/c7/c7_dynamics.json','w'),indent=1); print('WROTE c7_dynamics.json')
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
