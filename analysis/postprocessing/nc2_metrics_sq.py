#!/usr/bin/env python3
"""A안 원형 평가 — 정착창 [165,245], 조화적합 평균, finest 272k 대비."""
import glob, sys, json
from pathlib import Path
import numpy as np, pyvista as pv
CR=_roots.CR
import os
T0=float(os.environ.get('W0',165)); T1=float(os.environ.get('W1',245))
TS=np.arange(T0,T1+1e-9,1.0); DX=0.05
xl=np.arange(-1.5,20.0001,DX); yl=np.arange(-5,5.0001,DX)
Xl,Yl=np.meshgrid(xl,yl); gL=pv.StructuredGrid(Xl,Yl,np.zeros_like(Xl)); WL,HL=len(xl),len(yl)
def freq(d,lo=None,hi=None):
    lo=T0 if lo is None else lo; hi=T1 if hi is None else hi
    fs=sorted(glob.glob(str(CR/d/'postProcessing/forceCoeffs/*/coefficient.dat')))
    a=np.vstack([np.loadtxt(x,comments='#') for x in fs]); a=a[np.argsort(a[:,0])]
    _,i=np.unique(a[:,0],return_index=True); a=a[i]
    m=(a[:,0]>=lo)&(a[:,0]<=hi); t=a[m,0]; y=a[m,4]-a[m,4].mean()
    g=np.arange(t[0],t[-1],0.005); yg=np.interp(g,t,y)
    F=np.fft.rfft(yg*np.hanning(len(yg))); fr=np.fft.rfftfreq(len(yg),0.005)
    b=np.where((fr>0.05)&(fr<1.0))[0]; k=b[np.argmax(np.abs(F[b]))]
    a_,b_,c_=np.log(np.abs(F[k-1])),np.log(np.abs(F[k])),np.log(np.abs(F[k+1]))
    return float(fr[k]+0.5*(a_-c_)/(a_-2*b_+c_)*(fr[1]-fr[0]))
def forces(d,lo=None,hi=None):
    lo=T0 if lo is None else lo; hi=T1 if hi is None else hi
    fs=sorted(glob.glob(str(CR/d/'postProcessing/forceCoeffs/*/coefficient.dat')))
    a=np.vstack([np.loadtxt(x,comments='#') for x in fs]); a=a[np.argsort(a[:,0])]
    _,i=np.unique(a[:,0],return_index=True); a=a[i]
    m=(a[:,0]>=lo)&(a[:,0]<=hi)
    return float(a[m,1].mean()), float(a[m,1].std()), float(a[m,4].std())
def fit(d,f):
    p=CR/d/'open.foam'
    if not p.exists(): p.touch()
    r=pv.OpenFOAMReader(str(p)); av=np.array(r.time_values)
    U=[];V=[];N=[]
    for t in TS:
        r.set_active_time_value(float(av[np.abs(av-t).argmin()])); m=r.read()[0]
        N.append(m.n_cells)
        mm=m.cell_data_to_point_data() if 'U' in m.cell_data else m
        s=gL.sample(mm); u=np.asarray(s.point_data['U'])
        U.append(u[:,0].reshape(WL,HL).T); V.append(u[:,1].reshape(WL,HL).T)
    U=np.array(U,dtype=np.float64); V=np.array(V,dtype=np.float64)
    M=[np.ones_like(TS)]
    for n in (1,2): M+=[np.cos(2*np.pi*n*f*TS), np.sin(2*np.pi*n*f*TS)]
    M=np.array(M).T
    cu=np.linalg.lstsq(M,U.reshape(len(TS),-1),rcond=None)[0].reshape(5,WL,HL)
    cv=np.linalg.lstsq(M,V.reshape(len(TS),-1),rcond=None)[0].reshape(5,WL,HL)
    kk=0.5*((cu[1]**2+cu[2]**2+cu[3]**2+cu[4]**2)/2+(cv[1]**2+cv[2]**2+cv[3]**2+cv[4]**2)/2)
    uv=(cu[1]*cv[1]+cu[2]*cv[2]+cu[3]*cv[3]+cu[4]*cv[4])/2
    wz=np.gradient(cv[0],DX,axis=1)-np.gradient(cu[0],DX,axis=0)
    return dict(n=float(np.mean(N)),Ux=cu[0],Uy=cv[0],Wz=wz,k=kk,uv=uv)
def L2(a,b): return float(np.sqrt(((a-b)**2).mean()))
REF='grid_convergence/square_Re150_finer'
cases=sys.argv[1:] if len(sys.argv)>1 else ['NC2_coarse','NC2_gradU']
Sf=freq(REF); F=fit(REF,Sf)
print(f'기준 finest {F["n"]:,.0f}셀, St={Sf:.4f}, 창 [{T0:.0f},{T1:.0f}] {len(TS)}점 = {(T1-T0)*Sf:.1f}주기')
cdF,cdrF,clrF=forces(REF)
print(f"\n{'런':12s}{'셀':>9s}{'L2(Ux)':>9s}{'L2(Uy)':>9s}{'L2(wz)':>9s}{'L2(k)':>9s}{'L2(uv)':>9s}{'ΔSt%':>8s}{'Cd_rms':>9s}{'Cl_rms':>9s}")
print(f"{'finest':12s}{F['n']:9,.0f}{0:9.4f}{0:9.4f}{0:9.4f}{0:9.4f}{0:9.4f}{0:8.2f}{cdrF:9.4f}{clrF:9.4f}")
for d in cases:
    f=freq(d); m=fit(d,f); cd,cdr,clr=forces(d)
    print(f'{d:12s}{m["n"]:9,.0f}'+"".join(f'{L2(m[k],F[k]):9.4f}' for k in ['Ux','Uy','Wz','k','uv'])
          +f'{100*(f-Sf)/Sf:+8.2f}{cdr:9.4f}{clr:9.4f}')
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
