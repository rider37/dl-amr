#!/usr/bin/env python3
"""Stage-2 B+C: (B) harmonic-resolved vs raw fluctuation moments, (C) half-window sensitivity, all three shapes.
사용: stage2_moments_halfwin.py circ|sq|dia   (기존 49k 런 스냅샷만 사용, 새 계산 없음)"""
import os, sys, glob, json
from pathlib import Path
import numpy as np, pyvista as pv
K=sys.argv[1]
CR=_roots.CR; OUT=(_roots.ROOT/'analysis/output/eval/stage2')
CASES={'circ':dict(fine='grid_convergence/circular_Re200_finer',coarse='NC2_coarse',gradU='NC49k_gradU',vort='NC49k_vort',Q='NC49k_Q',sigma='NC49k_sigma',static='ST_static1_circ'),
       'sq':  dict(fine='grid_convergence/square_Re150_finer',coarse='NCsq_coarse_full',gradU='SQ49k_gradU',vort='SQ49k_vort',Q='SQ49k_Q',sigma='SQ49k_sigma',static='ST_static1_sq'),
       'dia': dict(fine='grid_convergence/diamond_Re150_finer',coarse='NCdia_coarse_full',gradU='DIA49k_gradU',vort='DIA49k_vort',Q='DIA49k_Q',sigma='DIA49k_sigma',static='ST_static1_dia')}[K]
LO,HI={'circ':(165,245),'sq':(237.5,346.5),'dia':(170,256)}[K]
xi=np.arange(-1.5,20.0001,0.05); yi=np.arange(-5.0,5.0001,0.05)
Xi,Yi=np.meshgrid(xi,yi); G=pv.StructuredGrid(Xi,Yi,np.zeros_like(Xi)); W,H=len(xi),len(yi)
body={'circ':Xi**2+Yi**2<0.55**2,'sq':(np.abs(Xi)<0.55)&(np.abs(Yi)<0.55),'dia':np.abs(Xi)+np.abs(Yi)<0.55}[K]
def freq(d,lo,hi):
    fs=sorted(glob.glob(str(CR/d/'postProcessing/forceCoeffs/*/coefficient.dat')))
    a=np.vstack([np.loadtxt(x,comments='#') for x in fs]); a=a[np.argsort(a[:,0])]
    _,i=np.unique(a[:,0],return_index=True); a=a[i]
    m=(a[:,0]>=lo)&(a[:,0]<=hi); t=a[m,0]; y=a[m,4]-a[m,4].mean()
    g=np.arange(t[0],t[-1],0.005); yg=np.interp(g,t,y)
    F=np.fft.rfft(yg*np.hanning(len(yg))); fr=np.fft.rfftfreq(len(yg),0.005)
    b=np.where((fr>0.05)&(fr<1.0))[0]; k=b[np.argmax(np.abs(F[b]))]
    A,B,C=np.log(np.abs(F[k-1])),np.log(np.abs(F[k])),np.log(np.abs(F[k+1]))
    return float(fr[k]+0.5*(A-C)/(A-2*B+C)*(fr[1]-fr[0]))
def snaps(d,TS):
    p=CR/d/'open.foam'; p.touch(exist_ok=True)
    r=pv.OpenFOAMReader(str(p)); av=np.array(r.time_values); U=[];V=[]
    for t in TS:
        r.set_active_time_value(float(av[np.abs(av-t).argmin()])); m=r.read()[0]
        mm=m.cell_data_to_point_data() if 'U' in m.cell_data else m
        s=G.sample(mm); u=np.asarray(s.point_data['U'])
        U.append(u[:,0].reshape(W,H).T); V.append(u[:,1].reshape(W,H).T)
    return np.array(U,float),np.array(V,float)
def harm(U,V,TS,f):
    M=[np.ones_like(TS)]
    for n in (1,2): M+=[np.cos(2*np.pi*n*f*TS), np.sin(2*np.pi*n*f*TS)]
    M=np.array(M).T
    cu=np.linalg.lstsq(M,U.reshape(len(TS),-1),rcond=None)[0]; cv=np.linalg.lstsq(M,V.reshape(len(TS),-1),rcond=None)[0]
    Um=cu[0].reshape(H,W)
    k=0.5*((cu[1]**2+cu[2]**2+cu[3]**2+cu[4]**2)/2+(cv[1]**2+cv[2]**2+cv[3]**2+cv[4]**2)/2).reshape(H,W)
    uv=((cu[1]*cv[1]+cu[2]*cv[2]+cu[3]*cv[3]+cu[4]*cv[4])/2).reshape(H,W)
    resid=(U.reshape(len(TS),-1)-M@cu); rv=(V.reshape(len(TS),-1)-M@cv)
    kres=0.5*((resid**2).mean(0)+(rv**2).mean(0)).reshape(H,W)      # 적합 잔차 에너지 (광대역+고차)
    return Um,k,uv,kres
def raw(U,V):
    um=U.mean(0); vm=V.mean(0); up=U-um; vp=V-vm
    return um, 0.5*((up**2).mean(0)+(vp**2).mean(0)), (up*vp).mean(0)
TS=np.arange(LO,HI+1e-9,1.0); mid=len(TS)//2
SL={'full':slice(None),'h1':slice(0,mid),'h2':slice(mid,None)}
D={}
for name,c in CASES.items():
    f=freq(c,LO,HI); U,V=snaps(c,TS); D[name]={'St':f}
    for w,s in SL.items():
        Um,k,uv,kres=harm(U[s],V[s],TS[s],f); um_r,k_r,uv_r=raw(U[s],V[s])
        if not os.environ.get('NOBODY'):
            for a in (Um,k,uv,kres,um_r,k_r,uv_r): a[body]=np.nan
        D[name][w]=dict(Um=Um,k=k,uv=uv,kres=kres,Umr=um_r,kr=k_r,uvr=uv_r)
    print('done',name,'St=%.4f'%f,flush=True)
def L2(a,b): return float(np.sqrt(np.nanmean((a-b)**2)))
res={}
print(f"\n== {K}  window [{LO},{HI}]  N={len(TS)} ==")
print(f"{'case':7s} {'L2Ux full':>10s} {'h1':>8s} {'h2':>8s} {'dev%':>6s} | {'L2k harm':>9s} {'L2k raw':>8s} | {'L2uv harm':>9s} {'L2uv raw':>8s} | {'kres/k fine':>11s}")
for name in ['coarse','gradU','vort','Q','sigma','static']:
    F=D['fine']; X=D[name]; r={}
    r['L2Ux']={w:L2(X[w]['Um'],F[w]['Um']) for w in SL}
    r['L2Ux_raw']={w:L2(X[w]['Umr'],F[w]['Umr']) for w in SL}
    r['L2k_h']=L2(X['full']['k'],F['full']['k']); r['L2k_r']=L2(X['full']['kr'],F['full']['kr'])
    r['L2uv_h']=L2(X['full']['uv'],F['full']['uv']); r['L2uv_r']=L2(X['full']['uvr'],F['full']['uvr'])
    r['kres_over_k']=float(np.nanmean(X['full']['kres'])/np.nanmean(X['full']['k']))
    r['St']=X['St']; res[name]=r
    dev=100*max(abs(r['L2Ux']['h1']-r['L2Ux']['full']),abs(r['L2Ux']['h2']-r['L2Ux']['full']))/r['L2Ux']['full']
    print(f"{name:7s} {r['L2Ux']['full']:10.4f} {r['L2Ux']['h1']:8.4f} {r['L2Ux']['h2']:8.4f} {dev:6.1f} | {r['L2k_h']:9.4f} {r['L2k_r']:8.4f} | {r['L2uv_h']:9.4f} {r['L2uv_r']:8.4f} | {r['kres_over_k']:11.3f}")
res['fine']={'St':D['fine']['St'],'kres_over_k':float(np.nanmean(D['fine']['full']['kres'])/np.nanmean(D['fine']['full']['k']))}
print('fine kres/k =',round(res['fine']['kres_over_k'],3))
SUF='_nobody' if os.environ.get('NOBODY') else ''
json.dump(res,open(OUT/f'moments_halfwin_{K}{SUF}.json','w'),indent=1); print('saved',OUT/f'moments_halfwin_{K}{SUF}.json')
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
