#!/usr/bin/env python3
"""G7 — 형상 전이 AUC 재계산 (gap 모델, 목표·layout 통일).
세 형상 모두 동일 정의: 점수 = σ̂,  양성 = |q_fine − q_coarse| 상위 q 분위.
원형은 학습 분포 내(단 Re200 은 외삽), 사각·마름모는 zero-shot.
"""
import json,sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
sys.path.insert(0, str(_roots.ROOT))
from ml.src.models.hetero_model import HeteroDeltaFullRes
R=_roots.ROOT; D=R/'ml/data/processed/nc_delta_uv_uni'
m=HeteroDeltaFullRes(in_ch=2,out_ch_mean=2,out_ch_logvar=1,base=32,logvar_min=-12.,logvar_max=6.)
m.load_state_dict(torch.load(R/'ml/runs/nc_delta_hetero_uni/best.pt',map_location='cpu',weights_only=True)); m.eval()
ns=json.load(open(D/'norm_stats.json')); tn=json.load(open(D/'target_norm.json'))
XM=torch.tensor(ns['mean']).view(1,-1,1,1); XS=torch.tensor(ns['std']).view(1,-1,1,1)
TSD=float(torch.tensor(tn['std']).mean())
mx=np.load(D/'grid_x.npy'); my=np.load(D/'grid_y.npy')
Gx,Gy=np.meshgrid(mx,my)
SETS=[('circular Re200 (test)', D/'test.pt', None),
      ('square Re150 (zero-shot)',  R/'ml/data/processed/sq_ft/all.pt', 'sq'),
      ('diamond Re150 (zero-shot)', R/'ml/data/processed/dia_ft/all.pt','dia')]
def body(sh):
    if sh=='sq':  return (np.abs(Gx)<0.55)&(np.abs(Gy)<0.55)
    if sh=='dia': return (np.abs(Gx)+np.abs(Gy))<0.6
    return np.hypot(Gx,Gy)<0.55
out={}
print(f"{'세트':<26}{'쌍':>5}{'AUC q90':>9}{'q95':>8}{'q98':>8}{'q99':>8}{'Spearman':>10}{'near AUC95':>11}")
for nm,p,sh in SETS:
    d=torch.load(p,map_location='cpu',weights_only=False)
    X,Y,MK=d['X'],d['y'],d['mask'][:,0].numpy()>0
    Xn=((X-XM)/XS).float(); SG=[]
    with torch.no_grad():
        for i in range(0,len(Xn),16):
            _,lv=m(Xn[i:i+16]); SG.append(torch.exp(0.5*lv)[:,0])
    sig=(torch.cat(SG)*TSD).numpy(); absd=Y.norm(dim=1).numpy()
    ok=MK & ~body(sh)[None]
    s=sig[ok]; a=absd[ok]
    au={q:float(roc_auc_score((a>=np.quantile(a,q/100)).astype(int),s)) for q in (90,95,98,99)}
    rho=float(spearmanr(s,a).statistic)
    near=(np.abs(Gx-6)<6)&(np.abs(Gy)<2.5)          # 근접후류 x∈[0,12], |y|<2.5
    okn=MK & near[None] & ~body(sh)[None]
    sn=sig[okn]; an=absd[okn]
    a95=float(roc_auc_score((an>=np.quantile(an,0.95)).astype(int),sn))
    out[nm]=dict(n=len(X),auc=au,spearman=rho,near_auc95=a95)
    print(f'{nm:<26}{len(X):>5}{au[90]:>9.4f}{au[95]:>8.4f}{au[98]:>8.4f}{au[99]:>8.4f}{rho:>+10.4f}{a95:>11.4f}')
json.dump(out,open(R/'analysis/output/eval/g7_transfer_uni.json','w'),indent=1)
print('\n저장:',R/'analysis/output/eval/g7_transfer_uni.json')
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
