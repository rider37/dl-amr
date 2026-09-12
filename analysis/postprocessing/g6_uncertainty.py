#!/usr/bin/env python3
"""G6 — 불확실도 검증 재계산 (gap 모델, 해상도 잔차 목표).
현행 fig10-13 / tab:roc-auc 대체용.
 (a) σ̂ vs |Δ| 비닝            (fig10)
 (b) 보정곡선: σ̂ vs RMS(Δ−μ̂)  (fig11) ← R1-C2(a)가 지적한 '평균 주변 조건부 산포'
 (c) AUC(q90/95/98/99)         (tab:roc-auc)
 (d) 예산(상위N) 민감도         (fig13 대체)
"""
import json,sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(_roots.ROOT))
from ml.src.models.hetero_model import HeteroDeltaFullRes
R=_roots.ROOT; D=R/'ml/data/processed/nc_delta_uv_gap'
m=HeteroDeltaFullRes(in_ch=2,out_ch_mean=2,out_ch_logvar=1,base=32,logvar_min=-12.,logvar_max=6.)
m.load_state_dict(torch.load(R/'ml/runs/nc_delta_hetero_gap/best.pt',map_location='cpu',weights_only=True)); m.eval()
ns=json.load(open(D/'norm_stats.json')); tn=json.load(open(D/'target_norm.json'))
XM=torch.tensor(ns['mean']).view(1,-1,1,1); XS=torch.tensor(ns['std']).view(1,-1,1,1)
YM=torch.tensor(tn['mean']).view(1,-1,1,1); YS=torch.tensor(tn['std']).view(1,-1,1,1)
TSD=float(YS.mean())
d=torch.load(D/'test.pt',map_location='cpu',weights_only=False)
X,Y,MK=d['X'],d['y'],d['mask'][:,0].numpy()>0
Xn=((X-XM)/XS).float()
MU=[];SG=[]
with torch.no_grad():
    for i in range(0,len(Xn),16):
        mu,lv=m(Xn[i:i+16]); MU.append(mu); SG.append(torch.exp(0.5*lv)[:,0])
MU=torch.cat(MU)*(YS+1e-8)+YM; SG=(torch.cat(SG)*TSD)
res=(MU-Y)                                  # 평균 헤드 잔차
absd=Y.norm(dim=1).numpy()                  # |Δ|  (해상도 잔차 크기)
rms=res.pow(2).mean(1).sqrt().numpy()       # RMS(Δ−μ̂)
sig=SG.numpy(); ok=MK
print(f'test {len(X)}쌍, 유효점 {ok.sum():,}')
s=sig[ok]; a=absd[ok]; r=rms[ok]
# (a)(b) 비닝
qs=np.quantile(s,np.linspace(0,1,21)); qs[-1]+=1e-12
bi=np.clip(np.digitize(s,qs)-1,0,19)
binned=[dict(sig=float(s[bi==k].mean()), absd=float(a[bi==k].mean()),
             rms=float(np.sqrt((r[bi==k]**2).mean())), n=int((bi==k).sum())) for k in range(20)]
# (c) AUC
auc={}
for q in (90,95,98,99):
    thr=np.quantile(a,q/100); auc[f'q{q}']=float(roc_auc_score((a>=thr).astype(int), s))
# (d) 예산 민감도: 상위 q 면적이 담는 오차질량
cov={}
for q in (0.05,0.10,0.15,0.20,0.30,0.40,0.50):
    t=np.quantile(s,1-q); cov[f'{q:.2f}']=float(a[s>=t].sum()/a.sum())
from scipy.stats import spearmanr,pearsonr
out=dict(n_pairs=len(X), n_pts=int(ok.sum()),
         spearman=float(spearmanr(s,a).statistic), pearson=float(pearsonr(s,a)[0]),
         spearman_rms=float(spearmanr(s,r).statistic),
         auc=auc, coverage=cov, binned=binned,
         mae=float((MU-Y).abs().mean()), sig_mean=float(s.mean()), absd_mean=float(a.mean()))
json.dump(out,open(R/'analysis/output/eval/g6_uncertainty.json','w'),indent=1)
print(f"\nσ̂ vs |Δ|      Spearman {out['spearman']:+.4f}   Pearson {out['pearson']:+.4f}")
print(f"σ̂ vs RMS(Δ−μ̂) Spearman {out['spearman_rms']:+.4f}   ← 이분산 모델이 실제로 예측해야 할 양")
print('\nAUC  ' + '  '.join(f'{k}={v:.4f}' for k,v in auc.items()))
print('\n상위 면적이 담는 오차질량:')
for k,v in cov.items(): print(f'  top {float(k)*100:4.0f}% → {100*v:5.1f}%')
print('\n저장:',R/'analysis/output/eval/g6_uncertainty.json')
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
