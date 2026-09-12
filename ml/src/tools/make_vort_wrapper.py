#!/usr/bin/env python3
"""고전 |ω| 지표를 학습 정책과 동일한 wrapper(격자 샘플링→격자 평가→bilinear 역보간→W1 후보→DL 레벨 가중)로 통과시키는 TorchScript 모듈.
솔버 입력 x[1,3,H,W] = z-score 정규화된 (u,v,p). 출력 (mean_dummy[1,2,H,W], logvar[1,1,H,W]) with logvar = log(ω_z^2+eps) → I_c = sqrt(exp(logvar)) = |ω_z|.
격자: W=432 (x∈[-2,25]) , H=160 (y∈[-5,5]); k=j*W+i (j=y 행, i=x 열)."""
import json, os, sys, torch, torch.nn as nn
from pathlib import Path
ROOT=Path(os.environ.get("DLAMR_ROOT", Path(__file__).resolve().parents[3]))
class VortScore(nn.Module):
    def __init__(self, umean, ustd, vmean, vstd, dx, dy):
        super().__init__(); self.umean=umean; self.ustd=ustd; self.vmean=vmean; self.vstd=vstd; self.dx=dx; self.dy=dy
    def forward(self, x):
        u = x[:, 0:1]*self.ustd + self.umean; v = x[:, 1:2]*self.vstd + self.vmean
        # central differences (one-sided at edges), axis 3 = x, axis 2 = y
        dvdx = torch.zeros_like(v); dudy = torch.zeros_like(u)
        dvdx[:, :, :, 1:-1] = (v[:, :, :, 2:] - v[:, :, :, :-2])/(2*self.dx)
        dvdx[:, :, :, 0] = (v[:, :, :, 1] - v[:, :, :, 0])/self.dx; dvdx[:, :, :, -1] = (v[:, :, :, -1] - v[:, :, :, -2])/self.dx
        dudy[:, :, 1:-1, :] = (u[:, :, 2:, :] - u[:, :, :-2, :])/(2*self.dy)
        dudy[:, :, 0, :] = (u[:, :, 1, :] - u[:, :, 0, :])/self.dy; dudy[:, :, -1, :] = (u[:, :, -1, :] - u[:, :, -2, :])/self.dy
        w = dvdx - dudy
        logvar = torch.log(w*w + 1e-12)
        mean = torch.zeros_like(x[:, :2])
        return mean, logvar
um,us=ns['mean'][0],ns['std'][0]; vm,vs=ns['mean'][1],ns['std'][1]
print('norm stats u',um,us,'v',vm,vs)
mod=torch.jit.script(VortScore(float(um),float(us),float(vm),float(vs),27.0/431,10.0/159).eval())
out=ROOT/'ml/sweep/deploy/vort_grid.ts'; mod.save(str(out))
# 검증: 테스트 스냅샷에서 |ω| vs numpy 계산
import numpy as np
te=torch.load(ROOT/'ml/data/processed/nc_delta_uv_uni/test.pt',map_location='cpu',weights_only=False)
X=te['X'][:2]; xn=(X-torch.tensor(ns['mean']).view(1,-1,1,1))/torch.tensor(ns['std']).view(1,-1,1,1); x3=torch.cat([xn,torch.zeros_like(xn[:,:1])],1)
with torch.no_grad(): m_,lv=torch.jit.load(str(out))(x3)
w_ts=torch.sqrt(torch.exp(lv))[:,0].numpy()
u=X[:,0].numpy().astype(float); v=X[:,1].numpy().astype(float); dudy,dudx=np.gradient(u,10/159,27/431,axis=(1,2)); dvdy,dvdx=np.gradient(v,10/159,27/431,axis=(1,2)); w_np=np.abs(dvdx-dudy)
print('shape',w_ts.shape,'max|Δ| vs numpy (interior)',float(np.abs(w_ts[:,1:-1,1:-1]-w_np[:,1:-1,1:-1]).max()),'max|ω|',float(w_np.max()))
print('saved',out)
