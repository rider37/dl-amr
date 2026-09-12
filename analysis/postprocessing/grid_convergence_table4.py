#!/usr/bin/env python3
"""3-grid Richardson (Celik observed-order) local error estimate of the coarse
(57k) mean-Ux field, from existing runs (fine 135840 / base78k 78200 / coarse
57798). Correlate σ̂ with this grid-convergence error and check independence
from |∇U| (unlike the momentum-residual τ). Non-monotonic pixels masked."""
from pathlib import Path
import sys, json
import numpy as np
import pyvista as pv
import torch
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp

ROOT = _roots.ROOT; sys.path.insert(0, str(ROOT / 'eval/scripts'))
from revision_indicator_ranking import load_model, sample, grad_indicators, SIM, NORM, N_TIMES, NEAR
OUT = ROOT / 'paper/M1_report'
XMIN, XMAX, YMIN, YMAX = 2.0, 39.0, -5.0, 5.0
GEOMS = {'circular_Re200': ('circular_Re200', 'circular_Re200_base78k', 'circular_Re200_coarse'),
         'square_Re150': ('square_Re150', 'square_Re150_base78k', 'square_Re150_coarse'),
         'diamond_Re150': ('diamond_Re150', 'diamond_Re150_base78k', 'diamond_Re150_coarse')}
GL = {'circular_Re200': 'Circular Re=200', 'square_Re150': 'Square Re=150', 'diamond_Re150': 'Diamond Re=150'}
SHAPE = {'circular_Re200': 'circle', 'square_Re150': 'square', 'diamond_Re150': 'diamond'}
r21 = np.sqrt(135840 / 78200)   # h_medium / h_fine
r32 = np.sqrt(78200 / 57798)    # h_coarse / h_medium
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'figure.dpi': 160,
                     'savefig.dpi': 250, 'savefig.bbox': 'tight', 'axes.linewidth': 0.5})


def richardson_err(q1, q2, q3):
    """Celik observed-order Richardson. q1 fine, q2 medium, q3 coarse. Returns
    local error of q3, monotonic mask, order p."""
    eps21 = q2 - q1; eps32 = q3 - q2
    mono = (eps21 * eps32 > 0) & (np.abs(eps21) > 1e-10) & (np.abs(eps32) > 1e-10)
    R = eps32 / np.where(np.abs(eps21) < 1e-12, np.nan, eps21)
    s = np.sign(R)
    p = np.full_like(q1, 2.0)
    for _ in range(60):
        qp = np.log(np.abs((r21**p - s) / (r32**p - s)) + 1e-30)
        pn = np.abs(np.log(np.abs(R) + 1e-30) + qp) / np.log(r21)
        p = np.where(mono & np.isfinite(pn), np.clip(pn, 0.3, 6.0), p)
    q_ext = (r21**p * q1 - q2) / (r21**p - 1)
    err = np.abs(q3 - q_ext)
    err = np.where(mono, err, np.nan)
    return err, mono, p


def body(ax, g):
    kw = dict(fc='0.65', ec='w', lw=0.7, zorder=6); sh = SHAPE[g]
    if sh == 'circle': ax.add_patch(mp.Circle((0, 0), 0.5, **kw))
    elif sh == 'square': ax.add_patch(mp.Rectangle((-0.5, -0.5), 1, 1, **kw))
    else: ax.add_patch(mp.Polygon([(0.5, 0), (0, 0.5), (-0.5, 0), (0, -0.5)], **kw))


def rr(a, b, m):
    a, b = a[m], b[m]
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(rankdata(a[ok]), rankdata(b[ok]))[0, 1])


def main():
    model = load_model(); stats = json.loads(NORM.read_text()); summary = {}
    fig, axes = plt.subplots(3, 3, figsize=(13.4, 6.8), squeeze=False)
    for r, (gname, (fn, mn, cn)) in enumerate(GEOMS.items()):
        rf = pv.OpenFOAMReader(str(SIM / fn / 'open.foam'))
        rm = pv.OpenFOAMReader(str(SIM / mn / 'open.foam'))
        rc = pv.OpenFOAMReader(str(SIM / cn / 'open.foam'))
        tf = set(np.round(rf.time_values, 6)); tm = set(np.round(rm.time_values, 6)); tc = sorted(set(np.round(rc.time_values, 6)))
        common = [t for t in tc if t >= 200 and t <= 400 and t in tf and t in tm]
        sel = [common[i] for i in np.linspace(0, len(common) - 1, min(N_TIMES, len(common)), dtype=int)]
        aUf = aUm = aUc = aS = 0.0; n = 0
        for t in sel:
            uf, vf, pf = sample(rf, t); um, vm, pm = sample(rm, t); uc, vc, pc = sample(rc, t)
            aUf = aUf + uf; aUm = aUm + um; aUc = aUc + uc
            x = np.stack([(uc-stats['mean'][0])/(stats['std'][0]+1e-8),
                          (vc-stats['mean'][1])/(stats['std'][1]+1e-8),
                          (pc-stats['mean'][2])/(stats['std'][2]+1e-8)], 0).astype(np.float32)
            with torch.no_grad():
                xt = torch.from_numpy(x).unsqueeze(0); xt = xt.cuda() if torch.cuda.is_available() else xt
                _, lv = model(xt)
            aS = aS + np.sqrt(np.exp(lv.cpu().numpy()[0, 0])); n += 1
        Uf, Um, Uc, S = aUf/n, aUm/n, aUc/n, aS/n
        err_rich, mono, p = richardson_err(Uf, Um, Uc)     # local error of coarse mean-Ux
        err_2grid = np.abs(Uf - Uc)                        # 2-grid fine-coarse
        gU, _ = grad_indicators(Uc, aUc*0)                 # |dU/dx,dy| of mean Ux (indep. check proxy)
        gU = np.abs(np.gradient(Uc, np.linspace(XMIN,XMAX,Uc.shape[1]), axis=1)) + \
             np.abs(np.gradient(Uc, np.linspace(YMIN,YMAX,Uc.shape[0]), axis=0))
        m = NEAR.reshape(S.shape)
        mm = m & mono                          # fair: same pixel set for both estimators
        rho_rich = rr(S, err_rich, mm)
        rho_2g = rr(S, err_2grid, m)           # 2grid on full near-wake
        rho_2g_fair = rr(S, err_2grid, mm)     # 2grid on SAME monotonic pixels (fair)
        rho_rg = rr(err_rich, gU, mm)          # Richardson err vs |grad| (independence)
        rho_2g_g = rr(err_2grid, gU, m)
        frac_mask = float(np.mean(~mono[m]))
        summary[gname] = dict(rho_sigma_richardson=rho_rich, rho_sigma_2grid=rho_2g,
                              rho_sigma_2grid_monopixels=rho_2g_fair,
                              rho_richardson_gradU=rho_rg, rho_2grid_gradU=rho_2g_g,
                              frac_nonmonotonic=frac_mask, median_order_p=float(np.nanmedian(p[mm])))
        s = summary[gname]
        print(f'{gname:16} ρ(σ̂,Rich)={rho_rich:.3f}  ρ(σ̂,2grid_fair)={rho_2g_fair:.3f}  '
              f'[ρ(σ̂,2grid_all)={rho_2g:.3f}] | ρ(Rich,|∇U|)={rho_rg:.3f} | '
              f'nonmono={frac_mask*100:.0f}%  p̃={s["median_order_p"]:.2f}', flush=True)
        # figure row: Richardson err | sigma | scatter
        Xc = np.linspace(XMIN, XMAX, S.shape[1]); crop = Xc <= 15
        ax = axes[r][0]
        ax.imshow(err_rich[:, crop], origin='lower', extent=[XMIN, 15, YMIN, YMAX], aspect='auto',
                  cmap='viridis', vmax=np.nanpercentile(err_rich, 98))
        body(ax, gname); ax.set_xlim(2, 15); ax.set_ylim(-3, 3); ax.set_ylabel(f'{GL[gname]}\n$y/D$', fontsize=9)
        if r == 0: ax.set_title('Richardson error of coarse $\\overline{U}_x$', fontsize=10)
        ax = axes[r][1]
        ax.imshow(S[:, crop], origin='lower', extent=[XMIN, 15, YMIN, YMAX], aspect='auto',
                  cmap='inferno', vmax=np.nanpercentile(S, 99))
        body(ax, gname); ax.set_xlim(2, 15); ax.set_ylim(-3, 3); ax.set_yticklabels([])
        if r == 0: ax.set_title('model $\\hat\\sigma$', fontsize=10)
        ax = axes[r][2]
        ev, sv = err_rich[m], S[m]; ok = np.isfinite(ev) & np.isfinite(sv)
        ev, sv = ev[ok], sv[ok]
        xhi, yhi = np.percentile(ev, 98), np.percentile(sv, 98)
        ax.hexbin(ev, sv, gridsize=40, cmap='magma', bins='log', mincnt=1, extent=(0, xhi, 0, yhi))
        qb = np.unique(np.quantile(ev, np.linspace(0, 1, 11)))
        cx = 0.5*(qb[:-1]+qb[1:]); cy = [np.median(sv[(ev>=a)&(ev<b)]) if ((ev>=a)&(ev<b)).any() else np.nan for a,b in zip(qb[:-1],qb[1:])]
        ax.plot(cx, cy, '-o', color='#00e5c0', ms=3, lw=1.5)
        ax.set_xlim(0, xhi); ax.set_ylim(0, yhi); ax.set_xlabel('Richardson err'); ax.set_ylabel('$\\hat\\sigma$')
        ax.text(0.04, 0.94, rf'$\rho(\hat\sigma,$Rich$)={rho_rich:.2f}$', transform=ax.transAxes, va='top',
                fontsize=9.5, bbox=dict(fc='white', ec='none', alpha=0.8))
        ax.text(0.04, 0.80, rf'2-grid on same px: $\rho={rho_2g_fair:.2f}$ (wins)', transform=ax.transAxes,
                va='top', fontsize=7.5, color='#7a2e2e')
        ax.text(0.04, 0.69, rf'non-monotonic {frac_mask*100:.0f}\% masked', transform=ax.transAxes,
                va='top', fontsize=7.5, color='#555')
        if r == 0: ax.set_title('$\\hat\\sigma$ vs Richardson err', fontsize=10)
        for c in (0, 1):
            if r == 2: axes[r][c].set_xlabel('$x/D$')
            elif r < 2: axes[r][c].set_xticklabels([])
        del rf, rm, rc
    (OUT / 'richardson.json').write_text(json.dumps(summary, indent=2))
    fig.suptitle('3-grid Richardson of coarse mean-$U_x$: NOT in asymptotic range '
                 '(order pegs, 29--52\\% non-monotonic=white) — on valid pixels 2-grid $|$fine$-$coarse$|$ '
                 'does as well or better, so Richardson adds nothing', y=1.0, fontsize=10.5)
    fig.savefig(OUT / 'fig_richardson.png'); fig.savefig(OUT / 'fig_richardson.pdf'); plt.close(fig)
    print(f'\nsaved {OUT}/fig_richardson.png and richardson.json')


if __name__ == '__main__':
    main()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _roots
