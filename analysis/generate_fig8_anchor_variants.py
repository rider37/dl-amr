#!/usr/bin/env python3
"""Fig 8 phase-matched variants with different anchor strategies.

Anchors:
  A: Fine CL max near a representative time (most neutral)
  B1: square uses t=210 for grad(U); others use max wake_frac
  B2: square uses t=340 for grad(U); others use max wake_frac
  C: DL-AMR best wake_frac time (would favour DL-AMR)

Overlay style (flow + mesh), zoom to x in [-2,10], y in [-2.5, 2.5].
Produces 3-row variant (Fine + DL-AMR + grad-U).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.collections import LineCollection
from scipy.interpolate import griddata
from scipy.signal import hilbert, argrelextrema
import pyvista as pv

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.linewidth': 0.6,
})

BASE = os.environ.get('DL_AMR_CASES', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cases'))
OUTDIR = os.environ.get('DL_AMR_OUTDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))

GEOM_ORDER = ['circular_Re200', 'square_Re150', 'diamond_Re150']
GEOMETRIES = {
    'circular_Re200': {'title': 'Circular ($Re=200$)', 'shape': 'circle'},
    'square_Re150':   {'title': 'Square ($Re=150$)',   'shape': 'square'},
    'diamond_Re150':  {'title': 'Diamond ($Re=150$)',  'shape': 'diamond'},
}
CASES = {
    'fine':     '',
    'dl_amr':   '_coarse_amr_dl_uvp_wake_nll',
    'grad_amr': '_coarse_amr_vorticity',
}

GRAD_BEST_TIMES = {
    'circular_Re200': 360.0,
    'square_Re150':   270.0,
    'diamond_Re150':  160.0,
}
DL_BEST_TIMES = {
    'circular_Re200': 215.0,
    'square_Re150':   285.0,
    'diamond_Re150':  400.0,
}

XLIM = (-2, 10)
YLIM = (-2.5, 2.5)


def case_dir(geom, key):
    return os.path.join(BASE, geom + CASES[key])


def add_obstacle(ax, shape, half_size=0.5):
    if shape == 'circle':
        p = Circle((0, 0), half_size, fill=True, fc='white', ec='black', lw=0.8, zorder=5)
    elif shape == 'square':
        p = Rectangle((-half_size, -half_size), 2*half_size, 2*half_size,
                       fill=True, fc='white', ec='black', lw=0.8, zorder=5)
    elif shape == 'diamond':
        verts = [(half_size, 0), (0, half_size), (-half_size, 0), (0, -half_size)]
        p = Polygon(verts, closed=True, fill=True, fc='white', ec='black', lw=0.8, zorder=5)
    else:
        return
    ax.add_patch(p)


def load_cl(geom, key):
    fpath = os.path.join(case_dir(geom, key),
                         'postProcessing/forceCoeffs/0/coefficient.dat')
    rows = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows.append([float(v) for v in line.split()])
    arr = np.array(rows)
    return arr[:, 0], arr[:, 4]


def compute_phase(t, cl, t_min=100):
    mask = t > t_min
    t_use, cl_use = t[mask], cl[mask]
    cl_d = cl_use - np.mean(cl_use)
    analytic = hilbert(cl_d)
    return t_use, np.angle(analytic)


def find_matching_time(t_ref_target, t_ref, phase_ref,
                       t_other, phase_other, window=15.0):
    idx = np.argmin(np.abs(t_ref - t_ref_target))
    phase_target = phase_ref[idx]
    mask = np.abs(t_other - t_ref_target) < window
    if not mask.any():
        mask = np.ones_like(t_other, dtype=bool)
    dphi = phase_other[mask] - phase_target
    dphi = np.mod(dphi + np.pi, 2*np.pi) - np.pi
    best_sub_idx = np.argmin(np.abs(dphi))
    return t_other[mask][best_sub_idx]


def find_cl_max_near(t, cl, t_target, window=10.0):
    mask = np.abs(t - t_target) < window
    t_m, cl_m = t[mask], cl[mask]
    idx = np.argmax(cl_m)
    return t_m[idx]


def build_matched(anchor_mode, square_grad_override=None):
    """Returns dict[geom] -> {'fine': t, 'dl_amr': t, 'grad_amr': t}."""
    matched = {}
    for geom in GEOM_ORDER:
        t_fi, cl_fi = load_cl(geom, 'fine')
        t_dl, cl_dl = load_cl(geom, 'dl_amr')
        t_gr, cl_gr = load_cl(geom, 'grad_amr')
        t_fi_p, ph_fi = compute_phase(t_fi, cl_fi)
        t_dl_p, ph_dl = compute_phase(t_dl, cl_dl)
        t_gr_p, ph_gr = compute_phase(t_gr, cl_gr)

        if anchor_mode == 'fine':
            # anchor: Fine's CL max near a representative time (use GRAD_BEST as starting region)
            t_anchor = find_cl_max_near(t_fi, cl_fi, GRAD_BEST_TIMES[geom])
            t_fine = t_anchor
            idx = np.argmin(np.abs(t_fi_p - t_anchor))
            phase_target = ph_fi[idx]
            # match other two
            def match(t_other, ph_other):
                mask = np.abs(t_other - t_anchor) < 15.0
                if not mask.any():
                    mask = np.ones_like(t_other, dtype=bool)
                dphi = ph_other[mask] - phase_target
                dphi = np.mod(dphi + np.pi, 2*np.pi) - np.pi
                i = np.argmin(np.abs(dphi))
                return t_other[mask][i]
            t_dl_match = match(t_dl_p, ph_dl)
            t_gr_match = match(t_gr_p, ph_gr)
        elif anchor_mode in ('grad', 'grad_b1', 'grad_b2'):
            if anchor_mode == 'grad':
                t_anchor = GRAD_BEST_TIMES[geom]
            elif anchor_mode == 'grad_b1' and geom == 'square_Re150':
                t_anchor = 210.0
            elif anchor_mode == 'grad_b2' and geom == 'square_Re150':
                t_anchor = 340.0
            else:
                t_anchor = GRAD_BEST_TIMES[geom]
            t_gr_match = t_anchor
            t_fine = find_matching_time(t_anchor, t_gr_p, ph_gr, t_fi_p, ph_fi)
            t_dl_match = find_matching_time(t_anchor, t_gr_p, ph_gr, t_dl_p, ph_dl)
        elif anchor_mode == 'dl':
            t_anchor = DL_BEST_TIMES[geom]
            t_dl_match = t_anchor
            t_fine = find_matching_time(t_anchor, t_dl_p, ph_dl, t_fi_p, ph_fi)
            t_gr_match = find_matching_time(t_anchor, t_dl_p, ph_dl, t_gr_p, ph_gr)
        else:
            raise ValueError(anchor_mode)
        matched[geom] = {'fine': t_fine, 'dl_amr': t_dl_match, 'grad_amr': t_gr_match}
    return matched


def load_U_at_time(geom, key, t_target):
    foam_file = os.path.join(case_dir(geom, key), 'open.foam')
    reader = pv.OpenFOAMReader(foam_file)
    times = np.array(reader.time_values)
    idx = np.argmin(np.abs(times - t_target))
    reader.set_active_time_value(times[idx])
    mesh = reader.read()['internalMesh']
    if 'U' in mesh.cell_data and 'U' not in mesh.point_data:
        mesh = mesh.cell_data_to_point_data()
    return mesh, times[idx]


def resample_ux(mesh, xlim, ylim, nx=500, ny=250, shape=None):
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    Ux = mesh.point_data['U'][:, 0]
    pad = 0.5
    m = ((x >= xlim[0]-pad) & (x <= xlim[1]+pad) &
         (y >= ylim[0]-pad) & (y <= ylim[1]+pad))
    x, y, Ux = x[m], y[m], Ux[m]
    xi = np.linspace(xlim[0], xlim[1], nx)
    yi = np.linspace(ylim[0], ylim[1], ny)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), Ux, (Xi, Yi), method='linear')
    nan_m = np.isnan(Zi)
    if nan_m.any():
        Zi[nan_m] = griddata((x, y), Ux, (Xi, Yi), method='nearest')[nan_m]
    hs = 0.55
    if shape == 'circle':
        obs = Xi**2 + Yi**2 < hs**2
    elif shape == 'square':
        obs = (np.abs(Xi) < hs) & (np.abs(Yi) < hs)
    elif shape == 'diamond':
        obs = np.abs(Xi) + np.abs(Yi) < hs
    else:
        obs = np.zeros_like(Xi, dtype=bool)
    Zi[obs] = np.nan
    return Xi, Yi, Zi


def extract_mesh_edges(geom, key, t_target, xlim, ylim):
    foam_file = os.path.join(case_dir(geom, key), 'open.foam')
    reader = pv.OpenFOAMReader(foam_file)
    times = np.array(reader.time_values)
    idx = np.argmin(np.abs(times - t_target))
    reader.set_active_time_value(times[idx])
    mesh = reader.read()['internalMesh']
    edges = mesh.extract_all_edges()
    pts = np.asarray(edges.points)
    if hasattr(edges, 'lines') and edges.lines is not None and len(edges.lines) > 0:
        lines = np.asarray(edges.lines).reshape(-1, 3)
        segs = []
        for L in lines:
            p0, p1 = pts[L[1]], pts[L[2]]
            if (xlim[0] - 0.5 <= p0[0] <= xlim[1] + 0.5 and
                ylim[0] - 0.5 <= p0[1] <= ylim[1] + 0.5):
                segs.append([(p0[0], p0[1]), (p1[0], p1[1])])
        return LineCollection(segs)
    return LineCollection([])


def global_vmax(matched):
    v = 0.0
    for geom in GEOM_ORDER:
        mesh, _ = load_U_at_time(geom, 'fine', matched[geom]['fine'])
        Xi, Yi, Ux = resample_ux(mesh, XLIM, YLIM, shape=GEOMETRIES[geom]['shape'])
        vals = Ux[~np.isnan(Ux)]
        if len(vals):
            v = max(v, np.percentile(np.abs(vals), 99))
    return -v, v


def generate_fig(matched, vmin, vmax, label):
    row_specs = [
        ('fine', False, 'Fine'),
        ('dl_amr', True, 'DL-AMR\n(flow + mesh)'),
        ('grad_amr', True, r'$|\nabla \mathbf{U}|$ AMR' + '\n(flow + mesh)'),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(8.0, 5.0),
                              gridspec_kw={'hspace': 0.22, 'wspace': 0.10})
    for row, (key, overlay_mesh, row_title) in enumerate(row_specs):
        for col, geom in enumerate(GEOM_ORDER):
            info = GEOMETRIES[geom]
            ax = axes[row, col]
            t_target = matched[geom][key]
            mesh, _ = load_U_at_time(geom, key, t_target)
            Xi, Yi, Ux = resample_ux(mesh, XLIM, YLIM, shape=info['shape'])
            ax.pcolormesh(Xi, Yi, Ux, cmap='coolwarm', vmin=vmin, vmax=vmax,
                          shading='auto', rasterized=True)
            if overlay_mesh:
                lc = extract_mesh_edges(geom, key, t_target, XLIM, YLIM)
                lc.set_color('black')
                lc.set_linewidth(0.10)
                lc.set_alpha(0.25)
                ax.add_collection(lc)
            add_obstacle(ax, info['shape'])
            ax.set_xlim(XLIM)
            ax.set_ylim(YLIM)
            ax.set_aspect('equal')
            if row == 0:
                ax.set_title(info['title'], fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{row_title}\n$y/D$')
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel('$x/D$')
            else:
                ax.set_xticklabels([])
    out = os.path.join(OUTDIR, f'fig8_anchor_{label}.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == '__main__':
    variants = [
        ('A_fine_anchor', 'fine'),
        ('B1_square_t210', 'grad_b1'),
        ('B2_square_t340', 'grad_b2'),
        ('C_dl_anchor', 'dl'),
    ]
    for label, mode in variants:
        print(f"\n=== {label} (mode={mode}) ===")
        matched = build_matched(mode)
        for g in GEOM_ORDER:
            m = matched[g]
            print(f"  {g}: fi={m['fine']:.1f}, dl={m['dl_amr']:.1f}, gr={m['grad_amr']:.1f}")
        vmin, vmax = global_vmax(matched)
        generate_fig(matched, vmin, vmax, label)
