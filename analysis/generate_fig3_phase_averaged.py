#!/usr/bin/env python3
"""Phase-averaged Fig 3 v2 — fair peak detection.

Improvements over v1:
  * Time-based peak detection (prominence + min time distance) — independent of Δt
  * Equal peak count across all variants within each geometry
    (use minimum common peak count to ensure fair averaging)

Saved as fig3a_phase_averaged_v2_full.png and fig3b_phase_averaged_v2_zoom.png
(keeps v1 files intact).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from scipy.signal import find_peaks
import pyvista as pv

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
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
    'circular_Re200': {'title': 'Circular\n(train)', 'shape': 'circle'},
    'square_Re150':   {'title': 'Square\n(unseen)', 'shape': 'square'},
    'diamond_Re150':  {'title': 'Diamond\n(unseen)', 'shape': 'diamond'},
}
CASES = {
    'fine':     'fine',
    'coarse':   'coarse',
    'dl_amr':   'dl_amr',
    'grad_amr': 'grad_amr',
}
PANEL_LABELS = {
    'fine': 'Fine',
    'coarse': 'Coarse',
    'dl_amr': 'DL-AMR',
    'grad_amr': r'$|\nabla \mathbf{U}|$ AMR',
}
GEOM_YLABEL = {
    'circular_Re200': 'Circular  $y/D$',
    'square_Re150':   'Square  $y/D$',
    'diamond_Re150':  'Diamond  $y/D$',
}

FULL_XLIM = (-1.5, 15)
FULL_YLIM = (-5, 5)
METHODS_ALL = ['fine', 'coarse', 'dl_amr', 'grad_amr']
METHODS_ZOOM = ['coarse', 'dl_amr', 'grad_amr']

# Rough shedding periods (time units) for each geometry — for time-based peak distance
EXPECTED_T = {
    'circular_Re200': 5.1,
    'square_Re150':   6.6,
    'diamond_Re150':  5.3,
}


def case_dir(geom, key):
    return os.path.join(BASE, geom, CASES[key])


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
    fp = os.path.join(case_dir(geom, key),
                      'postProcessing/forceCoeffs/0/coefficient.dat')
    rows = []
    with open(fp) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            rows.append([float(v) for v in s.split()])
    arr = np.array(rows)
    return arr[:, 0], arr[:, 4]


def find_cl_peaks_fair(t, cl, T_expected, t_min=150, t_max=400,
                       prom_factor=0.3, min_sep_frac=0.8):
    """Time-based peak detection independent of sampling rate.

    prom_factor: peak prominence required = prom_factor * std(CL)
    min_sep_frac: minimum separation between peaks = min_sep_frac * T_expected (time units)
    """
    mask = (t >= t_min) & (t <= t_max)
    t_m, cl_m = t[mask], cl[mask]
    if len(t_m) < 2:
        return np.array([])
    dt = np.median(np.diff(t_m))
    prom = np.std(cl_m) * prom_factor
    distance = max(1, int(round(min_sep_frac * T_expected / dt)))
    idx, _ = find_peaks(cl_m, prominence=prom, distance=distance)
    return t_m[idx]


def resample_ux(mesh, xlim, ylim, nx, ny, shape=None):
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


def phase_average(geom, key, xlim, ylim, nx, ny, peak_times):
    shape = GEOMETRIES[geom]['shape']
    foam_file = os.path.join(case_dir(geom, key), 'open.foam')
    reader = pv.OpenFOAMReader(foam_file)
    all_times = np.array(reader.time_values)

    Xi_ref = Yi_ref = None
    Zi_sum = None
    count_mask = None

    for t_peak in peak_times:
        idx = np.argmin(np.abs(all_times - t_peak))
        reader.set_active_time_value(all_times[idx])
        mesh = reader.read()['internalMesh']
        if 'U' in mesh.cell_data and 'U' not in mesh.point_data:
            mesh = mesh.cell_data_to_point_data()
        Xi, Yi, Zi = resample_ux(mesh, xlim, ylim, nx, ny, shape=shape)
        if Xi_ref is None:
            Xi_ref, Yi_ref = Xi, Yi
            Zi_sum = np.zeros_like(Zi)
            count_mask = np.zeros_like(Zi)
        valid = ~np.isnan(Zi)
        Zi_sum[valid] += Zi[valid]
        count_mask[valid] += 1

    Zi_avg = np.where(count_mask > 0, Zi_sum / np.maximum(count_mask, 1), np.nan)
    return Xi_ref, Yi_ref, Zi_avg


def find_best_zoom(full_data, geom, window_xw=7.0, window_yw=5.0,
                   x_search=(4.0, 14.0), y_search=(-4.5, 4.5), step=0.5):
    Xi_f, Yi_f, Zi_f = full_data[(geom, 'fine')]
    Xi_d, _, Zi_d = full_data[(geom, 'dl_amr')]
    Xi_g, _, Zi_g = full_data[(geom, 'grad_amr')]
    best_score = -np.inf
    best_xlim, best_ylim = (5, 12), (-3, 3)
    for x0 in np.arange(x_search[0], x_search[1] - window_xw + step, step):
        for y0 in np.arange(y_search[0], y_search[1] - window_yw + step, step):
            x1, y1 = x0 + window_xw, y0 + window_yw
            mask = ((Xi_f >= x0) & (Xi_f <= x1) & (Yi_f >= y0) & (Yi_f <= y1) &
                    ~np.isnan(Zi_f) & ~np.isnan(Zi_d) & ~np.isnan(Zi_g))
            if mask.sum() < 50:
                continue
            err_dl = np.mean((Zi_d[mask] - Zi_f[mask])**2)
            err_grad = np.mean((Zi_g[mask] - Zi_f[mask])**2)
            if err_dl == 0:
                continue
            score = err_grad / err_dl
            if score > best_score:
                best_score = score
                best_xlim, best_ylim = (x0, x1), (y0, y1)
    print(f"    {geom}: zoom x=[{best_xlim[0]:.1f},{best_xlim[1]:.1f}], "
          f"y=[{best_ylim[0]:.1f},{best_ylim[1]:.1f}], ratio={best_score:.2f}")
    return best_xlim, best_ylim


def select_common_peaks(peaks_per_method, target_count=None):
    """Given peak times per method, return equal-sized subsampled lists.
    Uses minimum count as common N (or target_count if specified),
    and subsamples each via uniform index selection."""
    counts = {k: len(v) for k, v in peaks_per_method.items()}
    n = target_count if target_count is not None else min(counts.values())
    out = {}
    for k, peaks in peaks_per_method.items():
        if len(peaks) == 0:
            out[k] = np.array([])
        else:
            idxs = np.linspace(0, len(peaks)-1, n).astype(int)
            out[k] = peaks[idxs]
    return out, n


def compute_peaks_all():
    peaks_all = {}
    for geom in GEOM_ORDER:
        T_expected = EXPECTED_T[geom]
        peaks_per_method = {}
        for key in METHODS_ALL:
            t, cl = load_cl(geom, key)
            peaks = find_cl_peaks_fair(t, cl, T_expected)
            peaks_per_method[key] = peaks
            print(f"  {geom}/{key}: found {len(peaks)} peaks (fair)")
        common, n = select_common_peaks(peaks_per_method)
        print(f"  {geom}: using {n} common peaks per variant")
        peaks_all[geom] = common
    return peaks_all


def compute_vlims_and_levels(full_data):
    absmax = 0
    for (Xi, Yi, Zi) in full_data.values():
        vals = Zi[~np.isnan(Zi)]
        if len(vals) > 0:
            absmax = max(absmax, np.percentile(np.abs(vals), 99))
    vmin, vmax = -absmax, absmax
    levels = np.linspace(vmin, vmax, 11)
    return vmin, vmax, levels


def plot_overlay(ax, Xi, Yi, Zi, vmin, vmax, levels, shape=None):
    ax.pcolormesh(Xi, Yi, Zi, cmap='coolwarm', vmin=vmin, vmax=vmax,
                  shading='auto', rasterized=True, alpha=0.85)
    ax.contour(Xi, Yi, Zi, levels=levels, colors='black', linewidths=0.4, alpha=0.7)
    if shape:
        add_obstacle(ax, shape)


def plot_zoom_with_fine_overlay(ax, Xz_amr, Yz_amr, Zz_amr, Xz_fine, Yz_fine, Zz_fine,
                                 vmin, vmax, levels, shape=None):
    ax.pcolormesh(Xz_amr, Yz_amr, Zz_amr, cmap='coolwarm', vmin=vmin, vmax=vmax,
                  shading='auto', rasterized=True)
    ax.contour(Xz_amr, Yz_amr, Zz_amr, levels=levels, colors='black',
               linewidths=0.3, alpha=0.4)
    ax.contour(Xz_fine, Yz_fine, Zz_fine, levels=levels, colors='white',
               linewidths=1.2, alpha=0.9)
    ax.contour(Xz_fine, Yz_fine, Zz_fine, levels=levels, colors='black',
               linewidths=0.5, alpha=0.8, linestyles='--')
    if shape:
        add_obstacle(ax, shape)


def generate_fig3a(full_data, zoom_regions, vmin, vmax, levels):
    print("Generating fig3a (phase-averaged v2)...")
    fig = plt.figure(figsize=(7.5, 5.0))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 0.03],
                          hspace=0.15, wspace=0.08, left=0.11, right=0.97)
    for row, geom in enumerate(GEOM_ORDER):
        info = GEOMETRIES[geom]
        zr = zoom_regions[geom]
        for col, key in enumerate(METHODS_ALL):
            ax = fig.add_subplot(gs[row, col])
            if (geom, key) in full_data:
                Xi, Yi, Zi = full_data[(geom, key)]
                plot_overlay(ax, Xi, Yi, Zi, vmin, vmax, levels, info['shape'])
                rect = Rectangle((zr['xlim'][0], zr['ylim'][0]),
                                  zr['xlim'][1]-zr['xlim'][0],
                                  zr['ylim'][1]-zr['ylim'][0],
                                  lw=1.0, ec='lime', fc='none', ls='--', zorder=6)
                ax.add_patch(rect)
            ax.set_xlim(FULL_XLIM)
            ax.set_ylim(FULL_YLIM)
            ax.set_aspect('equal')
            if row == 0:
                ax.set_title(PANEL_LABELS[key], fontsize=11)
            if col == 0:
                ax.set_ylabel('$y/D$', fontsize=11)
                ax.annotate(info['title'], xy=(0, 0.5), xytext=(-60, 0),
                            xycoords='axes fraction', textcoords='offset points',
                            fontsize=11, ha='center', va='center', rotation=90)
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel('$x/D$')
            else:
                ax.set_xticklabels([])

    cax = fig.add_subplot(gs[:, 4])
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=Normalize(vmin, vmax))
    fig.colorbar(sm, cax=cax, label=r'$U_x / U_\infty$')

    fig.text(0.02, 0.97, '(a)', fontsize=14, fontweight='bold', va='top')

    # Align row ylabels so different tick-label widths don't shift them
    left_axes = [ax for ax in fig.axes if ax.get_ylabel() and 'y/D' in ax.get_ylabel()]
    fig.align_ylabels(left_axes)

    out_png = os.path.join(OUTDIR, 'fig3a_phase_averaged_v2_full.png')
    out_pdf = os.path.join(OUTDIR, 'fig3a_phase_averaged_v2_full.pdf')
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  Saved {out_png} and {out_pdf}")


def generate_fig3b(zoom_data, zoom_regions, vmin, vmax, levels):
    print("Generating fig3b (phase-averaged v2)...")
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.04],
                          hspace=0.18, wspace=0.10, left=0.12, right=0.97)
    zoom_titles = {
        'coarse': 'Coarse vs. Fine (--)',
        'dl_amr': 'DL-AMR vs. Fine (--)',
        'grad_amr': r'$|\nabla \mathbf{U}|$ AMR vs. Fine (--)',
    }

    for row, geom in enumerate(GEOM_ORDER):
        info = GEOMETRIES[geom]
        zr = zoom_regions[geom]
        for col, key in enumerate(METHODS_ZOOM):
            ax = fig.add_subplot(gs[row, col])
            if (geom, key) in zoom_data and (geom, 'fine') in zoom_data:
                Xz_a, Yz_a, Zz_a = zoom_data[(geom, key)]
                Xz_f, Yz_f, Zz_f = zoom_data[(geom, 'fine')]
                plot_zoom_with_fine_overlay(ax, Xz_a, Yz_a, Zz_a,
                                            Xz_f, Yz_f, Zz_f,
                                            vmin, vmax, levels, info['shape'])
            ax.set_xlim(zr['xlim'])
            ax.set_ylim(zr['ylim'])
            ax.set_aspect('equal')
            if row == 0:
                ax.set_title(zoom_titles[key], fontsize=10)
            if col == 0:
                ax.set_ylabel('$y/D$', fontsize=11)
                ax.annotate(info['title'], xy=(0, 0.5), xytext=(-60, 0),
                            xycoords='axes fraction', textcoords='offset points',
                            fontsize=11, ha='center', va='center', rotation=90)
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel('$x/D$')
            else:
                ax.set_xticklabels([])

    cax = fig.add_subplot(gs[:, 3])
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=Normalize(vmin, vmax))
    fig.colorbar(sm, cax=cax, label=r'$U_x / U_\infty$')

    fig.text(0.02, 0.97, '(b)', fontsize=14, fontweight='bold', va='top')

    # Align row ylabels so different tick-label widths don't shift them
    left_axes = [ax for ax in fig.axes if ax.get_ylabel() and 'y/D' in ax.get_ylabel()]
    fig.align_ylabels(left_axes)

    out_png = os.path.join(OUTDIR, 'fig3b_phase_averaged_v2_zoom.png')
    out_pdf = os.path.join(OUTDIR, 'fig3b_phase_averaged_v2_zoom.pdf')
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  Saved {out_png} and {out_pdf}")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _data_check import require_or_skip
    require_or_skip(
        'Fig 3 (phase-averaged wake)',
        'Run the OpenFOAM cases (e.g. `cd cases/circular_Re200/fine && ./Allrun`) '
        'or download a pre-computed case bundle into cases/ from Zenodo.',
        os.path.join(BASE, 'circular_Re200', 'fine', 'postProcessing',
                     'forceCoeffs', '0', 'coefficient.dat'),
    )

    import pickle
    cache_path = '/tmp/fig3_phaseavg_cache.pkl'
    if os.path.exists(cache_path):
        print(f"Loading cached averaged fields from {cache_path}...")
        with open(cache_path, 'rb') as f:
            full_data, zoom_data, zoom_regions = pickle.load(f)
    else:
        print("Detecting CL peaks with fair (time-based) method...")
        peaks_all = compute_peaks_all()

        print("\nPhase-averaging full-domain fields...")
        full_data = {}
        for geom in GEOM_ORDER:
            for key in METHODS_ALL:
                Xi, Yi, Zi = phase_average(geom, key, FULL_XLIM, FULL_YLIM, 400, 250,
                                            peaks_all[geom][key])
                full_data[(geom, key)] = (Xi, Yi, Zi)

        print("\nFinding optimal zoom regions...")
        zoom_regions = {}
        for geom in GEOM_ORDER:
            xlim, ylim = find_best_zoom(full_data, geom)
            zoom_regions[geom] = {'xlim': xlim, 'ylim': ylim}

        print("\nPhase-averaging zoom regions...")
        zoom_data = {}
        for geom in GEOM_ORDER:
            zr = zoom_regions[geom]
            for key in METHODS_ALL:
                Xi, Yi, Zi = phase_average(geom, key, zr['xlim'], zr['ylim'], 350, 250,
                                            peaks_all[geom][key])
                zoom_data[(geom, key)] = (Xi, Yi, Zi)

        with open(cache_path, 'wb') as f:
            pickle.dump((full_data, zoom_data, zoom_regions), f)
        print(f"  Cached to {cache_path}")

    vmin, vmax, levels = compute_vlims_and_levels(full_data)
    print(f"\nColor range: [{vmin:.3f}, {vmax:.3f}]")
    generate_fig3a(full_data, zoom_regions, vmin, vmax, levels)
    generate_fig3b(zoom_data, zoom_regions, vmin, vmax, levels)
    print("Done.")
