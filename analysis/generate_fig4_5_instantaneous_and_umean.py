#!/usr/bin/env python3
"""Generate all new figures for the paper in one script:
  1. fig3a_phase_matched_full.png  — with (a) label
  2. fig3b_phase_matched_zoom.png  — with (b) label
  3. error_contour_UMean.png       — per-geometry colorbar, original style
  4. scatter_method_vs_fine.png    — original style
All with larger font sizes for JCP readability.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
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
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
})

BASE = os.environ.get('DL_AMR_CASES', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cases'))
OUTDIR = os.environ.get('DL_AMR_OUTDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))

GEOM_ORDER = ['circular_Re200', 'square_Re150', 'diamond_Re150']
GEOMETRIES = {
    'circular_Re200': {'title': 'Circular (train)', 'shape': 'circle'},
    'square_Re150':   {'title': 'Square (unseen)',   'shape': 'square'},
    'diamond_Re150':  {'title': 'Diamond (unseen)',  'shape': 'diamond'},
}

CASE_SUFFIXES = {
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

CL_MAX_TIMES = {
    'circular_Re200': {'fine': 110, 'coarse': 103, 'dl_amr': 95, 'grad_amr': 98},
    'square_Re150':   {'fine': 107, 'coarse': 110, 'dl_amr': 101, 'grad_amr': 110},
    'diamond_Re150':  {'fine': 147.7, 'coarse': 106, 'dl_amr': 146, 'grad_amr': 134},
}

FULL_XLIM = (-1.5, 15)
FULL_YLIM = (-5, 5)
UMEAN_XLIM = (-1.5, 20)
UMEAN_YLIM = (-5, 5)

METHODS_ALL = ['fine', 'coarse', 'dl_amr', 'grad_amr']
METHODS_ZOOM = ['coarse', 'dl_amr', 'grad_amr']
METHODS_ERR = ['coarse', 'dl_amr', 'grad_amr']


def case_dir(geom, key):
    return os.path.join(BASE, geom, CASE_SUFFIXES[key])


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


def resample_to_grid(mesh, field_name, component, xlim, ylim, nx, ny, shape=None):
    if field_name in mesh.cell_data and field_name not in mesh.point_data:
        mesh = mesh.cell_data_to_point_data()
    data = mesh.point_data[field_name]
    values = data[:, component] if data.ndim == 2 else data
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    pad = 0.5
    m = ((x >= xlim[0]-pad) & (x <= xlim[1]+pad) &
         (y >= ylim[0]-pad) & (y <= ylim[1]+pad))
    x, y, values = x[m], y[m], values[m]
    xi = np.linspace(xlim[0], xlim[1], nx)
    yi = np.linspace(ylim[0], ylim[1], ny)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), values, (Xi, Yi), method='linear')
    nan_m = np.isnan(Zi)
    if nan_m.any():
        Zi[nan_m] = griddata((x, y), values, (Xi, Yi), method='nearest')[nan_m]
    if shape:
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


def load_foam_mesh(geom, key, time_val=None):
    foam_file = os.path.join(case_dir(geom, key), 'open.foam')
    reader = pv.OpenFOAMReader(foam_file)
    if time_val is not None:
        times = reader.time_values
        idx = np.argmin(np.abs(np.array(times) - time_val))
        reader.set_active_time_value(times[idx])
    else:
        reader.set_active_time_value(400.0)
    mesh = reader.read()['internalMesh']
    return mesh


# ============================================================
# Phase-matched data loading
# ============================================================
def load_phase_matched_data():
    print("Loading phase-matched instantaneous fields...")
    full_data = {}
    meshes = {}
    for geom in GEOM_ORDER:
        shape = GEOMETRIES[geom]['shape']
        for key in METHODS_ALL:
            print(f"  {geom}/{key}...")
            t = CL_MAX_TIMES[geom][key]
            mesh = load_foam_mesh(geom, key, time_val=t)
            meshes[(geom, key)] = mesh
            Xi, Yi, Zi = resample_to_grid(mesh, 'U', 0, FULL_XLIM, FULL_YLIM,
                                           400, 250, shape)
            full_data[(geom, key)] = (Xi, Yi, Zi)

    # Find zoom regions
    print("Finding optimal zoom regions...")
    zoom_regions = {}
    for geom in GEOM_ORDER:
        xlim, ylim = find_best_zoom(full_data, geom)
        zoom_regions[geom] = {'xlim': xlim, 'ylim': ylim}

    zoom_data = {}
    for geom in GEOM_ORDER:
        shape = GEOMETRIES[geom]['shape']
        zr = zoom_regions[geom]
        for key in METHODS_ALL:
            if (geom, key) in meshes:
                Xi, Yi, Zi = resample_to_grid(meshes[(geom, key)], 'U', 0,
                                               zr['xlim'], zr['ylim'],
                                               350, 250, shape)
                zoom_data[(geom, key)] = (Xi, Yi, Zi)
    return full_data, zoom_data, zoom_regions


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
    print(f"    {geom}: x=[{best_xlim[0]:.1f},{best_xlim[1]:.1f}], "
          f"y=[{best_ylim[0]:.1f},{best_ylim[1]:.1f}], ratio={best_score:.2f}")
    return best_xlim, best_ylim


# ============================================================
# Fig 3(a): Full-domain phase-matched
# ============================================================
GEOM_YLABEL = {
    'circular_Re200': 'Circular  $y/D$',
    'square_Re150':   'Square  $y/D$',
    'diamond_Re150':  'Diamond  $y/D$',
}

def generate_fig3a(full_data, zoom_regions, vmin, vmax, levels):
    print("Generating fig3a...")
    fig = plt.figure(figsize=(7.5, 3.8))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 0.03],
                          hspace=0.15, wspace=0.08)
    for row, geom in enumerate(GEOM_ORDER):
        info = GEOMETRIES[geom]
        zr = zoom_regions[geom]
        for col, key in enumerate(METHODS_ALL):
            ax = fig.add_subplot(gs[row, col])
            if (geom, key) in full_data:
                Xi, Yi, Zi = full_data[(geom, key)]
                ax.pcolormesh(Xi, Yi, Zi, cmap='coolwarm', vmin=vmin, vmax=vmax,
                              shading='auto', rasterized=True, alpha=0.85)
                ax.contour(Xi, Yi, Zi, levels=levels, colors='black',
                           linewidths=0.4, alpha=0.7)
                add_obstacle(ax, info['shape'])
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
                ax.set_ylabel(GEOM_YLABEL[geom], fontsize=10)
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

    out = os.path.join(OUTDIR, 'fig3a_phase_matched_full.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================
# Fig 3(b): Zoom with Fine overlay
# ============================================================
def generate_fig3b(zoom_data, zoom_regions, vmin, vmax, levels):
    print("Generating fig3b...")
    fig = plt.figure(figsize=(7.0, 3.8))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.04],
                          hspace=0.20, wspace=0.10)
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
                ax.pcolormesh(Xz_a, Yz_a, Zz_a, cmap='coolwarm', vmin=vmin, vmax=vmax,
                              shading='auto', rasterized=True)
                ax.contour(Xz_a, Yz_a, Zz_a, levels=levels, colors='black',
                           linewidths=0.3, alpha=0.4)
                ax.contour(Xz_f, Yz_f, Zz_f, levels=levels, colors='white',
                           linewidths=1.2, alpha=0.9)
                ax.contour(Xz_f, Yz_f, Zz_f, levels=levels, colors='black',
                           linewidths=0.5, alpha=0.8, linestyles='--')
                add_obstacle(ax, info['shape'])
            ax.set_xlim(zr['xlim'])
            ax.set_ylim(zr['ylim'])
            ax.set_aspect('equal')
            if row == 0:
                ax.set_title(zoom_titles[key], fontsize=10)
            if col == 0:
                ax.set_ylabel(GEOM_YLABEL[geom], fontsize=10)
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel('$x/D$')
            else:
                ax.set_xticklabels([])

    cax = fig.add_subplot(gs[:, 3])
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=Normalize(vmin, vmax))
    fig.colorbar(sm, cax=cax, label=r'$U_x / U_\infty$')

    # (b) label
    fig.text(0.02, 0.97, '(b)', fontsize=14, fontweight='bold', va='top')

    out = os.path.join(OUTDIR, 'fig3b_phase_matched_zoom.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================
# Fig 4: Error contour (original style: contourf, per-row colorbar, near-wake)
# ============================================================
def generate_error_contour(umean_data):
    print("Generating error contour...")
    err_xlim = (-1.5, 12)
    err_ylim = (-4, 4)

    fig = plt.figure(figsize=(8.0, 5.5))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.05],
                          hspace=0.20, wspace=0.10)

    # Resample to near-wake grid for display
    for row, geom in enumerate(GEOM_ORDER):
        info = GEOMETRIES[geom]
        # Resample fine and methods to display grid
        fine_mesh = load_foam_mesh(geom, 'fine')
        Xi_f, Yi_f, Zi_f = resample_to_grid(fine_mesh, 'UMean', 0,
                                              err_xlim, err_ylim, 300, 200,
                                              info['shape'])
        row_vmax = 0
        errors = {}
        for key in METHODS_ERR:
            m = load_foam_mesh(geom, key)
            Xi, Yi, Zi = resample_to_grid(m, 'UMean', 0,
                                           err_xlim, err_ylim, 300, 200,
                                           info['shape'])
            err = np.abs(Zi - Zi_f)
            errors[key] = (Xi, Yi, err)
            vals = err[~np.isnan(err)]
            if len(vals) > 0:
                row_vmax = max(row_vmax, np.percentile(vals, 99))

        n_levels = 12
        clevels = np.linspace(0, row_vmax, n_levels)

        im = None
        for col, key in enumerate(METHODS_ERR):
            ax = fig.add_subplot(gs[row, col])
            Xi, Yi, err = errors[key]
            im = ax.contourf(Xi, Yi, np.nan_to_num(err, nan=0), levels=clevels,
                              cmap='YlOrBr', extend='max')
            add_obstacle(ax, info['shape'])
            ax.set_xlim(err_xlim)
            ax.set_ylim(err_ylim)
            ax.set_aspect('equal')
            if row == 0:
                ax.set_title(PANEL_LABELS[key])
            if col == 0:
                ax.set_ylabel('$y/D$')
                ax.annotate(info['title'], xy=(0, 0.5), xytext=(-58, 0),
                            xycoords='axes fraction', textcoords='offset points',
                            fontsize=10, ha='center', va='center', rotation=90)
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel('$x/D$')
            else:
                ax.set_xticklabels([])

        cax = fig.add_subplot(gs[row, 3])
        cb = fig.colorbar(im, cax=cax)
        if row == 1:
            cb.set_label(r'$|\overline{U}_x - \overline{U}_{x,\mathrm{ref}}|$',
                         fontsize=11)

    out = os.path.join(OUTDIR, 'error_contour_UMean.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================
# Fig 5: Scatter plot (original style: per-row colour, orange diagonal)
# ============================================================
def generate_scatter(umean_data):
    print("Generating scatter plot...")
    from matplotlib.colors import LogNorm
    row_cmaps = ['Blues', 'Oranges', 'Greens']
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.5))
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

    for row, geom in enumerate(GEOM_ORDER):
        info = GEOMETRIES[geom]
        Xi_f, Yi_f, Zi_f = umean_data[(geom, 'fine')]
        fine_flat = Zi_f.ravel()
        valid_f = ~np.isnan(fine_flat)

        for col, key in enumerate(METHODS_ERR):
            ax = axes[row, col]
            Xi, Yi, Zi = umean_data[(geom, key)]
            method_flat = Zi.ravel()
            valid = valid_f & ~np.isnan(method_flat)
            xv, yv = fine_flat[valid], method_flat[valid]
            rmse = np.sqrt(np.mean((yv - xv)**2))

            ax.hexbin(xv, yv, gridsize=80, cmap=row_cmaps[row],
                      mincnt=1, norm=LogNorm(), rasterized=True)
            lims = [min(xv.min(), yv.min()) - 0.05, max(xv.max(), yv.max()) + 0.05]
            ax.plot(lims, lims, '-', color='darkorange', lw=1.0, alpha=0.8)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')
            ax.text(0.05, 0.95, f'RMSE={rmse:.4f}', transform=ax.transAxes,
                    fontsize=10, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray',
                              alpha=0.9))

            if row == 0:
                ax.set_title(PANEL_LABELS[key])
            if col == 0:
                ax.set_ylabel(r'$\overline{U}_{x,\mathrm{method}}$')
                ax.annotate(info['title'], xy=(0, 0.5), xytext=(-62, 0),
                            xycoords='axes fraction', textcoords='offset points',
                            fontsize=10, ha='center', va='center', rotation=90)
            if row == 2:
                ax.set_xlabel(r'$\overline{U}_{x,\mathrm{fine}}$')

    out = os.path.join(OUTDIR, 'scatter_method_vs_fine.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _data_check import require_or_skip
    require_or_skip(
        'Figs 4, 5 (instantaneous wake + UMean)',
        'Run the OpenFOAM cases (e.g. `cd cases/circular_Re200/fine && ./Allrun`) '
        'or download a pre-computed case bundle into cases/ from Zenodo.',
        os.path.join(BASE, 'circular_Re200', 'fine', 'postProcessing',
                     'forceCoeffs', '0', 'coefficient.dat'),
    )

    # Phase-matched figures
    full_data, zoom_data, zoom_regions = load_phase_matched_data()
    absmax = 0
    for (Xi, Yi, Zi) in full_data.values():
        vals = Zi[~np.isnan(Zi)]
        if len(vals) > 0:
            absmax = max(absmax, np.percentile(np.abs(vals), 99))
    vmin, vmax = -absmax, absmax
    levels = np.linspace(vmin, vmax, 11)
    print(f"  Color range: [{vmin:.3f}, {vmax:.3f}]")

    generate_fig3a(full_data, zoom_regions, vmin, vmax, levels)
    generate_fig3b(zoom_data, zoom_regions, vmin, vmax, levels)

    # Time-averaged fields for error contour and scatter
    print("\nLoading time-averaged UMean fields...")
    umean_data = {}
    for geom in GEOM_ORDER:
        shape = GEOMETRIES[geom]['shape']
        for key in METHODS_ALL:
            print(f"  {geom}/{key}...")
            mesh = load_foam_mesh(geom, key)
            Xi, Yi, Zi = resample_to_grid(mesh, 'UMean', 0,
                                           UMEAN_XLIM, UMEAN_YLIM,
                                           400, 200, shape)
            umean_data[(geom, key)] = (Xi, Yi, Zi)

    generate_error_contour(umean_data)
    generate_scatter(umean_data)
    print("\nAll done.")
