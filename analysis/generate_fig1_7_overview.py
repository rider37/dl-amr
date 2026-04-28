#!/usr/bin/env python3
"""Generate ALL paper figures for the DL-AMR paper."""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

warnings.filterwarnings('ignore')

# ============================================================
# Global style settings
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.0,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
})

COLORS = {
    'fine': '#000000',
    'coarse': '#0072B2',
    'dl_wake': '#D55E00',
    'grad_amr': '#009E73',
}
STYLES = {
    'fine': '-',
    'coarse': '--',
    'dl_wake': '-',
    'grad_amr': '-.',
}
LW = {
    'fine': 1.2,
    'coarse': 1.0,
    'dl_wake': 1.2,
    'grad_amr': 0.9,
}
LABELS = {
    'fine': 'Fine',
    'coarse': 'Coarse',
    'dl_wake': 'DL-AMR',
    'grad_amr': r'$|\nabla \mathbf{U}|$ AMR',
}

BASE = os.environ.get('DL_AMR_CASES', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cases'))
OUTDIR = os.environ.get('DL_AMR_OUTDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
os.makedirs(OUTDIR, exist_ok=True)

GEOM_ORDER = ['circular_Re200', 'square_Re150', 'diamond_Re150']
GEOMETRIES = {
    'circular_Re200': {'title': 'Circular\n(train)', 'Re': 200, 'shape': 'circle'},
    'square_Re150':   {'title': 'Square\n(unseen)',   'Re': 150, 'shape': 'square'},
    'diamond_Re150':  {'title': 'Diamond\n(unseen)',  'Re': 150, 'shape': 'diamond'},
}

CASE_METHODS = {
    'fine':     'fine',
    'coarse':   'coarse',
    'dl_wake':  'dl_amr',
    'dl_amr':   'dl_amr',
    'grad_amr': 'grad_amr',
}


def case_dir(geom, key):
    return os.path.join(BASE, geom, CASE_METHODS[key])


# ============================================================
# Helper: load force coefficients
# ============================================================
def load_force_coeffs(filepath):
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = line.split()
            rows.append([float(v) for v in vals])
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 4]  # time, Cd, Cl


# ============================================================
# Figure 1: Force histories – one composite figure
# ============================================================
def generate_force_figures():
    print("Generating force history figures...")

    fig, all_axes = plt.subplots(2, 3, figsize=(6.5, 3.8), sharex=True)

    for col, geom in enumerate(GEOM_ORDER):
        info = GEOMETRIES[geom]
        ax_cd = all_axes[0, col]
        ax_cl = all_axes[1, col]

        # First pass: plot fine/coarse/DL wake to determine y-limits
        ylims_cd = []
        ylims_cl = []
        plot_data = {}
        for key in ['fine', 'coarse', 'dl_wake', 'grad_amr']:
            fpath = os.path.join(case_dir(geom, key),
                                 'postProcessing/forceCoeffs/0/coefficient.dat')
            try:
                t, cd, cl = load_force_coeffs(fpath)
                mask = t >= 150
                plot_data[key] = (t[mask], cd[mask], cl[mask])
                if key != 'grad_amr':
                    ylims_cd.extend([cd[mask].min(), cd[mask].max()])
                    ylims_cl.extend([cl[mask].min(), cl[mask].max()])
            except Exception as e:
                print(f"  Warning: could not load {fpath}: {e}")

        # Compute y-limits from fine/coarse/DL wake with padding
        if ylims_cd:
            cd_lo, cd_hi = min(ylims_cd), max(ylims_cd)
            cd_pad = (cd_hi - cd_lo) * 0.15
            cd_ylim = (cd_lo - cd_pad, cd_hi + cd_pad)
        else:
            cd_ylim = None

        if ylims_cl:
            cl_lo, cl_hi = min(ylims_cl), max(ylims_cl)
            cl_pad = (cl_hi - cl_lo) * 0.15
            cl_ylim = (cl_lo - cl_pad, cl_hi + cl_pad)
        else:
            cl_ylim = None

        # Plot all variants (grad_amr will be clipped by ylim)
        for key in ['fine', 'coarse', 'dl_wake', 'grad_amr']:
            if key not in plot_data:
                continue
            t, cd, cl = plot_data[key]
            ax_cd.plot(t, cd, color=COLORS[key], linestyle=STYLES[key],
                       linewidth=LW[key], label=LABELS[key], clip_on=True)
            ax_cl.plot(t, cl, color=COLORS[key], linestyle=STYLES[key],
                       linewidth=LW[key], label=LABELS[key], clip_on=True)

        ax_cd.set_xlim(150, 400)
        ax_cl.set_xlim(150, 400)
        if cd_ylim:
            ax_cd.set_ylim(cd_ylim)
        if cl_ylim:
            ax_cl.set_ylim(cl_ylim)

        ax_cd.set_title(info['title'], fontsize=9)
        if col == 0:
            ax_cd.set_ylabel(r'$C_D$')
            ax_cl.set_ylabel(r'$C_L$')
        ax_cl.set_xlabel(r'$t \cdot U_\infty / D$')

    # Single legend at top
    handles, labels = all_axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.03), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = os.path.join(OUTDIR, 'force_history_all.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ============================================================
# Helper: pyvista field loading
# ============================================================
def find_vtk_mean_internal(geom, key):
    # Try VTK_mean first
    d = os.path.join(case_dir(geom, key), 'VTK_mean')
    if os.path.isdir(d):
        subdirs = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
        if subdirs:
            vtu = os.path.join(d, subdirs[0], 'internal.vtu')
            if os.path.isfile(vtu):
                return vtu
    # Fallback: read via OpenFOAMReader at t=400
    foam_file = os.path.join(case_dir(geom, key), 'open.foam')
    if os.path.isfile(foam_file):
        return ('foam', foam_file)
    return None


def _load_mesh(vtu_path):
    """Load mesh from VTU file or OpenFOAM case."""
    import pyvista as pv
    if isinstance(vtu_path, tuple) and vtu_path[0] == 'foam':
        reader = pv.OpenFOAMReader(vtu_path[1])
        reader.set_active_time_value(400.0)
        multi = reader.read()
        if hasattr(multi, 'keys') and 'internalMesh' in multi.keys():
            return multi['internalMesh']
        return multi
    return pv.read(vtu_path)


def resample_field_to_grid(vtu_path, field_name, component, xlim, ylim,
                           nx=400, ny=200, obstacle_shape=None):
    mesh = _load_mesh(vtu_path)

    if field_name not in mesh.point_data and field_name not in mesh.cell_data:
        raise KeyError(f"Field '{field_name}' not found in {vtu_path}")

    if field_name in mesh.point_data:
        data = mesh.point_data[field_name]
        points = mesh.points
    else:
        mesh = mesh.cell_data_to_point_data()
        data = mesh.point_data[field_name]
        points = mesh.points

    if data.ndim == 1:
        values = data
    elif data.ndim == 2:
        values = data[:, component]
    else:
        raise ValueError(f"Unexpected field shape: {data.shape}")

    x, y = points[:, 0], points[:, 1]
    # Use slightly wider margins for interpolation to avoid edge artifacts
    pad = 0.5
    mask = (x >= xlim[0] - pad) & (x <= xlim[1] + pad) & \
           (y >= ylim[0] - pad) & (y <= ylim[1] + pad)
    x, y, values = x[mask], y[mask], values[mask]

    xi = np.linspace(xlim[0], xlim[1], nx)
    yi = np.linspace(ylim[0], ylim[1], ny)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), values, (Xi, Yi), method='linear')
    # Fill NaN edges with nearest-neighbour to avoid empty borders
    nan_mask = np.isnan(Zi)
    if nan_mask.any():
        Zi_nn = griddata((x, y), values, (Xi, Yi), method='nearest')
        Zi[nan_mask] = Zi_nn[nan_mask]

    # Mask out obstacle interior to prevent interpolation artifacts
    if obstacle_shape is not None:
        hs = 0.55  # slightly larger than half_size=0.5 to cover edges
        if obstacle_shape == 'circle':
            obs_mask = Xi**2 + Yi**2 < hs**2
        elif obstacle_shape == 'square':
            obs_mask = (np.abs(Xi) < hs) & (np.abs(Yi) < hs)
        elif obstacle_shape == 'diamond':
            obs_mask = np.abs(Xi) + np.abs(Yi) < hs
        else:
            obs_mask = None
        if obs_mask is not None:
            Zi[obs_mask] = np.nan

    return Xi, Yi, Zi


def extract_line_from_vtk(vtu_path, x_station, y_range=(-3, 3), n_points=200):
    """Extract a vertical line profile of UMean_x at a given x/D station."""
    import pyvista as pv
    mesh = _load_mesh(vtu_path)
    if 'UMean' in mesh.cell_data and 'UMean' not in mesh.point_data:
        mesh = mesh.cell_data_to_point_data()
    a = np.array([x_station, y_range[0], 0.0])
    b = np.array([x_station, y_range[1], 0.0])
    line = pv.Line(a, b, resolution=n_points - 1)
    sampled = line.sample(mesh)
    y_vals = sampled.points[:, 1]
    u_vals = sampled.point_data['UMean'][:, 0]
    return y_vals, u_vals


def add_obstacle(ax, shape, half_size=0.5):
    if shape == 'circle':
        patch = Circle((0, 0), half_size, fill=True, facecolor='white',
                        edgecolor='black', linewidth=0.8, zorder=5)
    elif shape == 'square':
        patch = Rectangle((-half_size, -half_size), 2*half_size, 2*half_size,
                           fill=True, facecolor='white', edgecolor='black',
                           linewidth=0.8, zorder=5)
    elif shape == 'diamond':
        verts = [(half_size, 0), (0, half_size), (-half_size, 0), (0, -half_size)]
        patch = Polygon(verts, closed=True, fill=True, facecolor='white',
                         edgecolor='black', linewidth=0.8, zorder=5)
    else:
        return
    ax.add_patch(patch)


# ============================================================
# Figures 2 & 3: Field comparisons – one composite per field
# 3 rows (geometries) × 3 columns (fine/coarse/DL wake) + colorbar
# ============================================================
def compute_manual_stats(geom, key, xlim, ylim, nx=400, ny=200,
                         t_start=300, t_end=400, t_step=2):
    """Compute time-averaged Ux and u'u' from instantaneous snapshots
    on a uniform grid. Avoids AMR field-remapping artifacts."""
    import pyvista as pv

    foam_file = os.path.join(case_dir(geom, key), 'open.foam')
    reader = pv.OpenFOAMReader(foam_file)

    xi = np.linspace(xlim[0], xlim[1], nx)
    yi = np.linspace(ylim[0], ylim[1], ny)
    Xi, Yi = np.meshgrid(xi, yi)

    Ux_sum = np.zeros((ny, nx))
    Ux2_sum = np.zeros((ny, nx))
    count = 0
    pad = 0.5

    for t in range(t_start, t_end + 1, t_step):
        try:
            reader.set_active_time_value(float(t))
            multi = reader.read()
            mesh = multi['internalMesh']
            if 'U' in mesh.cell_data and 'U' not in mesh.point_data:
                mesh = mesh.cell_data_to_point_data()
            x, y = mesh.points[:, 0], mesh.points[:, 1]
            u_data = mesh.point_data['U']
            m = ((x >= xlim[0]-pad) & (x <= xlim[1]+pad) &
                 (y >= ylim[0]-pad) & (y <= ylim[1]+pad))
            Ux = griddata((x[m], y[m]), u_data[m, 0], (Xi, Yi), method='linear')
            Ux = np.nan_to_num(Ux)
            Ux_sum += Ux
            Ux2_sum += Ux**2
            count += 1
        except Exception:
            pass

    if count == 0:
        return Xi, Yi, None, None

    Ux_mean = Ux_sum / count
    Ux_prime2 = Ux2_sum / count - Ux_mean**2
    Ux_prime2 = np.maximum(Ux_prime2, 0)  # numerical floor

    return Xi, Yi, Ux_mean, Ux_prime2


def generate_field_figures(style='color'):
    """Generate Fig 3 (UMean) and Fig 4 (UPrime2Mean).
    style: 'color' (filled colormesh), 'contour' (line contours only), 'overlay' (color + contour lines).
    """
    print(f"Generating field comparison figures (style={style})...")

    field_configs = [
        {
            'field': 'UMean', 'component': 0, 'suffix': 'UMean_x',
            'cmap': 'coolwarm', 'label': r'$\overline{U}_x / U_\infty$',
        },
        {
            'field': 'UPrime2Mean', 'component': 0, 'suffix': 'UPrime2Mean_xx',
            'cmap': 'inferno', 'label': r"$\overline{u'_x u'_x} / U_\infty^2$",
        },
    ]

    xlim = (-1.5, 12)
    ylim = (-4, 4)
    panels = ['fine', 'coarse', 'dl_wake', 'grad_amr']
    panel_titles = ['Fine', 'Coarse', 'DL-AMR', r'$|\nabla \mathbf{U}|$ AMR']

    for fc in field_configs:
        fig = plt.figure(figsize=(7.5, 5.0))
        if style == 'contour':
            # No colorbar column for pure contour
            gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1],
                                   hspace=0.08, wspace=0.08)
        else:
            gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 0.03],
                                   hspace=0.08, wspace=0.08)

        global_vmin = np.inf
        global_vmax = -np.inf

        # First pass: load all data and compute global color limits
        all_fields = {}
        # Pre-compute manual stats for AMR cases (avoid fieldAverage artifacts)
        manual_cache = {}
        for geom in GEOM_ORDER:
            for key in panels:
                if key in ('dl_wake', 'grad_amr'):
                    cache_key = (geom, key)
                    if cache_key not in manual_cache:
                        print(f"  Computing manual stats for {geom}/{key}...")
                        Xi, Yi, Ux_mean, Ux_prime2 = compute_manual_stats(
                            geom, key, xlim, ylim)
                        manual_cache[cache_key] = (Xi, Yi, Ux_mean, Ux_prime2)

        for row, geom in enumerate(GEOM_ORDER):
            info = GEOMETRIES[geom]
            for col, key in enumerate(panels):
                try:
                    if key in ('dl_wake', 'grad_amr') and (geom, key) in manual_cache:
                        Xi, Yi, Ux_mean, Ux_prime2 = manual_cache[(geom, key)]
                        if fc['field'] == 'UMean':
                            Zi = Ux_mean
                        else:
                            Zi = Ux_prime2
                        # Mask obstacle
                        hs = 0.55
                        if info['shape'] == 'circle':
                            mask = Xi**2 + Yi**2 < hs**2
                        elif info['shape'] == 'square':
                            mask = (np.abs(Xi) < hs) & (np.abs(Yi) < hs)
                        elif info['shape'] == 'diamond':
                            mask = np.abs(Xi) + np.abs(Yi) < hs
                        else:
                            mask = np.zeros_like(Xi, dtype=bool)
                        if Zi is not None:
                            Zi[mask] = np.nan
                    else:
                        vtu_path = find_vtk_mean_internal(geom, key)
                        if vtu_path is None:
                            continue
                        Xi, Yi, Zi = resample_field_to_grid(
                            vtu_path, fc['field'], fc['component'], xlim, ylim,
                            obstacle_shape=info['shape'])

                    all_fields[(row, col)] = (Xi, Yi, Zi)
                    vals = Zi[~np.isnan(Zi)]
                    if len(vals) > 0:
                        if fc['cmap'] == 'inferno':
                            global_vmax = max(global_vmax, np.percentile(vals, 99))
                            global_vmin = 0
                        else:
                            absmax = np.percentile(np.abs(vals), 99)
                            global_vmax = max(global_vmax, absmax)
                            global_vmin = -global_vmax
                except Exception as e:
                    print(f"  Warning: {geom}/{key}: {e}")

        if fc['cmap'] == 'coolwarm':
            global_vmin = -global_vmax

        # Build contour levels (10 evenly spaced values between vmin and vmax)
        if fc['cmap'] == 'inferno':
            contour_levels = np.linspace(global_vmin, global_vmax, 11)[1:]  # skip 0
        else:
            contour_levels = np.linspace(global_vmin, global_vmax, 11)

        # Second pass: plot
        im = None
        for row, geom in enumerate(GEOM_ORDER):
            info = GEOMETRIES[geom]
            for col, key in enumerate(panels):
                ax = fig.add_subplot(gs[row, col])
                if (row, col) in all_fields:
                    Xi, Yi, Zi = all_fields[(row, col)]
                    if style == 'color':
                        im = ax.pcolormesh(Xi, Yi, Zi, cmap=fc['cmap'],
                                            vmin=global_vmin, vmax=global_vmax,
                                            shading='auto', rasterized=True)
                    elif style == 'contour':
                        # White background, black contour lines with labels
                        ax.set_facecolor('white')
                        cs = ax.contour(Xi, Yi, Zi, levels=contour_levels,
                                         colors='black', linewidths=0.5)
                        # Label every 2nd contour with small text
                        ax.clabel(cs, levels=contour_levels[::2],
                                   inline=True, fontsize=5, fmt='%.2f')
                        # For colorbar reference (invisible)
                        im = ax.pcolormesh(Xi, Yi, Zi, cmap=fc['cmap'],
                                            vmin=global_vmin, vmax=global_vmax,
                                            shading='auto', rasterized=True, alpha=0)
                    elif style == 'overlay':
                        im = ax.pcolormesh(Xi, Yi, Zi, cmap=fc['cmap'],
                                            vmin=global_vmin, vmax=global_vmax,
                                            shading='auto', rasterized=True, alpha=0.85)
                        ax.contour(Xi, Yi, Zi, levels=contour_levels,
                                    colors='black', linewidths=0.4, alpha=0.7)
                    add_obstacle(ax, info['shape'])
                else:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                            ha='center', va='center', fontsize=8, color='gray')

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_aspect('equal')

                # Titles on top row only
                if row == 0:
                    ax.set_title(panel_titles[col], fontsize=9)

                # Row labels on left
                if col == 0:
                    ax.set_ylabel('$y/D$', fontsize=8)
                    ax.annotate(info['title'], xy=(0, 0.5),
                                xytext=(-55, 0), xycoords='axes fraction',
                                textcoords='offset points', fontsize=8,
                                ha='center', va='center', rotation=90)
                else:
                    ax.set_yticklabels([])

                # x-label on bottom row only
                if row == 2:
                    ax.set_xlabel(r'$x/D$')
                else:
                    ax.set_xticklabels([])

        # Colorbar (skip for pure contour style)
        if im is not None and style != 'contour':
            cax = fig.add_subplot(gs[:, 4])
            fig.colorbar(im, cax=cax, label=fc['label'])

        suffix_tag = '' if style == 'color' else f'_{style}'
        outpath = os.path.join(OUTDIR, f"field_{fc['suffix']}{suffix_tag}.png")
        fig.savefig(outpath)
        plt.close(fig)
        print(f"  Saved {outpath}")


# ============================================================
# Figure 4: Line sampling – one composite figure
# 3 columns (geometries) × 5 rows (stations)
# ============================================================
def generate_line_figures():
    print("Generating line sampling figures...")

    stations = ['x5D', 'x10D', 'x15D']
    station_labels = [r'$x/D = 5$', r'$x/D = 10$', r'$x/D = 15$']
    station_xD = [5.0, 10.0, 15.0]
    line_keys = ['fine', 'coarse', 'dl_wake', 'grad_amr']

    fig, axes = plt.subplots(3, 3, figsize=(6.5, 5.5), sharey=True)

    for col, geom in enumerate(GEOM_ORDER):
        info = GEOMETRIES[geom]
        for row, station in enumerate(stations):
            ax = axes[row, col]
            for key in line_keys:
                fpath = os.path.join(
                    case_dir(geom, key),
                    f'postProcessing/sampleMean/400/wake_{station}_pMean_UMean.csv'
                )
                try:
                    df = pd.read_csv(fpath)
                    y = df['y'].values
                    u = df['UMean_0'].values
                    ax.plot(u, y, color=COLORS[key], linestyle=STYLES[key],
                            linewidth=LW[key], label=LABELS[key])
                except Exception:
                    # Fallback: extract line profile from VTK_mean
                    try:
                        vtu_path = find_vtk_mean_internal(geom, key)
                        if vtu_path is not None:
                            y_prof, u_prof = extract_line_from_vtk(
                                vtu_path, station_xD[row], y_range=(-3, 3),
                                n_points=200)
                            ax.plot(u_prof, y_prof, color=COLORS[key],
                                    linestyle=STYLES[key], linewidth=LW[key],
                                    label=LABELS[key])
                    except Exception:
                        pass

            ax.set_ylim(-3, 3)
            ax.tick_params(labelsize=7)

            # Row labels on left
            if col == 0:
                ax.set_ylabel(station_labels[row] + '\n' + r'$y/D$', fontsize=8)

            # Column titles on top
            if row == 0:
                ax.set_title(info['title'], fontsize=9)

            # x-label on bottom only
            if row == 2:
                ax.set_xlabel(r'$\overline{U}_x / U_\infty$', fontsize=8)

    # Single legend
    handles, labels = [], []
    for key in line_keys:
        h, = axes[0, 0].plot([], [], color=COLORS[key], linestyle=STYLES[key],
                              linewidth=LW[key], label=LABELS[key])
        handles.append(h)
        labels.append(LABELS[key])
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outpath = os.path.join(OUTDIR, 'line_sampling_all.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ============================================================
# Figure: Baseline sweep threshold sensitivity
# ============================================================
def generate_baseline_sweep_figure():
    print("Generating baseline sweep threshold sensitivity figure...")

    geometries = [
        ('circular', 'Circular ($Re=200$)',
         os.path.join(os.environ.get('DL_AMR_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')), 'baseline_sweep_circular.csv')),
        ('square', 'Square ($Re=150$)',
         os.path.join(os.environ.get('DL_AMR_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')), 'baseline_sweep_square.csv')),
        ('diamond', 'Diamond ($Re=150$)',
         os.path.join(os.environ.get('DL_AMR_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')), 'baseline_sweep_diamond.csv')),
    ]

    # Load data for each geometry
    geo_data = {}
    for geo_key, geo_label, csv_path in geometries:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Warning: could not load {geo_key} data: {e}")
            continue

        lower_thresh = []
        upper_thresh = []
        for c in df['case']:
            parts = c.split('_')
            l_str = [p for p in parts if p.startswith('L')][0]
            u_str = [p for p in parts if p.startswith('U')][0]
            lower_thresh.append(float(l_str[1:].replace('p', '.')))
            upper_thresh.append(float(u_str[1:].replace('p', '.')))

        df['lower_thresh'] = lower_thresh
        df['upper_thresh'] = upper_thresh
        geo_data[geo_key] = (geo_label, df)

    if not geo_data:
        print("  No data loaded; skipping figure.")
        return

    n_geo = len(geo_data)
    fig, axes = plt.subplots(n_geo, 3, figsize=(6.5, 2.3 * n_geo),
                             squeeze=False)

    for row, (geo_key, (geo_label, df)) in enumerate(geo_data.items()):
        unique_lower = sorted(df['lower_thresh'].unique())
        cmap = plt.cm.viridis
        norm = Normalize(vmin=min(unique_lower), vmax=max(unique_lower))

        for lt in unique_lower:
            sub = df[df['lower_thresh'] == lt].sort_values('upper_thresh')
            lbl = f'L={lt:.1f}'
            c = cmap(norm(lt))
            axes[row, 0].plot(sub['upper_thresh'], sub['cells_mean'] / 1000,
                              'o-', color=c, label=lbl, markersize=3, linewidth=0.9)
            axes[row, 1].plot(sub['upper_thresh'], sub['l2_mean'],
                              'o-', color=c, label=lbl, markersize=3, linewidth=0.9)
            axes[row, 2].plot(sub['upper_thresh'], sub['CD_err%'],
                              'o-', color=c, label=lbl, markersize=3, linewidth=0.9)

        # Row label on the left
        axes[row, 0].set_ylabel('Mean cells (×1000)')
        axes[row, 1].set_ylabel(r'$L_2(\overline{U}_x)$')
        axes[row, 2].set_ylabel(r'$\overline{C}_D$ error (%)')

        # Geometry label on leftmost y-axis (as text annotation)
        axes[row, 0].annotate(geo_label, xy=(0, 0.5),
                              xytext=(-0.55, 0.5),
                              xycoords='axes fraction',
                              textcoords='axes fraction',
                              fontsize=9, fontweight='bold',
                              ha='center', va='center',
                              rotation=90)

        for col in range(3):
            axes[row, col].legend(fontsize=5.5, loc='best')
            if row == n_geo - 1:
                axes[row, col].set_xlabel('upperRefineLevel')

    # Column headers on the first row only
    col_titles = ['Mesh size', 'Wake-field accuracy', 'Force accuracy']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10)

    fig.tight_layout()
    outpath = os.path.join(OUTDIR, 'baseline_threshold_sensitivity.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ============================================================
# Helper: compute vorticity on a uniform grid (artifact-free)
# ============================================================
def resample_Ux_on_grid(foam_or_vtu, time_value, xlim, ylim,
                        nx=600, ny=300, obstacle_shape=None):
    """
    Resample Ux onto a uniform grid. No gradient computation, so no
    artifacts at AMR refinement boundaries.
    """
    import pyvista as pv

    if isinstance(foam_or_vtu, tuple) and foam_or_vtu[0] == 'foam':
        reader = pv.OpenFOAMReader(foam_or_vtu[1])
        reader.set_active_time_value(time_value)
        multi = reader.read()
        if hasattr(multi, 'keys') and 'internalMesh' in multi.keys():
            mesh = multi['internalMesh']
        else:
            mesh = multi
    else:
        mesh = pv.read(foam_or_vtu)

    if 'U' in mesh.cell_data and 'U' not in mesh.point_data:
        mesh = mesh.cell_data_to_point_data()

    x, y = mesh.points[:, 0], mesh.points[:, 1]
    u_data = mesh.point_data['U']
    pad = 0.5
    m = ((x >= xlim[0] - pad) & (x <= xlim[1] + pad) &
         (y >= ylim[0] - pad) & (y <= ylim[1] + pad))

    xi = np.linspace(xlim[0], xlim[1], nx)
    yi = np.linspace(ylim[0], ylim[1], ny)
    Xi, Yi = np.meshgrid(xi, yi)
    Ux = griddata((x[m], y[m]), u_data[m, 0], (Xi, Yi), method='linear')

    # Mask obstacle interior
    if obstacle_shape is not None:
        hs = 0.55
        if obstacle_shape == 'circle':
            mask = Xi**2 + Yi**2 < hs**2
        elif obstacle_shape == 'square':
            mask = (np.abs(Xi) < hs) & (np.abs(Yi) < hs)
        elif obstacle_shape == 'diamond':
            mask = np.abs(Xi) + np.abs(Yi) < hs
        else:
            mask = np.zeros_like(Xi, dtype=bool)
        Ux[mask] = np.nan

    return Xi, Yi, Ux


def compute_vorticity_on_grid(foam_or_vtu, time_value, xlim, ylim,
                              nx=600, ny=300, obstacle_shape=None):
    """
    Resample U onto a uniform grid, then compute omega_z = dv/dx - du/dy
    via finite differences. This avoids artifacts at AMR refinement boundaries.
    """
    import pyvista as pv

    if isinstance(foam_or_vtu, tuple) and foam_or_vtu[0] == 'foam':
        reader = pv.OpenFOAMReader(foam_or_vtu[1])
        reader.set_active_time_value(time_value)
        multi = reader.read()
        if hasattr(multi, 'keys') and 'internalMesh' in multi.keys():
            mesh = multi['internalMesh']
        else:
            mesh = multi
    else:
        mesh = pv.read(foam_or_vtu)

    if 'U' in mesh.cell_data and 'U' not in mesh.point_data:
        mesh = mesh.cell_data_to_point_data()

    x, y = mesh.points[:, 0], mesh.points[:, 1]
    u_data = mesh.point_data['U']
    pad = 0.5
    m = ((x >= xlim[0] - pad) & (x <= xlim[1] + pad) &
         (y >= ylim[0] - pad) & (y <= ylim[1] + pad))

    xi = np.linspace(xlim[0], xlim[1], nx)
    yi = np.linspace(ylim[0], ylim[1], ny)
    Xi, Yi = np.meshgrid(xi, yi)

    Ux = griddata((x[m], y[m]), u_data[m, 0], (Xi, Yi), method='linear')
    Uy = griddata((x[m], y[m]), u_data[m, 1], (Xi, Yi), method='linear')
    dx, dy = xi[1] - xi[0], yi[1] - yi[0]
    vort = np.gradient(Uy, dx, axis=1) - np.gradient(Ux, dy, axis=0)

    # Mask obstacle interior
    if obstacle_shape is not None:
        hs = 0.55
        if obstacle_shape == 'circle':
            obs = Xi**2 + Yi**2 < hs**2
        elif obstacle_shape == 'square':
            obs = (np.abs(Xi) < hs) & (np.abs(Yi) < hs)
        elif obstacle_shape == 'diamond':
            obs = np.abs(Xi) + np.abs(Yi) < hs
        else:
            obs = None
        if obs is not None:
            vort[obs] = np.nan

    return Xi, Yi, vort


def extract_mesh_edges(foam_or_vtu, time_value, xlim, ylim):
    """Extract mesh edges as line segments for grid visualization."""
    import pyvista as pv
    from matplotlib.collections import LineCollection

    if isinstance(foam_or_vtu, tuple) and foam_or_vtu[0] == 'foam':
        reader = pv.OpenFOAMReader(foam_or_vtu[1])
        reader.set_active_time_value(time_value)
        multi = reader.read()
        if hasattr(multi, 'keys') and 'internalMesh' in multi.keys():
            mesh = multi['internalMesh']
        else:
            mesh = multi
    else:
        mesh = pv.read(foam_or_vtu)

    edges = mesh.extract_all_edges()
    pts = edges.points
    lines = edges.lines

    segments = []
    i = 0
    pad = 0.3
    while i < len(lines):
        n = lines[i]
        if n == 2:
            p0, p1 = pts[lines[i + 1]], pts[lines[i + 2]]
            if (xlim[0] - pad <= p0[0] <= xlim[1] + pad and
                xlim[0] - pad <= p1[0] <= xlim[1] + pad and
                ylim[0] - pad <= p0[1] <= ylim[1] + pad and
                ylim[0] - pad <= p1[1] <= ylim[1] + pad):
                segments.append([(p0[0], p0[1]), (p1[0], p1[1])])
        i += n + 1

    return LineCollection(segments)


# ============================================================
# Figure 11: Refinement snapshots (vorticity + grid)
# 3 rows (geometries) × 3 columns (fine vort, DL-AMR vort, DL-AMR grid)
# ============================================================
def generate_refinement_snapshots():
    """Single combined figure: 5 rows (variants) x 3 columns (geometries).
    Rows 0-2: Ux velocity for Fine, DL-AMR, |grad U| AMR.
    Rows 3-4: adapted mesh for DL-AMR, |grad U| AMR.
    """
    print("Generating refinement snapshots (5 row x 3 col)...")
    from matplotlib.collections import LineCollection

    xlim = (-1.5, 15)
    ylim = (-4, 4)
    t_snap = 100.0

    row_specs = [
        ('fine',     'ux',   'Fine'),
        ('dl_wake',  'ux',   'DL-AMR'),
        ('grad_amr', 'ux',   r'$|\nabla \mathbf{U}|$ AMR'),
        ('dl_wake',  'grid', 'DL-AMR mesh'),
        ('grad_amr', 'grid', r'$|\nabla \mathbf{U}|$ AMR mesh'),
    ]
    n_rows = len(row_specs)

    panel_aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    col_w = 6.5 / 3
    row_h = col_w / panel_aspect
    fig_h = n_rows * row_h + 0.6
    fig, axes = plt.subplots(n_rows, 3, figsize=(6.5, fig_h),
                             gridspec_kw={'hspace': 0.15, 'wspace': 0.08})

    # Global Ux color range from fine meshes
    global_vmax = 0.0
    for geom in GEOM_ORDER:
        info = GEOMETRIES[geom]
        foam_path = ('foam', os.path.join(case_dir(geom, 'fine'), 'open.foam'))
        try:
            _, _, Ux = resample_Ux_on_grid(
                foam_path, t_snap, xlim, ylim,
                obstacle_shape=info['shape'])
            vals = Ux[~np.isnan(Ux)]
            global_vmax = max(global_vmax, np.percentile(np.abs(vals), 99))
        except Exception as e:
            print(f"  Warning computing Ux range for {geom}: {e}")
    global_vmin = -global_vmax

    for row, (key, kind, row_title) in enumerate(row_specs):
        for col, geom in enumerate(GEOM_ORDER):
            info = GEOMETRIES[geom]
            ax = axes[row, col]
            foam_path = ('foam', os.path.join(case_dir(geom, key), 'open.foam'))

            if kind == 'ux':
                try:
                    Xi, Yi, Ux = resample_Ux_on_grid(
                        foam_path, t_snap, xlim, ylim,
                        obstacle_shape=info['shape'])
                    ax.pcolormesh(Xi, Yi, Ux, cmap='coolwarm',
                                  vmin=global_vmin, vmax=global_vmax,
                                  shading='auto', rasterized=True)
                    add_obstacle(ax, info['shape'])
                except Exception as e:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                            ha='center', va='center', fontsize=8, color='gray')
                    print(f"  Warning: {geom}/{key} Ux: {e}")
            else:  # grid
                try:
                    lc = extract_mesh_edges(foam_path, t_snap, xlim, ylim)
                    lc.set_color('#1a1a3a')
                    lc.set_linewidth(0.15)
                    lc.set_alpha(0.8)
                    ax.set_facecolor('#d8e8f0')
                    ax.add_collection(lc)
                    add_obstacle(ax, info['shape'], half_size=0.5)
                except Exception as e:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                            ha='center', va='center', fontsize=8, color='gray')
                    print(f"  Warning: {geom}/{key} grid: {e}")

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal')

            # Top row: geometry titles
            if row == 0:
                ax.set_title(info["title"].replace('\n', ' '), fontsize=9)
            # Left column: variant row labels
            if col == 0:
                ax.set_ylabel(f'{row_title}\n$y/D$', fontsize=9)
            else:
                ax.set_yticklabels([])
            # Bottom row: x-label
            if row == n_rows - 1:
                ax.set_xlabel('$x/D$', fontsize=9)
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.10, right=0.99, top=0.96, bottom=0.05)
    outpath = os.path.join(OUTDIR, 'refinement_snapshots_t100.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ============================================================
# Figure 12: DL-AMR vs gradient-based AMR (circular only)
# 2×2: (DL vort, grad vort) / (DL grid, grad grid)
# ============================================================
def generate_dl_vs_grad_figure():
    print("Generating DL vs gradient AMR comparison (Fig 12)...")
    from matplotlib.collections import LineCollection

    xlim = (-1.5, 15)
    ylim = (-4, 4)
    t_snap = 100.0
    geom = 'circular_Re200'
    info = GEOMETRIES[geom]

    panel_aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    col_w = 6.5 / 2
    row_h = col_w / panel_aspect
    fig_h = 2 * row_h + 0.6
    fig, axes = plt.subplots(2, 2, figsize=(6.5, fig_h),
                             gridspec_kw={'hspace': 0.18, 'wspace': 0.08})

    keys = ['dl_wake', 'grad_amr']
    titles_top = ['DL-AMR', r'$|\nabla \mathbf{U}|$ AMR']
    titles_bot = ['DL-AMR grid', r'$|\nabla \mathbf{U}|$ AMR grid']

    # Compute shared Ux color range
    global_vmax = 0
    for key in keys:
        foam_path = ('foam', os.path.join(case_dir(geom, key), 'open.foam'))
        try:
            _, _, Ux = resample_Ux_on_grid(
                foam_path, t_snap, xlim, ylim,
                obstacle_shape=info['shape'])
            vals = Ux[~np.isnan(Ux)]
            absmax = np.percentile(np.abs(vals), 99)
            global_vmax = max(global_vmax, absmax)
        except Exception as e:
            print(f"  Warning: {key} Ux range: {e}")
    global_vmin = -global_vmax

    for col, key in enumerate(keys):
        foam_path = ('foam', os.path.join(case_dir(geom, key), 'open.foam'))

        # Top row: Ux
        ax = axes[0, col]
        try:
            Xi, Yi, Ux = resample_Ux_on_grid(
                foam_path, t_snap, xlim, ylim,
                obstacle_shape=info['shape'])
            ax.pcolormesh(Xi, Yi, Ux, cmap='coolwarm',
                          vmin=global_vmin, vmax=global_vmax,
                          shading='auto', rasterized=True)
            add_obstacle(ax, info['shape'])
        except Exception as e:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center')
            print(f"  Warning: {key} vorticity: {e}")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(titles_top[col], fontsize=9)
        if col == 0:
            ax.set_ylabel('$y/D$')
        else:
            ax.set_yticklabels([])
        ax.set_xticklabels([])

        # Bottom row: grid
        ax = axes[1, col]
        try:
            lc = extract_mesh_edges(foam_path, t_snap, xlim, ylim)
            lc.set_color('#1a1a3a')
            lc.set_linewidth(0.15)
            lc.set_alpha(0.8)
            ax.set_facecolor('#d8e8f0')
            ax.add_collection(lc)
            add_obstacle(ax, info['shape'], half_size=0.5)
        except Exception as e:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center')
            print(f"  Warning: {key} grid: {e}")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(titles_bot[col], fontsize=9)
        if col == 0:
            ax.set_ylabel('$y/D$')
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('$x/D$')

    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.09)
    outpath = os.path.join(OUTDIR, 'circular_dl_vs_grad_t100.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _data_check import require_or_skip
    require_or_skip(
        'Figs 1, 7 (overview + line sampling)',
        'Run the OpenFOAM cases (e.g. `cd cases/circular_Re200/fine && ./Allrun`) '
        'or download a pre-computed case bundle into cases/ from Zenodo.',
        os.path.join(BASE, 'circular_Re200', 'fine', 'postProcessing',
                     'forceCoeffs', '0', 'coefficient.dat'),
    )

    print("=" * 60)
    print("DL-AMR Paper Figure Generation")
    print("=" * 60)

    generate_force_figures()
    generate_field_figures()
    generate_line_figures()
    generate_baseline_sweep_figure()
    generate_refinement_snapshots()
    generate_dl_vs_grad_figure()

    print("\nDone. Figures saved to:", OUTDIR)
