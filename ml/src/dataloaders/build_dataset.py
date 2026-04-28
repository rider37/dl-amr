#!/usr/bin/env python3
"""Build the preprocessed training dataset from OpenFOAM circular-cylinder
reference simulations.

Reads instantaneous snapshots from a sequence of completed OpenFOAM cases
(e.g., cases/circular_Re100/fine, cases/circular_Re150/fine), resamples each
snapshot to a uniform Cartesian grid covering the wake region
x/D in [2, 39], y/D in [-5, 5] at 64 x 224 resolution, computes the
per-channel temporal residual Delta q_t = q_{t+1} - q_t, and writes
train.pt / val.pt / test.pt under <output_dir>.

Usage
-----
    python -m ml.src.dataloaders.build_dataset \\
        --case  cases/circular_Re100/fine  \\
        --case  cases/circular_Re150/fine  \\
        --t-start 100 --t-end 300 --dt 1 \\
        --split-ratios 0.8 0.1 0.1 \\
        --output ml/data/processed/cylinder_delta_star_uvp_wake_re100_150

The output directory will contain:
    train.pt, val.pt, test.pt   (torch.save dicts with keys X, y, mask)
    norm_stats.json             (per-channel mean/std for inputs)
    target_norm.json            (per-channel mean/std for targets)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import griddata


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    p.add_argument("--case", action="append", required=True,
                   help="Path to a completed OpenFOAM case (repeatable).")
    p.add_argument("--t-start", type=float, default=100.0,
                   help="Earliest snapshot time to include.")
    p.add_argument("--t-end", type=float, default=300.0,
                   help="Latest snapshot time to include.")
    p.add_argument("--dt", type=float, default=1.0,
                   help="Snapshot interval (time units).")
    p.add_argument("--xlim", type=float, nargs=2, default=(2.0, 39.0),
                   help="Wake-region x/D bounds (default: 2 39).")
    p.add_argument("--ylim", type=float, nargs=2, default=(-5.0, 5.0),
                   help="Wake-region y/D bounds (default: -5 5).")
    p.add_argument("--nx", type=int, default=224,
                   help="Grid resolution in x (default: 224).")
    p.add_argument("--ny", type=int, default=64,
                   help="Grid resolution in y (default: 64).")
    p.add_argument("--split-ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--output", required=True,
                   help="Output directory for train.pt/val.pt/test.pt.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_snapshot(case_dir, time_value):
    """Load (Ux, Uy, p) at a given time from an OpenFOAM case via PyVista."""
    import pyvista as pv  # imported lazily so unrelated tooling does not need it
    foam = os.path.join(case_dir, "open.foam")
    if not os.path.exists(foam):
        Path(foam).touch()
    reader = pv.OpenFOAMReader(foam)
    times = np.asarray(reader.time_values)
    idx = int(np.argmin(np.abs(times - time_value)))
    reader.set_active_time_value(times[idx])
    multi = reader.read()
    mesh = multi["internalMesh"]
    if "U" in mesh.cell_data and "U" not in mesh.point_data:
        mesh = mesh.cell_data_to_point_data()
    U = mesh.point_data["U"]
    p = mesh.point_data["p"] if "p" in mesh.point_data else np.zeros(len(mesh.points))
    return mesh.points, U[:, 0], U[:, 1], p, float(times[idx])


def resample_uniform(pts, vals, xlim, ylim, nx, ny):
    x, y = pts[:, 0], pts[:, 1]
    pad = 0.5
    m = ((x >= xlim[0] - pad) & (x <= xlim[1] + pad) &
         (y >= ylim[0] - pad) & (y <= ylim[1] + pad))
    xi = np.linspace(xlim[0], xlim[1], nx)
    yi = np.linspace(ylim[0], ylim[1], ny)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x[m], y[m]), vals[m], (Xi, Yi), method="linear")
    nan_mask = np.isnan(Zi)
    if nan_mask.any():
        Zi[nan_mask] = griddata(
            (x[m], y[m]), vals[m], (Xi, Yi), method="nearest"
        )[nan_mask]
    return Xi, Yi, Zi.astype(np.float32)


def build_per_case(case_dir, t_start, t_end, dt, xlim, ylim, nx, ny):
    """Return a stack of (3, ny, nx) arrays for q_t and Delta q_t."""
    times = np.arange(t_start, t_end + 1e-9, dt)
    print(f"  [{case_dir}] loading {len(times)} snapshots...", flush=True)
    snaps = []
    actuals = []
    for t in times:
        try:
            pts, ux, uy, p, t_actual = load_snapshot(case_dir, t)
        except Exception as exc:
            print(f"    skip t={t}: {exc}")
            continue
        _, _, gx = resample_uniform(pts, ux, xlim, ylim, nx, ny)
        _, _, gy = resample_uniform(pts, uy, xlim, ylim, nx, ny)
        _, _, gp = resample_uniform(pts, p,  xlim, ylim, nx, ny)
        snaps.append(np.stack([gx, gy, gp], axis=0))
        actuals.append(t_actual)
    return np.stack(snaps, axis=0), np.asarray(actuals)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    Xs, Ys = [], []
    for case_dir in args.case:
        snaps, actuals = build_per_case(
            case_dir, args.t_start, args.t_end, args.dt,
            tuple(args.xlim), tuple(args.ylim), args.nx, args.ny,
        )
        if len(snaps) < 2:
            raise RuntimeError(f"Need at least 2 snapshots from {case_dir}")
        # input q_t and target Delta q_t = q_{t+1} - q_t
        Xs.append(snaps[:-1].astype(np.float32))
        Ys.append((snaps[1:] - snaps[:-1]).astype(np.float32))

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    N = X.shape[0]
    print(f"Total samples: {N}, X={X.shape}, Y={Y.shape}")

    # Random split
    idx = np.arange(N)
    rng.shuffle(idx)
    r_train, r_val, _ = args.split_ratios
    n_tr = int(N * r_train)
    n_va = int(N * r_val)
    splits = {
        "train": idx[:n_tr],
        "val":   idx[n_tr:n_tr + n_va],
        "test":  idx[n_tr + n_va:],
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-channel normalisation stats from training split
    Xt = X[splits["train"]]
    Yt = Y[splits["train"]]
    norm_stats = {
        "channel_mean": Xt.mean(axis=(0, 2, 3)).tolist(),
        "channel_std":  Xt.std(axis=(0, 2, 3)).tolist(),
    }
    target_norm = {
        "channel_mean": Yt.mean(axis=(0, 2, 3)).tolist(),
        "channel_std":  Yt.std(axis=(0, 2, 3)).tolist(),
    }
    (out_dir / "norm_stats.json").write_text(json.dumps(norm_stats, indent=2))
    (out_dir / "target_norm.json").write_text(json.dumps(target_norm, indent=2))

    mask = np.ones((args.ny, args.nx), dtype=np.float32)
    for split, ids in splits.items():
        payload = {
            "X":    torch.from_numpy(X[ids]),
            "y":    torch.from_numpy(Y[ids]),
            "mask": torch.from_numpy(np.broadcast_to(mask, (len(ids), args.ny, args.nx)).copy()),
        }
        torch.save(payload, out_dir / f"{split}.pt")
        print(f"  wrote {split}.pt: N={len(ids)}, X={tuple(payload['X'].shape)}")

    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
