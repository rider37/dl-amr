"""Journal-aware CFD/AI figure creation and preflight.

Verified preset policy: 2026-07-13.

This module creates Matplotlib figures at intended publication size, audits
final-size readability and geometric collisions, exports vector/raster files,
and writes manuscript-scale previews and machine-readable audit reports.

The numerical width presets are working production presets. Journal-specific
instructions and current templates remain authoritative; see
``references/10_journal_figure_specs.md``.
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from cycler import cycler
from PIL import Image, ImageDraw

MM_PER_INCH = 25.4
_PT_PER_INCH = 72.0

# Okabe-Ito: colour-vision-deficiency-aware and printable.
SAFE_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]
SAFE_LINESTYLES = ["-", "--", "-.", ":"]
SAFE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
RISKY_CMAPS = {
    "jet", "rainbow", "gist_rainbow", "nipy_spectral", "hsv", "flag", "prism",
}

# official_width=False means a practical placement preset rather than a hard
# millimetre value printed in the current journal instructions.
JOURNAL_SPECS: dict[str, dict[str, Any]] = {
    "JCP": {
        "name": "Journal of Computational Physics",
        "publisher": "Elsevier",
        "width_mm": {"single": 90.0, "onehalf": 140.0, "double": 190.0},
        "official_width": False,
        "width_authority": "Elsevier working single/1.5/two-column presets; verify in current JCP template",
        "max_depth_mm": None,
        "dpi": {"line": 1000, "combo": 500, "half": 300},
        "vector": ["pdf", "eps"],
        "raster": ["tiff"],
        "color_mode": "rgb",
        "font_family": "sans-serif",
        "font_pt": 8.0,
        "min_font_pt": 7.0,
        "subscript_min_font_pt": 6.0,
        "line_pt": 1.0,
        "min_line_pt": 0.25,
        "absolute_min_line_pt": 0.1,
        "fonts_preferred": ["Arial", "Helvetica"],
        "verified_on": "2026-07-13",
    },
    "CPC": {
        "name": "Computer Physics Communications",
        "publisher": "Elsevier",
        "width_mm": {"single": 90.0, "onehalf": 140.0, "double": 190.0},
        "official_width": False,
        "width_authority": "Elsevier working single/1.5/two-column presets; verify in current CPC template",
        "max_depth_mm": None,
        "dpi": {"line": 1000, "combo": 500, "half": 300},
        "vector": ["pdf", "eps"],
        "raster": ["tiff"],
        "color_mode": "rgb",
        "font_family": "sans-serif",
        "font_pt": 8.0,
        "min_font_pt": 7.0,
        "subscript_min_font_pt": 6.0,
        "line_pt": 1.0,
        "min_line_pt": 0.25,
        "absolute_min_line_pt": 0.1,
        "fonts_preferred": ["Arial", "Helvetica"],
        "verified_on": "2026-07-13",
    },
    "JFM": {
        "name": "Journal of Fluid Mechanics",
        "publisher": "Cambridge University Press",
        "width_mm": {"single": 70.0, "onehalf": 105.0, "double": 140.0},
        "official_width": False,
        "width_authority": "Practical widths relative to the current JFM text measure; verify in compiled template",
        "max_depth_mm": 228.0,
        "dpi": {"line": 1000, "combo": 600, "half": 300},
        "vector": ["pdf", "eps"],
        "raster": ["tiff"],
        "color_mode": "rgb",  # current JFM-specific instruction
        "font_family": "sans-serif",
        "font_pt": 9.0,
        "min_font_pt": 9.0,
        "subscript_min_font_pt": 8.0,
        "line_pt": 1.0,
        "min_line_pt": 0.5,  # current JFM-specific instruction
        "absolute_min_line_pt": 0.5,
        "fonts_preferred": ["Arial", "Helvetica", "Times New Roman", "Times"],
        "verified_on": "2026-07-13",
    },
    "PoF": {
        "name": "Physics of Fluids",
        "publisher": "AIP Publishing",
        "width_mm": {"single": 85.0, "onehalf": 130.0, "double": 170.0},
        "official_width": True,
        "width_authority": "AIP: 85 mm one-column and 170 mm two-column; 130 mm is a working intermediate preset",
        "max_depth_mm": 211.0,
        "dpi": {"line": 600, "combo": 600, "half": 300},
        "vector": ["svg", "eps", "pdf"],
        "raster": ["tiff"],
        "color_mode": "rgb",
        "font_family": "sans-serif",
        "font_pt": 9.0,
        "min_font_pt": 8.0,
        "subscript_min_font_pt": 8.0,
        "line_pt": 1.0,
        "min_line_pt": 0.5,
        "absolute_min_line_pt": 0.5,
        "fonts_preferred": ["Arial", "Helvetica"],
        "verified_on": "2026-07-13",
    },
    "STRICT": {
        "name": "Conservative shared preset for JCP/CPC/JFM/PoF",
        "publisher": "Multiple",
        "width_mm": {"single": 70.0, "onehalf": 105.0, "double": 140.0},
        "official_width": False,
        "width_authority": "Smallest corresponding working width across supported journals",
        "max_depth_mm": 211.0,
        "dpi": {"line": 1000, "combo": 600, "half": 300},
        "vector": ["pdf", "eps"],
        "raster": ["tiff"],
        "color_mode": "rgb",
        "font_family": "sans-serif",
        "font_pt": 9.0,
        "min_font_pt": 9.0,
        "subscript_min_font_pt": 8.0,
        "line_pt": 1.0,
        "min_line_pt": 0.5,
        "absolute_min_line_pt": 0.5,
        "fonts_preferred": ["Arial", "Helvetica"],
        "verified_on": "2026-07-13",
    },
}

_FONT_FALLBACK = {
    "sans-serif": "DejaVu Sans",
    "serif": "DejaVu Serif",
}


@dataclass
class AuditIssue:
    severity: str
    code: str
    message: str
    objects: list[str] = field(default_factory=list)
    repair: str | None = None


@dataclass
class FigureAuditReport:
    journal: str
    width_mode: str
    native_size_mm: tuple[float, float]
    placed_width_mm: float
    scale_factor: float
    issues: list[AuditIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def add(self, severity: str, code: str, message: str,
            objects: Sequence[str] | None = None, repair: str | None = None) -> None:
        self.issues.append(AuditIssue(
            severity=severity,
            code=code,
            message=message,
            objects=list(objects or []),
            repair=repair,
        ))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["passed"] = self.passed
        data["error_count"] = len(self.errors)
        data["warning_count"] = len(self.warnings)
        return data

    def to_markdown(self) -> str:
        lines = [
            "# Figure preflight report",
            "",
            f"- Journal preset: `{self.journal}`",
            f"- Width mode: `{self.width_mode}`",
            f"- Native size: {self.native_size_mm[0]:.1f} × {self.native_size_mm[1]:.1f} mm",
            f"- Intended placed width: {self.placed_width_mm:.1f} mm",
            f"- Placement scale factor: {self.scale_factor:.3f}",
            f"- Status: **{'PASS' if self.passed else 'FAIL'}**",
            f"- Errors: {len(self.errors)}; warnings: {len(self.warnings)}",
            "",
        ]
        if not self.issues:
            lines.append("No issues detected by the automated checks. Manual scientific review is still required.")
            return "\n".join(lines) + "\n"
        lines += ["| Severity | Code | Finding | Repair |", "|---|---|---|---|"]
        for issue in self.issues:
            obj = f" Objects: {', '.join(issue.objects)}." if issue.objects else ""
            repair = (issue.repair or "Inspect manually").replace("|", "\\|")
            message = (issue.message + obj).replace("|", "\\|")
            lines.append(f"| {issue.severity} | `{issue.code}` | {message} | {repair} |")
        lines += [
            "",
            "## Limits of automation",
            "",
            "Bounding-box and pixel-density checks cannot determine which flow structure is scientifically important. Inspect the final-size and grayscale previews manually.",
        ]
        return "\n".join(lines) + "\n"


def list_journals() -> list[str]:
    return list(JOURNAL_SPECS)


def _resolve(journal: str) -> dict[str, Any]:
    want = journal.strip().lower()
    for key, value in JOURNAL_SPECS.items():
        if key.lower() == want:
            return value
    raise KeyError(f"Unknown journal {journal!r}; choose {', '.join(JOURNAL_SPECS)}")


def spec(journal: str) -> dict[str, Any]:
    return dict(_resolve(journal))


def figure_size(journal: str, width: str = "single", aspect: float = 0.72,
                height_mm: float | None = None,
                custom_width_mm: float | None = None) -> tuple[float, float]:
    s = _resolve(journal)
    if custom_width_mm is None:
        if width not in s["width_mm"]:
            raise ValueError(f"width must be one of {list(s['width_mm'])}")
        w_mm = float(s["width_mm"][width])
    else:
        if custom_width_mm <= 0:
            raise ValueError("custom_width_mm must be positive")
        w_mm = float(custom_width_mm)
    h_mm = float(height_mm) if height_mm is not None else w_mm * float(aspect)
    if s["max_depth_mm"] is not None and h_mm > s["max_depth_mm"]:
        warnings.warn(
            f"Requested height {h_mm:.1f} mm exceeds {journal} preset maximum "
            f"{s['max_depth_mm']:.1f} mm; clamping.", stacklevel=2)
        h_mm = float(s["max_depth_mm"])
    return w_mm / MM_PER_INCH, h_mm / MM_PER_INCH


def _font_stack(s: dict[str, Any]) -> list[str]:
    fam = s["font_family"]
    return list(s["fonts_preferred"]) + [_FONT_FALLBACK.get(fam, "DejaVu Sans")]


def apply_style(journal: str, *, font_pt: float | None = None,
                physics_ticks: bool = True, grid: bool = False) -> dict[str, Any]:
    s = _resolve(journal)
    base = max(float(font_pt or s["font_pt"]), float(s["min_font_pt"]))
    normal_small = max(base - 1.0, float(s["min_font_pt"]))
    stack = _font_stack(s)
    fam = s["font_family"]

    rc = {
        "font.family": fam,
        ("font.serif" if fam == "serif" else "font.sans-serif"): stack,
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": normal_small,
        "ytick.labelsize": normal_small,
        "legend.fontsize": normal_small,
        "figure.titlesize": base,
        "text.usetex": False,
        "mathtext.fontset": "stixsans" if fam != "serif" else "stix",
        "mathtext.default": "it",
        "axes.prop_cycle": cycler(color=SAFE_COLORS),
        "lines.linewidth": max(float(s["line_pt"]), float(s["min_line_pt"])),
        "lines.markersize": 4.5,
        "axes.linewidth": max(0.6, float(s["min_line_pt"])),
        "grid.linewidth": max(0.5, float(s["min_line_pt"])),
        "xtick.major.width": max(0.6, float(s["min_line_pt"])),
        "ytick.major.width": max(0.6, float(s["min_line_pt"])),
        "xtick.minor.width": max(0.5, float(s["min_line_pt"])),
        "ytick.minor.width": max(0.5, float(s["min_line_pt"])),
        "patch.linewidth": max(0.6, float(s["min_line_pt"])),
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": physics_ticks,
        "ytick.right": physics_ticks,
        "xtick.minor.visible": physics_ticks,
        "ytick.minor.visible": physics_ticks,
        "axes.grid": grid,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.02,
    }
    matplotlib.rcParams.update(rc)
    return s


def new_figure(journal: str, width: str = "single", aspect: float = 0.72,
               height_mm: float | None = None,
               custom_width_mm: float | None = None,
               layout: str = "constrained", **style_kwargs: Any):
    apply_style(journal, **style_kwargs)
    size = figure_size(journal, width, aspect, height_mm, custom_width_mm)
    fig, ax = plt.subplots(figsize=size, layout=layout)
    fig._jf_meta = {"journal": journal, "width": width, "custom_width_mm": custom_width_mm}  # type: ignore[attr-defined]
    return fig, ax


def new_figure_grid(journal: str, nrows: int, ncols: int, *, width: str = "double",
                    aspect: float = 0.62, height_mm: float | None = None,
                    custom_width_mm: float | None = None,
                    sharex: bool | str = False, sharey: bool | str = False,
                    layout: str = "constrained", squeeze: bool = True,
                    **style_kwargs: Any):
    if nrows < 1 or ncols < 1:
        raise ValueError("nrows and ncols must be positive")
    apply_style(journal, **style_kwargs)
    size = figure_size(journal, width, aspect, height_mm, custom_width_mm)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=size, layout=layout,
        sharex=sharex, sharey=sharey, squeeze=squeeze,
    )
    fig._jf_meta = {"journal": journal, "width": width, "custom_width_mm": custom_width_mm}  # type: ignore[attr-defined]
    return fig, axs


def style_series(lines: Sequence[Line2D], *, use_markers: bool = False,
                 marker_every: int | None = None) -> None:
    """Apply redundant colour/line-style/marker coding to existing lines."""
    for i, line in enumerate(lines):
        line.set_color(SAFE_COLORS[i % len(SAFE_COLORS)])
        line.set_linestyle(SAFE_LINESTYLES[(i // len(SAFE_COLORS)) % len(SAFE_LINESTYLES)])
        if use_markers:
            line.set_marker(SAFE_MARKERS[i % len(SAFE_MARKERS)])
            if marker_every is not None:
                line.set_markevery(marker_every)


def add_panel_labels(axs: Iterable[Axes] | Axes, *, labels: Sequence[str] | None = None,
                     x: float = 0.02, y: float = 0.98, fontsize: float | None = None,
                     weight: str = "bold", box: bool = False) -> list[Text]:
    axes = [axs] if isinstance(axs, Axes) else list(_flatten_axes(axs))
    if labels is None:
        labels = [f"({chr(97 + i)})" for i in range(len(axes))]
    if len(labels) != len(axes):
        raise ValueError("labels length must match number of axes")
    out = []
    for ax, label in zip(axes, labels):
        kwargs: dict[str, Any] = {}
        if box:
            kwargs["bbox"] = {"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0}
        t = ax.text(
            x, y, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=fontsize, fontweight=weight, clip_on=False, zorder=20, **kwargs,
        )
        t.set_gid("panel-label")
        out.append(t)
    return out


def mark_comparison_group(fig, axes: Sequence[Axes], *, require_xy_limits: bool = False,
                          require_clim: bool = True, require_cmap: bool = True,
                          require_equal_box: bool = True) -> None:
    groups = getattr(fig, "_jf_comparison_groups", [])
    groups.append({
        "axes": list(axes),
        "require_xy_limits": require_xy_limits,
        "require_clim": require_clim,
        "require_cmap": require_cmap,
        "require_equal_box": require_equal_box,
    })
    fig._jf_comparison_groups = groups  # type: ignore[attr-defined]


def set_shared_clim(mappables: Sequence[Artist], vmin: float | None = None,
                    vmax: float | None = None, *, symmetric: bool = False) -> tuple[float, float]:
    arrays = []
    for m in mappables:
        arr = getattr(m, "get_array", lambda: None)()
        if arr is not None:
            arrays.append(arr)
    if not arrays and (vmin is None or vmax is None):
        raise ValueError("No mappable arrays available to infer limits")
    if vmin is None:
        vmin = min(float(a.min()) for a in arrays)
    if vmax is None:
        vmax = max(float(a.max()) for a in arrays)
    if symmetric:
        bound = max(abs(vmin), abs(vmax))
        vmin, vmax = -bound, bound
    for m in mappables:
        setter = getattr(m, "set_clim", None)
        if setter:
            setter(vmin, vmax)
    return float(vmin), float(vmax)


def _flatten_axes(obj: Any) -> Iterable[Axes]:
    if isinstance(obj, Axes):
        yield obj
    elif hasattr(obj, "flat"):
        yield from obj.flat
    else:
        for item in obj:
            yield from _flatten_axes(item)


def _is_colorbar_axes(ax: Axes) -> bool:
    return ax.get_label() == "<colorbar>" or hasattr(ax, "_colorbar")


def _primary_axes(fig) -> list[Axes]:
    return [ax for ax in fig.axes if not _is_colorbar_axes(ax) and ax.get_visible()]


def _bbox_area(b: Bbox) -> float:
    return max(0.0, b.width) * max(0.0, b.height)


def _intersection_area(a: Bbox, b: Bbox) -> float:
    inter = Bbox.intersection(a, b)
    return 0.0 if inter is None else _bbox_area(inter)


def _contains(outer: Bbox, inner: Bbox, tol: float = 1.0) -> bool:
    return (inner.x0 >= outer.x0 - tol and inner.y0 >= outer.y0 - tol and
            inner.x1 <= outer.x1 + tol and inner.y1 <= outer.y1 + tol)


def _artist_name(artist: Artist, role: str | None = None) -> str:
    if isinstance(artist, Text):
        txt = artist.get_text().replace("\n", " ")
        txt = txt if len(txt) <= 40 else txt[:37] + "..."
        return f"{role or 'text'}:{txt!r}"
    return f"{role or artist.__class__.__name__}"


def _text_records(fig, renderer) -> list[dict[str, Any]]:
    roles: dict[int, tuple[str, Axes | None]] = {}
    for ax in fig.axes:
        roles[id(ax.title)] = ("title", ax)
        roles[id(ax.xaxis.label)] = ("xlabel", ax)
        roles[id(ax.yaxis.label)] = ("ylabel", ax)
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            roles[id(t)] = ("tick", ax)
        legend = ax.get_legend()
        if legend:
            for t in legend.get_texts():
                roles[id(t)] = ("legend", ax)
        for t in ax.texts:
            role = "panel-label" if t.get_gid() == "panel-label" else "annotation"
            roles[id(t)] = (role, ax)
    for t in fig.texts:
        roles.setdefault(id(t), ("figure-text", None))

    records = []
    for t in fig.findobj(match=lambda a: isinstance(a, Text)):
        if not t.get_visible() or not t.get_text().strip():
            continue
        try:
            bbox = t.get_window_extent(renderer=renderer)
        except Exception:
            continue
        role, owner = roles.get(id(t), ("text", getattr(t, "axes", None)))
        records.append({"artist": t, "bbox": bbox, "role": role, "axes": owner})
    return records



def _background_complexity(fig, records: Sequence[dict[str, Any]], renderer) -> dict[int, dict[str, float]]:
    """Heuristic pixel complexity beneath text/legends with text hidden.

    Returns non-white occupancy and luminance standard deviation for each text
    artist. This cannot identify scientific importance; it only flags visually
    busy backgrounds where text may hide or be confused with plotted content.
    """
    text_artists = [rec["artist"] for rec in records]
    legends = [a for a in fig.findobj(match=lambda a: isinstance(a, Legend)) if a.get_visible()]
    text_states = [(a, a.get_visible()) for a in text_artists]
    legend_states = [(a, a.get_visible()) for a in legends]
    try:
        for artist, _ in text_states:
            artist.set_visible(False)
        for artist, _ in legend_states:
            artist.set_visible(False)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        # Matplotlib display coordinates use lower-left origin; arrays use upper-left.
        height, width = rgba.shape[:2]
        result: dict[int, dict[str, float]] = {}
        for rec in records:
            b = rec["bbox"]
            x0 = max(0, int(math.floor(b.x0)))
            x1 = min(width, int(math.ceil(b.x1)))
            y0 = max(0, int(math.floor(height - b.y1)))
            y1 = min(height, int(math.ceil(height - b.y0)))
            if x1 <= x0 or y1 <= y0:
                continue
            crop = rgba[y0:y1, x0:x1, :3].astype(float) / 255.0
            lum = 0.2126 * crop[..., 0] + 0.7152 * crop[..., 1] + 0.0722 * crop[..., 2]
            occupancy = float(np.mean(lum < 0.97))
            result[id(rec["artist"])] = {
                "occupancy": occupancy,
                "luminance_std": float(np.std(lum)),
            }
        return result
    finally:
        for artist, state in text_states:
            artist.set_visible(state)
        for artist, state in legend_states:
            artist.set_visible(state)
        fig.canvas.draw()

def _native_mm(fig) -> tuple[float, float]:
    w, h = fig.get_size_inches()
    return float(w) * MM_PER_INCH, float(h) * MM_PER_INCH


def _axis_box_mm(ax: Axes, fig) -> tuple[float, float]:
    bbox = ax.get_window_extent()
    return bbox.width / fig.dpi * MM_PER_INCH, bbox.height / fig.dpi * MM_PER_INCH


def _resolved_font_names(records: Sequence[dict[str, Any]]) -> list[str]:
    names = set()
    for rec in records:
        try:
            names.add(rec["artist"].get_fontproperties().get_name())
        except Exception:
            pass
    return sorted(names)


def _pairwise_color_issues(ax: Axes, report: FigureAuditReport) -> None:
    lines = [ln for ln in ax.lines if ln.get_visible() and ln.get_label() != "_nolegend_"]
    if len(lines) < 2:
        return
    entries = []
    for ln in lines:
        try:
            rgb = mcolors.to_rgb(ln.get_color())
        except ValueError:
            continue
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        entries.append((ln, rgb, lum))
    for (l1, c1, y1), (l2, c2, y2) in itertools.combinations(entries, 2):
        dist = math.dist(c1, c2)
        same_style = (l1.get_linestyle() == l2.get_linestyle() and
                      l1.get_marker() == l2.get_marker())
        if (dist < 0.14 or abs(y1 - y2) < 0.06) and same_style:
            report.add(
                "WARNING", "SERIES_NOT_REDUNDANT",
                "Two plotted series have similar colour/luminance and identical line/marker coding.",
                [_artist_name(l1, "line"), _artist_name(l2, "line")],
                "Change line style or marker so the series remain distinguishable in grayscale and for colour-vision deficiency.",
            )


def _check_comparison_groups(fig, report: FigureAuditReport, renderer) -> None:
    for gi, group in enumerate(getattr(fig, "_jf_comparison_groups", []), start=1):
        axes = [ax for ax in group["axes"] if ax in fig.axes]
        if len(axes) < 2:
            continue
        if group.get("require_equal_box"):
            boxes = [ax.get_window_extent(renderer) for ax in axes]
            widths = [b.width for b in boxes]
            heights = [b.height for b in boxes]
            if max(widths) / min(widths) > 1.03 or max(heights) / min(heights) > 1.03:
                report.add("WARNING", "UNEQUAL_COMPARISON_BOXES",
                           f"Comparison group {gi} has unequal panel data-box dimensions.",
                           repair="Use a shared GridSpec/colour bar so directly compared panels have equal data areas.")
        if group.get("require_xy_limits"):
            xlims = [tuple(round(v, 12) for v in ax.get_xlim()) for ax in axes]
            ylims = [tuple(round(v, 12) for v in ax.get_ylim()) for ax in axes]
            if len(set(xlims)) > 1 or len(set(ylims)) > 1:
                report.add("WARNING", "INCONSISTENT_COMPARISON_LIMITS",
                           f"Comparison group {gi} does not use common x/y limits.",
                           repair="Use identical limits or explain why different domains/scales are required.")
        mappables = []
        for ax in axes:
            if ax.images:
                mappables.append(ax.images[0])
            elif ax.collections:
                candidates = [c for c in ax.collections if hasattr(c, "get_clim")]
                if candidates:
                    mappables.append(candidates[0])
        if len(mappables) >= 2:
            if group.get("require_clim"):
                clims = [tuple(round(v, 12) for v in m.get_clim()) for m in mappables]
                if len(set(clims)) > 1:
                    report.add("ERROR", "INCONSISTENT_CLIM",
                               f"Comparison group {gi} uses different colour limits.",
                               repair="Set common colour limits for magnitude comparison or explicitly remove the direct-comparison declaration.")
            if group.get("require_cmap"):
                cmaps = [m.get_cmap().name for m in mappables]
                if len(set(cmaps)) > 1:
                    report.add("WARNING", "INCONSISTENT_CMAP",
                               f"Comparison group {gi} uses different colormaps.",
                               repair="Use one colormap for the same variable or justify the distinction.")


def audit_figure(fig, journal: str, *, width: str = "single",
                 placed_width_mm: float | None = None,
                 width_tolerance_mm: float = 0.8,
                 min_panel_width_mm: float = 25.0,
                 min_panel_height_mm: float = 18.0,
                 min_data_area_fraction: float = 0.40,
                 check_missing_labels: bool = True) -> FigureAuditReport:
    """Audit final-size geometry, typography, lines, colour, and comparison groups.

    ``placed_width_mm`` models later scaling in LaTeX/Word. Prefer creating the
    source figure directly at the placed width so the scale factor is near one.
    """
    s = _resolve(journal)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    native_w, native_h = _native_mm(fig)
    target_w = float(placed_width_mm if placed_width_mm is not None else s["width_mm"][width])
    scale = target_w / native_w
    report = FigureAuditReport(
        journal=journal,
        width_mode=width,
        native_size_mm=(round(native_w, 3), round(native_h, 3)),
        placed_width_mm=target_w,
        scale_factor=scale,
    )

    target_preset = float(s["width_mm"][width])
    if placed_width_mm is None and abs(native_w - target_preset) > width_tolerance_mm:
        report.add("ERROR", "WIDTH_MISMATCH",
                   f"Native width {native_w:.1f} mm does not match the {journal} {width} preset {target_preset:.1f} mm.",
                   repair="Create the figure with new_figure/new_figure_grid or pass the actual intended placed_width_mm.")
    if abs(scale - 1.0) > 0.05:
        report.add("WARNING", "EXTERNAL_SCALING",
                   f"The figure will be scaled by {scale:.3f}; all fonts and strokes change by the same factor.",
                   repair="Regenerate at the intended placement width rather than relying on manuscript scaling.")
    placed_h = native_h * scale
    if s["max_depth_mm"] is not None and placed_h > float(s["max_depth_mm"]) + 0.5:
        report.add("ERROR", "HEIGHT_EXCEEDS_LIMIT",
                   f"Placed height {placed_h:.1f} mm exceeds the preset maximum {s['max_depth_mm']:.1f} mm.",
                   repair="Reduce height, reorganize panels, or split the figure.")

    records = _text_records(fig, renderer)
    font_sizes = [float(r["artist"].get_fontsize()) * scale for r in records]
    if font_sizes:
        min_font = min(font_sizes)
        report.metrics["effective_font_pt_min"] = round(min_font, 3)
        if min_font + 1e-6 < float(s["min_font_pt"]):
            report.add("ERROR", "FONT_TOO_SMALL",
                       f"Smallest effective text is {min_font:.2f} pt; preset minimum is {s['min_font_pt']:.1f} pt.",
                       repair="Increase the native font or create the figure at final placement size.")
    report.metrics["resolved_fonts"] = _resolved_font_names(records)

    canvas = Bbox.from_bounds(0, 0, fig.bbox.width, fig.bbox.height)
    for rec in records:
        b = rec["bbox"]
        if not _contains(canvas, b, tol=1.5):
            report.add("ERROR", "TEXT_CLIPPED",
                       "Visible text extends beyond the exported canvas.",
                       [_artist_name(rec["artist"], rec["role"])],
                       "Move the text inward, increase layout space, shorten the label, or reorganize the figure.")

    # Text-text collision check.
    for a, b in itertools.combinations(records, 2):
        area = _intersection_area(a["bbox"], b["bbox"])
        if area <= 0:
            continue
        min_area = min(_bbox_area(a["bbox"]), _bbox_area(b["bbox"]))
        if min_area <= 0 or area < max(4.0, 0.025 * min_area):
            continue
        report.add("ERROR", "TEXT_OVERLAP",
                   "Two visible text elements overlap at final render size.",
                   [_artist_name(a["artist"], a["role"]), _artist_name(b["artist"], b["role"])],
                   "Shorten labels, reduce tick density, move the legend/annotation, share labels, or enlarge/reorganize the layout.")

    # Text from one panel entering another panel's data rectangle.
    axis_boxes = {ax: ax.get_window_extent(renderer) for ax in fig.axes if ax.get_visible()}
    for rec in records:
        owner = rec["axes"]
        for ax, ab in axis_boxes.items():
            if owner is ax:
                continue
            area = _intersection_area(rec["bbox"], ab)
            if area > 0.05 * max(1.0, _bbox_area(rec["bbox"])):
                report.add("ERROR", "TEXT_CROSSES_PANEL",
                           "Text associated with one panel intrudes into another panel's data area.",
                           [_artist_name(rec["artist"], rec["role"])],
                           "Increase inter-panel spacing, shorten the label, or use shared figure-level labels.")
                break

    # Heuristic text/legend-on-data check. Only annotations, panel labels, and
    # legends inside their owning data axes are considered.
    complexity = _background_complexity(fig, records, renderer)
    for rec in records:
        if rec["role"] not in {"annotation", "panel-label", "legend"}:
            continue
        owner = rec["axes"]
        if owner is None or owner not in axis_boxes:
            continue
        if _intersection_area(rec["bbox"], axis_boxes[owner]) < 0.5 * max(1.0, _bbox_area(rec["bbox"])):
            continue
        stats = complexity.get(id(rec["artist"]))
        if not stats:
            continue
        patch = rec["artist"].get_bbox_patch() if isinstance(rec["artist"], Text) else None
        protected = patch is not None and (patch.get_alpha() is None or float(patch.get_alpha()) >= 0.65)
        if stats["occupancy"] > 0.10 or stats["luminance_std"] > 0.16:
            sev = "INFO" if protected and rec["role"] == "panel-label" else "WARNING"
            report.add(
                sev, "TEXT_ON_COMPLEX_DATA",
                f"{rec['role']} lies over a visually complex data region "
                f"(occupancy={stats['occupancy']:.0%}, luminance std={stats['luminance_std']:.2f}).",
                [_artist_name(rec["artist"], rec["role"])],
                "Move it to an empty margin/outside the axes, or use an opaque high-contrast box/halo only when covering the underlying data is acceptable.",
            )

    # Axes rectangle collisions. Full containment is treated as an intentional inset;
    # near-identical boxes are treated as twinned axes.
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    for ax1, ax2 in itertools.combinations(visible_axes, 2):
        b1, b2 = axis_boxes[ax1], axis_boxes[ax2]
        inter = _intersection_area(b1, b2)
        if inter <= 0:
            continue
        frac = inter / min(_bbox_area(b1), _bbox_area(b2))
        if frac > 0.95 or _contains(b1, b2, 1.0) or _contains(b2, b1, 1.0):
            continue
        report.add("ERROR", "AXES_OVERLAP",
                   f"Two axes/colour-bar rectangles overlap by {frac:.0%} of the smaller area.",
                   repair="Use GridSpec/constrained layout or a shared colour bar; do not position axes manually on top of each other.")

    primary = _primary_axes(fig)
    fig_area = max(1.0, fig.bbox.width * fig.bbox.height)
    primary_area = sum(_bbox_area(axis_boxes[ax]) for ax in primary)
    area_fraction = primary_area / fig_area
    report.metrics["primary_axes_area_fraction"] = round(area_fraction, 4)
    if area_fraction < min_data_area_fraction:
        report.add("WARNING", "LOW_DATA_AREA",
                   f"Primary data axes occupy only {area_fraction:.0%} of the figure canvas.",
                   repair="Remove repeated labels/titles, share legends/colour bars, reduce empty margins, or simplify/split the figure.")

    panel_sizes = []
    for idx, ax in enumerate(primary, start=1):
        w_mm, h_mm = _axis_box_mm(ax, fig)
        w_mm *= scale
        h_mm *= scale
        panel_sizes.append((round(w_mm, 2), round(h_mm, 2)))
        if w_mm < min_panel_width_mm:
            report.add("WARNING", "PANEL_TOO_NARROW",
                       f"Panel {idx} data region is only {w_mm:.1f} mm wide at final placement.",
                       repair="Use fewer columns, a wider placement, shared labels, or split the figure.")
        if h_mm < min_panel_height_mm:
            report.add("WARNING", "PANEL_TOO_SHORT",
                       f"Panel {idx} data region is only {h_mm:.1f} mm high at final placement.",
                       repair="Increase height, reduce rows, or split the figure.")
        ratio = w_mm / max(h_mm, 1e-9)
        if ratio > 5.0 or ratio < 0.20:
            report.add("WARNING", "EXTREME_PANEL_ASPECT",
                       f"Panel {idx} data-region aspect ratio is {ratio:.2f}.",
                       repair="Confirm that the geometry/phenomenon requires this aspect ratio; otherwise rebalance the panel.")

        # Prominent lines, spines, and markers after placement scaling.
        widths = [float(ln.get_linewidth()) * scale for ln in ax.lines if ln.get_visible()]
        widths += [float(sp.get_linewidth()) * scale for sp in ax.spines.values() if sp.get_visible()]
        nonzero = [w for w in widths if w > 0]
        if nonzero and min(nonzero) + 1e-6 < float(s["min_line_pt"]):
            report.add("ERROR", "LINE_TOO_THIN",
                       f"Panel {idx} contains an effective stroke of {min(nonzero):.2f} pt; minimum is {s['min_line_pt']:.2f} pt.",
                       repair="Increase line/spine width or regenerate at final size.")
        markers = [float(ln.get_markersize()) * scale for ln in ax.lines
                   if ln.get_visible() and ln.get_marker() not in (None, "None", "", " ")]
        if markers and min(markers) < 3.0:
            report.add("WARNING", "MARKER_TOO_SMALL",
                       f"Panel {idx} contains an effective marker size below 3 pt.",
                       repair="Increase marker size or reduce the number of markers.")

        # Legend footprint.
        leg = ax.get_legend()
        if leg and leg.get_visible():
            lb = leg.get_window_extent(renderer)
            ab = axis_boxes[ax]
            inside = _intersection_area(lb, ab) / max(1.0, _bbox_area(lb))
            footprint = _bbox_area(lb) / max(1.0, _bbox_area(ab))
            if inside > 0.9 and footprint > 0.22:
                report.add("WARNING", "LARGE_IN_AXES_LEGEND",
                           f"Panel {idx} legend occupies {footprint:.0%} of the data rectangle.",
                           repair="Move the legend outside, use a shared figure legend, shorten labels, or directly label curves.")

        _pairwise_color_issues(ax, report)

        for mappable in list(ax.images) + [c for c in ax.collections if hasattr(c, "get_cmap")]:
            try:
                name = mappable.get_cmap().name
            except Exception:
                continue
            if name in RISKY_CMAPS:
                report.add("WARNING", "RISKY_COLORMAP",
                           f"Panel {idx} uses the nonuniform/rainbow-style colormap `{name}`.",
                           repair="Use a perceptually uniform sequential/diverging/cyclic map appropriate to the physical variable.")

    report.metrics["panel_data_size_mm"] = panel_sizes

    if len(primary) > 1:
        panel_labels = [r for r in records if r["role"] == "panel-label" or re.fullmatch(r"\([a-zA-Z]\)", r["artist"].get_text().strip())]
        if len(panel_labels) < len(primary):
            report.add("WARNING", "MISSING_PANEL_LABELS",
                       f"Detected {len(primary)} primary panels but only {len(panel_labels)} panel labels.",
                       repair="Label all parts consistently as (a), (b), … in reading order.")

    if check_missing_labels and len(primary) == 1:
        ax = primary[0]
        has_data = bool(ax.lines or ax.images or ax.collections or ax.patches)
        if has_data and ax.axison:
            if not ax.get_xlabel().strip():
                report.add("WARNING", "MISSING_XLABEL", "The data axes has no x-axis label.",
                           repair="Add the physical quantity and units/nondimensionalization, or explicitly disable the axis for a pure field view.")
            if not ax.get_ylabel().strip():
                report.add("WARNING", "MISSING_YLABEL", "The data axes has no y-axis label.",
                           repair="Add the physical quantity and units/nondimensionalization, or explicitly disable the axis for a pure field view.")

    _check_comparison_groups(fig, report, renderer)
    report.metrics["native_size_mm"] = [round(native_w, 3), round(native_h, 3)]
    report.metrics["placed_size_mm"] = [round(target_w, 3), round(placed_h, 3)]
    report.metrics["width_authority"] = s["width_authority"]
    return report


def write_audit_report(report: FigureAuditReport, basename: str | os.PathLike[str]) -> dict[str, str]:
    base = str(basename)
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    json_path = base + ".json"
    md_path = base + ".md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report.to_markdown())
    return {"audit_json": json_path, "audit_md": md_path}


def _render_rgba(fig, dpi: int) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=False, bbox_inches=None)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def save_print_preview(fig, basename: str, *, dpi: int = 150,
                       grayscale: bool = True,
                       target_width_mm: float | None = None) -> dict[str, str]:
    Path(basename).parent.mkdir(parents=True, exist_ok=True)
    image = _render_rgba(fig, dpi)
    if target_width_mm is not None:
        target_px = max(1, round(target_width_mm / MM_PER_INCH * dpi))
        if target_px != image.width:
            target_h = max(1, round(image.height * target_px / image.width))
            image = image.resize((target_px, target_h), Image.Resampling.LANCZOS)
    path = basename + "_final-size.png"
    image.convert("RGB").save(path, dpi=(dpi, dpi))
    out = {"preview": path}
    if grayscale:
        gray_path = basename + "_grayscale.png"
        image.convert("L").save(gray_path, dpi=(dpi, dpi))
        out["preview_grayscale"] = gray_path
    return out


def save_manuscript_preview(fig, basename: str, journal: str, *, width: str,
                            dpi: int = 150, page_width_mm: float = 210.0,
                            page_height_mm: float = 297.0,
                            margin_mm: float = 10.0,
                            placed_width_mm: float | None = None) -> str:
    """Place the figure on a simple manuscript-page mockup at 100% size."""
    s = _resolve(journal)
    figure_img = _render_rgba(fig, dpi).convert("RGB")
    page_px = (round(page_width_mm / MM_PER_INCH * dpi),
               round(page_height_mm / MM_PER_INCH * dpi))
    page = Image.new("RGB", page_px, "white")
    draw = ImageDraw.Draw(page)
    margin_px = round(margin_mm / MM_PER_INCH * dpi)
    physical_width_mm = float(placed_width_mm if placed_width_mm is not None else s["width_mm"][width])
    col_width_px = round(physical_width_mm / MM_PER_INCH * dpi)
    x0 = margin_px
    x1 = min(page_px[0] - margin_px, x0 + col_width_px)
    y0 = margin_px * 2
    draw.rectangle([x0, margin_px, x1, page_px[1] - margin_px], outline=(220, 220, 220), width=1)
    # Preserve the native size in pixels at the preview DPI.
    if figure_img.width > x1 - x0:
        new_h = round(figure_img.height * (x1 - x0) / figure_img.width)
        figure_img = figure_img.resize((x1 - x0, new_h), Image.Resampling.LANCZOS)
    paste_x = x0 + max(0, (x1 - x0 - figure_img.width) // 2)
    page.paste(figure_img, (paste_x, y0))
    caption_y = min(page_px[1] - margin_px, y0 + figure_img.height + round(5 / MM_PER_INCH * dpi))
    # Neutral caption placeholder lines; no dependence on host fonts.
    for i, frac in enumerate((0.96, 0.92, 0.82)):
        yy = caption_y + i * max(2, round(3 / MM_PER_INCH * dpi))
        draw.line([x0, yy, x0 + round((x1 - x0) * frac), yy], fill=(185, 185, 185), width=1)
    path = basename + "_manuscript-preview.png"
    page.save(path, dpi=(dpi, dpi))
    return path


def _save_vectors(fig, base: str, formats: Sequence[str]) -> dict[str, str]:
    out = {}
    for fmt in formats:
        path = f"{base}.{fmt}"
        fig.savefig(path, format=fmt, transparent=False, bbox_inches=None)
        out[fmt] = path
    return out


def _save_tiff(fig, base: str, dpi: int, color_mode: str) -> str:
    image = _render_rgba(fig, dpi)
    mode = color_mode.lower()
    if mode in {"grayscale", "gray", "grey", "l"}:
        image = image.convert("L")
    elif mode == "cmyk":
        image = image.convert("RGB").convert("CMYK")
    else:
        image = image.convert("RGB")
    path = base + ".tiff"
    image.save(path, format="TIFF", compression="tiff_lzw", dpi=(float(dpi), float(dpi)))
    return path


def verify_exports(files: dict[str, Any], expected_dpi: int | None = None) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    tiff = files.get("tiff")
    if tiff and os.path.exists(tiff):
        with Image.open(tiff) as im:
            checks["tiff"] = {
                "mode": im.mode,
                "size_px": list(im.size),
                "dpi": list(im.info.get("dpi", (None, None))),
                "compression": im.info.get("compression"),
                "dpi_matches": expected_dpi is None or all(abs(float(d) - expected_dpi) < 0.5 for d in im.info.get("dpi", (0, 0))),
            }
    pdf = files.get("pdf")
    pdffonts = shutil.which("pdffonts")
    if pdf and os.path.exists(pdf):
        if pdffonts:
            proc = subprocess.run([pdffonts, pdf], capture_output=True, text=True, check=False)
            checks["pdf_fonts"] = {
                "checked": True,
                "returncode": proc.returncode,
                "output": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        else:
            checks["pdf_fonts"] = {"checked": False, "reason": "pdffonts command not available"}
    return checks


def save_figure(fig, basename: str, journal: str, *, width: str = "single",
                art_type: str = "combo", color_mode: str | None = None,
                vector: bool = True, tiff: bool = True,
                svg: bool | None = None, audit: bool = True,
                placed_width_mm: float | None = None,
                fail_on: str | None = "error", previews: bool = True,
                close: bool = False) -> dict[str, Any]:
    """Audit and export a figure without changing the fixed outer dimensions."""
    s = _resolve(journal)
    if art_type not in {"line", "combo", "half"}:
        raise ValueError("art_type must be line, combo, or half")
    base = str(basename)
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    dpi = int(s["dpi"][art_type])
    cmode = (color_mode or s["color_mode"]).lower()
    result: dict[str, Any] = {
        "journal": journal,
        "width_mode": width,
        "dpi": dpi,
        "color_mode": cmode,
        "size_mm": tuple(round(v, 2) for v in _native_mm(fig)),
    }

    report = None
    if audit:
        report = audit_figure(fig, journal, width=width, placed_width_mm=placed_width_mm)
        result.update(write_audit_report(report, base + "_audit"))
        result["audit_passed"] = report.passed
        result["audit_errors"] = len(report.errors)
        result["audit_warnings"] = len(report.warnings)
        if previews:
            preview_width = float(placed_width_mm if placed_width_mm is not None else s["width_mm"][width])
            result.update(save_print_preview(fig, base, target_width_mm=preview_width))
            result["manuscript_preview"] = save_manuscript_preview(
                fig, base, journal, width=width, placed_width_mm=preview_width
            )
        if fail_on and fail_on.lower() == "error" and report.errors:
            raise ValueError(
                f"Figure preflight failed with {len(report.errors)} error(s). "
                f"See {result['audit_md']} and the generated previews before exporting final files."
            )

    if vector:
        formats = list(s["vector"])
        if svg is True and "svg" not in formats:
            formats.append("svg")
        if svg is False and "svg" in formats:
            formats.remove("svg")
        result.update(_save_vectors(fig, base, formats))
    if tiff:
        result["tiff"] = _save_tiff(fig, base, dpi, cmode)
    if previews and "preview" not in result:
        preview_width = float(placed_width_mm if placed_width_mm is not None else s["width_mm"][width])
        result.update(save_print_preview(fig, base, target_width_mm=preview_width))
        result["manuscript_preview"] = save_manuscript_preview(
            fig, base, journal, width=width, placed_width_mm=preview_width
        )
    result["verification"] = verify_exports(result, expected_dpi=dpi)

    if close:
        plt.close(fig)
    return result


def _demo(outdir: str) -> None:
    import numpy as np

    Path(outdir).mkdir(parents=True, exist_ok=True)
    x = np.linspace(0, 2 * np.pi, 300)
    fig, axs = new_figure_grid("STRICT", 1, 2, width="double", aspect=0.48, sharex=False)
    lines = []
    for i, phase in enumerate((0.0, 0.4, 0.8)):
        lines += axs[0].plot(x, np.sin(x + phase), label=fr"case {i + 1}")
    style_series(lines, use_markers=True, marker_every=45)
    axs[0].set_xlim(0, 2 * np.pi)
    axs[0].set_ylim(-1.15, 1.15)
    axs[0].set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    axs[0].set_yticks([-1, -0.5, 0, 0.5, 1])
    axs[0].set_xlabel(r"$x/D$")
    axs[0].set_ylabel(r"$u/U_\infty$")
    axs[0].legend(frameon=False, ncol=1)

    xx, yy = np.meshgrid(np.linspace(-2, 4, 100), np.linspace(-2, 2, 70))
    field = np.sin(xx) * np.exp(-0.4 * yy**2)
    im = axs[1].pcolormesh(xx, yy, field, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axs[1].set_xlim(-2.2, 4.2)
    axs[1].set_ylim(-2.2, 2.2)
    axs[1].set_xticks([-2, 0, 2, 4])
    axs[1].set_yticks([-2, -1, 0, 1, 2])
    axs[1].set_xlabel(r"$x/D$")
    axs[1].set_ylabel(r"$y/D$")
    fig.colorbar(im, ax=axs[1], label=r"$\omega_z D/U_\infty$")
    add_panel_labels(axs, box=True)
    info = save_figure(fig, str(Path(outdir) / "demo"), "STRICT", width="double", art_type="combo", close=True)
    print(json.dumps(info, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Journal-aware CFD/AI figure formatter and preflight")
    parser.add_argument("--list", action="store_true", help="List presets")
    parser.add_argument("--demo", action="store_true", help="Create an audited demo")
    parser.add_argument("--outdir", default="demo_out")
    args = parser.parse_args()
    if args.list:
        for key, s in JOURNAL_SPECS.items():
            print(
                f"{key:7s} {s['name']} | widths={s['width_mm']} mm | "
                f"font>={s['min_font_pt']} pt | line>={s['min_line_pt']} pt | "
                f"dpi={s['dpi']} | colour={s['color_mode']}"
            )
    if args.demo:
        _demo(args.outdir)
    if not (args.list or args.demo):
        parser.print_help()


if __name__ == "__main__":
    main()
