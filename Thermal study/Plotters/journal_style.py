"""
journal_style.py
================
Shared style constants, colour maps, label helpers, and statistical utilities
for all standalone journal figure scripts.

All figure scripts must import from this module:
    from journal_style import *
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats

# ── Directories ────────────────────────────────────────────────────────────────
SCRIPTS_DIR   = Path(__file__).parent
OUTPUT_DIR    = SCRIPTS_DIR.parent          # Figures_Journal/
DATA_DIR      = SCRIPTS_DIR.parent / "data_exports"

# ── Figure sizes ───────────────────────────────────────────────────────────────
FIG_W_SINGLE_MM = 85
FIG_W_DOUBLE_MM = 178
FIG_H_SHORT_MM  = 65
FIG_H_TALL_MM   = 110

def mm_to_inch(*values):
    """Convert mm to inches for figsize."""
    if len(values) == 1:
        return values[0] / 25.4
    return tuple(v / 25.4 for v in values)

# ── Seaborn palette ────────────────────────────────────────────────────────────
SEABORN_PALETTE = "deep"

# ── Material system ────────────────────────────────────────────────────────────
# Canonical order and display labels.
# Internal key (material_full from clean pipeline): display label
MATERIAL_ORDER = ["PEEK-CF", "PPS-CF", "PPS neat"]
MATERIAL_LABELS = {
    "PEEK-CF":  "PEEK-CF",
    "PPS-CF":   "PPS-CF",
    "PPS neat": "PPS neat",
}

_palette = sns.color_palette(SEABORN_PALETTE, n_colors=len(MATERIAL_ORDER))
MATERIAL_COLORS = dict(zip(MATERIAL_ORDER, [c for c in _palette]))

# Polymer / fibre derived from material_full
def material_polymer(mat):
    return "PEEK" if "PEEK" in mat else "PPS"

def material_fibre(mat):
    return "CF" if "CF" in mat else "Neat"

# ── Orientation system ─────────────────────────────────────────────────────────
ORIENTATION_ORDER  = ["1", "2", "3"]
ORIENTATION_LABELS = {"1": "O1", "2": "O2", "3": "O3"}

# ── Crystallinity system ───────────────────────────────────────────────────────
CRYST_ORDER = ["High", "Low"]
CRYST_ALPHA = {"High": 1.0, "Low": 0.45}  # bar / fill opacity
CRYST_LS    = {"High": "-", "Low": "--"}   # line style

# ── CTE direction colours ──────────────────────────────────────────────────────
# Consistent across all figures involving α_zz / α_yy curves
COLOR_ZZ = "#1f3a93"   # dark navy — through-thickness direction
COLOR_YY = "#c0392b"   # crimson   — in-plane direction

# ── Specimen coordinate system ─────────────────────────────────────────────────
# Physical directions in the specimen frame.
# Mapping from DIC measurement direction to specimen direction per orientation:
#
#   Orientation 1:  DIC zz → build       DIC yy → transraster
#   Orientation 2:  DIC zz → build       DIC yy → raster
#   Orientation 3:  DIC zz → raster      DIC yy → build
#
SC_DIRECTION_ORDER  = ["raster", "transraster", "build"]
SC_DIRECTION_LABELS = {
    "build":       "Build",
    "raster":      "Raster",
    "transraster": "Transraster",
}
COLOR_BUILD       = "#1f3a93"   # navy
COLOR_RASTER      = "#c0392b"   # crimson
COLOR_TRANSRASTER = "#27ae60"   # green
SC_DIRECTION_COLORS = {
    "build":       COLOR_BUILD,
    "raster":      COLOR_RASTER,
    "transraster": COLOR_TRANSRASTER,
}

_ZZ_TO_SC = {"1": "build",  "2": "build",  "3": "raster"}
_YY_TO_SC = {"1": "transraster", "2": "raster", "3": "build"}

# Direction of heat flow through the specimen (through-thickness, perpendicular
# to the DIC camera face) for each orientation.
#   O1 face: (build, transraster) → heat flows through transraster
#   O2 face: (build, raster)      → heat flows through raster
#   O3 face: (raster, transraster)→ heat flows through build
_HEATED_DIRECTION = {"1": "transraster", "2": "raster", "3": "build"}

def zz_direction(ori):
    """Specimen direction measured by the DIC zz (vertical) sensors."""
    return _ZZ_TO_SC.get(str(ori), "zz")

def yy_direction(ori):
    """Specimen direction measured by the DIC yy (horizontal) sensors."""
    return _YY_TO_SC.get(str(ori), "yy")

def sc_direction_label(d):
    return SC_DIRECTION_LABELS.get(d, d)

def sc_direction_color(d):
    return SC_DIRECTION_COLORS.get(d, "0.5")

def heated_direction(ori):
    """Physical direction that heat flows through for this orientation.
    i.e. the specimen direction perpendicular to the DIC camera face.
    ⚠ Verify _HEATED_DIRECTION mapping against your experiment geometry."""
    return _HEATED_DIRECTION.get(str(ori), "unknown")

# ── Extensometer system ────────────────────────────────────────────────────────
VERT_SENSORS  = ["E0", "E1", "E2"]   # εzz — through thickness
HORIZ_SENSORS = ["E3", "E4", "E5"]   # εyy — transverse

EXT_COLORS_ZZ = {"E0": "#d62728", "E1": "#7f7f7f", "E2": "#1f77b4"}
EXT_COLORS_YY = {"E3": "#2ca02c", "E4": "#7f7f7f", "E5": "#ff7f0e"}
EXT_LABELS_ZZ = {
    "E0": "E0 (front, hot)",
    "E1": "E1 (mid)",
    "E2": "E2 (back, cool)",
}
EXT_LABELS_YY = {
    "E3": "E3 (top)",
    "E4": "E4 (mid)",
    "E5": "E5 (bottom, plate)",
}

# ── Physics constants ──────────────────────────────────────────────────────────
# Tg per material (material_full keys).  Fibre-reinforced and neat variants
# can differ — set each independently.  The polymer-level fallback dict is used
# if a material_full key is not found (e.g. for PEEK neat if it is re-added).
TG_MATERIAL = {
    "PEEK-CF":   143.0,   # °C  DSC
    "PPS-CF":     97.7,   # °C  DSC
    "PPS neat":   90.0,   # °C  DSC
    "PEEK neat": 143.0,   # °C  fallback
}
_TG_POLYMER_FALLBACK = {"PEEK": 143.0, "PPS": 90.0}   # °C
SPECIMEN_LENGTH_MM = 20.0

# ── Helper: label accessors ────────────────────────────────────────────────────
def material_color(mat):
    return MATERIAL_COLORS.get(mat, "0.5")

def material_label(mat):
    return MATERIAL_LABELS.get(mat, mat)

def orientation_label(ori):
    return ORIENTATION_LABELS.get(str(ori), f"O{ori}")

def tg_for(mat):
    """Return Tg [°C] for a material.  Looks up TG_MATERIAL by material_full
    first; falls back to polymer-level _TG_POLYMER_FALLBACK if not found."""
    if mat in TG_MATERIAL:
        return TG_MATERIAL[mat]
    return _TG_POLYMER_FALLBACK.get(material_polymer(mat), 100.0)

# ── Statistics ─────────────────────────────────────────────────────────────────
def ci95(values):
    """Return (mean, half-width of 95 % CI) for a 1-D array."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    if len(v) == 1:
        return float(v[0]), 0.0
    se = np.std(v, ddof=1) / np.sqrt(len(v))
    tc = stats.t.ppf(0.975, df=len(v) - 1)
    return float(np.mean(v)), float(tc * se)

def ci95_profile(stack):
    """Return (mean_profile, ci_halfwidth_profile) from (n_reps × n_t) array."""
    stack = np.asarray(stack, dtype=float)
    n = np.sum(np.isfinite(stack), axis=0).clip(min=1)
    mean = np.nanmean(stack, axis=0)
    sd   = np.nanstd(stack, axis=0, ddof=1)
    tc   = stats.t.ppf(0.975, df=np.maximum(n - 1, 1))
    ci   = np.where(n > 1, tc * sd / np.sqrt(np.maximum(n, 1)), 0.0)
    return mean, ci

def stable_rng(*args):
    """Seeded RNG from string args for reproducible jitter."""
    seed = hash(tuple(str(a) for a in args)) % (2**31)
    return np.random.default_rng(seed)

# ── Specimen name parser ───────────────────────────────────────────────────────
def parse_specimen_name(name):
    """
    Parse specimen name like PEEK_CF_H1a or PPS_NEAT_L23.
    Returns dict with material_full, polymer, fibre, crystallinity,
    orientation, replicate — or None on failure.
    """
    parts = str(name).split("_")
    if len(parts) < 3:
        return None
    raw_mat = "_".join(parts[:2]).upper()
    code    = parts[2]
    if len(code) < 2:
        return None
    c = code[0].upper()
    cryst = "High" if c == "H" else ("Low" if c == "L" else None)
    if cryst is None:
        return None
    ori = code[1] if code[1].isdigit() else None
    if ori is None:
        return None
    rep = code[2:] if len(code) > 2 else "1"

    # Map raw key → canonical material_full (clean pipeline convention)
    _mat_map = {
        "PEEK_CF":   "PEEK-CF",
        "PPS_CF":    "PPS-CF",
        "PPS_NEAT":  "PPS neat",
        "PEEK_NEAT": "PEEK neat",
    }
    mat_full = _mat_map.get(raw_mat, raw_mat)
    return dict(
        material_full = mat_full,
        polymer       = material_polymer(mat_full),
        fibre         = material_fibre(mat_full),
        crystallinity = cryst,
        orientation   = ori,
        replicate     = rep,
    )

# ── Temperature axis preference ────────────────────────────────────────────────
def temp_axis(res):
    """Prefer peak_temp (Q2) over mean_temp."""
    pk = res.get("peak_temp")
    if pk is not None and np.isfinite(np.asarray(pk, dtype=float)).any():
        return np.asarray(pk, dtype=float)
    mn = res.get("mean_temp")
    return np.asarray(mn, dtype=float) if mn is not None else None

# ── Mean strain from sensor group ─────────────────────────────────────────────
def mean_strain(res, sensors):
    """Average strain across sensors in the group."""
    vals = [res["strains"][s] for s in sensors if s in res.get("strains", {})]
    if not vals:
        return None
    return np.nanmean(np.vstack([np.asarray(v, float) for v in vals]), axis=0)

# ── Figure style ───────────────────────────────────────────────────────────────
def set_journal_style():
    sns.set_theme(context="paper", style="ticks", palette=SEABORN_PALETTE)
    plt.rcParams.update({
        "font.family":              "sans-serif",
        "font.sans-serif":          ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset":         "dejavusans",
        "font.size":                9,
        "axes.labelsize":           9,
        "axes.titlesize":           9,
        "xtick.labelsize":          9,
        "ytick.labelsize":          9,
        "legend.fontsize":          9,
        "axes.linewidth":           0.8,
        "lines.linewidth":          1.2,
        "lines.markersize":         4.0,
        "xtick.direction":          "out",
        "ytick.direction":          "out",
        "xtick.major.size":         3,
        "ytick.major.size":         3,
        "xtick.major.width":        0.8,
        "ytick.major.width":        0.8,
        "xtick.minor.size":         1.5,
        "ytick.minor.size":         1.5,
        "legend.frameon":           False,
        "legend.handlelength":      1.6,
        "legend.borderaxespad":     0.4,
        "axes.spines.top":          False,
        "axes.spines.right":        False,
        "axes.grid":                False,
        "figure.dpi":               150,
        "figure.constrained_layout.use": True,
        "savefig.dpi":              600,
        "savefig.bbox":             "tight",
        "savefig.pad_inches":       0.02,
        "axes.unicode_minus":       False,
        "pdf.fonttype":             42,
        "ps.fonttype":              42,
        "svg.fonttype":             "none",
    })

# ── Save helper ────────────────────────────────────────────────────────────────
def save_figure(fig, basename, out_dir=None):
    """Save PDF + SVG + PNG to out_dir (default OUTPUT_DIR)."""
    if out_dir is None:
        out_dir = OUTPUT_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{basename}.pdf")
    fig.savefig(out_dir / f"{basename}.svg")
    fig.savefig(out_dir / f"{basename}.png", dpi=600)
    print(f"Saved → {out_dir / basename}  [.pdf .svg .png]")

# ── Axis helpers ───────────────────────────────────────────────────────────────
def hide_top_right(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def label_panel(ax, letter, x=-0.12, y=1.04, fontsize=9):
    """Place a bold panel label (a), (b) etc. in axes coordinates."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            va="bottom", ha="right")

def significance_stars(p):
    if np.isnan(p):   return ""
    if p < 0.001:     return "***"
    if p < 0.01:      return "**"
    if p < 0.05:      return "*"
    return "ns"
