"""
fig_table_specimen_coord.py
============================
CTE in the SPECIMEN coordinate system (build / raster / transraster),
derived from TABLE_thermal_expansion.csv by remapping each measurement axis
to the appropriate specimen direction.

Orientation–to–specimen-direction mapping
------------------------------------------
  O1 : zz → build,      yy → transraster
  O2 : zz → build,      yy → raster
  O3 : yy → build,      zz → raster
  (transraster is measured only via O1_yy)

Each specimen direction is therefore sampled by multiple orientations:
  build      ← O1_zz,  O2_zz,  O3_yy
  raster     ← O2_yy,  O3_zz
  transraster← O1_yy

Layout
------
  2 rows   : glassy (30–75 °C) | rubbery (above Tg)
  3 columns: one per material
  x-axis   : specimen direction (Build / Raster / Transraster)
  Within each group: High χ (solid) and Low χ (hatched)
  Bar height  = mean across contributing orientations
  Error bar   = 95 % CI of the contributing orientation means (between-orientation SD)
  Scatter dots= individual orientation contributions (coloured by orientation)

Data: TABLE_thermal_expansion.csv (placed in the same folder as this script)

Run standalone:
    python3 fig_table_specimen_coord.py
"""

import sys
import re
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as _stats

sys.path.insert(0, str(Path(__file__).parent))
from journal_style import (
    set_journal_style, save_figure, mm_to_inch,
    material_color, material_label, DATA_DIR,
)

# ── Configuration ───────────────────────────────────────────────────────────────
FIGURE_BASENAME = "fig_table_specimen_coord"
FIG_WIDTH_MM    = 178
ROW_HEIGHT_MM   = 68

CSV_CANDIDATES = [
    Path(__file__).parent / "TABLE_thermal_expansion.csv",
    DATA_DIR / "TABLE_thermal_expansion.csv",
    Path(__file__).parent.parent / "TABLE_thermal_expansion.csv",
]

# ── Specimen-direction mapping ──────────────────────────────────────────────────
# Each entry: (orientation, measured_axis) → specimen_direction
MEAS_TO_SPEC = {
    ("O1", "zz"): "build",
    ("O1", "yy"): "transraster",
    ("O2", "zz"): "build",
    ("O2", "yy"): "raster",
    ("O3", "yy"): "build",
    ("O3", "zz"): "raster",
    # transraster is NOT directly measured in O2 or O3
}

SPEC_DIRS    = ["build", "raster", "transraster"]
SPEC_LABELS  = {"build": "Build", "raster": "Raster", "transraster": "Transraster"}
SPEC_COLORS  = {"build": "#1f3a93", "raster": "#c0392b", "transraster": "#27ae60"}

MAT_ORDER    = ["PEEK CF", "PPS CF", "PPS NEAT"]
MAT_LABELS   = {"PEEK CF": "PEEK-CF", "PPS CF": "PPS-CF", "PPS NEAT": "PPS neat"}
MAT_CLEAN    = {"PEEK CF": "PEEK-CF", "PPS CF": "PPS-CF", "PPS NEAT": "PPS neat"}
ORI_ORDER    = ["O1", "O2", "O3"]
CRYST_ORDER  = ["High", "Low"]

ORI_MARKERS  = {"O1": "o", "O2": "s", "O3": "^"}
ORI_MCOLORS  = {"O1": "#555555", "O2": "#888888", "O3": "#aaaaaa"}

CRYST_ALPHA  = {"High": 0.85, "Low": 0.30}
CRYST_HATCH  = {"High": "",   "Low": "///"}

REGIME_SPECS = [
    ("g", "Glassy\n(30–75 °C)"),
    ("r", "Rubbery\n(above $T_g$)"),
]

N_PER_GROUP  = 5   # replicates per (mat, ori, cryst) condition


# ── Parse helpers ───────────────────────────────────────────────────────────────
def _parse_val(s):
    """Return (mean, sd) from '93.4±17.7', or (nan, nan) from '—'."""
    s = str(s).strip()
    if s in ("—", "-", "", "nan", "—"):
        return np.nan, np.nan
    m = re.match(r"([+-]?\d+\.?\d*)\s*[±]\s*(\d+\.?\d*)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(s), 0.0
    except ValueError:
        return np.nan, np.nan


# ── Load CSV ────────────────────────────────────────────────────────────────────
csv_path = next((c for c in CSV_CANDIDATES if c.exists()), None)
if csv_path is None:
    raise FileNotFoundError(
        "TABLE_thermal_expansion.csv not found.\n"
        f"Tried: {[str(c) for c in CSV_CANDIDATES]}"
    )

records = []
with open(csv_path, newline="", encoding="utf-8-sig") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        mat   = row["Material"].strip()
        ori   = row["Orientation"].strip()
        cryst = row["Crystallinity"].strip()
        n_rep = int(row["n"].strip()) if row.get("n", "").strip().isdigit() else N_PER_GROUP
        zz_g_m, zz_g_s = _parse_val(row.get("α_zz glassy [ppm/K]", "—"))
        yy_g_m, yy_g_s = _parse_val(row.get("α_yy glassy [ppm/K]", "—"))
        zz_r_m, zz_r_s = _parse_val(row.get("α_zz rubbery [ppm/K]", "—"))
        yy_r_m, yy_r_s = _parse_val(row.get("α_yy rubbery [ppm/K]", "—"))
        records.append(dict(
            mat=mat, ori=ori, cryst=cryst, n=n_rep,
            zz_g=zz_g_m, zz_g_sd=zz_g_s,
            yy_g=yy_g_m, yy_g_sd=yy_g_s,
            zz_r=zz_r_m, zz_r_sd=zz_r_s,
            yy_r=yy_r_m, yy_r_sd=yy_r_s,
        ))


def _ori_value(rec, regime, meas_axis):
    """Return (mean, sem) for one orientation–axis–regime combination."""
    key = f"{meas_axis}_{regime}"        # e.g. "zz_g"
    mu  = rec.get(key, np.nan)
    sd  = rec.get(f"{key}_sd", 0.0)
    n   = rec["n"]
    sem = sd / np.sqrt(n) if (np.isfinite(sd) and n > 0) else 0.0
    return mu, sem


def _pool(values):
    """Pool a list of (mean, sem) tuples.
    Returns (grand_mean, 95% CI half-width) treating orientation means as observations."""
    vals = np.array([v for v, _ in values if np.isfinite(v)])
    if len(vals) == 0:
        return np.nan, 0.0
    if len(vals) == 1:
        return float(vals[0]), float(values[0][1] * _stats.t.ppf(0.975, df=max(values[0][1]-1, 1)))
    grand_mean = float(np.mean(vals))
    # Between-orientation SE
    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    tc = float(_stats.t.ppf(0.975, df=len(vals) - 1))
    return grand_mean, se * tc


# ── Collect per-direction values ─────────────────────────────────────────────────
def _direction_data(mat, cryst, regime, spec_dir):
    """
    Return (grand_mean, ci, ori_points) for one (mat, cryst, regime, spec_dir).
    ori_points: list of (ori_label, mean_val) for individual scatter dots.
    """
    contributing = [(ori, axis)
                    for (ori, axis), sd in MEAS_TO_SPEC.items()
                    if sd == spec_dir]
    ori_points = []
    pool_vals  = []
    for ori, axis in contributing:
        recs = [r for r in records
                if r["mat"] == mat and r["ori"] == ori and r["cryst"] == cryst]
        if not recs:
            continue
        rec = recs[0]
        mu, sem = _ori_value(rec, regime, axis)
        if np.isfinite(mu):
            pool_vals.append((mu, sem))
            ori_points.append((ori, mu))
    gm, ci = _pool(pool_vals)
    return gm, ci, ori_points


# ── Plot ─────────────────────────────────────────────────────────────────────────
set_journal_style()

mats_present = [m for m in MAT_ORDER if any(r["mat"] == m for r in records)]
n_r = len(REGIME_SPECS)
n_c = len(mats_present)

fig, axes = plt.subplots(
    n_r, n_c,
    figsize=mm_to_inch(FIG_WIDTH_MM, ROW_HEIGHT_MM * n_r),
    sharey=False, sharex=True,
    squeeze=False,
)

bar_w      = 0.28   # individual bar width
cryst_off  = 0.30   # offset between High / Low within one spec_dir
dir_gap    = 1.05   # spacing between spec_dir groups

for i_r, (regime_key, regime_lbl) in enumerate(REGIME_SPECS):
    for i_c, mat in enumerate(mats_present):
        ax = axes[i_r, i_c]
        mc = material_color(MAT_CLEAN.get(mat, mat))

        x_centres   = []   # centre of each spec_dir group
        x_tick_pos  = []
        x_tick_lbl  = []
        any_bar     = False

        for i_sd, spec_dir in enumerate(SPEC_DIRS):
            group_x = i_sd * dir_gap
            col     = SPEC_COLORS[spec_dir]

            for i_cryst, cryst in enumerate(CRYST_ORDER):
                gm, ci, ori_pts = _direction_data(mat, cryst, regime_key, spec_dir)
                if not np.isfinite(gm):
                    continue
                x = group_x + (i_cryst - 0.5) * cryst_off
                alpha = CRYST_ALPHA[cryst]
                hatch = CRYST_HATCH[cryst]
                ax.bar(x, gm, width=bar_w * 0.92,
                       color=col, alpha=alpha,
                       hatch=hatch, edgecolor="k", linewidth=0.5,
                       zorder=3)
                ax.errorbar(x, gm, yerr=ci,
                            fmt="none", ecolor="k", elinewidth=0.7,
                            capsize=2.5, capthick=0.7, zorder=4)
                # Individual orientation dots
                for ori, val in ori_pts:
                    ax.scatter(x, val,
                               marker=ORI_MARKERS[ori], s=14,
                               color=ORI_MCOLORS[ori], zorder=5,
                               edgecolors="white", linewidths=0.3)
                any_bar = True

            x_tick_pos.append(group_x)
            x_tick_lbl.append(SPEC_LABELS[spec_dir])

        if not any_bar and regime_key == "r":
            ax.text(0.5, 0.5, "rubbery plateau\nnot reached",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="0.55")

        ax.axhline(0, color="0.7", lw=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(x_tick_pos)
        ax.set_xticklabels(x_tick_lbl, fontsize=9)

        if i_r == 0:
            ax.set_title(MAT_LABELS.get(mat, mat), fontsize=9,
                         color=mc, pad=4)
        if i_c == 0:
            ax.set_ylabel(f"{regime_lbl}\n" + r"$\alpha$ [ppm K$^{-1}$]",
                          fontsize=9)

# ── Legend ────────────────────────────────────────────────────────────────────
# Direction colours
dir_patches = [
    mpatches.Patch(color=SPEC_COLORS[sd], alpha=0.85,
                   label=SPEC_LABELS[sd])
    for sd in SPEC_DIRS
]
# Crystallinity
cryst_patches = [
    mpatches.Patch(facecolor="0.55", alpha=CRYST_ALPHA["High"],
                   edgecolor="k", lw=0.5, label="High $T$"),
    mpatches.Patch(facecolor="0.55", alpha=CRYST_ALPHA["Low"],
                   edgecolor="k", lw=0.5, hatch="///", label="Low $T$"),
]
# Orientation markers
ori_handles = [
    plt.Line2D([0], [0], marker=ORI_MARKERS[o], color="w",
               markerfacecolor=ORI_MCOLORS[o], markeredgecolor="0.3",
               markersize=5, label=f"{o}")
    for o in ORI_ORDER
]

all_handles = dir_patches + cryst_patches + ori_handles
fig.legend(handles=all_handles, loc="lower center", frameon=False, fontsize=9,
                  handlelength=1.2, ncol=2)

# Mapping footnote
note = (r"O1: $\varepsilon_{zz}$=build, $\varepsilon_{yy}$=transraster  |  "
        r"O2: $\varepsilon_{zz}$=build, $\varepsilon_{yy}$=raster  |  "
        r"O3: $\varepsilon_{yy}$=build, $\varepsilon_{zz}$=raster.  "
        r"Bar = mean across orientations; error bar = 95 % CI; dots = per-orientation mean ($n$=5).")
fig.text(0.5, 0.002, note, ha="center", fontsize=6.5, color="0.45",
         transform=fig.transFigure)

fig.tight_layout(rect=(0, 0.1, 1, 1))
save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
print("Done.")
