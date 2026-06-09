"""
fig_specimen_coord_cte.py
==========================
Glassy and rubbery CTE expressed in the SPECIMEN coordinate system
(build direction vs raster direction), rather than the measurement frame.

The raw measurements are:
  εzz — through-thickness (always measured by vertical extensometers E0–E2)
  εyy — transverse (always measured by horizontal extensometers E3–E5)

Orientation mapping to specimen coordinates:
  O1: εzz → α_build,  εyy → α_raster
  O2: εzz → α_build,  εyy → α_raster
  O3: εzz → α_raster, εyy → α_build

Two panels per material:
  Left  — α_build   (glassy + rubbery side-by-side)
  Right — α_raster  (glassy + rubbery side-by-side)

Bar colours: material colour. Hatch/fill encodes crystallinity.
Error bars = 95 % CI. Replicate dots overlaid.

Data requirement: data_exports/metrics_df.csv

NOTE: Only the representation changes — the underlying data are unchanged.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from journal_style import (
    set_journal_style, save_figure, mm_to_inch,
    MATERIAL_ORDER, CRYST_ORDER,
    material_color, material_label,
    stable_rng, DATA_DIR,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIGURE_BASENAME = "fig_specimen_coord_cte"
FIG_WIDTH_MM    = 178
ROW_HEIGHT_MM   = 62

# Orientation → which measured direction maps to which specimen direction
# (zz = measured through-thickness, yy = measured transverse)
COORD_MAP = {
    "1": {"build": "zz", "raster": "yy"},
    "2": {"build": "zz", "raster": "yy"},
    "3": {"build": "yy", "raster": "zz"},
}

_MAT_INTERNAL = {"PEEK_CF": "PEEK-CF", "PPS_CF": "PPS-CF",
                 "PPS_NEAT": "PPS neat", "PEEK_NEAT": "PEEK neat"}

# ── Load data ──────────────────────────────────────────────────────────────────
csv = DATA_DIR / "metrics_df.csv"
if not csv.exists():
    raise FileNotFoundError(f"{csv}\nRun prepare_exports.py first.")

metrics_df = pd.read_csv(csv)
col_mat = "material_full" if "material_full" in metrics_df.columns else "material"
if metrics_df[col_mat].str.contains("_").any():
    metrics_df["material_full"] = metrics_df[col_mat].map(
        lambda x: _MAT_INTERNAL.get(x, x)
    )
else:
    metrics_df["material_full"] = metrics_df[col_mat]

# ── Build specimen-coordinate columns ──────────────────────────────────────────
def _specimen_cte(row, regime, spec_dir):
    """
    Return the CTE value in the given specimen direction (build or raster)
    for a given regime (glassy or rubbery), using COORD_MAP.
    """
    ori = str(row["orientation"])
    mapping = COORD_MAP.get(ori)
    if mapping is None:
        return np.nan
    meas_dir = mapping[spec_dir]          # "zz" or "yy"
    col = f"alpha_{regime}_{meas_dir}"    # e.g. alpha_lin_zz
    return row.get(col, np.nan)


for regime, tag in [("lin", "glassy"), ("rub", "rubbery")]:
    metrics_df[f"alpha_{tag}_build"]  = metrics_df.apply(
        lambda r: _specimen_cte(r, regime, "build"),  axis=1
    )
    metrics_df[f"alpha_{tag}_raster"] = metrics_df.apply(
        lambda r: _specimen_cte(r, regime, "raster"), axis=1
    )


# ── Helpers ────────────────────────────────────────────────────────────────────
def _ci95(vals):
    v = vals.dropna()
    if len(v) < 2:
        return 0.0
    se = v.std(ddof=1) / np.sqrt(len(v))
    return float(stats.t.ppf(0.975, df=len(v) - 1) * se)


def _bar_col(mat, cryst):
    mc = material_color(mat)
    return (mc, 0.85) if cryst == "High" else (mc, 0.35)


# ── Resolve grid ───────────────────────────────────────────────────────────────
materials = [m for m in MATERIAL_ORDER if m in metrics_df["material_full"].unique()]

SPEC_DIRS = [
    ("build",  r"$\alpha_{\rm build}$"),
    ("raster", r"$\alpha_{\rm raster}$"),
]
REGIMES = [
    ("glassy",   "Glassy"),
    ("rubbery",  "Rubbery"),
]

# ── Plot ───────────────────────────────────────────────────────────────────────
set_journal_style()

# Layout: n_mat rows × 2 cols (build | raster)
n_r = len(materials)
n_c = len(SPEC_DIRS)

fig, axes = plt.subplots(
    n_r, n_c,
    figsize=mm_to_inch(FIG_WIDTH_MM, ROW_HEIGHT_MM * n_r),
    sharey=False, sharex=False,
    squeeze=False,
)

bar_width = 0.25
regime_gap = 0.65   # space between glassy/rubbery groups
cryst_offset = 0.28  # offset between High/Low within a regime

for i_r, mat in enumerate(materials):
    mc      = material_color(mat)
    sub_mat = metrics_df[metrics_df["material_full"] == mat]
    rng     = stable_rng(mat, "spec_coord")

    for i_c, (spec_dir, ylabel) in enumerate(SPEC_DIRS):
        ax = axes[i_r, i_c]

        x_ticks  = []
        x_labels = []

        x_cursor = 0.0
        for i_regime, (regime_key, regime_lbl) in enumerate(REGIMES):
            dv = f"alpha_{regime_key}_{spec_dir}"

            # Check if this DV has any data at all
            if sub_mat[dv].isna().all():
                x_cursor += regime_gap
                continue

            for i_cryst, cryst in enumerate(CRYST_ORDER):
                sub_c = sub_mat[sub_mat["crystallinity"] == cryst][dv].dropna()
                if sub_c.empty:
                    continue
                x = x_cursor + i_cryst * cryst_offset
                col, al = _bar_col(mat, cryst)
                ax.bar(x, sub_c.mean(), width=bar_width * 0.9,
                       yerr=_ci95(sub_c), color=col, alpha=al,
                       edgecolor="k", linewidth=0.5, capsize=2, zorder=3)
                jit = rng.uniform(-0.03, 0.03, len(sub_c))
                ax.scatter(x + jit, sub_c.values, s=6, color="k",
                           alpha=0.55, edgecolor="white", linewidths=0.3,
                           zorder=4)

            x_mid = x_cursor + (len(CRYST_ORDER) - 1) * cryst_offset / 2
            x_ticks.append(x_mid)
            x_labels.append(regime_lbl)
            x_cursor += len(CRYST_ORDER) * cryst_offset + regime_gap

        ax.axhline(0, color="0.7", lw=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=9)

        if i_r == 0:
            ax.set_title(ylabel, fontsize=9, pad=4)
        if i_c == 0:
            ax.set_ylabel(
                f"{material_label(mat)}\n[ppm °C$^{{-1}}$]",
                fontsize=9, color=mc,
            )

# Orientation mapping note and legend
legend_handles = [
    Line2D([0], [0], color="0.3", marker="s", markersize=6,
           linestyle="None", alpha=0.85, label="High $T$"),
    Line2D([0], [0], color="0.3", marker="s", markersize=6,
           linestyle="None", alpha=0.35, label="Low $T$"),
]
fig.legend(handles=legend_handles, loc="lower center", frameon=False, ncol=2, fontsize=9)

# Add a small annotation explaining the coordinate mapping
note = ("Coord. map: O1,O2 → build=zz, raster=yy;   O3 → build=yy, raster=zz")
fig.text(0.5, 0.005, note, ha="center", fontsize=7, color="0.5",
         transform=fig.transFigure)

fig.tight_layout(rect=(0, 0.1, 1, 1))
save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
