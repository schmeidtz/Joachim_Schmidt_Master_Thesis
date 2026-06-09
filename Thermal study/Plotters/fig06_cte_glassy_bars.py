"""
fig06_cte_glassy_bars.py
=========================
Glassy-regime CTE bar chart (30–75 °C).
Two rows: αzz (top) and αyy (bottom).
Columns = materials.  Groups = orientations.
High χ = full colour, Low χ = muted.  Replicate dots overlaid.

Data requirement: data_exports/metrics_df.csv
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
    MATERIAL_ORDER, ORIENTATION_ORDER, CRYST_ORDER,
    material_color, material_label, orientation_label,
    stable_rng, DATA_DIR,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIGURE_BASENAME = "fig06_cte_glassy_bars"
FIG_WIDTH_MM    = 178
ROW_HEIGHT_MM   = 55

# Map clean-pipeline names → augmented internal names
_MAT_INTERNAL = {
    "PEEK-CF":  "PEEK_CF",
    "PPS-CF":   "PPS_CF",
    "PPS neat": "PPS_NEAT",
}

# ── Load data ──────────────────────────────────────────────────────────────────
csv = DATA_DIR / "metrics_df.csv"
if not csv.exists():
    raise FileNotFoundError(f"{csv}\nRun prepare_exports.py first.")

metrics_df = pd.read_csv(csv)
# Normalise material_full to clean-pipeline naming if stored as augmented form
_inv = {v: k for k, v in _MAT_INTERNAL.items()}
if metrics_df["material_full"].str.contains("_").any():
    metrics_df["material_full"] = metrics_df["material_full"].map(
        lambda x: _inv.get(x, x)
    )


# ── Helpers ────────────────────────────────────────────────────────────────────
def _ci95(vals):
    v = vals.dropna()
    if len(v) < 2:
        return 0.0
    se = v.std(ddof=1) / np.sqrt(len(v))
    return float(stats.t.ppf(0.975, df=len(v) - 1) * se)


def _bar_color(mat, cryst):
    mc = material_color(mat)
    if cryst == "High":
        return mc, 0.85
    else:
        return mc, 0.35


# ── Resolve grid ───────────────────────────────────────────────────────────────
materials    = [m for m in MATERIAL_ORDER if m in metrics_df["material_full"].unique()]
orientations = [o for o in ORIENTATION_ORDER
                if str(o) in metrics_df["orientation"].astype(str).unique()]

DVs = [("alpha_lin_zz", r"$\alpha_{zz}^{\rm glassy}$ [ppm °C$^{-1}$]"),
       ("alpha_lin_yy", r"$\alpha_{yy}^{\rm glassy}$ [ppm °C$^{-1}$]")]

# ── Plot ───────────────────────────────────────────────────────────────────────
set_journal_style()

n_r, n_c = len(DVs), len(materials)
fig, axes = plt.subplots(
    n_r, n_c,
    figsize=mm_to_inch(FIG_WIDTH_MM, ROW_HEIGHT_MM * n_r),
    sharex=False, sharey="row",
    squeeze=False,
)

bar_width = 0.32
group_gap = 0.80

for i_c, mat in enumerate(materials):
    sub_mat = metrics_df[metrics_df["material_full"] == mat]
    oris    = [o for o in orientations
               if str(o) in sub_mat["orientation"].astype(str).unique()]
    x_centres = np.arange(len(oris)) * group_gap

    for i_r, (dv, ylabel) in enumerate(DVs):
        ax = axes[i_r, i_c]
        rng = stable_rng(mat, dv)

        for i_o, ori in enumerate(oris):
            sub = sub_mat[sub_mat["orientation"].astype(str) == str(ori)]
            for i_cryst, cryst in enumerate(CRYST_ORDER):
                sub_c = sub[sub["crystallinity"] == cryst][dv].dropna()
                if sub_c.empty:
                    continue
                offset = (i_cryst - 0.5) * bar_width
                x      = x_centres[i_o] + offset
                col, al = _bar_color(mat, cryst)
                ax.bar(x, sub_c.mean(), width=bar_width * 0.9,
                       yerr=_ci95(sub_c), color=col, alpha=al,
                       edgecolor="k", linewidth=0.5, capsize=2, zorder=3)
                jit = rng.uniform(-0.04, 0.04, len(sub_c))
                ax.scatter(x + jit, sub_c.values, s=7, color="k",
                           alpha=0.6, edgecolor="white", linewidths=0.3,
                           zorder=4)

        ax.axhline(0, color="0.7", lw=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(x_centres)
        ax.set_xticklabels([orientation_label(o) for o in oris], fontsize=9)

        if i_r == 0:
            ax.set_title(material_label(mat), fontsize=9,
                         color=material_color(mat), pad=4)
        if i_c == 0:
            ax.set_ylabel(ylabel, fontsize=9)

# Legend — top-left panel
legend_handles = [
    Line2D([0], [0], color="0.3", marker="s", markersize=6,
           linestyle="None", alpha=0.85, label="High $T$"),
    Line2D([0], [0], color="0.3", marker="s", markersize=6,
           linestyle="None", alpha=0.35, label="Low $T$"),
]
fig.legend(handles=legend_handles, loc="lower center", frameon=False, ncol=2, fontsize=9)

fig.tight_layout(rect=(0, 0.1, 1, 1))
save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
