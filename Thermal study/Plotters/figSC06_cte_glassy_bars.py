"""
figSC06_cte_glassy_bars.py
===========================
Glassy-regime CTE bar chart in the SPECIMEN coordinate system.

Each specimen's measured α_zz / α_yy is remapped to α_build / α_raster /
α_transraster according to its orientation:
    O1 : zz → build,  yy → transraster
    O2 : zz → build,  yy → raster
    O3 : zz → raster, yy → build

Layout : one column per material, x-axis = physical direction.
High χ  = full colour bar,  Low χ = muted bar.
Replicate dots overlaid.  Error bars = 95 % CI.

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
    MATERIAL_ORDER, CRYST_ORDER,
    SC_DIRECTION_ORDER, SC_DIRECTION_COLORS, sc_direction_label,
    zz_direction, yy_direction,
    material_color, material_label,
    stable_rng, DATA_DIR,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIGURE_BASENAME = "figSC06_cte_glassy_bars"
FIG_WIDTH_MM    = 178
ROW_HEIGHT_MM   = 65

_MAT_INTERNAL = {"PEEK-CF": "PEEK_CF", "PPS-CF": "PPS_CF", "PPS neat": "PPS_NEAT"}

# ── Load & remap ───────────────────────────────────────────────────────────────
csv = DATA_DIR / "metrics_df.csv"
if not csv.exists():
    raise FileNotFoundError(f"{csv}\nRun prepare_exports.py first.")

metrics_df = pd.read_csv(csv)
_inv = {v: k for k, v in _MAT_INTERNAL.items()}
if metrics_df["material_full"].str.contains("_").any():
    metrics_df["material_full"] = metrics_df["material_full"].map(
        lambda x: _inv.get(x, x))

# Build long-form table: one row per (specimen, direction, dv)
rows = []
for _, r in metrics_df.iterrows():
    ori = str(r["orientation"])
    mat = r["material_full"]
    cryst = r["crystallinity"]
    pairs = [
        (zz_direction(ori), r.get("alpha_lin_zz", np.nan)),
        (yy_direction(ori), r.get("alpha_lin_yy", np.nan)),
    ]
    for direction, val in pairs:
        rows.append(dict(material=mat, crystallinity=cryst,
                         direction=direction, alpha=val))

sc_df = pd.DataFrame(rows)
sc_df = sc_df.dropna(subset=["alpha"])

# ── Helpers ────────────────────────────────────────────────────────────────────
def _ci95(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return 0.0
    se = v.std(ddof=1) / np.sqrt(len(v))
    return float(stats.t.ppf(0.975, df=len(v) - 1) * se)

# ── Plot ───────────────────────────────────────────────────────────────────────
set_journal_style()

materials = [m for m in MATERIAL_ORDER if m in sc_df["material"].unique()]
n_c = len(materials)

fig, axes = plt.subplots(
    1, n_c,
    figsize=mm_to_inch(FIG_WIDTH_MM, ROW_HEIGHT_MM),
    sharey=True,
    squeeze=False,
)
axes = axes[0]

bar_width = 0.30
group_gap = 1.0
directions = SC_DIRECTION_ORDER   # build, raster, transraster

for i_c, mat in enumerate(materials):
    ax  = axes[i_c]
    mc  = material_color(mat)
    sub = sc_df[sc_df["material"] == mat]
    rng = stable_rng(mat, "SC06")

    x_centres = np.arange(len(directions)) * group_gap

    for j_d, direction in enumerate(directions):
        sub_d = sub[sub["direction"] == direction]
        dir_col = SC_DIRECTION_COLORS[direction]

        for k_c, cryst in enumerate(CRYST_ORDER):
            sub_c = sub_d[sub_d["crystallinity"] == cryst]["alpha"].dropna()
            if sub_c.empty:
                continue
            alpha_bar = 0.85 if cryst == "High" else 0.35
            offset = (k_c - 0.5) * bar_width
            x = x_centres[j_d] + offset
            ax.bar(x, sub_c.mean(), width=bar_width * 0.92,
                   yerr=_ci95(sub_c), color=dir_col, alpha=alpha_bar,
                   edgecolor="k", linewidth=0.5, capsize=2, zorder=3)
            jit = rng.uniform(-0.04, 0.04, len(sub_c))
            ax.scatter(x + jit, sub_c.values, s=7, color="k",
                       alpha=0.6, edgecolor="white", linewidths=0.3, zorder=4)

    ax.axhline(0, color="0.7", lw=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x_centres)
    ax.set_xticklabels([sc_direction_label(d) for d in directions], fontsize=9)
    ax.set_title(material_label(mat), fontsize=9, color=mc, pad=4)
    if i_c == 0:
        ax.set_ylabel(
            r"$\alpha^{\rm glassy}$ [µm m$^{-1}$ K$^{-1}$]",
            fontsize=9)

# Legend
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
