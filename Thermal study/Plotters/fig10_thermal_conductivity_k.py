"""
fig10_thermal_conductivity_k.py
=================================
Apparent through-thickness thermal conductivity k [W m⁻¹ K⁻¹] per material,
grouped by orientation.  High χ = full colour, Low χ = muted.
Bar height = mean; error bar = 95 % CI; dots = individual replicates.

If k_W_mK is NaN for all specimens (heat flux not calibrated), falls back
to showing through-thickness temperature gradient ΔT [K] instead.

Data requirement: data_exports/prof_df.csv
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
FIGURE_BASENAME = "fig10_thermal_conductivity_k"
FIG_WIDTH_MM    = 178
FIG_HEIGHT_MM   = 70

_MAT_INTERNAL = {"PEEK_CF": "PEEK-CF", "PPS_CF": "PPS-CF",
                 "PPS_NEAT": "PPS neat", "PEEK_NEAT": "PEEK neat"}

# ── Load data ──────────────────────────────────────────────────────────────────
csv = DATA_DIR / "prof_df.csv"
if not csv.exists():
    raise FileNotFoundError(f"{csv}\nRun prepare_exports.py first.")

prof_df = pd.read_csv(csv)

# Normalise material column
_inv = {v: k for k, v in _MAT_INTERNAL.items()}
col_mat = "material_full" if "material_full" in prof_df.columns else "material"
if prof_df[col_mat].str.contains("_").any():
    prof_df["material_full"] = prof_df[col_mat].map(lambda x: _MAT_INTERNAL.get(x, x))
else:
    prof_df["material_full"] = prof_df[col_mat]

# Resolve k column — notebook may export 'k_W_mK_cal' or 'k_W_mK'
for _k_col in ("k_W_mK_cal", "k_W_mK"):
    if _k_col in prof_df.columns and prof_df[_k_col].notna().any():
        dv = _k_col
        break
else:
    raise KeyError(
        "prof_df has no k column (checked k_W_mK_cal, k_W_mK). "
        "Re-run prepare_exports.py after running the thermal-conductivity cells."
    )
y_label = r"$k$ [W m$^{-1}$ K$^{-1}$]"


# ── Helpers ────────────────────────────────────────────────────────────────────
def _ci95(vals):
    v = vals.dropna()
    if len(v) < 2:
        return 0.0
    se = v.std(ddof=1) / np.sqrt(len(v))
    return float(stats.t.ppf(0.975, df=len(v) - 1) * se)


def _bar_color(mat, cryst):
    mc = material_color(mat)
    return (mc, 0.85) if cryst == "High" else (mc, 0.35)


# ── Resolve grid ───────────────────────────────────────────────────────────────
materials    = [m for m in MATERIAL_ORDER if m in prof_df["material_full"].unique()]
orientations = [o for o in ORIENTATION_ORDER
                if str(o) in prof_df["orientation"].astype(str).unique()]

# ── Plot ───────────────────────────────────────────────────────────────────────
set_journal_style()

n_c = len(materials)
fig, axes = plt.subplots(
    1, n_c,
    figsize=mm_to_inch(FIG_WIDTH_MM, FIG_HEIGHT_MM),
    sharey=True,
    squeeze=False,
)
axes = axes[0]  # 1-D array of axes

bar_width = 0.32
group_gap = 0.80

for i_c, mat in enumerate(materials):
    ax      = axes[i_c]
    sub_mat = prof_df[prof_df["material_full"] == mat]
    oris    = [o for o in orientations
               if str(o) in sub_mat["orientation"].astype(str).unique()]
    x_centres = np.arange(len(oris)) * group_gap
    rng = stable_rng(mat, dv)

    for i_o, ori in enumerate(oris):
        sub = sub_mat[sub_mat["orientation"].astype(str) == str(ori)]
        crysts_present = [c for c in CRYST_ORDER
                          if c in sub["crystallinity"].values]
        for i_cryst, cryst in enumerate(crysts_present):
            sub_c = sub[sub["crystallinity"] == cryst][dv].dropna()
            if sub_c.empty:
                continue
            offset = (i_cryst - (len(crysts_present) - 1) / 2) * bar_width
            x      = x_centres[i_o] + offset
            col, al = _bar_color(mat, cryst)
            mean_k = sub_c.mean()
            ci     = _ci95(sub_c)
            ax.bar(x, mean_k, width=bar_width * 0.9,
                   yerr=ci, color=col, alpha=al,
                   edgecolor="k", linewidth=0.5, capsize=2, zorder=3)
            jit = rng.uniform(-0.04, 0.04, len(sub_c))
            ax.scatter(x + jit, sub_c.values, s=7, color="k",
                       alpha=0.6, edgecolor="white", linewidths=0.3, zorder=4)
            # numeric k annotation above bar
            ax.text(x, mean_k + ci + 0.005, f"{mean_k:.3f}",
                    ha="center", va="bottom", fontsize=6.5,
                    color="k", zorder=5)

    ax.set_xticks(x_centres)
    ax.set_xticklabels([orientation_label(o) for o in oris], fontsize=9)
    ax.set_title(material_label(mat), fontsize=9,
                 color=material_color(mat), pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="0.7", lw=0.5, zorder=0)
    if i_c == 0:
        ax.set_ylabel(y_label, fontsize=9)

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
