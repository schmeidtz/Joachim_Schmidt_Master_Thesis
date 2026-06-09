"""
figSC_anova_hc3.py
===================
HC3-robust two-way ANOVA on CTE expressed in the SPECIMEN coordinate system.

Each specimen's α_zz / α_yy measurements are first remapped to physical
directions (build / raster / transraster) based on orientation:
    O1 : zz → build,  yy → transraster
    O2 : zz → build,  yy → raster
    O3 : zz → raster, yy → build

Then for each direction a two-way ANOVA is run:
    α_direction ~ C(crystallinity) * C(material)

(Orientation is no longer a factor — it has been consumed by the remapping.)

Layout: one panel per physical direction.
Rows    = specimen-coordinate DVs
Columns = ANOVA terms  [Cryst. | Material | Cryst.×Material]
Colour  = −log₁₀(p_FDR_HC3)    Bubble area ∝ partial η²

Notes
-----
• transraster is measured only by O1 specimens — fewer replicates, interpret
  with caution.
• BH-FDR correction is applied per direction × DV group.

Data requirement: data_exports/metrics_df.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent))
from journal_style import (
    set_journal_style, save_figure, mm_to_inch,
    SC_DIRECTION_ORDER, SC_DIRECTION_COLORS, sc_direction_label,
    zz_direction, yy_direction,
    significance_stars, DATA_DIR,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIGURE_BASENAME = "figSC_anova_hc3"
FIG_WIDTH_MM    = 178

_MAT_INTERNAL = {"PEEK_CF": "PEEK-CF", "PPS_CF": "PPS-CF",
                 "PPS_NEAT": "PPS neat", "PEEK_NEAT": "PEEK neat"}

# Specimen-coordinate DVs built from each measurement pair
# (dv_label,  zz_source_col,      yy_source_col)
SC_DV_PAIRS = [
    ("alpha_lin",  "alpha_lin_zz",  "alpha_lin_yy"),
    ("alpha_rub",  "alpha_rub_zz",  "alpha_rub_yy"),
    ("peak_eps",   "peak_eps_zz",   "peak_eps_yy"),
]

# These DVs come from the zz sensor only (no yy counterpart)
SC_DV_ZZ_ONLY = []   # extend if needed

TERMS = {
    "C(crystallinity)":                         "Cryst.",
    "C(material)":                              "Material",
    "C(crystallinity):C(material)":             "Cryst.×Mat.",
}
TERM_ORDER = list(TERMS.values())

# ── Load & remap ───────────────────────────────────────────────────────────────
csv = DATA_DIR / "metrics_df.csv"
if not csv.exists():
    raise FileNotFoundError(f"{csv}\nRun prepare_exports.py first.")

metrics_df = pd.read_csv(csv)
if metrics_df["material_full"].str.contains("_").any():
    metrics_df["material_full"] = metrics_df["material_full"].map(
        lambda x: _MAT_INTERNAL.get(x, x))

# Build long-form specimen-coordinate table
rows = []
for _, r in metrics_df.iterrows():
    ori   = str(r["orientation"])
    mat   = r["material_full"]
    cryst = r["crystallinity"]
    sc_zz = zz_direction(ori)
    sc_yy = yy_direction(ori)

    for dv_label, zz_col, yy_col in SC_DV_PAIRS:
        if zz_col in r.index and pd.notna(r[zz_col]):
            rows.append(dict(material=mat, crystallinity=cryst,
                             direction=sc_zz,
                             dv=f"{dv_label}_{sc_zz}",
                             value=float(r[zz_col])))
        if yy_col in r.index and pd.notna(r[yy_col]):
            rows.append(dict(material=mat, crystallinity=cryst,
                             direction=sc_yy,
                             dv=f"{dv_label}_{sc_yy}",
                             value=float(r[yy_col])))

sc_df = pd.DataFrame(rows)

# DV order for display
DV_ORDER = []
for dv_label, _, _ in SC_DV_PAIRS:
    for d in SC_DIRECTION_ORDER:
        dv_key = f"{dv_label}_{d}"
        if dv_key in sc_df["dv"].unique():
            DV_ORDER.append(dv_key)
DV_ORDER = list(dict.fromkeys(DV_ORDER))   # deduplicate, preserve order

DV_LABELS = {}
_base_labels = {"alpha_lin": r"$\alpha^{\rm gl}$",
                "alpha_rub": r"$\alpha^{\rm rub}$",
                "peak_eps":  r"$\varepsilon^{\rm pk}$"}
for dv_key in DV_ORDER:
    for prefix, base in _base_labels.items():
        if dv_key.startswith(prefix):
            direction = dv_key[len(prefix)+1:]
            DV_LABELS[dv_key] = f"{base} {sc_direction_label(direction)}"
            break


# ── HC3 ANOVA ─────────────────────────────────────────────────────────────────
def _hc3_wald(formula, data, term_map):
    try:
        m     = ols(formula, data=data).fit(cov_type="HC3")
        m_ols = ols(formula, data=data).fit()
        aov   = sm.stats.anova_lm(m_ols, typ=2)
        ss_res = aov.loc["Residual", "sum_sq"]
    except Exception:
        return {}
    exog   = m.model.exog_names
    df_den = float(m.df_resid)
    out    = {}
    for term_str, nice in term_map.items():
        parts = term_str.split(":")
        idxs  = []
        for i, n in enumerate(exog):
            if n == "Intercept":
                continue
            nps = n.split(":")
            if len(nps) != len(parts):
                continue
            if all(any(pp.startswith(tp) for pp in nps) for tp in parts):
                idxs.append(i)
        if not idxs:
            continue
        R = np.zeros((len(idxs), len(exog)))
        for r, c in enumerate(idxs):
            R[r, c] = 1.0
        try:
            w = m.wald_test(R, scalar=True, use_f=True)
        except TypeError:
            w = m.wald_test(R, use_f=True)
        F_val = float(np.atleast_1d(w.statistic).squeeze())
        p_val = float(np.atleast_1d(w.pvalue).squeeze())
        eta_p = np.nan
        if term_str in aov.index:
            ss_t  = aov.loc[term_str, "sum_sq"]
            eta_p = float(ss_t / (ss_t + ss_res))
        out[nice] = (F_val, p_val, float(len(idxs)), df_den, eta_p)
    return out


anova_rows = []
formula = "value ~ C(crystallinity) * C(material)"

for direction in SC_DIRECTION_ORDER:
    sub_dir = sc_df[sc_df["direction"] == direction]
    if sub_dir.empty:
        continue
    for dv_key in DV_ORDER:
        if not dv_key.endswith(f"_{direction}"):
            continue
        dat = sub_dir[sub_dir["dv"] == dv_key].dropna(subset=["value"]).copy()
        if len(dat) < 6:
            continue
        cs = dat.groupby(["crystallinity", "material"]).size()
        if cs.min() < 2:
            continue
        res = _hc3_wald(formula, dat, TERMS)
        for nice, (F, p, df_num, df_den, eta_p) in res.items():
            anova_rows.append(dict(direction=direction, DV=dv_key, term=nice,
                                   F_HC3=F, p_HC3=p, eta_p=eta_p))

if not anova_rows:
    raise RuntimeError(
        "No ANOVA results — check metrics_df.csv for required columns."
    )

anova_df = pd.DataFrame(anova_rows)

# BH-FDR correction per direction × DV group
fdr_rows = []
for (direction, dv), grp in anova_df.groupby(["direction", "DV"]):
    ps = grp["p_HC3"].values
    _, ps_adj, _, _ = multipletests(ps, method="fdr_bh")
    fdr_rows.append(grp.assign(p_fdr=ps_adj))
anova_df = pd.concat(fdr_rows, ignore_index=True)

anova_df["neg_log10_p_fdr"] = -np.log10(anova_df["p_fdr"].clip(lower=1e-10))
anova_df["sig"]             = anova_df["p_fdr"].apply(significance_stars)


# ── Plot ───────────────────────────────────────────────────────────────────────
set_journal_style()

directions_present = [d for d in SC_DIRECTION_ORDER
                      if d in anova_df["direction"].unique()]
n_dir = len(directions_present)

# DVs present per direction
dvs_per_dir = {}
for d in directions_present:
    dvs_per_dir[d] = [dv for dv in DV_ORDER
                      if dv in anova_df[anova_df["direction"] == d]["DV"].unique()]

n_dv_max = max(len(v) for v in dvs_per_dir.values()) if dvs_per_dir else 1

FIG_H_MM = 20 + n_dv_max * 13
fig, axes = plt.subplots(
    1, n_dir,
    figsize=mm_to_inch(FIG_WIDTH_MM, FIG_H_MM),
    constrained_layout=True,
    squeeze=False,
)
axes = axes[0]

cmap = mcolors.LinearSegmentedColormap.from_list(
    "sig_cmap", ["#ffffff", "#fde0d0", "#f97c59", "#c0392b", "#7b0000"],
    N=256,
)
norm = mcolors.Normalize(vmin=0, vmax=4)

MAX_S = 350
MIN_S = 10


def _bubble_size(eta_p):
    if not np.isfinite(eta_p) or eta_p <= 0:
        return MIN_S
    return MIN_S + (MAX_S - MIN_S) * np.clip(eta_p, 0, 1)


for i_dir, direction in enumerate(directions_present):
    ax       = axes[i_dir]
    sub      = anova_df[anova_df["direction"] == direction]
    dir_dvs  = dvs_per_dir[direction]
    n_dv     = len(dir_dvs)
    n_term   = len(TERM_ORDER)
    dir_col  = SC_DIRECTION_COLORS[direction]

    for i_dv, dv in enumerate(dir_dvs):
        for j_term, term in enumerate(TERM_ORDER):
            r = sub[(sub["DV"] == dv) & (sub["term"] == term)]
            if r.empty:
                val = np.nan
                eta = np.nan
                sig = ""
            else:
                val = float(r.iloc[0]["neg_log10_p_fdr"])
                eta = float(r.iloc[0]["eta_p"])
                sig = r.iloc[0]["sig"]
            col = cmap(norm(val)) if np.isfinite(val) else (0.92, 0.92, 0.92, 1)
            s   = _bubble_size(eta)
            ax.scatter(j_term, n_dv - 1 - i_dv, s=s, c=[col],
                       marker="o", edgecolors="0.35", linewidths=0.4,
                       zorder=3)
            if sig and sig != "ns":
                ax.text(j_term, n_dv - 1 - i_dv, sig,
                        ha="center", va="center", fontsize=6,
                        color="white" if val > 2 else "0.2",
                        fontweight="bold", zorder=4)

    ax.set_xlim(-0.5, n_term - 0.5)
    ax.set_ylim(-0.5, n_dv - 0.5)
    ax.set_xticks(range(n_term))
    ax.set_xticklabels(TERM_ORDER, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n_dv))
    if i_dir == 0:
        ax.set_yticklabels(
            [DV_LABELS.get(dv, dv) for dv in reversed(dir_dvs)],
            fontsize=9,
        )
    else:
        ax.set_yticklabels([])
    ax.set_title(sc_direction_label(direction), fontsize=9, pad=4,
                 color=dir_col)

    for x in np.arange(-0.5, n_term, 1):
        ax.axvline(x, color="0.88", lw=0.5, zorder=0)
    for y in np.arange(-0.5, n_dv, 1):
        ax.axhline(y, color="0.88", lw=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

# Shared colourbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=list(axes), orientation="vertical",
                    fraction=0.015, pad=0.02, shrink=0.8)
cbar.set_label(r"$-\log_{10}(p_{\rm FDR})$", fontsize=9)
cbar.ax.tick_params(labelsize=6)
for p_thresh, label in [(0.05, "p=0.05"), (0.01, "p=0.01"), (0.001, "p=0.001")]:
    cbar.ax.axhline(-np.log10(p_thresh), color="0.4", lw=0.7, ls="--")
    cbar.ax.text(1.35, -np.log10(p_thresh), label,
                 transform=cbar.ax.transData, fontsize=6,
                 va="center", color="0.4")

# Bubble legend
legend_ax = axes[-1]
for eta_demo, lbl in [(0.05, r"$\eta^2=0.05$"),
                      (0.20, r"$\eta^2=0.20$"),
                      (0.50, r"$\eta^2=0.50$")]:
    legend_ax.scatter([], [], s=_bubble_size(eta_demo), c="0.6",
                      edgecolors="0.35", linewidths=0.4, label=lbl)
legend_ax.legend(loc="upper right", fontsize=9,
                 title=r"Bubble area $\propto \eta^2$",
                 title_fontsize=7, frameon=False,
                 markerscale=0.25, handletextpad=1.0, labelspacing=1.2)

save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
