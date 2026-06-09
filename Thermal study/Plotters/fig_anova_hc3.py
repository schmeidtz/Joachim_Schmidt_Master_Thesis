"""
fig_anova_hc3.py
=================
HC3-robust three-way ANOVA results visualised as a bubble heatmap.

Rows    = dependent variables (DVs)
Columns = ANOVA terms (main effects + interactions)
Colour  = −log₁₀(p_FDR_HC3)  →  white = ns, red = significant
Bubble size ∝ partial η²

Significance thresholds:
  *   p < 0.05
  **  p < 0.01
  *** p < 0.001

Data requirement:
  data_exports/metrics_df.csv   (per-specimen metrics produced by Cell A1)
  Runs the HC3 ANOVA internally — does NOT need anova3_df.csv.
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
    significance_stars, DATA_DIR,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIGURE_BASENAME = "fig_anova_hc3"
FIG_WIDTH_MM    = 178

_MAT_INTERNAL = {"PEEK_CF": "PEEK-CF", "PPS_CF": "PPS-CF",
                 "PPS_NEAT": "PPS neat", "PEEK_NEAT": "PEEK neat"}

DVs = [
    "alpha_lin_zz", "alpha_lin_yy",
    "alpha_rub_zz", "alpha_rub_yy",
    "anisotropy", "aniso_rubbery",
    "poisson_zy",
    "peak_eps_zz", "peak_eps_yy",
]

DV_LABELS = {
    "alpha_lin_zz":  r"$\alpha_{zz}^{\rm glassy}$",
    "alpha_lin_yy":  r"$\alpha_{yy}^{\rm glassy}$",
    "alpha_rub_zz":  r"$\alpha_{zz}^{\rm rubbery}$",
    "alpha_rub_yy":  r"$\alpha_{yy}^{\rm rubbery}$",
    "anisotropy":    r"$\alpha_{zz}/\alpha_{yy}$ (glassy)",
    "aniso_rubbery": r"$\alpha_{zz}/\alpha_{yy}$ (rubbery)",
    "poisson_zy":    r"$\nu_{zy}$",
    "peak_eps_zz":   r"$\varepsilon_{zz}^{\rm peak}$",
    "peak_eps_yy":   r"$\varepsilon_{yy}^{\rm peak}$",
}

TERMS = {
    "C(crystallinity)":                              "Cryst.",
    "C(fibre)":                                      "Fibre",
    "C(orientation)":                                "Ori.",
    "C(crystallinity):C(fibre)":                     "Cryst.×Fibre",
    "C(crystallinity):C(orientation)":               "Cryst.×Ori.",
    "C(fibre):C(orientation)":                       "Fibre×Ori.",
    "C(crystallinity):C(fibre):C(orientation)":      "Cryst.×Fibre×Ori.",
}

TERM_ORDER = list(TERMS.values())
DV_ORDER   = DVs


# ── Load metrics ───────────────────────────────────────────────────────────────
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

# Ensure factor columns exist
if "polymer" not in metrics_df.columns:
    metrics_df["polymer"] = metrics_df["material_full"].str.split("-").str[0]
if "fibre" not in metrics_df.columns:
    metrics_df["fibre"] = np.where(
        metrics_df["material_full"].str.contains("CF"), "CF", "Neat"
    )


# ── HC3 ANOVA ─────────────────────────────────────────────────────────────────
def _hc3_wald(formula, data, term_map):
    """Return {nice_name: (F, p, df_num, df_den, eta_p)} for each term."""
    try:
        m   = ols(formula, data=data).fit(cov_type="HC3")
        m_ols = ols(formula, data=data).fit()   # for eta²
        aov = sm.stats.anova_lm(m_ols, typ=2)
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


rows = []

# Run per-material ANOVAs.
# - All materials: two-way Crystallinity × Orientation
# - PPS only (has CF + Neat): add Fibre as a third factor
TERMS_2WAY = {
    "C(crystallinity)":                    "Cryst.",
    "C(orientation)":                      "Ori.",
    "C(crystallinity):C(orientation)":     "Cryst.×Ori.",
}
TERMS_3WAY = {
    "C(crystallinity)":                              "Cryst.",
    "C(fibre)":                                      "Fibre",
    "C(orientation)":                                "Ori.",
    "C(crystallinity):C(fibre)":                     "Cryst.×Fibre",
    "C(crystallinity):C(orientation)":               "Cryst.×Ori.",
    "C(fibre):C(orientation)":                       "Fibre×Ori.",
    "C(crystallinity):C(fibre):C(orientation)":      "Cryst.×Fibre×Ori.",
}

for mat, sub in metrics_df.groupby("material_full"):
    sub = sub.dropna(subset=["crystallinity", "orientation"])
    if sub["crystallinity"].nunique() < 2 or sub["orientation"].nunique() < 2:
        continue
    has_fibre = sub["fibre"].nunique() >= 2
    term_map  = TERMS_3WAY if has_fibre else TERMS_2WAY
    base_factors = ["crystallinity", "fibre", "orientation"] if has_fibre else ["crystallinity", "orientation"]
    formula_tpl = ("{dv} ~ C(crystallinity)*C(fibre)*C(orientation)"
                   if has_fibre else
                   "{dv} ~ C(crystallinity)*C(orientation)")
    for dv in DVs:
        dat = sub.dropna(subset=[dv]).copy()
        if len(dat) < 6:
            continue
        cs = dat.groupby(base_factors).size()
        if cs.min() < 2:
            continue
        formula = formula_tpl.format(dv=dv)
        res = _hc3_wald(formula, dat, term_map)
        for nice, (F, p, df_num, df_den, eta_p) in res.items():
            rows.append(dict(material=mat, DV=dv, term=nice,
                             F_HC3=F, p_HC3=p, eta_p=eta_p))

if not rows:
    raise RuntimeError("No ANOVA results — check metrics_df.csv for required columns "
                       "(crystallinity, fibre, orientation, alpha_lin_zz, …).")

anova_df = pd.DataFrame(rows)

# BH-FDR correction per material × DV group
fdr_rows = []
for (mat, dv), grp in anova_df.groupby(["material", "DV"]):
    ps = grp["p_HC3"].values
    _, ps_adj, _, _ = multipletests(ps, method="fdr_bh")
    fdr_rows.append(grp.assign(p_fdr=ps_adj))
anova_df = pd.concat(fdr_rows, ignore_index=True)

anova_df["neg_log10_p_fdr"] = -np.log10(anova_df["p_fdr"].clip(lower=1e-10))
anova_df["sig"]             = anova_df["p_fdr"].apply(significance_stars)


# ── Plot ───────────────────────────────────────────────────────────────────────
set_journal_style()

# Use material order from journal_style if available, else sort
try:
    from journal_style import MATERIAL_ORDER as _MAT_ORDER
    materials = [m for m in _MAT_ORDER if m in anova_df["material"].unique()]
    # add any materials not in MATERIAL_ORDER
    for m in sorted(anova_df["material"].unique()):
        if m not in materials:
            materials.append(m)
except ImportError:
    materials = sorted(anova_df["material"].unique())

n_mat = len(materials)

# Figure: one sub-grid per material side-by-side
FIG_H_MM = 20 + len(DV_ORDER) * 11
fig, axes = plt.subplots(
    1, n_mat,
    figsize=mm_to_inch(FIG_WIDTH_MM, FIG_H_MM),
    constrained_layout=True,
    squeeze=False,
)
axes = axes[0]

# Colour scale: white → red over 0 … 4 (−log₁₀ p)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "sig_cmap", ["#ffffff", "#fde0d0", "#f97c59", "#c0392b", "#7b0000"],
    N=256,
)
norm = mcolors.Normalize(vmin=0, vmax=4)

# Bubble size: η² → area in points²
MAX_S = 350
MIN_S = 10


def _bubble_size(eta_p):
    if not np.isfinite(eta_p) or eta_p <= 0:
        return MIN_S
    return MIN_S + (MAX_S - MIN_S) * np.clip(eta_p, 0, 1)


for i_mat, mat in enumerate(materials):
    ax  = axes[i_mat]
    sub = anova_df[anova_df["material"] == mat]

    n_dv   = len(DV_ORDER)
    n_term = len(TERM_ORDER)

    # Background grid cells
    for i_dv, dv in enumerate(DV_ORDER):
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
            col   = cmap(norm(val)) if np.isfinite(val) else (0.92, 0.92, 0.92, 1)
            s     = _bubble_size(eta)
            ax.scatter(j_term, n_dv - 1 - i_dv, s=s, c=[col],
                       marker="o", edgecolors="0.35", linewidths=0.4,
                       zorder=3)
            if sig and sig != "ns":
                ax.text(j_term, n_dv - 1 - i_dv, sig,
                        ha="center", va="center", fontsize=6,
                        color="white" if val > 2 else "0.2", fontweight="bold",
                        zorder=4)

    ax.set_xlim(-0.5, n_term - 0.5)
    ax.set_ylim(-0.5, n_dv - 0.5)
    ax.set_xticks(range(n_term))
    ax.set_xticklabels(TERM_ORDER, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n_dv))
    # Only show y-tick labels on the leftmost panel
    if i_mat == 0:
        ax.set_yticklabels(
            [DV_LABELS.get(dv, dv) for dv in reversed(DV_ORDER)],
            fontsize=9,
        )
    else:
        ax.set_yticklabels([])
    ax.set_title(mat, fontsize=9, pad=4)

    # Light grid lines
    for x in np.arange(-0.5, n_term, 1):
        ax.axvline(x, color="0.88", lw=0.5, zorder=0)
    for y in np.arange(-0.5, n_dv, 1):
        ax.axhline(y, color="0.88", lw=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

# Shared colourbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=list(axes), orientation="vertical",
                    fraction=0.015, pad=0.02, shrink=0.8)
cbar.set_label(r"$-\log_{10}(p_{\rm FDR})$", fontsize=9)
cbar.ax.tick_params(labelsize=6)
# Threshold lines
for p_thresh, label in [(0.05, "p=0.05"), (0.01, "p=0.01"), (0.001, "p=0.001")]:
    cbar.ax.axhline(-np.log10(p_thresh), color="0.4", lw=0.7, ls="--")
    cbar.ax.text(1.35, -np.log10(p_thresh), label,
                 transform=cbar.ax.transData, fontsize=6,
                 va="center", color="0.4")

# Bubble size legend (η²) — on the last panel
if n_mat > 0:
    legend_ax = axes[-1]
    for eta_demo, lbl in [(0.05, r"$\eta^2=0.05$"),
                          (0.20, r"$\eta^2=0.20$"),
                          (0.50, r"$\eta^2=0.50$")]:
        legend_ax.scatter([], [], s=_bubble_size(eta_demo), c="0.6",
                          edgecolors="0.35", linewidths=0.4,
                          label=lbl)
    legend_ax.legend(loc="upper right", fontsize=9, title=r"Bubble area $\propto \eta^2$",
                     title_fontsize=7, frameon=False,
                     markerscale=0.25, handletextpad=1.0, labelspacing=1.2)

save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
