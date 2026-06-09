"""
figSC_ext_strain_directions.py
================================
Combined per-direction strain ε(t) in the SPECIMEN coordinate system.

All DIC extensometer channels that measure the same physical direction are
pooled together into a single mean ± 95 % CI band per direction:

    Raster      : VERT (E0–E2) from O3  +  HORIZ (E3–E5) from O2
    Transraster : HORIZ (E3–E5) from O1
    Build       : VERT (E0–E2) from O1+O2  +  HORIZ (E3–E5) from O3

Grid layout:
    Rows    = materials
    Columns = Raster | Transraster | Build

Line colour = direction colour.  High χ: solid / dark; Low χ: dashed / muted.
Shaded band = 95 % CI across all pooled sensor × replicate traces.

Data requirement: data_exports/processed_results.pkl
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from journal_style import (
    set_journal_style, save_figure, mm_to_inch,
    MATERIAL_ORDER, CRYST_ORDER,
    VERT_SENSORS, HORIZ_SENSORS,
    SC_DIRECTION_ORDER, SC_DIRECTION_COLORS, sc_direction_label,
    zz_direction, yy_direction,
    material_color, material_label,
    parse_specimen_name, ci95_profile, DATA_DIR,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIGURE_BASENAME = "figSC_ext_strain_directions"
FIG_WIDTH_MM    = 178
ROW_HEIGHT_MM   = 48
N_TNORM         = 300

# ── Load data ──────────────────────────────────────────────────────────────────
pkl = DATA_DIR / "processed_results.pkl"
if not pkl.exists():
    raise FileNotFoundError(f"{pkl}\nRun prepare_exports.py first.")

with open(pkl, "rb") as f:
    processed_results = pickle.load(f)

t_grid = np.linspace(0, 1, N_TNORM)

# ── Pool all sensors measuring the same physical direction ─────────────────────
# key: (material, sc_direction, crystallinity, ori_grp) → list of traces
# ori_grp: "O12" (orientations 1+2 pooled) or "O3" (orientation 3 alone)
buckets = {}

for name, res in processed_results.items():
    meta = parse_specimen_name(name)
    if meta is None:
        continue
    mat      = meta["material_full"]
    ori      = meta["orientation"]
    cryst    = meta["crystallinity"]
    ori_grp  = "O3" if ori == "3" else "O12"

    t_raw = np.asarray(res.get("time", []), float)
    if t_raw.size < 5:
        continue
    t_norm = (t_raw - t_raw[0]) / max(t_raw[-1] - t_raw[0], 1.0)

    # VERT sensors → zz_direction(ori)
    for s in VERT_SENSORS:
        eps = np.asarray(res.get("strains", {}).get(s, []), float)
        if eps.size != t_raw.size:
            continue
        ok = np.isfinite(t_norm) & np.isfinite(eps)
        if ok.sum() < 5:
            continue
        arr = np.interp(t_grid, t_norm[ok], eps[ok],
                        left=np.nan, right=np.nan) * 1e6
        key = (mat, zz_direction(ori), cryst, ori_grp)
        buckets.setdefault(key, []).append(arr)

    # HORIZ sensors → yy_direction(ori)
    for s in HORIZ_SENSORS:
        eps = np.asarray(res.get("strains", {}).get(s, []), float)
        if eps.size != t_raw.size:
            continue
        ok = np.isfinite(t_norm) & np.isfinite(eps)
        if ok.sum() < 5:
            continue
        arr = np.interp(t_grid, t_norm[ok], eps[ok],
                        left=np.nan, right=np.nan) * 1e6
        key = (mat, yy_direction(ori), cryst, ori_grp)
        buckets.setdefault(key, []).append(arr)

means = {}
cis   = {}
for k, v in buckets.items():
    m, ci = ci95_profile(np.vstack(v))
    means[k] = m
    cis[k]   = ci

# ── Resolve grid ───────────────────────────────────────────────────────────────
all_meta  = [parse_specimen_name(n) for n in processed_results]
all_meta  = [m for m in all_meta if m is not None]
materials  = [m for m in MATERIAL_ORDER
              if any(x["material_full"] == m for x in all_meta)]
dirs_present = [d for d in SC_DIRECTION_ORDER
                if any(k[1] == d for k in means)]

# ── Plot ───────────────────────────────────────────────────────────────────────
set_journal_style()

n_r = len(materials)
n_c = len(dirs_present)

fig, axes = plt.subplots(
    n_r, n_c,
    figsize=mm_to_inch(FIG_WIDTH_MM, ROW_HEIGHT_MM * n_r),
    sharex=True, sharey="row",
    squeeze=False,
)

for i_r, mat in enumerate(materials):
    mc = material_color(mat)
    for i_c, direction in enumerate(dirs_present):
        ax      = axes[i_r, i_c]
        dir_col = SC_DIRECTION_COLORS[direction]

        # O1+O2 first (main, direction colour), then O3 (grey, secondary)
        for ori_grp, lw, a_base, a_f_base, col in [
            ("O12", 1.4, 1.0,  0.16, dir_col),
            ("O3",  1.0, 0.85, 0.12, "0.55"),
        ]:
            for cryst in CRYST_ORDER:
                ls  = "-"  if cryst == "High" else "--"
                a_l = a_base       if cryst == "High" else a_base * 0.65
                a_f = a_f_base     if cryst == "High" else a_f_base * 0.5
                key = (mat, direction, cryst, ori_grp)
                if key not in means:
                    continue
                mu    = means[key]
                ci    = cis[key]
                valid = np.isfinite(mu)
                ax.plot(t_grid[valid], mu[valid],
                        color=col, ls=ls, lw=lw, alpha=a_l, zorder=3)
                ax.fill_between(
                    t_grid[valid],
                    (mu - ci)[valid], (mu + ci)[valid],
                    color=col, alpha=a_f, linewidth=0, zorder=2,
                )

        ax.axhline(0, color="0.7", lw=0.5, zorder=0)
        ax.set_ylim(-1000, 8000)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if i_r == 0:
            ax.set_title(sc_direction_label(direction), pad=4, fontsize=11,
                         color=dir_col)
        if i_r == n_r - 1:
            ax.set_xlabel("Normalised time [—]", fontsize=10)
        else:
            ax.tick_params(labelbottom=False)
        ax.tick_params(labelsize=9)
        if i_c == 0:
            ax.set_ylabel(
                f"{material_label(mat)}\nStrain [µm m$^{{-1}}$]",
                fontsize=10, color=mc,
            )
        else:
            ax.tick_params(labelleft=False)

# Legend — fig.tight_layout(rect) reserves bottom space so there's no overlap
fig.legend(
    handles=[
        Line2D([0], [0], color="0.35", ls="-",  lw=1.4,
               label="High $T$, O1+O2  (mean ± 95 % CI)"),
        Line2D([0], [0], color="0.35", ls="--", lw=1.4, alpha=0.65,
               label="Low $T$, O1+O2"),
        Line2D([0], [0], color="0.55", ls="-",  lw=1.0,
               label="High $T$, O3"),
        Line2D([0], [0], color="0.55", ls="--", lw=1.0, alpha=0.65,
               label="Low $T$, O3"),
    ],
    loc="lower center",
    ncol=4,
    fontsize=9,
    handlelength=1.6,
    frameon=False,
)
fig.tight_layout(rect=(0, 0.07, 1, 1))

save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
