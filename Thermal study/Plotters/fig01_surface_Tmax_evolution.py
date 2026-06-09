"""
fig01_surface_Tmax_evolution.py
================================
Heated surface temperature evolution from Camera 1 (Test 5 specimens).
T_max per frame, normalised heating time, mean ± 95 % CI across replicates.

Grid: N_materials rows × N_orientations cols.
High $T$: solid line.  Low $T$: dashed line.

Data requirement: data_exports/surf_fields.pkl
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
    MATERIAL_ORDER, ORIENTATION_ORDER, CRYST_ORDER, CRYST_LS, CRYST_ALPHA,
    material_color, material_label, orientation_label,
    ci95_profile, DATA_DIR,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIGURE_BASENAME = "fig01_surface_Tmax_evolution"
FIG_WIDTH_MM    = 178
ROW_HEIGHT_MM   = 48   # per material row

N_TNORM = 200          # interpolation grid points on [0, 1]

# ── Load data ──────────────────────────────────────────────────────────────────
pkl = DATA_DIR / "surf_fields.pkl"
if not pkl.exists():
    raise FileNotFoundError(f"{pkl}\nRun prepare_exports.py from the notebook first.")

with open(pkl, "rb") as f:
    surf_fields = pickle.load(f)

# ── Normalise material names in surf_fields (augmented uses PEEK_CF, clean uses PEEK-CF)
_MAT_NORM = {"PEEK_CF": "PEEK-CF", "PPS_CF": "PPS-CF",
             "PPS_NEAT": "PPS neat", "PEEK_NEAT": "PEEK neat"}
for _d in surf_fields.values():
    _c = _d["cond"]
    if "material_full" not in _c and "material" in _c:
        _c["material_full"] = _MAT_NORM.get(_c["material"], _c["material"])
    elif "material_full" in _c:
        _c["material_full"] = _MAT_NORM.get(_c["material_full"], _c["material_full"])

# ── Resolve materials / orientations present in the data ──────────────────────
materials    = [m for m in MATERIAL_ORDER
                if any(d["cond"].get("material_full") == m for d in surf_fields.values())]
orientations = [o for o in ORIENTATION_ORDER
                if any(str(d["cond"].get("orientation", "")) == str(o)
                       for d in surf_fields.values())]

if not materials or not orientations:
    # Print what's actually in the data to help diagnose
    _sample = next(iter(surf_fields.values()))["cond"]
    raise RuntimeError(
        f"surf_fields contains no matching materials/orientations.\n"
        f"  Sample cond keys: {list(_sample.keys())}\n"
        f"  Sample cond values: {_sample}\n"
        f"  MATERIAL_ORDER: {MATERIAL_ORDER}\n"
        f"  ORIENTATION_ORDER: {ORIENTATION_ORDER}"
    )

t_grid = np.linspace(0, 1, N_TNORM)

# ── Plot ───────────────────────────────────────────────────────────────────────
set_journal_style()

n_r, n_c = len(materials), len(orientations)
fig, axes = plt.subplots(
    n_r, n_c,
    figsize=mm_to_inch(FIG_WIDTH_MM, ROW_HEIGHT_MM * n_r),
    sharex=True,
    squeeze=False,
)

for i_r, mat in enumerate(materials):
    mc = material_color(mat)
    for i_c, ori in enumerate(orientations):
        ax = axes[i_r, i_c]

        for cryst in CRYST_ORDER:
            ls  = CRYST_LS[cryst]
            a_l = CRYST_ALPHA[cryst]
            a_f = 0.18 if cryst == "High" else 0.10

            specs = [
                d for d in surf_fields.values()
                if d["cond"]["material_full"] == mat
                and d["cond"]["orientation"]  == ori
                and d["cond"]["crystallinity"] == cryst
            ]
            if not specs:
                continue

            curves = []
            for d in specs:
                t = np.asarray(d["t_s"], float)
                T = np.asarray(d["T_max"], float)
                t_n = (t - t[0]) / max(t[-1] - t[0], 1.0)
                curves.append(np.interp(t_grid, t_n, T))

            stack   = np.vstack(curves)
            mean_T, ci_T = ci95_profile(stack)

            ax.plot(t_grid, mean_T,
                    color=mc, ls=ls, lw=1.3, alpha=a_l, zorder=3)
            ax.fill_between(t_grid, mean_T - ci_T, mean_T + ci_T,
                            color=mc, alpha=a_f, linewidth=0, zorder=2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Column header — top row only
        if i_r == 0:
            ax.set_title(orientation_label(ori), pad=4, fontsize=9)

        # x label — bottom row only
        if i_r == n_r - 1:
            ax.set_xlabel("Normalised heating time [—]", fontsize=9)
        else:
            ax.tick_params(labelbottom=False)

        # y label — left column only
        if i_c == 0:
            ax.set_ylabel(
                f"{material_label(mat)}\n$T_{{\\rm max}}$ [°C]",
                fontsize=9, color=mc,
            )
        else:
            ax.tick_params(labelleft=False)

# Shared legend — top-left panel
fig.legend(
    handles=[
        Line2D([0], [0], color="0.3", ls="-",  lw=1.3, label="High $T$"),
        Line2D([0], [0], color="0.3", ls="--", lw=1.3, alpha=0.65,
               label="Low $T$"),
    ],
    loc="lower center", frameon=False, ncol=2,
    fontsize=9,
    handlelength=1.6,
)

fig.tight_layout(rect=(0, 0.1, 1, 1))
save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
