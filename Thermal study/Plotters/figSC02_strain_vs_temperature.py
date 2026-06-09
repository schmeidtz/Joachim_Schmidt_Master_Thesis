"""
figSC02_strain_vs_temperature.py
==================================
Strain ε(T) curves in the SPECIMEN coordinate system.

For each specimen the heating (solid, mean ± 95 % CI) and cooling (dashed,
lighter) curves from VERT sensors (DIC zz) and HORIZ sensors (DIC yy) are
remapped to physical specimen directions:
    O1 : zz → build,  yy → transraster
    O2 : zz → build,  yy → raster
    O3 : zz → raster, yy → build

Specimens from all contributing orientations are pooled per direction.

Grid layout:
    Rows    = materials
    Columns = physical direction (build | raster | transraster)

Line colour = direction colour.  High χ = solid/bright, Low χ = dashed/muted.
Tg is marked with a vertical dashed line.

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
    parse_specimen_name, temp_axis, mean_strain,
    ci95_profile, DATA_DIR, tg_for,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIGURE_BASENAME = "figSC02_strain_vs_temperature"
FIG_WIDTH_MM    = 178
ROW_HEIGHT_MM   = 52

T_GRID = np.linspace(20, 220, 400)

# ── Load data ──────────────────────────────────────────────────────────────────
pkl = DATA_DIR / "processed_results.pkl"
if not pkl.exists():
    raise FileNotFoundError(f"{pkl}\nRun prepare_exports.py first.")

with open(pkl, "rb") as f:
    processed_results = pickle.load(f)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _heating_cooling_eps(res, sensors):
    T   = temp_axis(res)
    eps = mean_strain(res, sensors)
    if T is None or eps is None:
        return None, None, None, None
    T   = np.asarray(T,   float)
    eps = np.asarray(eps, float)
    ok  = np.isfinite(T) & np.isfinite(eps)
    if ok.sum() < 10:
        return None, None, None, None
    T, eps = T[ok], eps[ok]
    i_pk = int(np.argmax(T))
    return T[:i_pk+1], eps[:i_pk+1], T[i_pk:], eps[i_pk:]


def _interp(T_seg, eps_seg):
    if T_seg is None or len(T_seg) < 4:
        return np.full(len(T_GRID), np.nan)
    idx = np.argsort(T_seg)
    _, u = np.unique(T_seg[idx], return_index=True)
    Ts, Es = T_seg[idx][u], eps_seg[idx][u]
    return np.interp(T_GRID, Ts, Es, left=np.nan, right=np.nan)


# ── Pool per (mat, sc_direction, cryst) ───────────────────────────────────────
heat_pool = {}
cool_pool = {}

for name, res in processed_results.items():
    meta = parse_specimen_name(name)
    if meta is None:
        continue
    mat      = meta["material_full"]
    ori      = meta["orientation"]
    cryst    = meta["crystallinity"]
    ori_grp  = "O3" if ori == "3" else "O12"
    for sensors, dic_dir in [(VERT_SENSORS, "zz"), (HORIZ_SENSORS, "yy")]:
        sc_dir = zz_direction(ori) if dic_dir == "zz" else yy_direction(ori)
        Th, eh, Tc, ec = _heating_cooling_eps(res, sensors)
        key = (mat, sc_dir, cryst, ori_grp)
        heat_pool.setdefault(key, []).append(_interp(Th, eh))
        cool_pool.setdefault(key, []).append(_interp(Tc, ec))

# ── Resolve present materials ──────────────────────────────────────────────────
all_meta  = [parse_specimen_name(n) for n in processed_results]
all_meta  = [m for m in all_meta if m is not None]
materials = [m for m in MATERIAL_ORDER
             if any(x["material_full"] == m for x in all_meta)]

# ── Plot ───────────────────────────────────────────────────────────────────────
set_journal_style()

n_r = len(materials)
n_c = len(SC_DIRECTION_ORDER)

fig, axes = plt.subplots(
    n_r, n_c,
    figsize=mm_to_inch(FIG_WIDTH_MM, ROW_HEIGHT_MM * n_r),
    sharex=True, sharey="row",
    squeeze=False,
)

for i_r, mat in enumerate(materials):
    mc = material_color(mat)
    tg = tg_for(mat)
    for i_c, direction in enumerate(SC_DIRECTION_ORDER):
        ax      = axes[i_r, i_c]
        dir_col = SC_DIRECTION_COLORS[direction]

        # O1+O2 first (main, full weight), then O3 (secondary, muted)
        for ori_grp, lw_h, a_base, a_f_base in [("O12", 1.2, 1.0, 0.14),
                                                  ("O3",  0.8, 0.50, 0.07)]:
            for cryst in CRYST_ORDER:
                ls  = "-"  if cryst == "High" else "--"
                a_l = a_base       if cryst == "High" else a_base * 0.65
                a_f = a_f_base     if cryst == "High" else a_f_base * 0.5

                key = (mat, direction, cryst, ori_grp)
                hc  = heat_pool.get(key, [])
                cc  = cool_pool.get(key, [])
                if not hc:
                    continue

                scale   = 1e6
                stack_h = np.vstack(hc)
                mu_h, ci_h = ci95_profile(stack_h)
                ax.plot(T_GRID, mu_h * scale,
                        color=dir_col, ls=ls, lw=lw_h, alpha=a_l, zorder=3)
                ax.fill_between(
                    T_GRID,
                    (mu_h - ci_h) * scale,
                    (mu_h + ci_h) * scale,
                    color=dir_col, alpha=a_f, linewidth=0, zorder=2,
                )
                if cc:
                    stack_c = np.vstack(cc)
                    mu_c, _ = ci95_profile(stack_c)
                    ax.plot(T_GRID, mu_c * scale,
                            color=dir_col, ls=ls, lw=lw_h * 0.7,
                            alpha=a_l * 0.55, zorder=2)

        ax.axvline(tg, color="#6a0dad", ls="--", lw=0.7, alpha=0.7, zorder=1)
        ax.axhline(0,  color="0.7",    lw=0.5, zorder=0)
        ax.set_ylim(-1000, 8000)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if i_r == 0:
            ax.set_title(sc_direction_label(direction), pad=4, fontsize=9,
                         color=dir_col)
        if i_r == n_r - 1:
            ax.set_xlabel("Temperature [°C]", fontsize=9)
        else:
            ax.tick_params(labelbottom=False)
        if i_c == 0:
            ax.set_ylabel(
                f"{material_label(mat)}\nStrain [µm m$^{{-1}}$]",
                fontsize=9, color=mc,
            )
        else:
            ax.tick_params(labelleft=False)

fig.legend(
    handles=[
        Line2D([0], [0], color="0.35", ls="-",  lw=1.2,
               label="Heating, High $T$, O1+O2  (mean ± 95 % CI)"),
        Line2D([0], [0], color="0.35", ls="--", lw=1.2, alpha=0.65,
               label="Heating, Low $T$, O1+O2"),
        Line2D([0], [0], color="0.35", ls="-",  lw=0.8, alpha=0.55,
               label="Cooling, O1+O2"),
        Line2D([0], [0], color="0.55", ls="-",  lw=0.8,
               label="Heating, O3  (grey)"),
        Line2D([0], [0], color="#6a0dad", ls="--", lw=0.7, label="$T_g$"),
    ],
    loc="lower center",
    ncol=5,
    fontsize=9,
    handlelength=1.8,
    frameon=False,
)
fig.tight_layout(rect=(0, 0.06, 1, 1))

save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
