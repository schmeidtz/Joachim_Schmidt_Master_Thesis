"""
fig02_strain_vs_temperature.py
================================
Strain ε(T) curves — heating phase (solid, mean ± 95 % CI) and
cooling phase (dashed, mean ± 95 % CI) for εzz and εyy.

Grid: N_materials rows × N_orientations cols.
Colour encodes direction (εzz navy, εyy crimson).
Crystallinity: High = solid, Low = dashed.

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
    MATERIAL_ORDER, ORIENTATION_ORDER, CRYST_ORDER,
    VERT_SENSORS, HORIZ_SENSORS, COLOR_ZZ, COLOR_YY,
    material_color, material_label, orientation_label,
    parse_specimen_name, temp_axis, mean_strain,
    ci95_profile, DATA_DIR, tg_for,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIGURE_BASENAME = "fig02_strain_vs_temperature"
FIG_WIDTH_MM    = 178
ROW_HEIGHT_MM   = 52

T_GRID = np.linspace(20, 220, 400)   # °C interpolation grid

# ── Load data ──────────────────────────────────────────────────────────────────
pkl = DATA_DIR / "processed_results.pkl"
if not pkl.exists():
    raise FileNotFoundError(f"{pkl}\nRun prepare_exports.py from the notebook first.")

with open(pkl, "rb") as f:
    processed_results = pickle.load(f)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _heating_cooling_eps(res, sensors):
    """
    Return (T_heat, eps_heat, T_cool, eps_cool) — sorted by temperature,
    split at the peak temperature index.
    """
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


def _interp_on_grid(T_seg, eps_seg):
    if T_seg is None or len(T_seg) < 4:
        return np.full(len(T_GRID), np.nan)
    idx = np.argsort(T_seg)
    _, u = np.unique(T_seg[idx], return_index=True)
    Ts, Es = T_seg[idx][u], eps_seg[idx][u]
    return np.interp(T_GRID, Ts, Es, left=np.nan, right=np.nan)


def _pool(processed_results, mat, ori, cryst, sensors):
    """Collect (heat_curves, cool_curves) for one condition."""
    heat, cool = [], []
    for name, res in processed_results.items():
        meta = parse_specimen_name(name)
        if meta is None:
            continue
        if (meta["material_full"] != mat or meta["orientation"] != str(ori)
                or meta["crystallinity"] != cryst):
            continue
        Th, eh, Tc, ec = _heating_cooling_eps(res, sensors)
        heat.append(_interp_on_grid(Th, eh))
        cool.append(_interp_on_grid(Tc, ec))
    return heat, cool


# ── Resolve materials / orientations ──────────────────────────────────────────
all_meta = [parse_specimen_name(n) for n in processed_results]
all_meta = [m for m in all_meta if m is not None]
materials    = [m for m in MATERIAL_ORDER if any(x["material_full"] == m for x in all_meta)]
orientations = [o for o in ORIENTATION_ORDER if any(x["orientation"] == o for x in all_meta)]

# ── Plot ───────────────────────────────────────────────────────────────────────
set_journal_style()

n_r, n_c = len(materials), len(orientations)
fig, axes = plt.subplots(
    n_r, n_c,
    figsize=mm_to_inch(FIG_WIDTH_MM, ROW_HEIGHT_MM * n_r),
    sharex=True, sharey="row",
    squeeze=False,
)

for i_r, mat in enumerate(materials):
    mc = material_color(mat)
    tg = tg_for(mat)
    for i_c, ori in enumerate(orientations):
        ax = axes[i_r, i_c]

        for sensors, col, dir_lbl in [
            (VERT_SENSORS,  COLOR_ZZ, "zz"),
            (HORIZ_SENSORS, COLOR_YY, "yy"),
        ]:
            for cryst in CRYST_ORDER:
                ls  = "-"  if cryst == "High" else "--"
                a_l = 1.0  if cryst == "High" else 0.65
                a_f = 0.14 if cryst == "High" else 0.07

                heat_curves, cool_curves = _pool(processed_results, mat, ori, cryst, sensors)
                if not heat_curves:
                    continue

                stack_h = np.vstack(heat_curves)
                stack_c = np.vstack(cool_curves)
                mu_h, ci_h = ci95_profile(stack_h)
                mu_c, ci_c = ci95_profile(stack_c)

                # Convert to µm/m
                scale = 1e6
                ax.plot(T_GRID, mu_h * scale,
                        color=col, ls=ls, lw=1.2, alpha=a_l, zorder=3)
                ax.fill_between(
                    T_GRID,
                    (mu_h - ci_h) * scale,
                    (mu_h + ci_h) * scale,
                    color=col, alpha=a_f, linewidth=0, zorder=2,
                )
                # Cooling — same colour, lighter
                ax.plot(T_GRID, mu_c * scale,
                        color=col, ls=ls, lw=0.8, alpha=a_l * 0.55, zorder=2)

        ax.axvline(tg, color="#6a0dad", ls="--", lw=0.7, alpha=0.7, zorder=1)
        ax.axhline(0,  color="0.7",    lw=0.5, zorder=0)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if i_r == 0:
            ax.set_title(orientation_label(ori), pad=4, fontsize=9)
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

# Shared legend — top-left panel
fig.legend(
    handles=[
        Line2D([0], [0], color=COLOR_ZZ, ls="-",  lw=1.2,
               label=r"$\varepsilon_{zz}$ (heating, mean ± 95 % CI)"),
        Line2D([0], [0], color=COLOR_YY, ls="-",  lw=1.2,
               label=r"$\varepsilon_{yy}$ (heating, mean ± 95 % CI)"),
        Line2D([0], [0], color="0.4",   ls="--", lw=0.8, alpha=0.6,
               label="Cooling"),
        Line2D([0], [0], color="#6a0dad", ls="--", lw=0.7, label="$T_g$"),
    ],
    loc="lower center", frameon=False, ncol=4, fontsize=9, handlelength=1.8,
)

fig.tight_layout(rect=(0, 0.1, 1, 1))
save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
