"""
fig05_cte_vs_temperature.py
============================
Continuous CTE α(T) curves — mean ± 95 % CI, heating phase only.
αzz (navy) and αyy (crimson). High χ: solid; Low χ: dashed.
Tg marked with dashed vertical line.

Savitzky-Golay smoothed derivative of the ε(T) interpolant.

Grid: N_materials rows × N_orientations cols.

Data requirement: data_exports/processed_results.pkl
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

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
FIGURE_BASENAME = "fig05_cte_vs_temperature"
FIG_WIDTH_MM    = 178
ROW_HEIGHT_MM   = 52

T_GRID   = np.linspace(25, 220, 220)   # °C
SG_WIN   = 41
SG_POLY  = 3

# ── Load data ──────────────────────────────────────────────────────────────────
pkl = DATA_DIR / "processed_results.pkl"
if not pkl.exists():
    raise FileNotFoundError(f"{pkl}\nRun prepare_exports.py first.")

with open(pkl, "rb") as f:
    processed_results = pickle.load(f)


# ── CTE computation ────────────────────────────────────────────────────────────
def _sg_adaptive(arr, win, poly):
    """Savitzky-Golay with fallback for short arrays."""
    n = np.sum(np.isfinite(arr))
    w = min(win, n - 1) if n > poly + 2 else None
    if w is None or w < poly + 2:
        return arr
    if w % 2 == 0:
        w -= 1
    filled = pd.Series(arr).interpolate("linear").bfill().ffill().values
    return savgol_filter(filled, w, poly)


def cte_on_grid(res, sensors):
    """Return smoothed α(T) on T_GRID [ppm/°C], heating segment only."""
    T   = temp_axis(res)
    eps = mean_strain(res, sensors)
    if T is None or eps is None:
        return None
    T   = np.asarray(T,   float)
    eps = np.asarray(eps, float)
    ok  = np.isfinite(T) & np.isfinite(eps)
    if ok.sum() < 10:
        return None
    T, eps = T[ok], eps[ok]
    # Heating segment only (up to peak T)
    i_pk = int(np.argmax(T))
    T, eps = T[:i_pk + 1], eps[:i_pk + 1]
    idx = np.argsort(T)
    _, u = np.unique(T[idx], return_index=True)
    Ts, Es = T[idx][u], eps[idx][u]
    e_on = interp1d(Ts, Es, bounds_error=False, fill_value=np.nan)(T_GRID)
    dT = np.gradient(T_GRID)
    de = np.gradient(e_on)
    with np.errstate(divide="ignore", invalid="ignore"):
        cte = np.where(np.abs(dT) > 1e-9, de / dT, np.nan) * 1e6  # ppm/°C
    ok2 = np.isfinite(cte)
    if ok2.sum() > SG_WIN + 5:
        cte = _sg_adaptive(cte, SG_WIN, SG_POLY)
        cte[~ok2] = np.nan
    return cte


# ── Pool per (mat, ori, cryst, direction) ─────────────────────────────────────
pool = {}
for name, res in processed_results.items():
    meta = parse_specimen_name(name)
    if meta is None:
        continue
    mat   = meta["material_full"]
    ori   = meta["orientation"]
    cryst = meta["crystallinity"]
    for sensors, direction in [(VERT_SENSORS, "zz"), (HORIZ_SENSORS, "yy")]:
        arr = cte_on_grid(res, sensors)
        if arr is not None:
            pool.setdefault((mat, ori, cryst, direction), []).append(arr)

# ── Resolve grid ───────────────────────────────────────────────────────────────
all_meta  = [parse_specimen_name(n) for n in processed_results]
all_meta  = [m for m in all_meta if m is not None]
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
    mc  = material_color(mat)
    tg  = tg_for(mat)
    for i_c, ori in enumerate(orientations):
        ax = axes[i_r, i_c]

        for direction, col in [("zz", COLOR_ZZ), ("yy", COLOR_YY)]:
            for cryst in CRYST_ORDER:
                ls  = "-"  if cryst == "High" else "--"
                a_l = 1.0  if cryst == "High" else 0.65
                a_f = 0.14 if cryst == "High" else 0.07
                key = (mat, ori, cryst, direction)
                curves = pool.get(key, [])
                if not curves:
                    continue
                stack = np.vstack(curves)
                mu, ci = ci95_profile(stack)
                valid = np.isfinite(mu)
                ax.plot(T_GRID[valid], mu[valid],
                        color=col, ls=ls, lw=1.2, alpha=a_l, zorder=3)
                ax.fill_between(
                    T_GRID[valid],
                    (mu - ci)[valid], (mu + ci)[valid],
                    color=col, alpha=a_f, linewidth=0, zorder=2,
                )

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
                f"{material_label(mat)}\n"
                r"$\alpha$ [ppm °C$^{-1}$]",
                fontsize=9, color=mc,
            )
        else:
            ax.tick_params(labelleft=False)

# Shared legend
fig.legend(
    handles=[
        Line2D([0], [0], color=COLOR_ZZ, ls="-", lw=1.2,
               label=r"$\alpha_{zz}$"),
        Line2D([0], [0], color=COLOR_YY, ls="-", lw=1.2,
               label=r"$\alpha_{yy}$"),
        Line2D([0], [0], color="0.4",    ls="--", lw=1.0, alpha=0.65,
               label="Low $T$"),
        Line2D([0], [0], color="#6a0dad", ls="--", lw=0.7,
               label="$T_g$"),
    ],
    loc="lower center", frameon=False, ncol=4, fontsize=9, handlelength=1.8,
)

fig.tight_layout(rect=(0, 0.1, 1, 1))
save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
