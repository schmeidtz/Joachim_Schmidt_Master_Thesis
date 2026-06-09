"""
fig04_ext_strain_yy.py
========================
Per-extensometer εyy strain vs normalised time.
E3 = top, E4 = mid, E5 = bottom (plate side).
Mean ± 95 % CI bands.  High χ: solid; Low χ: dashed.

Grid: N_materials rows × N_orientations cols.

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
    HORIZ_SENSORS, EXT_COLORS_YY, EXT_LABELS_YY,
    material_color, material_label, orientation_label,
    parse_specimen_name, ci95_profile, DATA_DIR,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIGURE_BASENAME = "fig04_ext_strain_yy"
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

# ── Pool per (mat, ori, cryst, sensor) ────────────────────────────────────────
buckets = {}
for name, res in processed_results.items():
    meta = parse_specimen_name(name)
    if meta is None:
        continue
    mat   = meta["material_full"]
    ori   = meta["orientation"]
    cryst = meta["crystallinity"]
    t_raw = np.asarray(res.get("time", []), float)
    if t_raw.size < 5:
        continue
    t_norm = (t_raw - t_raw[0]) / max(t_raw[-1] - t_raw[0], 1.0)

    for s in HORIZ_SENSORS:
        eps = np.asarray(res.get("strains", {}).get(s, []), float)
        if eps.size != t_raw.size:
            continue
        ok  = np.isfinite(t_norm) & np.isfinite(eps)
        if ok.sum() < 5:
            continue
        arr = np.interp(t_grid, t_norm[ok], eps[ok],
                        left=np.nan, right=np.nan) * 1e6  # µm/m
        buckets.setdefault((mat, ori, cryst, s), []).append(arr)

means = {}
cis   = {}
for k, v in buckets.items():
    stack = np.vstack(v)
    m, ci = ci95_profile(stack)
    means[k] = m
    cis[k]   = ci

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
    mc = material_color(mat)
    for i_c, ori in enumerate(orientations):
        ax = axes[i_r, i_c]

        for s in HORIZ_SENSORS:
            col = EXT_COLORS_YY[s]
            for cryst in CRYST_ORDER:
                ls  = "-"  if cryst == "High" else "--"
                a_l = 1.0  if cryst == "High" else 0.65
                a_f = 0.14 if cryst == "High" else 0.07
                key = (mat, ori, cryst, s)
                if key not in means:
                    continue
                mu = means[key]
                ci = cis[key]
                valid = np.isfinite(mu)
                ax.plot(t_grid[valid], mu[valid],
                        color=col, ls=ls, lw=1.2, alpha=a_l, zorder=3)
                ax.fill_between(
                    t_grid[valid],
                    (mu - ci)[valid], (mu + ci)[valid],
                    color=col, alpha=a_f, linewidth=0, zorder=2,
                )

        ax.axhline(0, color="0.7", lw=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if i_r == 0:
            ax.set_title(orientation_label(ori), pad=4, fontsize=9)
        if i_r == n_r - 1:
            ax.set_xlabel("Normalised time [—]", fontsize=9)
        else:
            ax.tick_params(labelbottom=False)
        if i_c == 0:
            ax.set_ylabel(
                f"{material_label(mat)}\n"
                r"$\varepsilon_{yy}$ [µm m$^{-1}$]",
                fontsize=9, color=mc,
            )
        else:
            ax.tick_params(labelleft=False)

# Sensor legend (top-left), crystallinity note
legend_handles = [
    Line2D([0], [0], color=EXT_COLORS_YY[s], lw=1.3, label=EXT_LABELS_YY[s])
    for s in HORIZ_SENSORS
] + [
    Line2D([0], [0], color="0.4", ls="-",  lw=1.1, label="High $T$"),
    Line2D([0], [0], color="0.4", ls="--", lw=1.1, label="Low $T$"),
]
fig.legend(handles=legend_handles, loc="lower center", frameon=False, ncol=5, fontsize=9,
                  handlelength=1.6)

fig.tight_layout(rect=(0, 0.1, 1, 1))
save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
