"""
figSC08_tyt_heatmaps.py
========================
Through-thickness temperature heatmaps T(y, t_norm) and ΔT(y, t_norm) in the
SPECIMEN coordinate system.

Layout is identical to fig08 (N_materials rows × N_orientations cols) but the
strain-heatmap rows C and D are now labelled and coloured by PHYSICAL DIRECTION
rather than DIC axis label:

    Orientation 1:  VERT (E0–E2) → Δε_build      HORIZ (E3–E5) → Δε_transraster
    Orientation 2:  VERT (E0–E2) → Δε_build       HORIZ (E3–E5) → Δε_raster
    Orientation 3:  VERT (E0–E2) → Δε_raster      HORIZ (E3–E5) → Δε_build

Colour maps for the strain blocks use the direction colour:
    build       → Blues
    raster      → Reds
    transraster → Greens

Data requirements:
  data_exports/spec_fields.pkl
  data_exports/processed_results.pkl
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
from journal_style import (
    set_journal_style, save_figure, mm_to_inch,
    MATERIAL_ORDER, ORIENTATION_ORDER,
    VERT_SENSORS, HORIZ_SENSORS,
    SC_DIRECTION_COLORS, sc_direction_label,
    zz_direction, yy_direction,
    material_color, material_label,
    parse_specimen_name, DATA_DIR,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIGURE_BASENAME = "figSC08_tyt_heatmaps"
FIG_WIDTH_MM    = 178
N_TIME          = 101
L_MM            = 20.0
N_COLS          = 30

_ZZ_ORDER = ["E0", "E1", "E2"]
_YY_ORDER = ["E3", "E4", "E5"]

_MAT_INTERNAL = {"PEEK_CF": "PEEK-CF", "PPS_CF": "PPS-CF",
                 "PPS_NEAT": "PPS neat", "PEEK_NEAT": "PEEK neat"}

# Direction → matplotlib colormap name
_DIR_CMAP = {
    "build":       "Blues",
    "raster":      "Reds",
    "transraster": "Greens",
}

# ── Load data ──────────────────────────────────────────────────────────────────
spec_pkl = DATA_DIR / "spec_fields.pkl"
proc_pkl = DATA_DIR / "processed_results.pkl"
if not spec_pkl.exists():
    raise FileNotFoundError(f"{spec_pkl}\nRun prepare_exports.py first.")
if not proc_pkl.exists():
    raise FileNotFoundError(f"{proc_pkl}\nRun prepare_exports.py first.")

with open(spec_pkl, "rb") as f:
    spec_fields = pickle.load(f)
with open(proc_pkl, "rb") as f:
    processed_results = pickle.load(f)

t_grid = np.linspace(0, 1, N_TIME)
x_mm   = np.linspace(0, L_MM, N_COLS)


# ── Helpers (same as fig08) ────────────────────────────────────────────────────
def _cond_key(cond):
    mat_raw = cond.get("material_full") or cond.get("material", "")
    mat = _MAT_INTERNAL.get(mat_raw, mat_raw)
    ori = str(cond.get("orientation", ""))
    return mat, ori


def _pool_T_fields(sf, mat, ori):
    stacks = []
    for d in sf.values():
        m, o = _cond_key(d["cond"])
        if m != mat or o != ori:
            continue
        field = np.asarray(d.get("field", d.get("T_field", [])), float)
        if field.ndim != 2 or field.shape[1] != N_COLS:
            continue
        nt = field.shape[0]
        if nt != N_TIME:
            t_raw = np.linspace(0, 1, nt)
            field = np.vstack(
                [np.interp(t_grid, t_raw, field[:, c]) for c in range(N_COLS)]
            ).T
        stacks.append(field)
    if not stacks:
        return None
    return np.nanmean(np.stack(stacks, axis=0), axis=0)


def _pool_strain(pr, mat, ori, sensors):
    pool = {s: [] for s in sensors}
    for name, res in pr.items():
        meta = parse_specimen_name(name)
        if meta is None or meta["material_full"] != mat or meta["orientation"] != ori:
            continue
        t_raw = np.asarray(res.get("time", []), float)
        if t_raw.size < 5:
            continue
        t_norm = (t_raw - t_raw[0]) / max(t_raw[-1] - t_raw[0], 1.0)
        for s in sensors:
            eps = np.asarray(res.get("strains", {}).get(s, []), float)
            if eps.size != t_raw.size:
                continue
            ok = np.isfinite(t_norm) & np.isfinite(eps)
            if ok.sum() < 5:
                continue
            arr = np.interp(t_grid, t_norm[ok], eps[ok] * 1e6,
                            left=np.nan, right=np.nan)
            arr -= np.nanmean(arr[:3])
            pool[s].append(arr)
    cols = []
    for s in sensors:
        if pool[s]:
            cols.append(np.nanmean(np.vstack(pool[s]), axis=0))
        else:
            cols.append(np.full(N_TIME, np.nan))
    return np.column_stack(cols)


# ── Resolve which (mat, ori) exist ─────────────────────────────────────────────
all_meta     = [parse_specimen_name(n) for n in processed_results]
all_meta     = [m for m in all_meta if m is not None]
materials    = [m for m in MATERIAL_ORDER if any(x["material_full"] == m for x in all_meta)]
orientations = [o for o in ORIENTATION_ORDER if any(x["orientation"] == o for x in all_meta)]

# ── Plot ───────────────────────────────────────────────────────────────────────
set_journal_style()

n_mat = len(materials)
n_ori = len(orientations)

BLOCK_MM = 18
GAP_MM   = 4
TOTAL_H  = n_mat * (4 * BLOCK_MM + 3 * GAP_MM) + 15

fig = plt.figure(figsize=mm_to_inch(FIG_WIDTH_MM, TOTAL_H))

outer = gridspec.GridSpec(n_mat, n_ori, figure=fig,
                          hspace=0.55, wspace=0.30,
                          left=0.10, right=0.92, top=0.97, bottom=0.04)

for i_r, mat in enumerate(materials):
    mc = material_color(mat)
    for i_c, ori in enumerate(orientations):

        # Physical directions for this orientation
        dir_zz  = zz_direction(ori)   # direction measured by VERT sensors
        dir_yy  = yy_direction(ori)   # direction measured by HORIZ sensors

        inner = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=outer[i_r, i_c],
            hspace=0.08, height_ratios=[3, 3, 1.5, 1.5],
        )
        ax_T    = fig.add_subplot(inner[0])
        ax_dT   = fig.add_subplot(inner[1])
        ax_ezz  = fig.add_subplot(inner[2])
        ax_eyy  = fig.add_subplot(inner[3])

        # ── T(y, t) ──────────────────────────────────────────────────────────
        T_field = _pool_T_fields(spec_fields, mat, ori)
        if T_field is not None:
            ax_T.imshow(
                T_field.T, aspect="auto", origin="lower",
                extent=[0, 1, 0, L_MM],
                cmap="plasma", interpolation="bilinear",
            )
        else:
            ax_T.text(0.5, 0.5, "no data", ha="center", va="center",
                      transform=ax_T.transAxes, fontsize=9, color="0.5")
        ax_T.set_ylabel("$y$ [mm]", fontsize=7, labelpad=2)
        ax_T.tick_params(labelbottom=False, labelsize=5)

        # ── ΔT(y, t) ─────────────────────────────────────────────────────────
        if T_field is not None:
            dT_field = T_field - T_field[0, :]
            vext = np.nanpercentile(np.abs(dT_field), 98)
            ax_dT.imshow(
                dT_field.T, aspect="auto", origin="lower",
                extent=[0, 1, 0, L_MM],
                cmap="RdBu_r", vmin=-vext, vmax=vext,
                interpolation="bilinear",
            )
        else:
            ax_dT.text(0.5, 0.5, "no data", ha="center", va="center",
                       transform=ax_dT.transAxes, fontsize=9, color="0.5")
        ax_dT.set_ylabel("$y$ [mm]", fontsize=7, labelpad=2)
        ax_dT.tick_params(labelbottom=False, labelsize=5)

        # ── Δε for the VERT-sensor direction (zz → dir_zz) ───────────────────
        e_vert = _pool_strain(processed_results, mat, ori, _ZZ_ORDER)
        vv     = np.nanpercentile(np.abs(e_vert), 98) if np.any(np.isfinite(e_vert)) else 1
        ax_ezz.imshow(
            e_vert.T, aspect="auto", origin="lower",
            extent=[0, 1, -0.5, len(_ZZ_ORDER) - 0.5],
            cmap=_DIR_CMAP.get(dir_zz, "Blues"),
            vmin=0, vmax=max(vv, 1e-9),
            interpolation="nearest",
        )
        ax_ezz.set_yticks(range(len(_ZZ_ORDER)))
        ax_ezz.set_yticklabels(["E2", "E1", "E0"], fontsize=6)
        ax_ezz.tick_params(labelbottom=False, labelsize=5)
        dir_zz_col = SC_DIRECTION_COLORS.get(dir_zz, "0.3")
        ax_ezz.set_ylabel(
            rf"$\Delta\varepsilon_{{\rm {dir_zz}}}$",
            fontsize=9, labelpad=2, color=dir_zz_col,
        )

        # ── Δε for the HORIZ-sensor direction (yy → dir_yy) ──────────────────
        e_horiz = _pool_strain(processed_results, mat, ori, _YY_ORDER)
        vy      = np.nanpercentile(np.abs(e_horiz), 98) if np.any(np.isfinite(e_horiz)) else 1
        ax_eyy.imshow(
            e_horiz.T, aspect="auto", origin="lower",
            extent=[0, 1, -0.5, len(_YY_ORDER) - 0.5],
            cmap=_DIR_CMAP.get(dir_yy, "Greens"),
            vmin=0, vmax=max(vy, 1e-9),
            interpolation="nearest",
        )
        ax_eyy.set_yticks(range(len(_YY_ORDER)))
        ax_eyy.set_yticklabels(["E5", "E4", "E3"], fontsize=6)
        ax_eyy.set_xlabel("Norm. time", fontsize=7, labelpad=2)
        ax_eyy.tick_params(labelsize=5)
        dir_yy_col = SC_DIRECTION_COLORS.get(dir_yy, "0.3")
        ax_eyy.set_ylabel(
            rf"$\Delta\varepsilon_{{\rm {dir_yy}}}$",
            fontsize=9, labelpad=2, color=dir_yy_col,
        )

        # Column / row labels
        if i_r == 0:
            ax_T.set_title(
                f"{sc_direction_label(dir_zz)} | {sc_direction_label(dir_yy)}",
                fontsize=9, pad=3,
            )
        if i_c == 0:
            ax_T.text(-0.22, 0.5, material_label(mat), transform=ax_T.transAxes,
                      fontsize=9, color=mc, rotation=90, va="center", ha="right")

save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
