"""
fig08_tyt_heatmaps.py
======================
Through-thickness temperature heatmaps T(y, t_norm) and ΔT(y, t_norm).

Four blocks per (material × orientation) panel:
  Row A: mean T(y, t_norm)      — plasma colormap
  Row B: ΔT = T − T(t=0)       — RdBu_r colormap
  Row C: Δεzz(sensor, t_norm)  — Blues
  Row D: Δεyy(sensor, t_norm)  — Greens

Outer grid: N_materials rows × N_orientations cols.

Data requirements:
  data_exports/spec_fields.pkl        (through-thickness T fields)
  data_exports/processed_results.pkl  (extensometer strains)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import make_axes

sys.path.insert(0, str(Path(__file__).parent))
from journal_style import (
    set_journal_style, save_figure, mm_to_inch,
    MATERIAL_ORDER, ORIENTATION_ORDER, CRYST_ORDER,
    VERT_SENSORS, HORIZ_SENSORS,
    material_color, material_label, orientation_label,
    parse_specimen_name, DATA_DIR,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIGURE_BASENAME  = "fig08_tyt_heatmaps"
FIG_WIDTH_MM     = 178
N_TIME           = 101
L_MM             = 20.0    # specimen thickness
N_COLS           = 30      # through-thickness columns in spec_fields

# Extensometer order: hot→cool for zz, top→bottom for yy
_ZZ_ORDER  = ["E0", "E1", "E2"]   # E0 front/hot, E2 back/cool
_YY_ORDER  = ["E3", "E4", "E5"]   # E3 top, E5 plate

# Map augmented naming → clean pipeline naming
_MAT_INTERNAL = {"PEEK_CF": "PEEK-CF", "PPS_CF": "PPS-CF",
                 "PPS_NEAT": "PPS neat", "PEEK_NEAT": "PEEK neat"}

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


# ── Helpers ────────────────────────────────────────────────────────────────────
def _cond_key(cond):
    """Extract (material_full, orientation) from a condition dict."""
    mat_raw = cond.get("material_full") or cond.get("material", "")
    mat = _MAT_INTERNAL.get(mat_raw, mat_raw)
    ori = str(cond.get("orientation", ""))
    return mat, ori


def _pool_T_fields(spec_fields, mat, ori):
    """Return stacked T fields (n_reps, N_TIME, N_COLS) for (mat, ori)."""
    stacks = []
    for d in spec_fields.values():
        m, o = _cond_key(d["cond"])
        if m != mat or o != ori:
            continue
        field = np.asarray(d.get("field", d.get("T_field", [])), float)
        if field.ndim != 2 or field.shape[1] != N_COLS:
            continue
        # Resample to common time grid if needed
        nt = field.shape[0]
        if nt != N_TIME:
            t_raw = np.linspace(0, 1, nt)
            field = np.vstack(
                [np.interp(t_grid, t_raw, field[:, c]) for c in range(N_COLS)]
            ).T
        stacks.append(field)
    if not stacks:
        return None
    return np.nanmean(np.stack(stacks, axis=0), axis=0)  # (N_TIME, N_COLS)


def _pool_strain(processed_results, mat, ori, sensors):
    """Return Δε heatmap (N_TIME, N_sensors) for (mat, ori), µm/m."""
    pool = {s: [] for s in sensors}
    for name, res in processed_results.items():
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
            arr -= np.nanmean(arr[:3])  # Δε from baseline
            pool[s].append(arr)
    cols = []
    for s in sensors:
        if pool[s]:
            cols.append(np.nanmean(np.vstack(pool[s]), axis=0))
        else:
            cols.append(np.full(N_TIME, np.nan))
    return np.column_stack(cols)  # (N_TIME, N_sensors)


# ── Resolve which (mat, ori) combinations exist ────────────────────────────────
all_meta = [parse_specimen_name(n) for n in processed_results]
all_meta = [m for m in all_meta if m is not None]
materials    = [m for m in MATERIAL_ORDER if any(x["material_full"] == m for x in all_meta)]
orientations = [o for o in ORIENTATION_ORDER if any(x["orientation"] == o for x in all_meta)]

# ── Plot ───────────────────────────────────────────────────────────────────────
set_journal_style()

n_mat = len(materials)
n_ori = len(orientations)

# Figure height: 4 blocks per cell, each ~18 mm, plus spacing
BLOCK_MM = 18
GAP_MM   = 4
TOTAL_H  = n_mat * (4 * BLOCK_MM + 3 * GAP_MM) + 15

fig = plt.figure(figsize=mm_to_inch(FIG_WIDTH_MM, TOTAL_H))

# Outer grid: n_mat rows × n_ori cols
outer = gridspec.GridSpec(n_mat, n_ori, figure=fig,
                          hspace=0.55, wspace=0.30,
                          left=0.10, right=0.92, top=0.97, bottom=0.04)

for i_r, mat in enumerate(materials):
    mc = material_color(mat)
    for i_c, ori in enumerate(orientations):
        # Inner 4-row grid within this cell
        inner = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=outer[i_r, i_c],
            hspace=0.08, height_ratios=[3, 3, 1.5, 1.5],
        )
        ax_T   = fig.add_subplot(inner[0])
        ax_dT  = fig.add_subplot(inner[1])
        ax_ezz = fig.add_subplot(inner[2])
        ax_eyy = fig.add_subplot(inner[3])

        # ── T(y, t) ──────────────────────────────────────────────────────────
        T_field = _pool_T_fields(spec_fields, mat, ori)
        if T_field is not None:
            im = ax_T.imshow(
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
            dT_field = T_field - T_field[0, :]  # relative to first frame
            vext = np.nanpercentile(np.abs(dT_field), 98)
            im2 = ax_dT.imshow(
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

        # ── Δεzz heatmap ─────────────────────────────────────────────────────
        ezz = _pool_strain(processed_results, mat, ori, _ZZ_ORDER)
        vz = np.nanpercentile(np.abs(ezz), 98) if np.any(np.isfinite(ezz)) else 1
        ax_ezz.imshow(
            ezz.T, aspect="auto", origin="lower",
            extent=[0, 1, -0.5, len(_ZZ_ORDER) - 0.5],
            cmap="Blues", vmin=0, vmax=max(vz, 1e-9),
            interpolation="nearest",
        )
        ax_ezz.set_yticks(range(len(_ZZ_ORDER)))
        ax_ezz.set_yticklabels(["E2", "E1", "E0"], fontsize=6)
        ax_ezz.tick_params(labelbottom=False, labelsize=5)
        ax_ezz.set_ylabel(r"$\Delta\varepsilon_{zz}$", fontsize=7, labelpad=2)

        # ── Δεyy heatmap ─────────────────────────────────────────────────────
        eyy = _pool_strain(processed_results, mat, ori, _YY_ORDER)
        vy = np.nanpercentile(np.abs(eyy), 98) if np.any(np.isfinite(eyy)) else 1
        ax_eyy.imshow(
            eyy.T, aspect="auto", origin="lower",
            extent=[0, 1, -0.5, len(_YY_ORDER) - 0.5],
            cmap="Greens", vmin=0, vmax=max(vy, 1e-9),
            interpolation="nearest",
        )
        ax_eyy.set_yticks(range(len(_YY_ORDER)))
        ax_eyy.set_yticklabels(["E5", "E4", "E3"], fontsize=6)
        ax_eyy.set_xlabel("Norm. time", fontsize=7, labelpad=2)
        ax_eyy.tick_params(labelsize=5)
        ax_eyy.set_ylabel(r"$\Delta\varepsilon_{yy}$", fontsize=7, labelpad=2)

        # Column / row labels
        if i_r == 0:
            ax_T.set_title(orientation_label(ori), fontsize=9, pad=3)
        if i_c == 0:
            ax_T.text(-0.22, 0.5, material_label(mat), transform=ax_T.transAxes,
                      fontsize=9, color=mc, rotation=90, va="center", ha="right")

save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
