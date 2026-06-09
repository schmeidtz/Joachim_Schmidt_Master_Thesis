"""
figSC_gradient_lamp_off.py
===========================
2-D heatmap of the through-thickness temperature field T(y, t) and spatial
gradient ΔT(y, t) = T(y, t) − T(cool face, t) in a ±30 s window centred on
the lamp-off event (detected as the frame of peak mean specimen temperature).

Layout:  N_materials rows × N_orientations cols.
Each cell: two stacked imshow panels
  Top    — T(y, t) [°C]          colormap: plasma
  Bottom — ΔT(y, t) [°C]         colormap: RdBu_r, symmetric about 0

x-axis  : time relative to lamp-off [s], −30 → +30
y-axis  : through-thickness position y [mm], 0 (hot face) → 20 mm (cool face)

A dashed vertical line marks t = 0 (lamp-off) in every panel.
One colorbar per panel type on the rightmost column of each material row.

Mean across all replicates (High + Low χ pooled) per (material, orientation).

Time axis: uses t_s / time key from spec_fields if present; otherwise infers
from cond["fps"] (default 25 Hz).

Data requirement: data_exports/spec_fields.pkl
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).parent))
from journal_style import (
    set_journal_style, save_figure, mm_to_inch,
    MATERIAL_ORDER, ORIENTATION_ORDER,
    material_color, material_label, orientation_label,
    parse_specimen_name, DATA_DIR,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIGURE_BASENAME  = "figSC_gradient_lamp_off"
FIG_WIDTH_MM     = 178
ROW_HEIGHT_MM    = 68    # per-material row
L_MM             = 20.0
N_COLS           = 30
T_WINDOW_S       = 30    # seconds each side of lamp-off
N_TIME_DENSE     = 121   # grid points → 0.5 s resolution

t_dense = np.linspace(-T_WINDOW_S, T_WINDOW_S, N_TIME_DENSE)  # relative to lamp-off
x_mm    = np.linspace(0, L_MM, N_COLS)

# Map augmented naming → clean pipeline
_MAT_INTERNAL = {"PEEK_CF": "PEEK-CF", "PPS_CF": "PPS-CF",
                 "PPS_NEAT": "PPS neat", "PEEK_NEAT": "PEEK neat"}

# ── Load data ──────────────────────────────────────────────────────────────────
pkl = DATA_DIR / "spec_fields.pkl"
if not pkl.exists():
    raise FileNotFoundError(f"{pkl}\nRun prepare_exports.py first.")

with open(pkl, "rb") as f:
    spec_fields = pickle.load(f)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _resolve_meta(name):
    meta = parse_specimen_name(name)
    if meta is not None:
        return meta
    parts = name.split("_")
    if len(parts) < 3:
        return None
    raw = f"{parts[0]}_{parts[1]}".upper()
    mat = _MAT_INTERNAL.get(raw, raw)
    code = parts[2]
    if len(code) < 2:
        return None
    c = code[0].upper()
    cryst = "High" if c == "H" else ("Low" if c == "L" else None)
    if cryst is None:
        return None
    ori = code[1] if code[1].isdigit() else None
    if ori is None:
        return None
    return dict(material_full=mat, orientation=ori, crystallinity=cryst)


def _get_field_and_time(d):
    """Return (T_field, t_s) from a spec_fields entry, t_s zeroed at start."""
    T_field = np.asarray(d.get("field", d.get("T_field", [])), float)
    if T_field.ndim != 2 or T_field.shape[1] < N_COLS:
        return None, None
    for key in ("t_s", "time", "t"):
        v = d.get(key)
        if v is not None:
            arr = np.asarray(v, float)
            if arr.size == T_field.shape[0] and np.isfinite(arr).all():
                return T_field, arr - arr[0]
    cond = d.get("cond", {})
    fps  = float(cond.get("fps", cond.get("frame_rate", 25.0)))
    return T_field, np.arange(T_field.shape[0]) / fps


def _lamp_off_idx(T_field):
    return int(np.nanargmax(np.nanmean(T_field[:, :N_COLS], axis=1)))


def _profile_at(T_field, t_s, lamp_idx, dt_s):
    """T(y) at t_lamp_off + dt_s via linear interpolation; NaN if out of range."""
    t_target = t_s[lamp_idx] + dt_s
    if t_target < t_s[0] or t_target > t_s[-1]:
        return np.full(N_COLS, np.nan)
    idx = int(np.searchsorted(t_s, t_target, side="left"))
    idx = np.clip(idx, 1, len(t_s) - 1)
    t0, t1 = t_s[idx - 1], t_s[idx]
    alpha = float(np.clip((t_target - t0) / max(t1 - t0, 1e-9), 0, 1))
    return (1.0 - alpha) * T_field[idx - 1, :N_COLS] + alpha * T_field[idx, :N_COLS]


# ── Pool per (mat, ori): build 2-D T(t, y) field per specimen then average ─────
# buckets[(mat, ori)] → list of (N_TIME_DENSE, N_COLS) arrays
buckets_2d = defaultdict(list)

for name, d in spec_fields.items():
    meta = _resolve_meta(name)
    if meta is None:
        continue
    T_field, t_s = _get_field_and_time(d)
    if T_field is None:
        continue
    lamp_idx = _lamp_off_idx(T_field)
    mat_full = _MAT_INTERNAL.get(meta["material_full"], meta["material_full"])
    key = (mat_full, meta["orientation"])

    # Build dense T(t, y) for this specimen
    T_2d = np.array([_profile_at(T_field, t_s, lamp_idx, dt) for dt in t_dense])
    buckets_2d[key].append(T_2d)

# Mean across replicates
mean_2d = {
    key: np.nanmean(np.stack(lst, axis=0), axis=0)   # (N_TIME_DENSE, N_COLS)
    for key, lst in buckets_2d.items()
}

# ── Resolve grid ───────────────────────────────────────────────────────────────
all_mats = {k[0] for k in buckets_2d}
all_oris = {k[1] for k in buckets_2d}
materials    = [m for m in MATERIAL_ORDER if m in all_mats]
orientations = [o for o in ORIENTATION_ORDER if o in all_oris]
n_r, n_c = len(materials), len(orientations)

if n_r == 0 or n_c == 0:
    raise RuntimeError("No data found in spec_fields — run prepare_exports.py first.")

# ── Pre-compute per-material colour scales ─────────────────────────────────────
# T: vmin/vmax from the data across all orientations for that material
# ΔT: symmetric ± vmax (95th percentile of |ΔT|)
T_scales  = {}   # mat → (vmin, vmax)
dT_scales = {}   # mat → vmax (symmetric)
for mat in materials:
    all_T  = [mean_2d[(mat, o)] for o in orientations if (mat, o) in mean_2d]
    all_dT = []
    for arr in all_T:
        T_cool = np.nanmean(arr[:, -3:], axis=1, keepdims=True)
        all_dT.append(arr - T_cool)
    flat_T  = np.concatenate([a.ravel() for a in all_T])
    flat_dT = np.concatenate([a.ravel() for a in all_dT])
    T_scales[mat]  = (np.nanpercentile(flat_T, 2), np.nanpercentile(flat_T, 98))
    dT_scales[mat] = np.nanpercentile(np.abs(flat_dT[np.isfinite(flat_dT)]), 95)

# ── Plot ───────────────────────────────────────────────────────────────────────
set_journal_style()

fig = plt.figure(figsize=mm_to_inch(FIG_WIDTH_MM, ROW_HEIGHT_MM * n_r))

outer = gridspec.GridSpec(
    n_r, n_c, figure=fig,
    hspace=0.55, wspace=0.28,
    left=0.10, right=0.97, top=0.96, bottom=0.06,
)

for i_r, mat in enumerate(materials):
    mc         = material_color(mat)
    T_vmin, T_vmax = T_scales[mat]
    dT_vmax    = dT_scales[mat]

    for i_c, ori in enumerate(orientations):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[i_r, i_c],
            hspace=0.10, height_ratios=[1, 1],
        )
        ax_T  = fig.add_subplot(inner[0])
        ax_dT = fig.add_subplot(inner[1], sharex=ax_T)

        key = (mat, ori)
        if key not in mean_2d:
            for ax in (ax_T, ax_dT):
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="0.5")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            continue

        T_2d   = mean_2d[key]                                         # (N_TIME, N_COLS)
        T_cool = np.nanmean(T_2d[:, -3:], axis=1, keepdims=True)     # (N_TIME, 1)
        dT_2d  = T_2d - T_cool                                        # (N_TIME, N_COLS)

        # extent: [left, right, bottom, top] = [t_min, t_max, y_min, y_max]
        extent = [-T_WINDOW_S, T_WINDOW_S, 0, L_MM]

        im_T = ax_T.imshow(
            T_2d.T, aspect="auto", origin="lower", extent=extent,
            cmap="plasma", vmin=T_vmin, vmax=T_vmax,
            interpolation="bilinear",
        )
        im_dT = ax_dT.imshow(
            dT_2d.T, aspect="auto", origin="lower", extent=extent,
            cmap="RdBu_r", vmin=-dT_vmax, vmax=dT_vmax,
            interpolation="bilinear",
        )

        # Lamp-off marker
        for ax in (ax_T, ax_dT):
            ax.axvline(0, color="white", lw=0.9, ls="--", alpha=0.75, zorder=5)
            ax.set_xlim(-T_WINDOW_S, T_WINDOW_S)
            ax.set_ylim(0, L_MM)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax_T.tick_params(labelbottom=False)

        # Colorbars on rightmost column only
        if i_c == n_c - 1:
            div_T  = make_axes_locatable(ax_T)
            cax_T  = div_T.append_axes("right", size="5%", pad=0.04)
            cb_T   = fig.colorbar(im_T, cax=cax_T)
            cb_T.set_label("$T$ [°C]", fontsize=7)
            cb_T.ax.tick_params(labelsize=6)

            div_dT = make_axes_locatable(ax_dT)
            cax_dT = div_dT.append_axes("right", size="5%", pad=0.04)
            cb_dT  = fig.colorbar(im_dT, cax=cax_dT)
            cb_dT.set_label(r"$\Delta T$ [°C]", fontsize=7)
            cb_dT.ax.tick_params(labelsize=6)

        # ── Labels ─────────────────────────────────────────────────────────
        if i_r == 0:
            ax_T.set_title(orientation_label(ori), pad=4, fontsize=9)
        if i_r == n_r - 1:
            ax_dT.set_xlabel("Time relative to lamp-off [s]", fontsize=9)
        else:
            ax_dT.tick_params(labelbottom=False)
        if i_c == 0:
            ax_T.set_ylabel(
                f"{material_label(mat)}\n$y$ [mm]",
                fontsize=9, color=mc,
            )
            ax_dT.set_ylabel("$y$ [mm]", fontsize=9)
        else:
            ax_T.tick_params(labelleft=False)
            ax_dT.tick_params(labelleft=False)

        # hot / cool face labels on leftmost column, bottom material row
        if i_c == 0 and i_r == n_r - 1:
            ax_dT.annotate("hot",  xy=(0, 0),    xytext=(-0.16, 0.04),
                           textcoords="axes fraction", fontsize=7,
                           color="0.5", ha="left", va="bottom")
            ax_dT.annotate("cool", xy=(0, L_MM), xytext=(-0.16, 0.94),
                           textcoords="axes fraction", fontsize=7,
                           color="0.5", ha="left", va="top")

# Shared caption note
fig.text(0.01, 0.01,
         "Dashed line = lamp-off  |  Mean across replicates (High + Low $T$ pooled)",
         fontsize=7, color="0.5", va="bottom")

save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
