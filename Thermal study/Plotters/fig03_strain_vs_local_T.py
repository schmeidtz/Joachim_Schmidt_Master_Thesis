"""
fig03_strain_vs_local_T.py
============================
Per-extensometer εzz strain vs LOCAL temperature at each sensor position,
split into heating (solid) and cooling (dashed) phases.

Temperature source
------------------
Each VERT extensometer (E0–E2) is assigned the temperature from the
spatial columns of the through-thickness thermal camera field (spec_fields):
    E0  →  columns  0– 2  (front / hot face)
    E1  →  columns 13–15  (mid-plane)
    E2  →  columns 27–29  (back / cool face)

Heating phase: temperatures come directly from spec_fields T_field.
Cooling phase: temperatures are reconstructed from the per-frame hot-face
    peak_temp (Camera 1, 99th percentile) scaled by the ΔT ratio each
    sensor had at the heating peak:
        T_Ei_cool(t)  =  T_room  +  (peak_temp(t) - T_room)
                          × ΔT_Ei_peak / ΔT_E0_peak
    where T_room is the specimen's initial temperature.
    This assumes the spatial temperature profile scales proportionally
    during cooling (valid for constant thermal resistance), which is a
    reasonable first approximation.

Grid layout:
    Rows    = materials
    Columns = orientations (O1 | O2 | O3)

Crystallinity is pooled (mean over High and Low χ together).
Mean ± 95 % CI across replicates is shown as a shaded band.

Data requirements:
    data_exports/processed_results.pkl
    data_exports/spec_fields.pkl
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
    MATERIAL_ORDER, ORIENTATION_ORDER,
    VERT_SENSORS, EXT_COLORS_ZZ, EXT_LABELS_ZZ,
    material_color, material_label, orientation_label,
    parse_specimen_name, ci95_profile, DATA_DIR,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIGURE_BASENAME  = "fig03_strain_vs_local_T"
FIG_WIDTH_MM     = 178
ROW_HEIGHT_MM    = 55
T_GRID           = np.linspace(15, 215, 400)   # common temperature axis [°C]
N_ROOM_FRAMES    = 5                             # frames used to estimate T_room

# Per-material y-axis limits [µm/m]
YLIM = {
    "PEEK-CF":  (-1000, 10500),
    "PPS-CF":   (-1000, 15000),
    "PPS neat": (-1000, 13000),
}

# Thermal camera column slices for each VERT extensometer
SENSOR_COLS = {"E0": (0, 3), "E1": (13, 16), "E2": (27, 30)}

_MAT_INTERNAL = {"PEEK_CF": "PEEK-CF", "PPS_CF": "PPS-CF",
                 "PPS_NEAT": "PPS neat", "PEEK_NEAT": "PEEK neat"}

# ── Load data ──────────────────────────────────────────────────────────────────
proc_pkl = DATA_DIR / "processed_results.pkl"
spec_pkl = DATA_DIR / "spec_fields.pkl"
for p in (proc_pkl, spec_pkl):
    if not p.exists():
        raise FileNotFoundError(f"{p}\nRun prepare_exports.py first.")

with open(proc_pkl, "rb") as f:
    processed_results = pickle.load(f)
with open(spec_pkl, "rb") as f:
    spec_fields = pickle.load(f)

# ── Build per-sensor (T, ε) curves split by heating/cooling ───────────────────
# Buckets: (mat, ori, sensor) → list of 1-D arrays on T_GRID
heat_buckets = {}
cool_buckets = {}

for name, d in spec_fields.items():
    if name not in processed_results:
        continue
    meta = parse_specimen_name(name)
    if meta is None:
        # Try mapping underscore names
        raw = name.split("_")
        if len(raw) >= 2:
            key = f"{raw[0]}_{raw[1]}".upper()
            name2 = _MAT_INTERNAL.get(key, name)
            meta = parse_specimen_name(name2)
        if meta is None:
            continue

    mat   = meta["material_full"]
    ori   = meta["orientation"]
    res   = processed_results[name]

    # ── Spec_fields temperature field (heating only, n_heat × 30 cols) ──────
    T_field = np.asarray(d.get("field", d.get("T_field", [])), float)
    if T_field.ndim != 2 or T_field.shape[1] < 30:
        continue
    n_heat   = T_field.shape[0]
    t_sf_norm = np.linspace(0, 1, n_heat)   # 0=start, 1=peak

    # Per-sensor T during heating from spec_fields
    T_sens_heat = {s: T_field[:, lo:hi].mean(axis=1)
                   for s, (lo, hi) in SENSOR_COLS.items()}

    # ── Processed-results: strains + temperature (full cycle) ────────────────
    t_raw  = np.asarray(res.get("time", []), float)
    if t_raw.size < 5:
        continue
    pt_arr = np.asarray(res.get("peak_temp",
                                res.get("mean_temp", [])), float)
    if pt_arr.size != t_raw.size or not np.isfinite(pt_arr).any():
        continue

    i_peak = int(np.nanargmax(pt_arr))
    t_end  = t_raw[-1] - t_raw[0]
    if t_end < 1.0 or i_peak < 3:
        continue

    # Normalise time for heating segment: 0→1
    t_heat_raw = t_raw[:i_peak + 1]
    t_heat_norm = (t_heat_raw - t_heat_raw[0]) / max(
        t_heat_raw[-1] - t_heat_raw[0], 1.0)

    # Room temperature estimate
    T_room = float(np.nanmean(pt_arr[:N_ROOM_FRAMES]))

    # ΔT of E0 at peak for cooling reconstruction
    T_E0_peak = float(T_field[-1, SENSOR_COLS["E0"][0]:
                                   SENSOR_COLS["E0"][1]].mean())
    DT_E0_peak = T_E0_peak - T_room

    # Cooling temperature: peak_temp[i_peak:] scaled per sensor
    pt_cool = pt_arr[i_peak:]

    for s in VERT_SENSORS:
        lo, hi = SENSOR_COLS[s]
        eps = np.asarray(res.get("strains", {}).get(s, []), float)
        if eps.size != t_raw.size:
            continue

        eps_um = eps * 1e6   # µm/m

        # ── Heating: align to spec_fields normalised time, then (T, ε) ──────
        eps_heat_raw  = eps_um[:i_peak + 1]
        # Interpolate strain onto spec_fields t-grid (0→1)
        eps_heat_sf   = np.interp(t_sf_norm, t_heat_norm, eps_heat_raw)
        T_heat        = T_sens_heat[s]   # already on t_sf_norm

        # Interpolate both onto T_GRID (heating: T increases → sort ascending)
        sort_h = np.argsort(T_heat)
        Tu_h, idx_h = np.unique(T_heat[sort_h], return_index=True)
        Eu_h        = eps_heat_sf[sort_h][idx_h]
        eps_on_Tgrid_heat = np.interp(T_GRID, Tu_h, Eu_h,
                                      left=np.nan, right=np.nan)

        # ── Cooling: reconstruct T per sensor ────────────────────────────────
        T_s_peak  = float(T_field[-1, lo:hi].mean())
        DT_s_peak = T_s_peak - T_room
        ratio     = (DT_s_peak / DT_E0_peak) if abs(DT_E0_peak) > 0.5 else 1.0
        T_cool    = T_room + (pt_cool - T_room) * ratio

        eps_cool  = eps_um[i_peak:]
        n_cool    = min(len(T_cool), len(eps_cool))
        T_cool    = T_cool[:n_cool]
        eps_cool  = eps_cool[:n_cool]

        # Cooling: T decreases; flip so we interpolate on ascending T
        valid_c   = np.isfinite(T_cool) & np.isfinite(eps_cool)
        if valid_c.sum() < 4:
            eps_on_Tgrid_cool = np.full(len(T_GRID), np.nan)
        else:
            sort_c = np.argsort(T_cool[valid_c])
            Tu_c, idx_c = np.unique(T_cool[valid_c][sort_c], return_index=True)
            Eu_c        = eps_cool[valid_c][sort_c][idx_c]
            eps_on_Tgrid_cool = np.interp(T_GRID, Tu_c, Eu_c,
                                           left=np.nan, right=np.nan)

        bkey = (mat, ori, s)
        heat_buckets.setdefault(bkey, []).append(eps_on_Tgrid_heat)
        cool_buckets.setdefault(bkey, []).append(eps_on_Tgrid_cool)

# ── Compute mean ± CI on T_GRID for each (mat, ori, sensor) ──────────────────
def _pool(buckets, key):
    curves = buckets.get(key, [])
    if not curves:
        return None, None
    stack = np.vstack(curves)
    return ci95_profile(stack)

# ── Resolve grid ───────────────────────────────────────────────────────────────
all_meta  = [parse_specimen_name(n) for n in processed_results]
all_meta  = [m for m in all_meta if m is not None]
materials    = [m for m in MATERIAL_ORDER
                if any(x["material_full"] == m for x in all_meta)]
orientations = [o for o in ORIENTATION_ORDER
                if any(x["orientation"] == o for x in all_meta)]

# ── Plot ───────────────────────────────────────────────────────────────────────
set_journal_style()

n_r = len(materials)
n_c = len(orientations)

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

        for s in VERT_SENSORS:
            col  = EXT_COLORS_ZZ[s]
            bkey = (mat, ori, s)

            # Heating
            mu_h, ci_h = _pool(heat_buckets, bkey)
            if mu_h is not None:
                v = np.isfinite(mu_h)
                ax.plot(T_GRID[v], mu_h[v],
                        color=col, ls="-", lw=1.4, alpha=1.0, zorder=3)
                ax.fill_between(T_GRID[v],
                                (mu_h - ci_h)[v], (mu_h + ci_h)[v],
                                color=col, alpha=0.14, linewidth=0, zorder=2)

            # Cooling
            mu_c, ci_c = _pool(cool_buckets, bkey)
            if mu_c is not None:
                v = np.isfinite(mu_c)
                ax.plot(T_GRID[v], mu_c[v],
                        color=col, ls="--", lw=1.0, alpha=0.70, zorder=3)
                ax.fill_between(T_GRID[v],
                                (mu_c - ci_c)[v], (mu_c + ci_c)[v],
                                color=col, alpha=0.07, linewidth=0, zorder=2)

        ax.axhline(0, color="0.7", lw=0.5, zorder=0)
        if mat in YLIM:
            ax.set_ylim(YLIM[mat])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

        if i_r == 0:
            ax.set_title(orientation_label(ori), pad=4, fontsize=10)
        if i_r == n_r - 1:
            ax.set_xlabel("Temperature [°C]", fontsize=10)
        else:
            ax.tick_params(labelbottom=False)
        if i_c == 0:
            ax.set_ylabel(
                f"{material_label(mat)}\n"
                r"$\varepsilon_{zz}$ [µm m$^{-1}$]",
                fontsize=10, color=mc,
            )
        else:
            ax.tick_params(labelleft=False)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color=EXT_COLORS_ZZ[s], lw=1.4, label=EXT_LABELS_ZZ[s])
    for s in VERT_SENSORS
] + [
    Line2D([0], [0], color="0.35", ls="-",  lw=1.4, label="Heating"),
    Line2D([0], [0], color="0.35", ls="--", lw=1.0, alpha=0.70, label="Cooling"),
]
fig.legend(handles=legend_handles, loc="lower center", frameon=False, ncol=5,
                  fontsize=9, handlelength=1.8)

fig.tight_layout(rect=(0, 0.1, 1, 1))
save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
