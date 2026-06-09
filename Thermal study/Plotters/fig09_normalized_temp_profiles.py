"""
fig09_normalized_temp_profiles.py
===================================
Normalised through-thickness temperature profile θ(y) at peak heating.

θ = (T − T_min) / (T_max − T_min) per specimen, where T(y) is averaged
over the 3-second window (75 frames at 25 Hz) immediately before the
peak temperature frame.

Mean ± 95 % CI across replicates. High χ: solid; Low χ: dashed.

Grid: N_materials rows × N_orientations cols.

Data requirement: data_exports/profile_per_spec.pkl
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
    material_color, material_label, orientation_label,
    parse_specimen_name, ci95_profile, DATA_DIR,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIGURE_BASENAME = "fig09_normalized_temp_profiles"
FIG_WIDTH_MM    = 178
ROW_HEIGHT_MM   = 55
L_MM            = 20.0   # specimen through-thickness length
N_COLS          = 30     # spatial columns

# Map augmented naming → clean pipeline
_MAT_INTERNAL = {"PEEK_CF": "PEEK-CF", "PPS_CF": "PPS-CF",
                 "PPS_NEAT": "PPS neat", "PEEK_NEAT": "PEEK neat"}

x_mm = np.linspace(0, L_MM, N_COLS)

# ── Load data ──────────────────────────────────────────────────────────────────
pkl = DATA_DIR / "profile_per_spec.pkl"
if not pkl.exists():
    raise FileNotFoundError(f"{pkl}\nRun prepare_exports.py first.")

with open(pkl, "rb") as f:
    profile_per_spec = pickle.load(f)

# profile_per_spec: name → dict with "T_x" (1-D, length N_COLS) or "T_x_grid"
# Fall back to spec_fields if profile_per_spec lacks θ
spec_pkl = DATA_DIR / "spec_fields.pkl"
spec_fields = None
if spec_pkl.exists():
    with open(spec_pkl, "rb") as f:
        spec_fields = pickle.load(f)


def _get_theta(name):
    """Return normalised θ(y) profile for this specimen."""
    data = profile_per_spec.get(name, {})
    # Try direct θ
    for key in ("theta", "T_x_norm", "theta_x"):
        if key in data:
            arr = np.asarray(data[key], float)
            if arr.size == N_COLS:
                return arr
    # Try raw T_x and normalise
    for key in ("T_x", "T_x_grid", "T_profile"):
        if key in data:
            arr = np.asarray(data[key], float)
            if arr.size == N_COLS and np.isfinite(arr).any():
                lo, hi = np.nanmin(arr), np.nanmax(arr)
                if hi - lo > 0.1:
                    return (arr - lo) / (hi - lo)
    # Fall back to spec_fields T field at peak
    if spec_fields is not None and name in spec_fields:
        field = np.asarray(spec_fields[name].get("field",
                           spec_fields[name].get("T_field", [])), float)
        if field.ndim == 2 and field.shape[1] == N_COLS:
            T_peak = field[-1, :]  # last time step = peak
            lo, hi = np.nanmin(T_peak), np.nanmax(T_peak)
            if hi - lo > 0.1:
                return (T_peak - lo) / (hi - lo)
    return None


# ── Pool per (mat, ori, cryst) ─────────────────────────────────────────────────
def _resolve_mat(name):
    """Parse name, handling both augmented and clean-pipeline conventions."""
    meta = parse_specimen_name(name)
    if meta is not None:
        return meta
    # Augmented naming PEEK_CF_H1a
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


buckets = {}
for name in profile_per_spec:
    meta = _resolve_mat(name)
    if meta is None:
        continue
    theta = _get_theta(name)
    if theta is None:
        continue
    key = (meta["material_full"], meta["orientation"], meta["crystallinity"])
    buckets.setdefault(key, []).append(theta)

means = {}
cis   = {}
for k, v in buckets.items():
    stack = np.vstack(v)
    m, ci = ci95_profile(stack)
    means[k] = m
    cis[k]   = ci

# ── Resolve grid ───────────────────────────────────────────────────────────────
all_mats = {k[0] for k in buckets}
all_oris = {k[1] for k in buckets}
materials    = [m for m in MATERIAL_ORDER if m in all_mats]
orientations = [o for o in ORIENTATION_ORDER if o in all_oris]

# ── Plot ───────────────────────────────────────────────────────────────────────
set_journal_style()

n_r, n_c = len(materials), len(orientations)
fig, axes = plt.subplots(
    n_r, n_c,
    figsize=mm_to_inch(FIG_WIDTH_MM, ROW_HEIGHT_MM * n_r),
    sharex=True, sharey=True,
    squeeze=False,
)

for i_r, mat in enumerate(materials):
    mc = material_color(mat)
    for i_c, ori in enumerate(orientations):
        ax = axes[i_r, i_c]

        for cryst in CRYST_ORDER:
            ls  = "-"  if cryst == "High" else "--"
            a_l = 1.0  if cryst == "High" else 0.65
            a_f = 0.16 if cryst == "High" else 0.08
            key = (mat, ori, cryst)
            if key not in means:
                continue
            mu = means[key]
            ci = cis[key]
            valid = np.isfinite(mu)
            ax.plot(x_mm[valid], mu[valid],
                    color=mc, ls=ls, lw=1.3, alpha=a_l, zorder=3)
            ax.fill_between(
                x_mm[valid],
                (mu - ci)[valid], (mu + ci)[valid],
                color=mc, alpha=a_f, linewidth=0, zorder=2,
            )

        ax.set_xlim(0, L_MM)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="0.7", lw=0.5, zorder=0)
        ax.axhline(1, color="0.7", lw=0.5, ls=":", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if i_r == 0:
            ax.set_title(orientation_label(ori), pad=4, fontsize=9)
        if i_r == n_r - 1:
            ax.set_xlabel("Through-thickness position $y$ [mm]", fontsize=9)
        else:
            ax.tick_params(labelbottom=False)
        if i_c == 0:
            ax.set_ylabel(
                f"{material_label(mat)}\n"
                r"$\theta = (T - T_{\min})/(T_{\max} - T_{\min})$ [—]",
                fontsize=9, color=mc,
            )
        else:
            ax.tick_params(labelleft=False)

    # Side label for hot/cool
    if n_c > 0:
        axes[i_r, 0].annotate("hot", xy=(0, 0), xytext=(-0.01, -0.07),
                               textcoords="axes fraction", fontsize=9,
                               color="0.5", ha="left")
        axes[i_r, -1].annotate("cool", xy=(1, 0), xytext=(1.0, -0.07),
                                textcoords="axes fraction", fontsize=9,
                                color="0.5", ha="right")

# Legend
fig.legend(
    handles=[
        Line2D([0], [0], color="0.3", ls="-",  lw=1.3,
               label="High $T$"),
        Line2D([0], [0], color="0.3", ls="--", lw=1.3, alpha=0.65,
               label="Low $T$"),
    ],
    loc="lower center", frameon=False, ncol=2, fontsize=9, handlelength=1.6,
)

fig.tight_layout(rect=(0, 0.06, 1, 0.98))
save_figure(fig, FIGURE_BASENAME)
plt.close(fig)
