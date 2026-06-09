"""
Plot component study results — select materials and temperature levels interactively.

Usage:
    python plot_components.py                          # plots all materials
    python plot_components.py --materials PEEK_CF PPS_CF
    python plot_components.py --materials PA6_CF --temps LOW Tg
    python plot_components.py --excel my_data.xlsx     # custom excel path
    python plot_components.py --sheet Sheet2           # specific sheet
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 25
LEGENDSIZE = FONTSIZE * 0.85


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_EXCEL = "Temp_Measurements.xlsx"  # adjust extension if needed (.xls, .xlsm)

MATERIAL_COLORS = {
    "PEEK_CF": "#BE0000",
    "PPS_CF":  "#b35300",
    "PA6_CF":  "#0035C7",
    "PLA":     "#0d0d0d",
}

TEMP_MARKERS = {
    "LOW":  "s",
    "Tg":   "^",
    "HIGH": "D",
    "N/A":  "o",
}

TEMP_ORDER = ["LOW", "Tg", "HIGH", "N/A"]

MATERIAL_LINESTYLES = {
    "PEEK_CF": "-",
    "PPS_CF":  "--",
    "PA6_CF":  (0, (3, 1, 1, 1)),  # dash-dot
    "PLA":     (0, (1, 1)),         # dotted
}

# ── Load & filter ─────────────────────────────────────────────────────────────
def load_data(excel_path, sheet_name=0, materials=None, temps=None):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    df["Temperature"] = df["Temperature"].fillna("N/A")

    # Detect wide pixel format: any column that isn't a known metadata column
    known_meta = {"Material", "Temperature", "T_Point", "T_MEAN", "T_MIN", "T_MAX", "T_Value"}
    pixel_cols = [c for c in df.columns if c not in known_meta]

    if pixel_cols:
        # Wide format: melt pixel columns into T_Value rows
        df = df.melt(
            id_vars=["Material", "Temperature", "T_Point"],
            value_vars=pixel_cols,
            var_name="_px",
            value_name="T_Value",
        ).drop(columns="_px")
        df["T_Value"] = pd.to_numeric(df["T_Value"], errors="coerce")
        df = df.dropna(subset=["T_Value"])

        stats = (df.groupby(["Material", "Temperature", "T_Point"])["T_Value"]
                   .agg(T_MEAN="mean", T_MIN="min", T_MAX="max")
                   .reset_index())
        df = df.merge(stats, on=["Material", "Temperature", "T_Point"], how="left")

    elif "T_Value" in df.columns:
        # Long format: one row per pixel
        df["T_Value"] = pd.to_numeric(df["T_Value"], errors="coerce")
        df = df.dropna(subset=["T_Value"])

        stats = (df.groupby(["Material", "Temperature", "T_Point"])["T_Value"]
                   .agg(T_MEAN="mean", T_MIN="min", T_MAX="max")
                   .reset_index())
        df = df.merge(stats, on=["Material", "Temperature", "T_Point"], how="left")

    else:
        # Old format: summary columns only
        df["T_MEAN"] = pd.to_numeric(df["T_MEAN"], errors="coerce")
        if "T_MIN" in df.columns:
            df["T_MIN"] = pd.to_numeric(df["T_MIN"], errors="coerce")
        if "T_MAX" in df.columns:
            df["T_MAX"] = pd.to_numeric(df["T_MAX"], errors="coerce")
        df = df.dropna(subset=["T_MEAN"])

    if materials:
        df = df[df["Material"].isin(materials)]
    if temps:
        df = df[df["Temperature"].isin(temps)]

    return df


# ── Plotting ──────────────────────────────────────────────────────────────────
def _summary(df):
    """One row per (Material, Temperature, T_Point) with T_MEAN/T_MIN/T_MAX.
    Needed when df has been melted into per-pixel rows."""
    if "T_Value" in df.columns:
        return df.drop_duplicates(subset=["Material", "Temperature", "T_Point"])
    return df


def plot_by_test_point(df, title="Component Study Results"):
    """Bar-style plot: T vs T_Point, grouped by Material & Temperature."""
    fig, ax = plt.subplots(figsize=(12, 6))

    materials = [m for m in MATERIAL_COLORS if m in df["Material"].unique()]
    temps = [t for t in TEMP_ORDER if t in df["Temperature"].unique()]

    n_groups = len(materials)
    n_temps = len(temps)
    total_bars = n_groups * n_temps
    bar_width = 0.8 / max(total_bars, 1)

    t_points = sorted(df["T_Point"].unique())
    x = np.arange(len(t_points))

    sdf = _summary(df)
    idx = 0
    for mat in materials:
        for temp in temps:
            subset = sdf[(sdf["Material"] == mat) & (sdf["Temperature"] == temp)]
            values = subset.set_index("T_Point").reindex(t_points)["T_MEAN"]
            offset = (idx - total_bars / 2 + 0.5) * bar_width
            ax.bar(
                x + offset, values, bar_width,
                label=f"{mat} – {temp}",
                color=MATERIAL_COLORS.get(mat, "#999"),
                alpha=0.5 + 0.5 * (temps.index(temp) / max(len(temps) - 1, 1)),
                edgecolor="white", linewidth=0.5,
            )
            idx += 1

    ax.set_xlabel("Sample Point", fontsize=FONTSIZE)
    ax.set_ylabel("T", fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(t_points)
    ax.legend(fontsize=LEGENDSIZE, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_line(df, title="Component Study Results"):
    """Line plot: median per T_Point with box-and-whisker spread (when pixel data available),
    or mean line with min-max shading (summary data fallback)."""
    fig, ax = plt.subplots(figsize=(14, 7))

    materials = [m for m in MATERIAL_COLORS if m in df["Material"].unique()]
    temps = [t for t in TEMP_ORDER if t in df["Temperature"].unique()]
    temps = temps[::-1]  # low → high
    use_raw = "T_Value" in df.columns

    t_points = sorted(df["T_Point"].unique())
    try:
        x_centers = np.array([float(tp) for tp in t_points], dtype=float)
    except (TypeError, ValueError):
        x_centers = np.arange(1, len(t_points) + 1, dtype=float)
    point_positions = dict(zip(t_points, x_centers))
    combos = [(mat, temp) for mat in materials for temp in temps
              if not df[(df["Material"] == mat) & (df["Temperature"] == temp)].empty]
    point_gap = np.min(np.diff(x_centers)) if len(x_centers) > 1 else 1.0
    box_width = 0.18 * point_gap
    _legend_groups = {}

    for mat, temp in combos:
        subset = df[(df["Material"] == mat) & (df["Temperature"] == temp)]
        if subset.empty:
            continue
        subset = subset.sort_values("T_Point")
        color = MATERIAL_COLORS.get(mat, "#999")
        linestyle = MATERIAL_LINESTYLES.get(mat, "-")
        # Temperature level encoded via alpha: LOW=0.5, Tg=0.75, HIGH=1.0, N/A=1.0
        temp_alpha = {"LOW": 0.5, "Tg": 0.75, "HIGH": 1.0, "N/A": 1.0}.get(temp, 1.0)
        markersize = 9

        if use_raw:
            # Box = IQR (25–75 %), whiskers = 1.5×IQR, line through medians
            medians = []
            positions = []
            means = []
            for i, tp in enumerate(t_points):
                vals = subset[subset["T_Point"] == tp]["T_Value"].dropna().values
                if len(vals) == 0:
                    medians.append(np.nan)
                    continue
                pos = x_centers[i]
                bp = ax.boxplot(
                    vals, positions=[pos], widths=box_width * 0.9,
                    patch_artist=True, manage_ticks=False,
                    whiskerprops=dict(color=color, linewidth=2.0),
                    capprops=dict(color=color, linewidth=2.0),
                    medianprops=dict(color="white", linewidth=2.8),
                    flierprops=dict(marker="o", markerfacecolor="white",
                                    markeredgecolor=color, alpha=0.7, markersize=4),
                    boxprops=dict(linewidth=2.2),
                )
                bp["boxes"][0].set_facecolor(color)
                bp["boxes"][0].set_alpha(temp_alpha)
                bp["boxes"][0].set_edgecolor("#1f1f1f")
                medians.append(float(np.median(vals)))
                means.append(float(np.mean(vals)))
                positions.append(pos)

            # Interpolation line: straight line from first to last mean, drawn behind boxes
            if len(positions) >= 2:
                ax.plot([positions[0], positions[-1]], [means[0], means[-1]],
                        color=color, linestyle=linestyle, linewidth=2,
                        alpha=temp_alpha * 0.6, zorder=2)

            # Line through medians — positions matches non-nan medians exactly
            med_vals = [m for m in medians if not np.isnan(m)]
            line, = ax.plot(positions, med_vals, color=color, linestyle=linestyle,
                            linewidth=2.0, marker=TEMP_MARKERS.get(temp, "o"),
                            markersize=markersize, alpha=temp_alpha, zorder=5)
        else:
            # Fallback: mean line + shaded min-max
            subset = _summary(subset).set_index("T_Point").reindex(t_points).dropna(subset=["T_MEAN"])
            x = np.array([point_positions[tp] for tp in subset.index], dtype=float)
            line, = ax.plot(x, subset["T_MEAN"], color=color,
                            marker=TEMP_MARKERS.get(temp, "o"), markersize=markersize,
                            linestyle=linestyle, linewidth=2, alpha=temp_alpha)
            if "T_MIN" in df.columns and "T_MAX" in df.columns:
                ax.fill_between(x, subset["T_MIN"], subset["T_MAX"],
                                color=color, alpha=0.15 * temp_alpha)

        # Collect handle for grouped legend
        if mat not in _legend_groups:
            _legend_groups[mat] = []
        _legend_groups[mat].append((temp, line))

    legend_handles = []
    legend_labels = []
    for mat in materials:
        if mat not in _legend_groups:
            continue
        for temp, line in _legend_groups[mat]:
            legend_handles.append(line)
            legend_labels.append(f"{mat.replace('_', ' ')} {temp}")

    ax.set_xticks(x_centers)
    ax.set_xticklabels(t_points)
    ax.set_xlim(x_centers[0] - point_gap * 0.8, x_centers[-1] + point_gap * 0.8)
    ax.set_xlabel("Sample Point", fontsize=FONTSIZE)
    ax.set_ylabel('Temperature [°C]', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.legend(legend_handles, legend_labels, fontsize=LEGENDSIZE, ncol=(len(legend_handles) + 1) // 2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig

def plot_scatter(df, title="Component Study Results"):
    """Scatter plot: T vs T_Point with marker per temperature level."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sdf = _summary(df)
    materials = [m for m in MATERIAL_COLORS if m in sdf["Material"].unique()]
    temps = [t for t in TEMP_ORDER if t in sdf["Temperature"].unique()]

    for mat in materials:
        for temp in temps:
            subset = sdf[(sdf["Material"] == mat) & (sdf["Temperature"] == temp)]
            if subset.empty:
                continue
            ax.scatter(
                subset["T_Point"], subset["T_MEAN"],
                color=MATERIAL_COLORS.get(mat, "#999"),
                marker=TEMP_MARKERS.get(temp, "o"),
                s=80, label=f"{mat} – {temp}",
                edgecolors="white", linewidths=0.5,
            )

    ax.set_xlabel("Sample Point", fontsize=FONTSIZE)
    ax.set_ylabel('Temperature [°C]', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.legend(fontsize=LEGENDSIZE, ncol=max(1, (len(materials) * len(temps) + 1) // 2), loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_horizontal(df, title="Temperature Profile"):
    """Horizontal plot: Temperature on x-axis, T_Point (T1–T9) on y-axis.

    Designed to correlate visually with the component image showing
    thermocouple positions T1–T9 from bottom to top.
    Creates one subplot per material for easy comparison.
    """
    materials = [m for m in MATERIAL_COLORS if m in df["Material"].unique()]
    temps = [t for t in TEMP_ORDER if t in df["Temperature"].unique()]
    t_points = sorted(df["T_Point"].unique())
    y_positions = {tp: i for i, tp in enumerate(t_points)}

    n_mats = len(materials)
    fig, axes = plt.subplots(1, n_mats, figsize=(6 * n_mats, 8), sharey=True)
    if n_mats == 1:
        axes = [axes]

    sdf = _summary(df)
    t_min = sdf["T_MEAN"].min()
    t_max = sdf["T_MEAN"].max()
    margin = (t_max - t_min) * 0.05
    xlim = (t_min - margin, t_max + margin)

    for ax, mat in zip(axes, materials):
        for temp in temps:
            subset = sdf[(sdf["Material"] == mat) & (sdf["Temperature"] == temp)]
            if subset.empty:
                continue
            subset = subset.sort_values("T_Point")
            y_vals = [y_positions[tp] for tp in subset["T_Point"]]
            ax.plot(
                subset["T_MEAN"].values, y_vals,
                color=MATERIAL_COLORS.get(mat, "#999"),
                marker=TEMP_MARKERS.get(temp, "o"),
                linestyle="--" if temp == "Tg" else "-",
                linewidth=1.8, markersize=8,
                label=temp,
            )

        ax.set_xlabel("Temperature [°C]", fontsize=FONTSIZE)
        ax.set_xlim(xlim)
        ax.legend(fontsize=FONTSIZE*0.9, ncol=max(1, (len(temps) + 1) // 2), loc="upper center", bbox_to_anchor=(0.5, -0.15))
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Measurement Point", fontsize=FONTSIZE)
    axes[0].set_yticks(range(len(t_points)))
    axes[0].set_yticklabels(t_points)
    fig.tight_layout()
    return fig


def plot_interp(df, title="Component Study Results"):
    """Line plot with linear interpolation through mean values — no shading, markers only."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sdf = _summary(df)
    materials = [m for m in MATERIAL_COLORS if m in sdf["Material"].unique()]
    temps = [t for t in TEMP_ORDER if t in sdf["Temperature"].unique()]

    for mat in materials:
        for temp in temps:
            subset = sdf[(sdf["Material"] == mat) & (sdf["Temperature"] == temp)]
            if subset.empty:
                continue

            subset = subset.sort_values("T_Point")
            x_vals = subset["T_Point"].values
            y_vals = subset["T_MEAN"].values

            color = MATERIAL_COLORS.get(mat, "#999")
            linestyle = "--" if temp == "Tg" else "-"

            # Single straight line from first to last datapoint
            ax.plot(
                [x_vals[0], x_vals[-1]], [y_vals[0], y_vals[-1]],
                color=color,
                linestyle=linestyle,
                linewidth=2,
            )
            # Markers at each measured point
            ax.plot(
                x_vals, y_vals,
                color=color,
                marker=TEMP_MARKERS.get(temp, "o"),
                linestyle="none",
                linewidth=0,
                label=f"{mat} – {temp}",
            )

    ax.set_xlabel("Sample Point", fontsize=FONTSIZE)
    ax.set_ylabel('Temperature [°C]', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE, ncol=max(1, (len(materials) * len(temps) + 1) // 2), loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_box(df, title="Component Study — Distribution"):
    """Grouped box plot: one box per (Material×Temperature) per T_Point.

    Uses raw T_Value pixel data when available (new format), otherwise T_MEAN.
    Handles unequal numbers of data points per group naturally.
    """
    materials = [m for m in MATERIAL_COLORS if m in df["Material"].unique()]
    temps = [t for t in TEMP_ORDER if t in df["Temperature"].unique()]
    t_points = sorted(df["T_Point"].unique())
    combos = [(mat, temp) for mat in materials for temp in temps]

    use_raw = "T_Value" in df.columns
    value_col = "T_Value" if use_raw else "T_MEAN"

    fig, ax = plt.subplots(figsize=(max(8, len(t_points) * max(len(combos), 1) * 0.35), 6))

    all_data, all_positions, all_colors, all_widths = [], [], [], []
    # mat → list of (temp, patch) for grouped legend
    legend_groups = {mat: [] for mat in materials}
    temp_alphas = {"LOW": 0.5, "Tg": 0.75, "HIGH": 1.0, "N/A": 1.0}

    for i, tp in enumerate(t_points):
        present = []
        for mat, temp in combos:
            subset = df[
                (df["Material"] == mat) &
                (df["Temperature"] == temp) &
                (df["T_Point"] == tp)
            ]
            vals = subset[value_col].dropna().values
            if len(vals):
                present.append((mat, temp, vals))

        if not present:
            continue

        group_width = 0.8
        local_width = group_width / len(present)

        for j, (mat, temp, vals) in enumerate(present):
            pos = i + (j - len(present) / 2 + 0.5) * local_width
            all_data.append(vals)
            all_positions.append(pos)
            all_colors.append((MATERIAL_COLORS.get(mat, "#999"), temp))
            all_widths.append(max(0.08, local_width * 0.85))

    if all_data:
        bp = ax.boxplot(
            all_data,
            positions=all_positions,
            widths=all_widths,
            patch_artist=True,
            manage_ticks=False,
            boxprops=dict(linewidth=2.2),
            whiskerprops=dict(linewidth=2.2),
            capprops=dict(linewidth=2.2),
            medianprops=dict(color="white", linewidth=2.6),
            flierprops=dict(marker="o", markersize=5, alpha=0.4),
        )
        for patch, (c, temp) in zip(bp["boxes"], all_colors):
            patch.set_facecolor(c)
            patch.set_edgecolor(c)
            patch.set_alpha(temp_alphas.get(temp, 0.55))

    # Collect one representative patch per (mat, temp) for grouped legend
    seen = set()
    combo_idx = 0
    for i, tp in enumerate(t_points):
        for mat, temp in combos:
            subset = df[
                (df["Material"] == mat) &
                (df["Temperature"] == temp) &
                (df["T_Point"] == tp)
            ]
            if not subset[value_col].dropna().empty:
                key = (mat, temp)
                if key not in seen and combo_idx < len(bp["boxes"]):
                    seen.add(key)
                    legend_groups[mat].append((temp, bp["boxes"][combo_idx]))
                combo_idx += 1

    legend_handles = []
    legend_labels = []
    for mat in materials:
        if not legend_groups[mat]:
            continue
        for temp, patch in legend_groups[mat]:
            legend_handles.append(patch)
            legend_labels.append(f"{mat.replace('_', ' ')} {temp}")

    ax.set_xticks(range(len(t_points)))
    ax.set_xticklabels(t_points)
    ax.set_xlabel("Sample Point", fontsize=FONTSIZE)
    ax.set_ylabel('Temperature [°C]', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.legend(legend_handles, legend_labels, fontsize=LEGENDSIZE, ncol=(len(legend_handles) + 1) // 2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_condition_box(df, title="Component Study — Distribution"):
    """One box per (Material, Temperature), using T_MEAN over all T_Points.

    This matches an overall condition-level spread plot, where each box summarizes
    the mean temperature values across the measurement points for that condition.
    """
    sdf = _summary(df)
    materials = [m for m in MATERIAL_COLORS if m in sdf["Material"].unique()]
    temps = [t for t in TEMP_ORDER if t in sdf["Temperature"].unique()]
    combos = [
        (mat, temp) for mat in materials for temp in temps
        if not sdf[(sdf["Material"] == mat) & (sdf["Temperature"] == temp)].empty
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    plot_data = []
    labels = []
    colors = []
    combo_order = []  # tracks (mat, temp) for each box in order

    for mat, temp in combos:
        vals = sdf[(sdf["Material"] == mat) & (sdf["Temperature"] == temp)]["T_MEAN"].dropna().values
        if len(vals) == 0:
            continue
        plot_data.append(vals)
        labels.append(f"{mat.replace('_', ' ')}\n{temp}")
        colors.append(MATERIAL_COLORS.get(mat, "#999"))
        combo_order.append((mat, temp))

    if plot_data:
        bp = ax.boxplot(
            plot_data,
            widths=0.55,
            patch_artist=True,
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(color="#ff9d00", linewidth=1.6),
            flierprops=dict(marker="o", markersize=4, alpha=0.45),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#222222")
            patch.set_alpha(0.65)

    legend_groups = {}
    for (mat, temp), patch in zip(combo_order, bp["boxes"]):
        if mat not in legend_groups:
            legend_groups[mat] = []
        legend_groups[mat].append((temp, patch))

    legend_handles = []
    legend_labels = []
    for mat in materials:
        if mat not in legend_groups:
            continue
        for temp, patch in legend_groups[mat]:
            legend_handles.append(patch)
            legend_labels.append(f"{mat.replace('_', ' ')} {temp}")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_xlabel("", fontsize=FONTSIZE)
    ax.set_ylabel("Temperature [°C]", fontsize=FONTSIZE)
    ax.tick_params(axis="both", labelsize=FONTSIZE)
    ax.legend(legend_handles, legend_labels, fontsize=LEGENDSIZE, ncol=(len(legend_handles) + 1) // 2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ── Delta export ──────────────────────────────────────────────────────────────
def _sort_key(val):
    """Numeric sort key for T_Point labels: extracts leading/trailing number if present."""
    try:
        return (0, float(val))
    except (TypeError, ValueError):
        import re
        m = re.search(r"(\d+)", str(val))
        return (0, float(m.group(1))) if m else (1, str(val))


def compute_deltas(df):
    """Return long-format DataFrame of consecutive point-to-point T_MEAN deltas.

    Columns: Material, Temperature, Transition (e.g. '1→2'), Delta
    """
    sdf = _summary(df)[["Material", "Temperature", "T_Point", "T_MEAN"]].copy()

    rows = []
    for (mat, temp), grp in sdf.groupby(["Material", "Temperature"], sort=False):
        grp = grp.copy()
        grp["_sort"] = grp["T_Point"].map(_sort_key)
        grp = grp.sort_values("_sort").reset_index(drop=True)
        for i in range(len(grp) - 1):
            mean_from = grp.loc[i,   "T_MEAN"]
            mean_to   = grp.loc[i+1, "T_MEAN"]
            rows.append({
                "Material":    mat,
                "Temperature": temp,
                "Transition":  f"{grp.loc[i, 'T_Point']}→{grp.loc[i+1, 'T_Point']}",
                "T_MEAN_From": round(mean_from, 4),
                "T_MEAN_To":   round(mean_to,   4),
                "Delta":       round(mean_to - mean_from, 4),
            })
    return pd.DataFrame(rows)


def export_deltas_xlsx(df, output_path="Temp_Deltas.xlsx"):
    """Write two sheets to *output_path*:

    - Deltas          : all consecutive point-to-point deltas
    - Max_Differential: row per (Material, Temperature) with the largest |Delta|
    """
    deltas = compute_deltas(df)
    if deltas.empty:
        print("No delta data to export.")
        return

    grp = deltas.groupby(["Material", "Temperature"], sort=False)

    idx_abs = deltas["Delta"].abs().groupby([deltas["Material"], deltas["Temperature"]]).idxmax()
    idx_min = grp["Delta"].idxmin()
    idx_max = grp["Delta"].idxmax()

    def _pick(idx_series, delta_col, trans_col):
        sub = deltas.loc[idx_series.values, ["Material", "Temperature", trans_col, delta_col]]
        return sub.reset_index(drop=True)

    abs_df = _pick(idx_abs, "Delta", "Transition").rename(columns={"Delta": "Max_Abs_Delta", "Transition": "Max_Abs_Transition"})
    min_df = _pick(idx_min, "Delta", "Transition").rename(columns={"Delta": "Min_Delta",     "Transition": "Min_Transition"})
    max_df = _pick(idx_max, "Delta", "Transition").rename(columns={"Delta": "Max_Delta",     "Transition": "Max_Transition"})

    max_diff = (abs_df
                .merge(min_df, on=["Material", "Temperature"])
                .merge(max_df, on=["Material", "Temperature"])
               )[["Material", "Temperature",
                  "Max_Abs_Transition", "Max_Abs_Delta",
                  "Min_Transition",     "Min_Delta",
                  "Max_Transition",     "Max_Delta"]]
    max_diff = max_diff.sort_values(["Material", "Temperature"]).reset_index(drop=True)

    # Minimum T_MEAN across all T_Points per (Material, Temperature)
    sdf = _summary(df)[["Material", "Temperature", "T_Point", "T_MEAN"]].copy()
    idx_tmin = sdf.groupby(["Material", "Temperature"])["T_MEAN"].idxmin()
    tmin_df = sdf.loc[idx_tmin.values, ["Material", "Temperature", "T_Point", "T_MEAN"]].rename(
        columns={"T_Point": "Min_Temp_Point", "T_MEAN": "Min_Temp"}
    ).reset_index(drop=True)
    max_diff = max_diff.merge(tmin_df, on=["Material", "Temperature"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        deltas.to_excel(writer, sheet_name="Deltas", index=False)
        max_diff.to_excel(writer, sheet_name="Max_Differential", index=False)

    print(f"Saved: {output_path}  ({len(deltas)} delta rows, {len(max_diff)} max-diff rows)")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot component study results")
    parser.add_argument("--excel", default=DEFAULT_EXCEL, help="Path to Excel file")
    parser.add_argument("--sheet", default=0, help="Sheet name or index (default: first sheet)")
    parser.add_argument("--materials", nargs="+", help="Materials to include (e.g. PEEK_CF PPS_CF)")
    parser.add_argument("--temps", nargs="+", help="Temperature levels to include (e.g. LOW Tg HIGH)")
    parser.add_argument("--plot", choices=["bar", "scatter", "box", "condition_box", "line", "interp", "horizontal", "all"], default="all")
    _default_deltas = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Temp_Deltas.xlsx")
    parser.add_argument("--export-deltas", nargs="?", const=_default_deltas, default=_default_deltas, metavar="FILE",
                        help=f"Export point-to-point deltas xlsx (default: {_default_deltas})")
    args = parser.parse_args()

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet

    df = load_data(args.excel, sheet, args.materials, args.temps)

    if df.empty:
        print("No data to plot after filtering. Check your --materials / --temps flags.")
        return

    mat_str = ", ".join(args.materials) if args.materials else "All"
    title = f"Component Study — {mat_str}"

    if args.export_deltas:
        export_deltas_xlsx(df, args.export_deltas)

    plots = {
        "bar": plot_by_test_point,
        "scatter": plot_scatter,
        "box": plot_box,
        "condition_box": plot_condition_box,
        "line": plot_line,
        "interp": plot_interp,
        "horizontal": plot_horizontal,
    }

    if args.plot == "all":
        for fn in plots.values():
            fn(df, title)
    else:
        plots[args.plot](df, title)

    plt.show()


if __name__ == "__main__":
    main()
