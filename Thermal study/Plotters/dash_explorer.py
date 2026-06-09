"""
dash_explorer.py
================
Interactive data explorer for the thermal expansion study.

Reads from  Figures_Journal/data_exports/metrics_df.csv
(synced automatically via Dropbox once you run prepare_exports.py on the PC).

Usage
-----
    pip install dash plotly pandas numpy statsmodels
    python Figures_Journal/scripts/dash_explorer.py
    # then open  http://127.0.0.1:8050  in your browser

Tabs
----
  1. Metrics Explorer  — bar charts, any DV, grouped/coloured by condition
  2. Scatter           — any DV vs any DV, coloured by material / orientation
  3. ANOVA Heatmap     — interactive bubble heatmap (HC3-robust, BH-FDR corrected)
  4. Data Table        — filterable table of all per-specimen metrics
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
DATA_DIR     = SCRIPT_DIR.parent / "data_exports"
METRICS_CSV  = DATA_DIR / "metrics_df.csv"

if not METRICS_CSV.exists():
    sys.exit(
        f"\n  ✗  {METRICS_CSV} not found.\n"
        "     Run  %run -i Figures_Journal/scripts/prepare_exports.py  "
        "in the notebook on your PC first,\n"
        "     then wait for Dropbox to sync.\n"
    )

# ── Load & clean metrics ───────────────────────────────────────────────────────
_MAT_NORM = {"PEEK_CF": "PEEK-CF", "PPS_CF": "PPS-CF",
             "PPS_NEAT": "PPS neat", "PEEK_NEAT": "PEEK neat"}

df = pd.read_csv(METRICS_CSV)

# Normalise material name
col_mat = "material_full" if "material_full" in df.columns else "material"
df["material"] = df[col_mat].map(lambda x: _MAT_NORM.get(x, x))

# Ensure factor columns
if "fibre" not in df.columns:
    df["fibre"] = np.where(df["material"].str.contains("CF"), "CF", "Neat")
if "polymer" not in df.columns:
    df["polymer"] = df["material"].str.split("-").str[0].str.split(" ").str[0]

# Coerce orientation to string
df["orientation"] = df["orientation"].astype(str)
df["crystallinity"] = df["crystallinity"].astype(str)

# Ordered categories
MAT_ORDER    = ["PEEK-CF", "PEEK neat", "PPS-CF", "PPS neat"]
CRYST_ORDER  = ["High", "Low"]
ORI_ORDER    = sorted(df["orientation"].unique())

MATERIALS    = [m for m in MAT_ORDER if m in df["material"].unique()] + \
               [m for m in df["material"].unique() if m not in MAT_ORDER]

# ── Dependent variables ────────────────────────────────────────────────────────
DV_COLS = [c for c in [
    "alpha_lin_zz", "alpha_lin_yy",
    "alpha_rub_zz",  "alpha_rub_yy",
    "anisotropy",    "aniso_rubbery",
    "poisson_zy",
    "peak_eps_zz",   "peak_eps_yy",
] if c in df.columns]

DV_LABELS = {
    "alpha_lin_zz":  "α_zz glassy [ppm/°C]",
    "alpha_lin_yy":  "α_yy glassy [ppm/°C]",
    "alpha_rub_zz":  "α_zz rubbery [ppm/°C]",
    "alpha_rub_yy":  "α_yy rubbery [ppm/°C]",
    "anisotropy":    "α_zz/α_yy (glassy) [—]",
    "aniso_rubbery": "α_zz/α_yy (rubbery) [—]",
    "poisson_zy":    "ν_zy [—]",
    "peak_eps_zz":   "ε_zz peak [%]",
    "peak_eps_yy":   "ε_yy peak [%]",
}

# ── ANOVA (computed once at startup) ──────────────────────────────────────────
def _run_anova(data):
    """HC3-robust Wald F-tests + BH-FDR. Returns long-form DataFrame."""
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        warnings.warn("statsmodels not installed — ANOVA tab disabled.")
        return pd.DataFrame()

    TERMS_2WAY = {
        "C(crystallinity)":                 "Cryst.",
        "C(orientation)":                   "Ori.",
        "C(crystallinity):C(orientation)":  "Cryst.×Ori.",
    }
    TERMS_3WAY = {
        "C(crystallinity)":                              "Cryst.",
        "C(fibre)":                                      "Fibre",
        "C(orientation)":                                "Ori.",
        "C(crystallinity):C(fibre)":                     "Cryst.×Fibre",
        "C(crystallinity):C(orientation)":               "Cryst.×Ori.",
        "C(fibre):C(orientation)":                       "Fibre×Ori.",
        "C(crystallinity):C(fibre):C(orientation)":      "Cryst.×Fibre×Ori.",
    }

    def _wald(formula, dat, term_map):
        try:
            m     = ols(formula, data=dat).fit(cov_type="HC3")
            m_ols = ols(formula, data=dat).fit()
            aov   = sm.stats.anova_lm(m_ols, typ=2)
            ss_res = aov.loc["Residual", "sum_sq"]
        except Exception:
            return {}
        exog   = m.model.exog_names
        df_den = float(m.df_resid)
        out    = {}
        for term_str, nice in term_map.items():
            parts = term_str.split(":")
            idxs  = [i for i, n in enumerate(exog)
                     if n != "Intercept"
                     and len(n.split(":")) == len(parts)
                     and all(any(pp.startswith(tp)
                                 for pp in n.split(":"))
                             for tp in parts)]
            if not idxs:
                continue
            R = np.zeros((len(idxs), len(exog)))
            for r, c in enumerate(idxs):
                R[r, c] = 1.0
            try:
                w = m.wald_test(R, scalar=True, use_f=True)
            except TypeError:
                w = m.wald_test(R, use_f=True)
            F_val = float(np.atleast_1d(w.statistic).squeeze())
            p_val = float(np.atleast_1d(w.pvalue).squeeze())
            eta_p = np.nan
            if term_str in aov.index:
                ss_t  = aov.loc[term_str, "sum_sq"]
                eta_p = float(ss_t / (ss_t + ss_res))
            out[nice] = (F_val, p_val, float(len(idxs)), df_den, eta_p)
        return out

    rows = []
    for mat, sub in data.groupby("material"):
        sub = sub.dropna(subset=["crystallinity", "orientation"])
        if sub["crystallinity"].nunique() < 2 or sub["orientation"].nunique() < 2:
            continue
        has_fibre   = sub["fibre"].nunique() >= 2
        term_map    = TERMS_3WAY if has_fibre else TERMS_2WAY
        base_facs   = ["crystallinity", "fibre", "orientation"] if has_fibre \
                      else ["crystallinity", "orientation"]
        formula_tpl = ("{dv} ~ C(crystallinity)*C(fibre)*C(orientation)"
                       if has_fibre else
                       "{dv} ~ C(crystallinity)*C(orientation)")
        for dv in DV_COLS:
            dat = sub.dropna(subset=[dv]).copy()
            if len(dat) < 6:
                continue
            cs = dat.groupby(base_facs).size()
            if cs.min() < 2:
                continue
            res = _wald(formula_tpl.format(dv=dv), dat, term_map)
            for nice, (F, p, df_num, df_den, eta_p) in res.items():
                rows.append(dict(material=mat, DV=dv, term=nice,
                                 F_HC3=round(F, 3), p_HC3=p, eta_p=eta_p))

    if not rows:
        return pd.DataFrame()

    aov = pd.DataFrame(rows)
    fdr_rows = []
    for (mat, dv), grp in aov.groupby(["material", "DV"]):
        _, ps_adj, _, _ = multipletests(grp["p_HC3"].values, method="fdr_bh")
        fdr_rows.append(grp.assign(p_fdr=ps_adj))
    aov = pd.concat(fdr_rows, ignore_index=True)
    aov["neg_log10_p_fdr"] = -np.log10(aov["p_fdr"].clip(lower=1e-10))

    def _stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    aov["sig"] = aov["p_fdr"].apply(_stars)
    return aov


print("  Computing ANOVA (HC3-robust, BH-FDR) …", end="", flush=True)
anova_df = _run_anova(df)
print("  done." if not anova_df.empty else "  statsmodels missing, ANOVA tab disabled.")

ANOVA_TERM_ORDER = [
    "Cryst.", "Fibre", "Ori.",
    "Cryst.×Fibre", "Cryst.×Ori.", "Fibre×Ori.",
    "Cryst.×Fibre×Ori.",
]

# ── Colour helpers ─────────────────────────────────────────────────────────────
MAT_COLORS = {
    "PEEK-CF":  "#1f77b4",
    "PEEK neat": "#aec7e8",
    "PPS-CF":   "#d62728",
    "PPS neat": "#f7b6b6",
}
CRYST_COLORS = {"High": "#333333", "Low": "#aaaaaa"}
ORI_SEQ = px.colors.sequential.Viridis

# ── App layout ─────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Thermal Expansion Explorer")

_tab_style      = {"padding": "6px 14px", "fontFamily": "sans-serif", "fontSize": 13}
_tab_sel_style  = {**_tab_style, "borderTop": "3px solid #1f77b4", "fontWeight": "bold"}

app.layout = html.Div([

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        html.H2("Thermal Expansion — Data Explorer",
                style={"margin": "0 0 4px 0", "color": "#222"}),
        html.P(f"Loaded {len(df)} specimens  ·  {len(DV_COLS)} metrics  ·  "
               f"{df['material'].nunique()} materials",
               style={"margin": 0, "color": "#666", "fontSize": 13}),
    ], style={"background": "#f8f8f8", "padding": "16px 24px",
              "borderBottom": "1px solid #ddd"}),

    # ── Tabs ──────────────────────────────────────────────────────────────────
    dcc.Tabs(id="tabs", value="bar", children=[

        # ── Tab 1: Bar chart explorer ─────────────────────────────────────────
        dcc.Tab(label="📊 Metrics Explorer", value="bar",
                style=_tab_style, selected_style=_tab_sel_style,
                children=[
            html.Div([
                # Controls row
                html.Div([
                    html.Div([
                        html.Label("Metric (y-axis)", style={"fontSize": 12}),
                        dcc.Dropdown(id="bar-dv",
                            options=[{"label": DV_LABELS.get(c, c), "value": c}
                                     for c in DV_COLS],
                            value=DV_COLS[0], clearable=False),
                    ], style={"width": "24%"}),
                    html.Div([
                        html.Label("Group x-axis by", style={"fontSize": 12}),
                        dcc.Dropdown(id="bar-xgroup",
                            options=[{"label": "Orientation",   "value": "orientation"},
                                     {"label": "Crystallinity", "value": "crystallinity"},
                                     {"label": "Material",      "value": "material"}],
                            value="orientation", clearable=False),
                    ], style={"width": "18%"}),
                    html.Div([
                        html.Label("Colour by", style={"fontSize": 12}),
                        dcc.Dropdown(id="bar-colour",
                            options=[{"label": "Crystallinity", "value": "crystallinity"},
                                     {"label": "Orientation",   "value": "orientation"},
                                     {"label": "Material",      "value": "material"}],
                            value="crystallinity", clearable=False),
                    ], style={"width": "18%"}),
                    html.Div([
                        html.Label("Materials", style={"fontSize": 12}),
                        dcc.Checklist(id="bar-mats",
                            options=[{"label": f" {m}", "value": m} for m in MATERIALS],
                            value=MATERIALS, inline=True,
                            labelStyle={"marginRight": 12, "fontSize": 12}),
                    ], style={"width": "36%"}),
                ], style={"display": "flex", "gap": "16px", "alignItems": "flex-end",
                          "padding": "12px 16px", "background": "#fafafa",
                          "borderBottom": "1px solid #eee"}),

                dcc.Graph(id="bar-chart",
                          style={"height": "540px"},
                          config={"displayModeBar": True,
                                  "toImageButtonOptions": {"format": "svg"}}),
            ]),
        ]),

        # ── Tab 2: Scatter ────────────────────────────────────────────────────
        dcc.Tab(label="🔵 Scatter", value="scatter",
                style=_tab_style, selected_style=_tab_sel_style,
                children=[
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("X axis", style={"fontSize": 12}),
                        dcc.Dropdown(id="sc-x",
                            options=[{"label": DV_LABELS.get(c, c), "value": c}
                                     for c in DV_COLS],
                            value=DV_COLS[0], clearable=False),
                    ], style={"width": "24%"}),
                    html.Div([
                        html.Label("Y axis", style={"fontSize": 12}),
                        dcc.Dropdown(id="sc-y",
                            options=[{"label": DV_LABELS.get(c, c), "value": c}
                                     for c in DV_COLS],
                            value=DV_COLS[1] if len(DV_COLS) > 1 else DV_COLS[0],
                            clearable=False),
                    ], style={"width": "24%"}),
                    html.Div([
                        html.Label("Colour by", style={"fontSize": 12}),
                        dcc.Dropdown(id="sc-colour",
                            options=[{"label": "Material",      "value": "material"},
                                     {"label": "Orientation",   "value": "orientation"},
                                     {"label": "Crystallinity", "value": "crystallinity"}],
                            value="material", clearable=False),
                    ], style={"width": "18%"}),
                    html.Div([
                        html.Label("Symbol by", style={"fontSize": 12}),
                        dcc.Dropdown(id="sc-symbol",
                            options=[{"label": "None",          "value": "none"},
                                     {"label": "Orientation",   "value": "orientation"},
                                     {"label": "Crystallinity", "value": "crystallinity"}],
                            value="crystallinity", clearable=False),
                    ], style={"width": "18%"}),
                ], style={"display": "flex", "gap": "16px", "alignItems": "flex-end",
                          "padding": "12px 16px", "background": "#fafafa",
                          "borderBottom": "1px solid #eee"}),
                dcc.Graph(id="scatter-chart",
                          style={"height": "540px"},
                          config={"displayModeBar": True,
                                  "toImageButtonOptions": {"format": "svg"}}),
            ]),
        ]),

        # ── Tab 3: ANOVA heatmap ───────────────────────────────────────────────
        dcc.Tab(label="🔬 ANOVA Heatmap", value="anova",
                style=_tab_style, selected_style=_tab_sel_style,
                children=[
            html.Div([
                html.Div([
                    html.Label("Select material(s):", style={"fontSize": 12}),
                    dcc.Checklist(id="anova-mats",
                        options=[{"label": f" {m}", "value": m}
                                 for m in (anova_df["material"].unique()
                                           if not anova_df.empty else [])],
                        value=list(anova_df["material"].unique())
                              if not anova_df.empty else [],
                        inline=True,
                        labelStyle={"marginRight": 14, "fontSize": 12}),
                ], style={"padding": "12px 16px", "background": "#fafafa",
                          "borderBottom": "1px solid #eee"}),
                html.P("Bubble area ∝ η²  ·  Colour = −log₁₀(p_FDR)  ·  "
                       "Hover for exact values  ·  "
                       "Grey = term not in model (2-way for PEEK, 3-way for PPS)",
                       style={"padding": "6px 16px 0", "fontSize": 12, "color": "#666"}),
                dcc.Graph(id="anova-chart",
                          style={"height": "600px"},
                          config={"displayModeBar": True,
                                  "toImageButtonOptions": {"format": "svg"}}),
            ]),
        ]),

        # ── Tab 4: Data table ──────────────────────────────────────────────────
        dcc.Tab(label="📋 Data Table", value="table",
                style=_tab_style, selected_style=_tab_sel_style,
                children=[
            html.Div([
                html.Div([
                    html.Label("Filter by material:", style={"fontSize": 12}),
                    dcc.Checklist(id="tbl-mats",
                        options=[{"label": f" {m}", "value": m} for m in MATERIALS],
                        value=MATERIALS, inline=True,
                        labelStyle={"marginRight": 12, "fontSize": 12}),
                ], style={"padding": "10px 16px", "background": "#fafafa",
                          "borderBottom": "1px solid #eee"}),
                html.Div(id="data-table-container",
                         style={"padding": "8px 16px"}),
            ]),
        ]),

    ], style={"fontFamily": "sans-serif"}),

], style={"fontFamily": "sans-serif", "maxWidth": "1400px", "margin": "0 auto"})


# ── Callbacks ──────────────────────────────────────────────────────────────────

# Tab 1 — Bar chart
@app.callback(
    Output("bar-chart", "figure"),
    Input("bar-dv",     "value"),
    Input("bar-xgroup", "value"),
    Input("bar-colour", "value"),
    Input("bar-mats",   "value"),
)
def update_bar(dv, xgroup, colour_by, sel_mats):
    sub = df[df["material"].isin(sel_mats or MATERIALS)].copy()
    label_y = DV_LABELS.get(dv, dv)

    # Aggregate: mean ± std per (xgroup, colour_by, material if not already)
    group_cols = list({xgroup, colour_by, "material"})
    agg = (sub.groupby(group_cols)[dv]
             .agg(["mean", "std", "count"])
             .reset_index()
             .rename(columns={"mean": "mean_val", "std": "std_val",
                               "count": "n"}))
    agg["se"] = agg["std_val"] / np.sqrt(agg["n"].clip(lower=1))

    # Facet by material (one subplot per material)
    n_mat = len(sel_mats or MATERIALS)
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=max(n_mat, 1),
        subplot_titles=[m for m in MATERIALS if m in (sel_mats or MATERIALS)],
        shared_yaxes=True,
    )

    colour_vals = sorted(agg[colour_by].unique())

    for i_m, mat in enumerate([m for m in MATERIALS if m in (sel_mats or MATERIALS)], 1):
        sub_m = agg[agg["material"] == mat]
        for j_c, cv in enumerate(colour_vals):
            sub_c = sub_m[sub_m[colour_by] == cv].sort_values(xgroup)
            colour = (CRYST_COLORS.get(cv) if colour_by == "crystallinity"
                      else MAT_COLORS.get(cv)
                      if colour_by == "material"
                      else ORI_SEQ[int(j_c * 256 / max(len(colour_vals), 1))]
                           if len(colour_vals) > 1 else "#555")
            fig.add_trace(go.Bar(
                x=sub_c[xgroup].astype(str),
                y=sub_c["mean_val"],
                error_y=dict(type="data", array=sub_c["se"].values, visible=True),
                name=str(cv),
                marker_color=colour,
                showlegend=(i_m == 1),
                hovertemplate=(
                    f"<b>{mat}</b><br>"
                    f"{xgroup}: %{{x}}<br>"
                    f"{colour_by}: {cv}<br>"
                    f"{label_y}: %{{y:.3f}} ± %{{error_y.array:.3f}}<br>"
                    "n = %{customdata}<extra></extra>"
                ),
                customdata=sub_c["n"].values,
            ), row=1, col=i_m)

    fig.update_layout(
        barmode="group",
        title_text=label_y,
        yaxis_title=label_y,
        height=500,
        legend_title=colour_by.capitalize(),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(size=12),
    )
    fig.update_xaxes(title_text=xgroup.capitalize())
    return fig


# Tab 2 — Scatter
@app.callback(
    Output("scatter-chart", "figure"),
    Input("sc-x",      "value"),
    Input("sc-y",      "value"),
    Input("sc-colour", "value"),
    Input("sc-symbol", "value"),
)
def update_scatter(x_col, y_col, colour_by, symbol_by):
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    lx = DV_LABELS.get(x_col, x_col)
    ly = DV_LABELS.get(y_col, y_col)

    sym_col  = None if symbol_by == "none" else symbol_by
    hover_id = "specimen" if "specimen" in plot_df.columns else None

    fig = px.scatter(
        plot_df,
        x=x_col, y=y_col,
        color=colour_by,
        symbol=sym_col,
        color_discrete_map={**MAT_COLORS, **CRYST_COLORS},
        hover_data={"material": True, "orientation": True,
                    "crystallinity": True, x_col: ":.4f", y_col: ":.4f"},
        labels={x_col: lx, y_col: ly, colour_by: colour_by.capitalize()},
        trendline="ols",
        trendline_scope="overall",
    )
    fig.update_traces(marker_size=8, marker_line_width=0.5,
                      marker_line_color="rgba(0,0,0,0.4)")
    fig.update_layout(height=500, plot_bgcolor="#fff", paper_bgcolor="#fff",
                      font=dict(size=12))
    return fig


# Tab 3 — ANOVA heatmap
@app.callback(
    Output("anova-chart", "figure"),
    Input("anova-mats", "value"),
)
def update_anova(sel_mats):
    if anova_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="statsmodels not installed — pip install statsmodels",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=16))
        return fig

    sel_mats = sel_mats or list(anova_df["material"].unique())
    sub = anova_df[anova_df["material"].isin(sel_mats)]

    mats_present = [m for m in MATERIALS if m in sub["material"].unique()] + \
                   [m for m in sub["material"].unique() if m not in MATERIALS]

    dvs_present = [d for d in [
        "alpha_lin_zz", "alpha_lin_yy", "alpha_rub_zz", "alpha_rub_yy",
        "anisotropy", "aniso_rubbery", "poisson_zy",
        "peak_eps_zz", "peak_eps_yy",
    ] if d in sub["DV"].unique()]

    dv_labels_short = {
        "alpha_lin_zz":  "α_zz glassy",  "alpha_lin_yy":  "α_yy glassy",
        "alpha_rub_zz":  "α_zz rubbery", "alpha_rub_yy":  "α_yy rubbery",
        "anisotropy":    "Aniso. glassy", "aniso_rubbery": "Aniso. rubbery",
        "poisson_zy":    "ν_zy",
        "peak_eps_zz":   "ε_zz peak",    "peak_eps_yy":   "ε_yy peak",
    }

    terms = [t for t in ANOVA_TERM_ORDER
             if t in sub["term"].unique()]

    from plotly.subplots import make_subplots
    n_mat = len(mats_present)
    if n_mat == 0:
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=n_mat,
        subplot_titles=mats_present,
        shared_yaxes=True,
        horizontal_spacing=0.04,
    )

    MAX_S = 60
    MIN_S = 4
    shown_cbar = False

    for i_m, mat in enumerate(mats_present, 1):
        sub_m = sub[sub["material"] == mat]

        for i_dv, dv in enumerate(reversed(dvs_present)):
            for j_t, term in enumerate(terms):
                r = sub_m[(sub_m["DV"] == dv) & (sub_m["term"] == term)]
                if r.empty:
                    # Grey dot = term not in model
                    fig.add_trace(go.Scatter(
                        x=[term], y=[dv_labels_short.get(dv, dv)],
                        mode="markers",
                        marker=dict(size=MIN_S + 2, color="#dddddd",
                                    line=dict(color="#aaa", width=0.5)),
                        showlegend=False,
                        hoverinfo="skip",
                    ), row=1, col=i_m)
                    continue

                row_r    = r.iloc[0]
                val      = float(row_r["neg_log10_p_fdr"])
                eta_p    = float(row_r["eta_p"]) if pd.notna(row_r["eta_p"]) else 0.0
                sig      = row_r["sig"]
                p_fdr    = float(row_r["p_fdr"])
                F_val    = float(row_r["F_HC3"])

                # Bubble size proportional to η²
                size = MIN_S + (MAX_S - MIN_S) * np.clip(eta_p, 0, 1)

                # Colour by −log10(p_FDR), capped at 4
                color_val = min(val, 4.0)

                hover = (
                    f"<b>{mat}  ·  {dv_labels_short.get(dv, dv)}</b><br>"
                    f"Term: {term}<br>"
                    f"F = {F_val:.2f}<br>"
                    f"p_FDR = {p_fdr:.4f}  {sig}<br>"
                    f"η² = {eta_p:.3f}<extra></extra>"
                )

                fig.add_trace(go.Scatter(
                    x=[term],
                    y=[dv_labels_short.get(dv, dv)],
                    mode="markers+text",
                    marker=dict(
                        size=size,
                        color=color_val,
                        colorscale=[[0, "#ffffff"], [0.2, "#fde0d0"],
                                    [0.5, "#f97c59"], [0.8, "#c0392b"],
                                    [1.0, "#7b0000"]],
                        cmin=0, cmax=4,
                        showscale=(not shown_cbar),
                        colorbar=dict(
                            title="−log₁₀(p_FDR)",
                            thickness=14, len=0.7,
                            tickvals=[0, 1, 1.301, 2, 3, 4],
                            ticktext=["0", "1", "p=0.05", "2", "3", "≥4"],
                        ) if not shown_cbar else None,
                        line=dict(color="#555", width=0.5),
                    ),
                    text=["" if sig == "ns" else sig],
                    textfont=dict(size=8,
                                  color="white" if val > 2 else "#333"),
                    textposition="middle center",
                    hovertemplate=hover,
                    showlegend=False,
                ), row=1, col=i_m)
                shown_cbar = True

    fig.update_layout(
        height=560,
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(size=11),
    )
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


# Tab 4 — Data table
@app.callback(
    Output("data-table-container", "children"),
    Input("tbl-mats", "value"),
)
def update_table(sel_mats):
    sub = df[df["material"].isin(sel_mats or MATERIALS)].copy()

    display_cols = (["material", "crystallinity", "orientation"]
                    + DV_COLS
                    + [c for c in ["fibre", "specimen"]
                       if c in sub.columns])

    # Round numeric columns
    for c in DV_COLS:
        sub[c] = sub[c].round(4)

    return dash_table.DataTable(
        data=sub[display_cols].to_dict("records"),
        columns=[{"name": DV_LABELS.get(c, c.replace("_", " ").title()),
                  "id": c}
                 for c in display_cols],
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        page_size=20,
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": 12, "padding": "4px 10px",
                    "fontFamily": "monospace"},
        style_header={"fontWeight": "bold", "background": "#f0f4f8",
                      "fontSize": 12},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"},
        ],
        export_format="csv",
        export_headers="display",
    )


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Thermal Expansion Explorer")
    print(f"  Data: {METRICS_CSV}")
    print(f"  Specimens: {len(df)}  ·  Materials: {df['material'].nunique()}")
    print("\n  Open your browser at  http://127.0.0.1:8050\n")
    app.run(debug=False, host="127.0.0.1", port=8050)
