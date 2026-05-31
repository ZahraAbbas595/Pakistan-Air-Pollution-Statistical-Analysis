"""
utils/charts.py
Reusable Plotly chart builders -- all using the dark dashboard theme.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.data_loader import CITY_PALETTE, CAT_COLORS, AQI_CATEGORY_ORDER

# ── Theme defaults ─────────────────────────────────────────────────────────────
PLOT_BG    = "#161b22"
PAPER_BG   = "#161b22"
GRID_COLOR = "#30363d"
FONT_COLOR = "#e6edf3"
FONT_FAMILY = "Sora, sans-serif"

LAYOUT_BASE = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family=FONT_FAMILY, color=FONT_COLOR, size=12),
    margin=dict(t=50, b=40, l=60, r=30),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=GRID_COLOR,
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(
        gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
        linecolor=GRID_COLOR, tickfont=dict(size=11),
    ),
    yaxis=dict(
        gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
        linecolor=GRID_COLOR, tickfont=dict(size=11),
    ),
    hoverlabel=dict(bgcolor="#1c2128", bordercolor="#30363d", font_size=12),
)


def theme(**kwargs):
    d = {**LAYOUT_BASE}
    d.update(kwargs)
    return d


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert a 6-digit hex color to an rgba() string Plotly accepts."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Helper: box plot per city ──────────────────────────────────────────────────
def city_boxplot(df, col, title="", ylabel=""):
    order = df.groupby("city")[col].median().sort_values(ascending=False).index.tolist()
    fig = go.Figure()
    for i, city in enumerate(order):
        color = CITY_PALETTE[i % len(CITY_PALETTE)]
        fig.add_trace(go.Box(
            y=df[df["city"] == city][col].dropna(),
            name=city,
            marker_color=color,
            line_color=color,
            fillcolor=hex_to_rgba(color, 0.2),
            boxmean=True,
        ))
    fig.update_layout(**theme(
        title=dict(text=title, font=dict(size=14, weight=700)),
        yaxis_title=ylabel,
        showlegend=False,
        height=380,
    ))
    return fig


# ── Helper: AQI category stacked bar ──────────────────────────────────────────
def aqi_stacked_bar(df, group_col="city", title=""):
    ct = (
        df.groupby([group_col, "aqi_category"], observed=True)
        .size().unstack(fill_value=0)
    )
    ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100)
    present = [c for c in AQI_CATEGORY_ORDER if c in ct_pct.columns]
    ct_pct = ct_pct[present]

    fig = go.Figure()
    for cat in present:
        fig.add_trace(go.Bar(
            name=cat,
            x=ct_pct.index,
            y=ct_pct[cat],
            marker_color=CAT_COLORS.get(cat, "#888"),
            hovertemplate=f"<b>%{{x}}</b><br>{cat}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(**theme(
        barmode="stack",
        title=dict(text=title, font=dict(size=14, weight=700)),
        yaxis_title="% of Observations",
        yaxis_range=[0, 101],
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    ))
    return fig


# ── Helper: horizontal bar (ranking) ──────────────────────────────────────────
def city_ranking_bar(df, col, title="", xlab=""):
    stats = df.groupby("city")[col].mean().sort_values(ascending=True)
    colors = [
        "#f87171" if v > 100 else "#fbbf24" if v > 60 else "#4ade80"
        for v in stats.values
    ]
    fig = go.Figure(go.Bar(
        x=stats.values,
        y=stats.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in stats.values],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b>: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(**theme(
        title=dict(text=title, font=dict(size=14, weight=700)),
        xaxis_title=xlab,
        height=380,
        showlegend=False,
    ))
    return fig


# ── Helper: correlation heatmap ───────────────────────────────────────────────
def corr_heatmap(df, cols, method="pearson", title=""):
    corr = df[cols].corr(method=method).round(3)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    z = corr.where(~mask).values

    fig = go.Figure(go.Heatmap(
        z=z,
        x=corr.columns,
        y=corr.index,
        colorscale=[
            [0, "#2563eb"], [0.5, "#161b22"], [1, "#dc2626"]
        ],
        zmid=0,
        text=[[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11),
        showscale=True,
        colorbar=dict(tickfont=dict(size=10)),
    ))
    fig.update_layout(**theme(
        title=dict(text=title, font=dict(size=14, weight=700)),
        height=440,
    ))
    return fig


# ── Helper: time series line ───────────────────────────────────────────────────
def ts_line(ts: pd.Series, title="", ylab=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.index, y=ts.values,
        mode="lines",
        line=dict(color="#38bdf8", width=2),
        name=ylab,
        hovertemplate="<b>%{x}</b>: %{y:.1f}<extra></extra>",
    ))
    if len(ts) >= 7:
        roll = ts.rolling(7, center=True).mean()
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll.values,
            mode="lines",
            line=dict(color="#f97316", width=2.5, dash="dash"),
            name="7-day rolling mean",
        ))
    fig.add_hline(y=15, line_dash="dot", line_color="#fbbf24",
                  annotation_text="WHO 24-hr (15)", annotation_font_size=10)
    fig.add_hline(y=5, line_dash="dot", line_color="#4ade80",
                  annotation_text="WHO annual (5)", annotation_font_size=10)
    fig.update_layout(**theme(
        title=dict(text=title, font=dict(size=14, weight=700)),
        yaxis_title=ylab,
        height=360,
    ))
    return fig


# ── Helper: histogram ─────────────────────────────────────────────────────────
def pm25_histogram(data, title="PM2.5 Distribution"):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=60,
        histnorm="probability density",
        marker_color="#38bdf8",
        marker_line=dict(color="#1c2128", width=0.5),
        opacity=0.7,
        name="Observed",
    ))
    fig.update_layout(**theme(
        title=dict(text=title, font=dict(size=14, weight=700)),
        xaxis_title="PM2.5 (µg/m³)",
        yaxis_title="Probability Density",
        height=360,
    ))
    return fig


# ── Helper: scatter ───────────────────────────────────────────────────────────
def scatter_plot(x, y, xlabel, ylabel, title=""):
    fig = go.Figure(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(color="#38bdf8", size=4, opacity=0.35),
        hovertemplate=f"{xlabel}: %{{x:.2f}}<br>{ylabel}: %{{y:.2f}}<extra></extra>",
    ))
    fig.update_layout(**theme(
        title=dict(text=title, font=dict(size=14, weight=700)),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=360,
        showlegend=False,
    ))
    return fig


# ── Helper: mean CI bar ───────────────────────────────────────────────────────
def mean_ci_bar(cities, means, cis, title="", f_stat=None, p_val=None, eta_sq=None):
    colors = [
        "#f87171" if m > 200 else "#fbbf24" if m > 125 else "#4ade80"
        for m in means
    ]
    subtitle = ""
    if f_stat is not None:
        subtitle = f"F = {f_stat:.2f}  |  p = {p_val:.2e}  |  eta2 = {eta_sq:.3f}"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(cities),
        y=list(means),
        error_y=dict(type="data", array=list(cis), visible=True,
                     color="#e6edf3", thickness=1.5, width=5),
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Mean: %{y:.1f}<extra></extra>",
    ))
    fig.add_hline(y=float(np.mean(means)), line_dash="dash", line_color="#8b949e",
                  annotation_text="Grand mean", annotation_font_size=10)
    fig.update_layout(**theme(
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>",
                   font=dict(size=13, weight=700)),
        yaxis_title="Mean AQI (Numeric Proxy) +/- 95% CI",
        height=390,
        showlegend=False,
    ))
    return fig
