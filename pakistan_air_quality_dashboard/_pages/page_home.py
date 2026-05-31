"""pages/page_home.py — Overview & dataset upload"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_loader import load_data, AQI_CATEGORY_ORDER, CAT_COLORS, CITY_PALETTE
from utils.ui import page_header, metric_row, section_title, callout, no_data_message
from utils.charts import aqi_stacked_bar, city_ranking_bar, theme


def render():
    page_header(
        eyebrow="STAT222 · BSDS-02 · Pakistan Air Quality Analysis",
        title="Air Quality Crisis in Pakistan",
        description=(
            "A rigorous statistical analysis across 10 major cities using One-Way ANOVA, "
            "Two-Way ANOVA, Distribution Fitting, Multiple Regression, ARIMA, "
            "and Nonparametric methods. Data: 21,840 hourly observations, Nov 2025 – Feb 2026."
        ),
    )

    # ── File upload ──────────────────────────────────────────────────────────
    with st.expander("📂  Upload Dataset  (click to expand)", expanded=True):
        uploaded = st.file_uploader(
            "Upload pakistan_air_quality_final_clean.csv",
            type=["csv"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.session_state["uploaded_file"] = uploaded

    uploaded_file = st.session_state.get("uploaded_file", None)
    df = load_data(uploaded_file)

    if df is None:
        st.markdown("---")
        no_data_message()
        callout(
            "Download the dataset from: "
            "<a href='https://www.kaggle.com/datasets/ahsanneural/pakistan-air-quality-and-weather-10-cities' "
            "target='_blank' style='color:#38bdf8'>Kaggle — Pakistan Air Quality</a> "
            "then upload it above.",
            "info",
        )
        _render_demo_content()
        return

    st.session_state["df"] = df
    _render_with_data(df)


def _render_with_data(df):
    # ── Top metrics ──────────────────────────────────────────────────────────
    section_title("Dataset Overview", "📋")

    n_obs   = len(df)
    n_cities = df["city"].nunique() if "city" in df.columns else 0
    date_range = ""
    if "date" in df.columns:
        mn, mx = df["date"].min(), df["date"].max()
        date_range = f"{mn.strftime('%b %Y')} – {mx.strftime('%b %Y')}"

    pm25_mean = df["pm25"].mean() if "pm25" in df.columns else None
    who_ratio = f"{pm25_mean/5:.1f}×" if pm25_mean else "–"

    pct_haz = (
        (df["aqi_category"] == "Hazardous").mean() * 100
        if "aqi_category" in df.columns else 0
    )

    metric_row([
        {"label": "Total Observations", "value": f"{n_obs:,}",      "sub": "Hourly records"},
        {"label": "Cities Monitored",   "value": str(n_cities),     "sub": "Major Pakistan cities"},
        {"label": "Study Period",        "value": date_range,        "sub": "Peak pollution season"},
        {"label": "Mean PM2.5",          "value": f"{pm25_mean:.1f} µg/m³" if pm25_mean else "–",
         "sub": f"{who_ratio} WHO annual guideline", "color": "danger"},
        {"label": "Hazardous Days %",    "value": f"{pct_haz:.1f}%",
         "sub": "AQI category: Hazardous", "color": "danger" if pct_haz > 5 else "warning"},
    ])

    # ── AQI category breakdown ───────────────────────────────────────────────
    st.markdown("")
    c1, c2 = st.columns(2)

    with c1:
        section_title("AQI Category Distribution", "🎨")
        if "aqi_category" in df.columns:
            freq = df["aqi_category"].value_counts().reindex(AQI_CATEGORY_ORDER).dropna()
            pct  = freq / len(df) * 100
            fig = go.Figure(go.Bar(
                x=freq.index,
                y=freq.values,
                marker_color=[CAT_COLORS.get(c, "#888") for c in freq.index],
                text=[f"{p:.1f}%" for p in pct.values],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
            ))
            fig.update_layout(**theme(
                title=dict(text="Overall AQI Category Frequency", font=dict(size=13)),
                yaxis_title="Observations",
                height=360,
                showlegend=False,
            ))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        section_title("City Pollution Ranking", "🏙️")
        if "pm25" in df.columns and "city" in df.columns:
            st.plotly_chart(
                city_ranking_bar(df, "pm25", "Mean PM2.5 by City",
                                 "Mean PM2.5 (µg/m³)"),
                use_container_width=True,
            )

    # ── Stacked city breakdown ───────────────────────────────────────────────
    if "aqi_category" in df.columns and "city" in df.columns:
        section_title("AQI Mix by City", "📊")
        st.plotly_chart(
            aqi_stacked_bar(df, "city", "AQI Category Proportion by City"),
            use_container_width=True,
        )

    # ── Data preview ─────────────────────────────────────────────────────────
    with st.expander("🔍  Raw Data Preview"):
        st.dataframe(
            df.head(200).style.format(precision=2),
            use_container_width=True,
            height=320,
        )

    # ── Quick stats ──────────────────────────────────────────────────────────
    with st.expander("📈  Quick Descriptive Statistics"):
        focus = [c for c in ["pm25","pm10","no2","so2","co","temperature","humidity","wind_speed"]
                 if c in df.columns]
        if focus:
            desc = df[focus].describe().T
            desc["cv_%"] = (desc["std"] / desc["mean"] * 100).round(1)
            desc["skew"] = df[focus].skew().round(3)
            st.dataframe(desc.style.format(precision=3), use_container_width=True)


def _render_demo_content():
    """Show placeholder cards when no data is loaded."""
    section_title("What this dashboard covers", "📚")
    items = [
        ("🔬  One-Way ANOVA",           "Tests whether mean AQI differs across 10 cities."),
        ("🔀  Two-Way ANOVA",           "City × Season interaction on AQI."),
        ("📐  Distribution Fitting",    "Identifies if PM2.5 follows Lognormal, Gamma, Weibull, or Normal."),
        ("📈  Multiple Regression",     "Quantifies meteorological & pollutant predictors of PM2.5."),
        ("⏱️  Time Series & ARIMA",     "STL decomposition + ARIMA forecasting of daily PM2.5."),
        ("🔢  Nonparametric Tests",     "Kruskal-Wallis + Mann-Whitney U as robust confirmations."),
        ("🧪  Extra Hypothesis Tests",  "Welch's T, Chi-Square, Mann-Kendall, KS, Diurnal ANOVA."),
    ]
    cols = st.columns(3)
    for i, (title, desc) in enumerate(items):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card" style="min-height:90px">
                <div style="font-weight:700;font-size:.9rem;margin-bottom:6px">{title}</div>
                <div style="font-size:.78rem;color:#8b949e">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
