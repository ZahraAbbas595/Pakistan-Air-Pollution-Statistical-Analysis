"""pages/page_eda.py — Exploratory Data Analysis"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.data_loader import AQI_CATEGORY_ORDER, CAT_COLORS, CITY_PALETTE, get_focus_cols
from utils.ui import page_header, section_title, callout, no_data_message
from utils.charts import corr_heatmap, ts_line, city_boxplot, theme, hex_to_rgba as _hex_rgba


def render():
    page_header(
        eyebrow="Section 2 · Exploratory Data Analysis",
        title="Exploring the Data",
        description=(
            "Descriptive statistics, distribution visualisations, correlation analysis, "
            "temporal trends, and outlier detection across all variables."
        ),
    )

    df: pd.DataFrame | None = st.session_state.get("df", None)
    if df is None:
        no_data_message()
        return

    focus_cols = get_focus_cols(df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Descriptive Stats",
        "📦 Distributions",
        "🔗 Correlations",
        "📅 Temporal Trends",
        "🚨 Outlier Detection",
    ])

    # ── Tab 1: Descriptive stats ─────────────────────────────────────────────
    with tab1:
        section_title("Extended Descriptive Statistics")
        numeric = df[focus_cols].select_dtypes(include=np.number)
        desc = numeric.describe().T
        desc["cv_%"]     = (desc["std"] / desc["mean"] * 100).round(1)
        desc["skewness"] = numeric.skew().round(3)
        desc["kurtosis"] = numeric.kurt().round(3)
        desc["IQR"]      = (numeric.quantile(.75) - numeric.quantile(.25)).round(3)
        show_cols = ["mean","50%","std","IQR","cv_%","skewness","kurtosis","min","max"]
        st.dataframe(desc[show_cols].style.format(precision=3)
                     .background_gradient(subset=["cv_%","skewness"], cmap="RdYlGn_r"),
                     use_container_width=True)

        if "aqi_category" in df.columns:
            section_title("AQI Category Frequency Table")
            freq = df["aqi_category"].value_counts().reindex(AQI_CATEGORY_ORDER).dropna()
            pct  = (freq / len(df) * 100).round(2)
            ftab = pd.DataFrame({"Count": freq, "Pct %": pct,
                                  "Cumulative %": pct.cumsum().round(2)})
            st.dataframe(ftab.style.format({"Count": "{:,}", "Pct %": "{:.2f}",
                                            "Cumulative %": "{:.2f}"}),
                         use_container_width=True)

        if "city" in df.columns and "aqi_category" in df.columns:
            section_title("City-Wise AQI Breakdown (%)")
            city_cat = (
                df.groupby("city")["aqi_category"]
                .value_counts(normalize=True).mul(100).round(1)
                .rename("pct").reset_index()
                .pivot(index="city", columns="aqi_category", values="pct").fillna(0)
            )
            city_cat = city_cat.reindex(columns=[c for c in AQI_CATEGORY_ORDER if c in city_cat.columns])
            st.dataframe(city_cat.style.format("{:.1f}").background_gradient(cmap="YlOrRd"),
                         use_container_width=True)

    # ── Tab 2: Distributions ─────────────────────────────────────────────────
    with tab2:
        c1, c2 = st.columns(2)
        poll_cols = [c for c in ["pm25","pm10","no2","so2"] if c in df.columns]
        sel_col = c1.selectbox("Variable", poll_cols + ["aqi_numeric"], key="eda_dist_col")
        group_by = c2.selectbox("Colour by", ["None", "city", "season"], key="eda_dist_grp")

        if sel_col and sel_col in df.columns:
            data_col = df[sel_col].dropna()

            # histogram
            if group_by != "None" and group_by in df.columns:
                fig = px.histogram(
                    df.dropna(subset=[sel_col, group_by]),
                    x=sel_col, color=group_by,
                    nbins=60, histnorm="probability density",
                    opacity=.65, barmode="overlay",
                    color_discrete_sequence=CITY_PALETTE,
                )
            else:
                fig = go.Figure(go.Histogram(
                    x=data_col, nbinsx=60,
                    histnorm="probability density",
                    marker_color="#38bdf8", opacity=.7,
                ))
            fig.update_layout(**theme(
                title=dict(text=f"Distribution of {sel_col}", font=dict(size=13)),
                xaxis_title=sel_col, yaxis_title="Density", height=340,
            ))
            st.plotly_chart(fig, use_container_width=True)

        if "city" in df.columns:
            section_title("Box Plots by City")
            sel_box = st.selectbox("Variable for box plot", poll_cols, key="eda_box_col")
            if sel_box:
                st.plotly_chart(city_boxplot(df, sel_box, f"{sel_box.upper()} by City",
                                             f"{sel_box} (µg/m³)"),
                                use_container_width=True)

        # Violin — PM2.5 by AQI category
        if "aqi_category" in df.columns and "pm25" in df.columns:
            section_title("PM2.5 Within AQI Categories")
            present = [c for c in AQI_CATEGORY_ORDER if c in df["aqi_category"].cat.categories]
            fig = go.Figure()
            for cat in present:
                fig.add_trace(go.Violin(
                    y=df[df["aqi_category"] == cat]["pm25"].dropna(),
                    name=cat,
                    box_visible=True, meanline_visible=True,
                    line_color=CAT_COLORS.get(cat, "#888"),
                    fillcolor=_hex_rgba(CAT_COLORS.get(cat, "#888888"), 0.2),
                ))
            fig.update_layout(**theme(
                title=dict(text="PM2.5 Within AQI Category", font=dict(size=13)),
                yaxis_title="PM2.5 (µg/m³)", height=400, showlegend=False,
            ))
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Correlations ──────────────────────────────────────────────────
    with tab3:
        numeric_df = df[focus_cols].dropna()
        if len(numeric_df) > 10:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(corr_heatmap(numeric_df, focus_cols, "pearson",
                                             "Pearson Correlation"),
                                use_container_width=True)
            with c2:
                st.plotly_chart(corr_heatmap(numeric_df, focus_cols, "spearman",
                                             "Spearman Correlation"),
                                use_container_width=True)

            if "aqi_numeric" in numeric_df.columns:
                section_title("Spearman ρ with AQI")
                spear = numeric_df.corr("spearman")["aqi_numeric"].drop("aqi_numeric")
                spear = spear.sort_values(key=abs, ascending=True)
                colors = ["#f87171" if v > 0 else "#38bdf8" for v in spear.values]
                fig = go.Figure(go.Bar(
                    x=spear.values, y=spear.index, orientation="h",
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in spear.values], textposition="outside",
                ))
                fig.update_layout(**theme(
                    title=dict(text="Spearman Correlation with AQI", font=dict(size=13)),
                    xaxis_title="Spearman ρ", height=340,
                ))
                st.plotly_chart(fig, use_container_width=True)

            callout(
                "<strong>Interpretation:</strong> Spearman ρ is preferred here — "
                "PM2.5 data is right-skewed, violating the linearity assumption of Pearson. "
                "Where Spearman >> Pearson, the relationship is monotonic but nonlinear.",
                "info",
            )

    # ── Tab 4: Temporal Trends ───────────────────────────────────────────────
    with tab4:
        if "date" not in df.columns:
            callout("No date column found.", "warning")
        else:
            if "pm25" in df.columns:
                section_title("Monthly PM2.5 Trend")
                monthly = df.set_index("date")["pm25"].resample("ME").mean().dropna()
                st.plotly_chart(ts_line(monthly, "Monthly Mean PM2.5", "PM2.5 (µg/m³)"),
                                use_container_width=True)

            if "pm25" in df.columns and "city" in df.columns:
                section_title("PM2.5 Trend by City")
                fig = go.Figure()
                for i, city in enumerate(sorted(df["city"].unique())):
                    ts = (df[df["city"] == city]
                          .set_index("date")["pm25"].resample("D").mean().dropna())
                    fig.add_trace(go.Scatter(
                        x=ts.index, y=ts.values, mode="lines",
                        name=city,
                        line=dict(color=CITY_PALETTE[i % len(CITY_PALETTE)], width=1.5),
                        opacity=.8,
                    ))
                fig.update_layout(**theme(
                    title=dict(text="Daily PM2.5 per City", font=dict(size=13)),
                    yaxis_title="PM2.5 (µg/m³)", height=420,
                ))
                st.plotly_chart(fig, use_container_width=True)

            if "season" in df.columns and "pm25" in df.columns:
                section_title("PM2.5 by Season")
                avail = [s for s in ["Winter","Spring","Monsoon","Autumn/Post-Monsoon"]
                          if s in df["season"].unique()]
                fig = go.Figure()
                for i, s in enumerate(avail):
                    fig.add_trace(go.Box(
                        y=df[df["season"] == s]["pm25"].dropna(),
                        name=s, marker_color=CITY_PALETTE[i],
                        boxmean=True,
                    ))
                fig.update_layout(**theme(
                    title=dict(text="PM2.5 by Season", font=dict(size=13)),
                    yaxis_title="PM2.5 (µg/m³)", height=360, showlegend=False,
                ))
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Outlier detection ─────────────────────────────────────────────
    with tab5:
        section_title("IQR Outlier Detection")
        rows = []
        numeric_df2 = df[focus_cols].select_dtypes(include=np.number)
        for col in numeric_df2.columns:
            Q1, Q3 = numeric_df2[col].quantile([.25, .75])
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            n_out = ((numeric_df2[col] < lo) | (numeric_df2[col] > hi)).sum()
            rows.append({
                "Variable": col, "Q1": round(Q1, 2), "Q3": round(Q3, 2),
                "IQR": round(IQR, 2), "Lower Fence": round(lo, 2),
                "Upper Fence": round(hi, 2), "N Outliers": n_out,
                "Outlier %": round(n_out / len(numeric_df2) * 100, 2),
            })
        out_df = pd.DataFrame(rows)
        st.dataframe(out_df.style.format({"N Outliers": "{:,}", "Outlier %": "{:.2f}"})
                     .background_gradient(subset=["Outlier %"], cmap="YlOrRd"),
                     use_container_width=True)

        fig = go.Figure(go.Bar(
            x=out_df["Variable"],
            y=out_df["Outlier %"],
            marker_color=["#f87171" if v > 5 else "#fbbf24" if v > 1 else "#4ade80"
                          for v in out_df["Outlier %"]],
            text=[f"{v:.1f}%" for v in out_df["Outlier %"]],
            textposition="outside",
        ))
        fig.add_hline(y=5, line_dash="dash", line_color="#f87171",
                      annotation_text="5% threshold")
        fig.update_layout(**theme(
            title=dict(text="Outlier % by Variable (IQR Method)", font=dict(size=13)),
            yaxis_title="Outlier Percentage (%)", height=340,
        ))
        st.plotly_chart(fig, use_container_width=True)
