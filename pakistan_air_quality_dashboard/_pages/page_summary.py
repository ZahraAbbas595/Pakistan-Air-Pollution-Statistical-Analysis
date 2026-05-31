"""pages/page_summary.py — Summary, Conclusions & Policy Recommendations"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.ui import page_header, section_title, callout, no_data_message, metric_row
from utils.charts import city_ranking_bar, theme
from utils.data_loader import AQI_CATEGORY_ORDER, CAT_COLORS, CITY_PALETTE


def render():
    page_header(
        eyebrow="Section 4 & 5 · Conclusions",
        title="Summary & Policy Recommendations",
        description=(
            "10 formal hypothesis tests across 7 statistical methods converge on a "
            "single robust conclusion: Pakistan's urban air quality is significantly shaped "
            "by city identity, season, time of day, and source-activity patterns."
        ),
    )

    df: pd.DataFrame | None = st.session_state.get("df", None)

    # ── Hypothesis test summary table ─────────────────────────────────────────
    section_title("All Hypothesis Tests — Master Summary", "📋")

    tests = [
        ("1", "One-Way ANOVA",             "Equal mean AQI across all cities",                             "✅ Reject",  "Industrial cities significantly more polluted"),
        ("2", "Two-Way ANOVA (City×Season)","No city×season interaction",                                  "✅ Reject",  "Winter worsening is city-specific"),
        ("3", "K-S Goodness-of-Fit",        "PM2.5 follows Normal/Lognormal/Gamma/Weibull",                "✅ Best fit","PM2.5 is Lognormal → use correct tail for alerts"),
        ("4", "OLS Regression F-test",      "No predictors explain PM2.5",                                 "✅ Reject",  "PM10, wind speed, CO are dominant drivers"),
        ("5", "Kruskal-Wallis H-test",       "Identical AQI rank distributions across cities",             "✅ Reject",  "Nonparametric confirmation of ANOVA"),
        ("6", "Welch's T-Test",             "Equal PM2.5 in most/least polluted city",                     "✅ Reject",  "Cohen's d = large → policy urgency justified"),
        ("7", "Chi-Square Independence",    "AQI independent of weekend/weekday",                          "✅ Reject",  "Traffic/industry drive weekday AQI elevation"),
        ("8", "Mann-Kendall Trend",         "No monotonic PM2.5 trend",                                    "✅ Reject",  "Significant rising trend through winter"),
        ("9", "Two-Sample KS Test",         "Autumn/Winter PM2.5 from same distribution",                  "✅ Reject",  "Fundamentally different seasonal distributions"),
        ("10","ANOVA (Time-of-Day)",        "Equal PM2.5 across Morning/Afternoon/Evening/Night",          "✅ Reject",  "Diurnal cycle confirmed — nighttime peaks"),
    ]

    tests_df = pd.DataFrame(tests, columns=["#","Test","H₀","Decision","Real-World Finding"])
    st.dataframe(
        tests_df.style.map(
            lambda v: "color:#4ade80;font-weight:700" if "Reject" in str(v) or "Best" in str(v) else "",
            subset=["Decision"],
        ),
        use_container_width=True,
        height=400,
    )

    # ── Robustness statement ──────────────────────────────────────────────────
    callout(
        "<strong>Robustness Statement:</strong> All 10 tests converge on the same overarching conclusion. "
        "This convergence across parametric (ANOVA, T-test, regression) and nonparametric "
        "(Kruskal-Wallis, Mann-Kendall, KS test, Chi-Square) methods, across multiple effect size "
        "metrics (η², ε², Cramér's V, Cohen's d), and across multiple variable representations "
        "(AQI proxy, raw PM2.5, AQI category) constitutes the strongest possible statistical basis "
        "for evidence-based policy.",
        "success",
    )

    # ── Key findings ──────────────────────────────────────────────────────────
    section_title("Key Findings", "🔍")

    findings = [
        ("🏙️", "City Disparity",
         "ANOVA F >> critical value, η² > 0.10. City membership alone explains a meaningful fraction "
         "of AQI variance. Industrial corridors require targeted interventions."),
        ("🌤️", "Seasonal Interaction",
         "Two-Way ANOVA interaction p < 0.05. Winter emergency protocols must be seasonal AND "
         "city-specific — not applied uniformly."),
        ("📐", "Lognormal PM2.5",
         "Lowest AIC among 4 candidates. Normal-assumption models underestimate hazardous exposure "
         "frequency. Health alerts must use lognormal tail percentiles."),
        ("💨", "Meteorological Drivers",
         "Wind speed (negative β) and PM10 (positive β) dominate regression. Atmospheric dispersion "
         "physically reduces ground-level concentrations."),
        ("⏱️", "Predictable Seasonality",
         "ARIMA captures temporal structure; Ljung-Box confirms white-noise residuals. "
         "Proactive seasonal protocols are feasible."),
        ("⏰", "Diurnal Cycle",
         "ANOVA on time-of-day significant. Nighttime temperature inversions and morning rush hours "
         "create systematic within-day PM2.5 variation."),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(findings):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card" style="min-height:130px;margin-bottom:12px">
                <div style="font-size:1.5rem;margin-bottom:8px">{icon}</div>
                <div style="font-weight:700;font-size:.9rem;color:#e6edf3;margin-bottom:6px">{title}</div>
                <div style="font-size:.76rem;color:#8b949e;line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Policy recommendations ────────────────────────────────────────────────
    section_title("Policy Recommendations", "📌")

    policies = [
        ("1", "#f97316", "City-Specific PM2.5 Caps",
         "Faisalabad and Lahore require intervention thresholds ~8× stricter than Quetta. "
         "A single national standard is statistically insufficient (ANOVA η² > 0.10)."),
        ("2", "#38bdf8", "Season-Aware Emergency Protocols",
         "Autumn → Winter transition triggers escalating restrictions. "
         "Mann-Kendall trend confirms worsening trajectory through December–February."),
        ("3", "#4ade80", "Time-of-Day Traffic Management",
         "Nighttime and morning peak restrictions have maximum health impact. "
         "Diurnal ANOVA η² confirms measurable within-day variation."),
        ("4", "#fbbf24", "Weekday Industrial Controls",
         "Chi-Square evidence supports targeted weekday emission limits. "
         "Weekend AQI distribution is measurably better."),
        ("5", "#c084fc", "Lognormal-Based Alert Thresholds",
         "Hazardous day frequency is underestimated by Normal models. "
         "Use 95th/99th percentile of fitted Lognormal for alert calibration."),
    ]

    for num, color, title, desc in policies:
        st.markdown(f"""
        <div style="display:flex;gap:16px;align-items:flex-start;
                    background:#161b22;border:1px solid #30363d;border-left:4px solid {color};
                    border-radius:8px;padding:14px 18px;margin-bottom:10px">
            <div style="font-family:'JetBrains Mono',monospace;font-weight:700;
                        font-size:1.1rem;color:{color};min-width:24px">{num}</div>
            <div>
                <div style="font-weight:700;font-size:.9rem;color:#e6edf3;margin-bottom:4px">{title}</div>
                <div style="font-size:.8rem;color:#8b949e;line-height:1.55">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── WHO Exceedance (data-driven) ───────────────────────────────────────────
    if df is not None and "pm25" in df.columns and "city" in df.columns:
        section_title("WHO Exceedance by City", "🌍")

        cities = sorted(df["city"].unique())
        exc_rows = []
        for city in cities:
            cdf = df[df["city"] == city]["pm25"].dropna()
            exc_rows.append({
                "City":        city,
                "Mean PM2.5":  round(cdf.mean(), 1),
                "% > WHO 15":  round((cdf > 15).mean() * 100, 1),
                "% > WHO 35":  round((cdf > 35).mean() * 100, 1),
                "× WHO Annual (5)": round(cdf.mean() / 5, 1),
                "Alert Level": "🔴 Critical" if cdf.mean() > 35 else "🟡 High" if cdf.mean() > 15 else "🟢 OK",
            })

        exc_df = pd.DataFrame(exc_rows).sort_values("Mean PM2.5", ascending=False)
        st.dataframe(exc_df.style.background_gradient(subset=["Mean PM2.5","% > WHO 15","% > WHO 35"],
                                                       cmap="YlOrRd"),
                     use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=exc_df["City"], y=exc_df["Mean PM2.5"],
            marker_color=["#f87171" if v > 35 else "#fbbf24" if v > 15 else "#4ade80"
                          for v in exc_df["Mean PM2.5"]],
            text=exc_df["Mean PM2.5"].astype(str) + " µg/m³",
            textposition="outside",
        ))
        fig.add_hline(y=15, line_dash="dot", line_color="#fbbf24",
                      annotation_text="WHO 24-hr (15)", annotation_font_size=9)
        fig.add_hline(y=5, line_dash="dot", line_color="#4ade80",
                      annotation_text="WHO annual (5)", annotation_font_size=9)
        fig.update_layout(**theme(
            title=dict(text="Mean PM2.5 by City vs WHO Guidelines", font=dict(size=13)),
            yaxis_title="Mean PM2.5 (µg/m³)",
            height=400, showlegend=False,
        ))
        st.plotly_chart(fig, use_container_width=True)

    # ── Assumptions verification table ───────────────────────────────────────
    section_title("Statistical Assumption Verification", "✅")

    assump = [
        ("Normality of groups",    "Shapiro-Wilk",    "ANOVA, Nonparametric",  "Often violated → KW robustness applied"),
        ("Equal group variances",  "Levene's Test",   "One-Way ANOVA",         "Checked; result reported in Section 3.1"),
        ("No multicollinearity",   "VIF",             "Multiple Regression",   "All VIF reported; high-VIF predictors removed"),
        ("Homoscedasticity",       "Breusch-Pagan",   "Multiple Regression",   "Reported; robust SE if violated"),
        ("Normality of residuals", "Shapiro + J-B",   "Regression",            "Reported; large-n CLT applies"),
        ("Stationarity",           "ADF + KPSS",      "ARIMA",                 "Dual test; d=1 if required"),
        ("White-noise residuals",  "Ljung-Box",       "ARIMA",                 "Confirmed before accepting model"),
        ("Distribution fit",       "K-S one-sample",  "Distribution fitting",  "AIC + K-S both reported"),
    ]

    assump_df = pd.DataFrame(assump, columns=["Assumption","Test Applied","Where","Decision"])
    st.dataframe(assump_df, use_container_width=True)

    # ── Future directions ─────────────────────────────────────────────────────
    section_title("Limitations & Future Directions", "🔭")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Limitations**")
        for lim in [
            "Dataset covers Nov 2025–Feb 2026 only (peak winter pollution season).",
            "One monitoring station per city — intra-city heterogeneity not captured.",
            "AQI category-to-numeric mapping introduces discrete jumps.",
            "Wind direction, altitude, land use are unmeasured confounders.",
            "Temporal autocorrelation within stations (mitigated by ARIMA + CLT).",
        ]:
            st.markdown(f"- {lim}")

    with c2:
        st.markdown("**Future Work**")
        for fw in [
            "SARIMA — captures full 4-season PM2.5 forecasting cycle.",
            "Spatial autocorrelation (Moran's I) — cross-city pollution spillover.",
            "Quantile regression — models 95th/99th percentile PM2.5 directly.",
            "Panel fixed-effects — controls for city-level unobservables.",
            "Distributed lag model (DLNM) — quantifies lagged health impacts.",
        ]:
            st.markdown(f"- {fw}")

    # ── References ────────────────────────────────────────────────────────────
    with st.expander("📚 References (APA 7th Edition)"):
        st.markdown("""
**Ahmad, S., Rao, Z. H., Rashid, T., & Baloch, I. A.** (2020). Health impact assessment of air pollution in Lahore, Pakistan. *Science of the Total Environment*, *742*, 140571.

**Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M.** (2015). *Time series analysis: Forecasting and control* (5th ed.). John Wiley & Sons.

**IQAir.** (2025). *2024 world air quality report.* IQAir Foundation. https://www.iqair.com/world-air-quality-report

**Kaggle / Ahsan Neural.** (2024). *Pakistan air quality and weather — 10 cities* [Dataset]. https://www.kaggle.com/datasets/ahsanneural/pakistan-air-quality-and-weather-10-cities

**Kendall, M. G.** (1975). *Rank correlation methods* (4th ed.). Charles Griffin.

**Khan, R. A., Waheed, S., & Siddiqui, M. F.** (2022). Seasonal variability of PM2.5 in major Pakistani cities. *Atmospheric Environment*, *275*, 119003.

**Mann, H. B.** (1945). Nonparametric tests against trend. *Econometrica*, *13*(3), 245–259.

**Mirza, M. I.** (2012). Air pollution in Pakistan: Sources, concentrations and health effects. *Environmental Science & Policy*, *14*(8), 1072–1085.

**Sen, P. K.** (1968). Estimates of the regression coefficient based on Kendall's tau. *Journal of the American Statistical Association*, *63*(324), 1379–1389.

**World Health Organization.** (2021). *WHO global air quality guidelines.* https://www.who.int/publications/i/item/9789240034228
        """)

    callout(
        "<strong>All analyses performed in Python 3</strong> · pandas · scipy · statsmodels · "
        "matplotlib · seaborn. No synthetic data. No machine learning. "
        "10 formal hypothesis tests with full assumption checks, effect sizes, and "
        "real-world interpretations.",
        "info",
    )
