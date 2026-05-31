"""pages/page_anova.py — One-Way & Two-Way ANOVA"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import shapiro, levene, f_oneway, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from utils.data_loader import AQI_CATEGORY_ORDER, CAT_COLORS, CITY_PALETTE, get_city_groups
from utils.ui import (page_header, section_title, hypothesis_box,
                       result_banner, callout, stat_grid, no_data_message)
from utils.charts import mean_ci_bar, aqi_stacked_bar, theme


def render():
    page_header(
        eyebrow="Section 3.1 & 3.2 · Statistical Methods",
        title="ANOVA Tests",
        description=(
            "One-Way ANOVA tests whether mean AQI differs across cities. "
            "Two-Way ANOVA (City × Season) tests interaction effects."
        ),
    )

    df: pd.DataFrame | None = st.session_state.get("df", None)
    if df is None:
        no_data_message()
        return

    tab1, tab2 = st.tabs(["🏙️  One-Way ANOVA — City", "🌤️  Two-Way ANOVA — City × Season"])

    # ═══════════════════════════════════════════════════════════════════════════
    # ONE-WAY ANOVA
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        section_title("Hypothesis Framework")
        hypothesis_box(
            h0="μ₁ = μ₂ = … = μ₁₀  —  All 10 cities have equal mean AQI",
            h1="At least one city's mean AQI differs significantly from the others",
            alpha=0.05,
            test_name="One-Way ANOVA",
        )

        if "city" not in df.columns or "aqi_numeric" not in df.columns:
            callout("Required columns (city, aqi_numeric) not found.", "warning")
            return

        cities, city_groups = get_city_groups(df, "aqi_numeric")

        # ── Assumption 1: Normality ──────────────────────────────────────────
        section_title("Assumption Checks")

        norm_rows = []
        for city, group in zip(cities, city_groups):
            if len(group) < 3:
                continue
            sample = group if len(group) <= 5000 else np.random.choice(group, 5000, replace=False)
            w, p = shapiro(sample)
            norm_rows.append({
                "City": city, "n": len(group), "W": round(w, 4),
                "p-value": round(p, 4),
                "Normal?": "✅ Yes" if p > .05 else "❌ No",
            })
        norm_df = pd.DataFrame(norm_rows)

        with st.expander("Shapiro-Wilk Normality Test (per city)", expanded=True):
            st.dataframe(norm_df.style.map(
                lambda v: "color:#f87171" if "No" in str(v) else "color:#4ade80",
                subset=["Normal?"]), use_container_width=True)
            n_fail = (norm_df["Normal?"].str.contains("No")).sum()
            if n_fail:
                callout(
                    f"<strong>Normality violated</strong> in {n_fail}/{len(norm_df)} cities. "
                    "ANOVA proceeds — CLT ensures asymptotic normality for large n. "
                    "Kruskal-Wallis (Section 6) provides distribution-free confirmation.",
                    "warning",
                )
            else:
                callout("All groups satisfy normality ✅. ANOVA assumption fully met.", "success")

        # ── Assumption 2: Levene's ───────────────────────────────────────────
        lev_stat, lev_p = levene(*city_groups)
        with st.expander("Levene's Test — Homogeneity of Variance"):
            stat_grid([
                {"label": "Levene Statistic", "value": f"{lev_stat:.4f}"},
                {"label": "p-value",           "value": f"{lev_p:.6f}"},
                {"label": "Equal Variances?",  "value": "✅ Yes" if lev_p > .05 else "❌ No"},
            ])
            if lev_p <= .05:
                callout(
                    "Unequal variances detected. Industrial cities exhibit higher AQI variance "
                    "(erratic emission events). Welch's F-test addresses this in Section 3.8.",
                    "warning",
                )

        # ── One-Way ANOVA ────────────────────────────────────────────────────
        section_title("One-Way ANOVA Result")
        f_stat, anova_p = f_oneway(*city_groups)
        grand_mean = df["aqi_numeric"].mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in city_groups)
        ss_total   = sum((df["aqi_numeric"].dropna() - grand_mean)**2)
        eta_sq     = ss_between / ss_total
        n_all      = sum(len(g) for g in city_groups)
        omega_sq   = max(0, (ss_between - (len(cities)-1)*(ss_total/n_all)) /
                         (ss_total + ss_total/n_all))

        result_banner(anova_p < .05, f_stat, anova_p,
                      f"η² = {eta_sq:.3f} ({'Large' if eta_sq>.14 else 'Medium' if eta_sq>.06 else 'Small'})")

        stat_grid([
            {"label": "F-Statistic",  "value": f"{f_stat:.4f}"},
            {"label": "p-value",      "value": f"{anova_p:.2e}"},
            {"label": "η² (eta-sq)",  "value": f"{eta_sq:.4f}"},
            {"label": "ω² (omega-sq)","value": f"{omega_sq:.4f}"},
            {"label": "Effect Size",  "value": "Large" if eta_sq > .14 else "Medium" if eta_sq > .06 else "Small"},
        ])

        if anova_p < .05:
            callout(
                "🔴 <strong>City membership alone explains a meaningful fraction of AQI variance.</strong> "
                "Industrial centres (Lahore, Faisalabad, Gujranwala) show the highest AQI "
                "reflecting brick kilns, textile factories, and dense traffic. "
                "A uniform national standard is statistically insufficient.",
                "danger",
            )

        # ── Mean CI bar chart ────────────────────────────────────────────────
        city_stats = sorted(
            [(c, np.mean(g), 1.96*np.std(g)/np.sqrt(len(g)))
             for c, g in zip(cities, city_groups)],
            key=lambda x: x[1], reverse=True,
        )
        c_names, c_means, c_ci = zip(*city_stats)
        st.plotly_chart(
            mean_ci_bar(c_names, c_means, c_ci,
                        "Mean AQI by City ± 95% CI", f_stat, anova_p, eta_sq),
            use_container_width=True,
        )

        # ── Tukey HSD ────────────────────────────────────────────────────────
        section_title("Tukey HSD Post-hoc")
        with st.expander("Pairwise comparisons — which city pairs differ?"):
            tukey_data = df[["city","aqi_numeric"]].dropna()
            tukey = pairwise_tukeyhsd(
                endog=tukey_data["aqi_numeric"],
                groups=tukey_data["city"], alpha=.05,
            )
            tukey_df = pd.DataFrame(
                data=tukey._results_table.data[1:],
                columns=tukey._results_table.data[0],
            )
            st.dataframe(tukey_df.style.map(
                lambda v: "color:#f87171" if v is True else ("color:#4ade80" if v is False else ""),
                subset=["reject"]), use_container_width=True)

        # ── Chi-square ───────────────────────────────────────────────────────
        if "aqi_category" in df.columns:
            section_title("Chi-Square Confirmation")
            contingency = pd.crosstab(df["city"], df["aqi_category"])
            chi2_val, chi2_p, dof, _ = chi2_contingency(contingency)
            cramers_v = np.sqrt(chi2_val / (len(df) * (min(contingency.shape)-1)))
            stat_grid([
                {"label": "Chi-Square",    "value": f"{chi2_val:.4f}"},
                {"label": "df",            "value": str(dof)},
                {"label": "p-value",       "value": f"{chi2_p:.2e}"},
                {"label": "Cramér's V",    "value": f"{cramers_v:.4f}"},
                {"label": "Association",   "value": "Strong" if cramers_v>.5 else "Moderate" if cramers_v>.3 else "Weak"},
            ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TWO-WAY ANOVA
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        section_title("Hypothesis Framework")
        hypothesis_box(
            h0="City × Season interaction does NOT exist — seasonal patterns are uniform across cities",
            h1="City and season interact — the degree of seasonal AQI variation differs by city",
            alpha=0.05,
            test_name="Two-Way ANOVA (City × Season)",
        )

        if "season" not in df.columns:
            callout("Season column not found in dataset.", "warning")
            return

        twoway_df = df[["city","season","aqi_numeric"]].dropna()

        if len(twoway_df) < 50:
            callout("Insufficient data for two-way ANOVA.", "warning")
            return

        formula = "aqi_numeric ~ C(city) + C(season) + C(city):C(season)"
        try:
            model_2way = ols(formula, data=twoway_df).fit()
            anova_table = anova_lm(model_2way, typ=2)
        except Exception as e:
            callout(f"ANOVA failed: {e}", "danger")
            return

        section_title("ANOVA Table")
        st.dataframe(anova_table.style.format({"sum_sq":"{:.2f}","df":"{:.0f}",
                                               "F":"{:.4f}","PR(>F)":"{:.4e}"}),
                     use_container_width=True)

        p_city     = anova_table.loc["C(city)", "PR(>F)"]
        p_season   = anova_table.loc["C(season)", "PR(>F)"]
        p_interact = anova_table.loc["C(city):C(season)", "PR(>F)"]

        col1, col2, col3 = st.columns(3)
        with col1:
            result_banner(p_city < .05, p_val=p_city)
            st.caption("Main effect: City")
        with col2:
            result_banner(p_season < .05, p_val=p_season)
            st.caption("Main effect: Season")
        with col3:
            result_banner(p_interact < .05, p_val=p_interact)
            st.caption("Interaction: City × Season")

        if p_interact < .05:
            callout(
                "✅ <strong>Significant interaction:</strong> The degree of seasonal variation in AQI "
                "differs by city. Industrial cities (Lahore, Faisalabad) show extreme winter spikes "
                "due to temperature inversions; coastal Karachi has less seasonal variation. "
                "<strong>Policy implication:</strong> Seasonal pollution controls must be city-tailored.",
                "success",
            )

        # ── Interaction plot ─────────────────────────────────────────────────
        section_title("Interaction Plot")
        callout(
            "<strong>Reading the chart:</strong> Non-parallel lines indicate an interaction. "
            "If lines cross, the city ranking changes across seasons.",
            "info",
        )

        interaction_df = twoway_df.groupby(["city","season"])["aqi_numeric"].mean().reset_index()
        season_order = [s for s in ["Winter","Spring","Monsoon","Autumn/Post-Monsoon"]
                        if s in df["season"].unique()]

        fig = go.Figure()
        for i, city in enumerate(sorted(df["city"].unique())):
            cd = interaction_df[interaction_df["city"] == city].set_index("season").reindex(season_order)
            fig.add_trace(go.Scatter(
                x=season_order, y=cd["aqi_numeric"].values,
                mode="lines+markers",
                name=city,
                line=dict(color=CITY_PALETTE[i % len(CITY_PALETTE)], width=2),
                marker=dict(size=8),
            ))
        fig.update_layout(**theme(
            title=dict(text="AQI by City and Season — Interaction Plot<br><sup>Non-parallel lines = interaction exists</sup>",
                       font=dict(size=13)),
            yaxis_title="Mean AQI (Numeric Proxy)", height=420,
        ))
        st.plotly_chart(fig, use_container_width=True)

        # ── Heatmap ──────────────────────────────────────────────────────────
        section_title("Mean AQI Heatmap — City × Season")
        pivot = interaction_df.pivot(index="city", columns="season", values="aqi_numeric")
        pivot = pivot.reindex(columns=[s for s in season_order if s in pivot.columns])

        fig2 = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0,"#4ade80"],[.5,"#fbbf24"],[1,"#f87171"]],
            text=np.round(pivot.values, 1),
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="Mean AQI"),
        ))
        fig2.update_layout(**theme(
            title=dict(text="Mean AQI Heatmap (brighter = more polluted)", font=dict(size=13)),
            height=360,
        ))
        st.plotly_chart(fig2, use_container_width=True)
