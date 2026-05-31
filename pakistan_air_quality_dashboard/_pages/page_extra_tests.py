"""pages/page_extra_tests.py — 5 Extra Hypothesis Tests (Section 3.8)"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import ttest_ind, chi2_contingency, ks_2samp, shapiro, f_oneway, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from utils.data_loader import AQI_CATEGORY_ORDER, CAT_COLORS, CITY_PALETTE
from utils.ui import (page_header, section_title, hypothesis_box,
                       result_banner, stat_grid, callout, no_data_message)
from utils.charts import theme, hex_to_rgba as _hex_rgba


def render():
    page_header(
        eyebrow="Section 3.8 · Additional Hypothesis Tests",
        title="Five Extra Hypothesis Tests",
        description=(
            "Welch's T-Test · Chi-Square Independence · Mann-Kendall Trend · "
            "Two-Sample KS Test · One-Way ANOVA (Diurnal). "
            "Each test is framed with H₀/H₁, assumptions, and real-world policy conclusions."
        ),
    )

    df: pd.DataFrame | None = st.session_state.get("df", None)
    if df is None:
        no_data_message()
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "T-Test",
        "Chi-Square",
        "Mann-Kendall",
        "KS Seasonal",
        "Diurnal ANOVA",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: WELCH'S T-TEST
    # ═══════════════════════════════════════════════════════════════════════
    with tab1:
        section_title("Test 3.8.1 — Welch's Independent Samples T-Test")
        hypothesis_box(
            h0="Mean PM2.5 is equal in the most and least polluted city",
            h1="Mean PM2.5 differs significantly between the two cities",
            alpha=0.05,
            test_name="Welch's T-Test",
        )

        if "city" not in df.columns or "pm25" not in df.columns:
            callout("Required columns not found.", "warning")
        else:
            city_means = df.groupby("city")["pm25"].mean().sort_values()
            best_city  = city_means.index[0]
            worst_city = city_means.index[-1]

            g_best  = df[df["city"] == best_city]["pm25"].dropna()
            g_worst = df[df["city"] == worst_city]["pm25"].dropna()

            # Normality check
            sw_b = shapiro(g_best.sample(min(5000, len(g_best)), random_state=42))
            sw_w = shapiro(g_worst.sample(min(5000, len(g_worst)), random_state=42))

            with st.expander("Normality assumption (Shapiro-Wilk)"):
                stat_grid([
                    {"label": f"{best_city} W",  "value": f"{sw_b[0]:.4f}"},
                    {"label": f"{best_city} p",  "value": f"{sw_b[1]:.4f}"},
                    {"label": f"{worst_city} W", "value": f"{sw_w[0]:.4f}"},
                    {"label": f"{worst_city} p", "value": f"{sw_w[1]:.4f}"},
                ])
                callout(
                    "Welch's T-Test is robust to non-normality for large n (CLT applies).",
                    "info",
                )

            t_stat, p_val = ttest_ind(g_worst, g_best, equal_var=False)
            pooled_sd = np.sqrt((g_worst.std()**2 + g_best.std()**2) / 2)
            cohens_d  = (g_worst.mean() - g_best.mean()) / pooled_sd
            diff = g_worst.mean() - g_best.mean()

            result_banner(p_val < .05, t_stat, p_val,
                          f"Cohen's d = {cohens_d:.4f} ({'Large' if abs(cohens_d)>.8 else 'Medium' if abs(cohens_d)>.5 else 'Small'})")

            stat_grid([
                {"label": f"{worst_city} mean", "value": f"{g_worst.mean():.2f} µg/m³"},
                {"label": f"{best_city} mean",  "value": f"{g_best.mean():.2f} µg/m³"},
                {"label": "Difference",         "value": f"{diff:.2f} µg/m³"},
                {"label": "t-statistic",        "value": f"{t_stat:.4f}"},
                {"label": "p-value",            "value": f"{p_val:.2e}"},
                {"label": "Cohen's d",          "value": f"{cohens_d:.4f}"},
            ])

            if p_val < .05:
                callout(
                    f"🔴 PM2.5 in <strong>{worst_city}</strong> ({g_worst.mean():.1f} µg/m³) is "
                    f"significantly higher than in <strong>{best_city}</strong> ({g_best.mean():.1f} µg/m³). "
                    f"The difference ({diff:.0f} µg/m³) exceeds the WHO annual guideline (5 µg/m³) "
                    f"by {diff/5:.0f}×. Cohen's d = {cohens_d:.2f} confirms a <em>large</em> practical effect.",
                    "danger",
                )

            # Violin plot
            fig = go.Figure()
            for city, data, color in [(worst_city, g_worst, "#f87171"),
                                       (best_city,  g_best,  "#4ade80")]:
                fig.add_trace(go.Violin(
                    y=data, name=city,
                    box_visible=True, meanline_visible=True,
                    line_color=color, fillcolor=_hex_rgba(color, 0.2),
                ))
            fig.update_layout(**theme(
                title=dict(text=f"PM2.5 Distribution: {worst_city} vs {best_city}",
                           font=dict(size=13)),
                yaxis_title="PM2.5 (µg/m³)",
                height=380, showlegend=False,
            ))
            st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: CHI-SQUARE INDEPENDENCE
    # ═══════════════════════════════════════════════════════════════════════
    with tab2:
        section_title("Test 3.8.2 — Chi-Square Test of Independence")
        hypothesis_box(
            h0="AQI category distribution is INDEPENDENT of weekend/weekday status",
            h1="AQI category distribution DEPENDS on weekend vs weekday",
            alpha=0.05,
            test_name="Chi-Square Test of Independence",
        )

        if "is_weekend" not in df.columns or "aqi_category" not in df.columns:
            callout("is_weekend or aqi_category column not found.", "warning")
        else:
            contingency = pd.crosstab(
                df["is_weekend"].map({0: "Weekday", 1: "Weekend"}),
                df["aqi_category"],
            )
            contingency = contingency.reindex(
                columns=[c for c in AQI_CATEGORY_ORDER if c in contingency.columns]
            )
            chi2_val, chi2_p, dof, expected = chi2_contingency(contingency)
            n_total   = len(df)
            cramers_v = np.sqrt(chi2_val / (n_total * (min(contingency.shape)-1)))

            result_banner(chi2_p < .05, chi2_val, chi2_p,
                          f"Cramér's V = {cramers_v:.4f}")

            stat_grid([
                {"label": "Chi-Square",  "value": f"{chi2_val:.4f}"},
                {"label": "df",          "value": str(dof)},
                {"label": "p-value",     "value": f"{chi2_p:.4f}"},
                {"label": "Cramér's V",  "value": f"{cramers_v:.4f}"},
                {"label": "Association", "value": "Strong" if cramers_v>.5 else "Medium" if cramers_v>.3 else "Weak"},
            ])

            if chi2_p < .05:
                callout(
                    "✅ AQI category distribution differs significantly between weekdays and weekends. "
                    "Traffic and industrial activity reductions on weekends produce measurable AQI shifts. "
                    "<strong>Policy:</strong> Weekday odd-even traffic restrictions are statistically justified.",
                    "success",
                )
            else:
                callout(
                    "AQI distribution is similar on weekdays and weekends. "
                    "Other emission sources (domestic heating, agriculture burning) may dominate "
                    "over traffic/industry in this winter dataset period.",
                    "info",
                )

            with st.expander("Contingency Table"):
                st.dataframe(contingency, use_container_width=True)

            # Stacked bar
            ct_pct = contingency.div(contingency.sum(axis=1), axis=0).mul(100)
            fig = go.Figure()
            for cat in ct_pct.columns:
                fig.add_trace(go.Bar(
                    name=cat, x=ct_pct.index, y=ct_pct[cat],
                    marker_color=CAT_COLORS.get(cat, "#888"),
                ))
            fig.update_layout(**theme(
                barmode="stack",
                title=dict(text="AQI Category % — Weekday vs Weekend", font=dict(size=13)),
                yaxis_title="% of Observations", yaxis_range=[0, 101],
                height=360,
            ))
            st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 3: MANN-KENDALL TREND
    # ═══════════════════════════════════════════════════════════════════════
    with tab3:
        section_title("Test 3.8.3 — Mann-Kendall Trend Test + Sen's Slope")
        hypothesis_box(
            h0="No monotonic trend in daily mean PM2.5 over the study period",
            h1="A significant monotonic (upward or downward) trend exists in PM2.5",
            alpha=0.05,
            test_name="Mann-Kendall Non-Parametric Trend Test",
        )

        if "date" not in df.columns or "pm25" not in df.columns:
            callout("Required columns not found.", "warning")
        else:
            daily_ts = df.set_index("date")["pm25"].resample("D").mean().dropna()
            x = daily_ts.values
            n = len(x)

            with st.spinner("Computing Mann-Kendall S statistic…"):
                s = 0
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        diff = x[j] - x[i]
                        if diff > 0:   s += 1
                        elif diff < 0: s -= 1

                var_s = (n * (n-1) * (2*n+5)) / 18
                from scipy.stats import norm as norm_dist
                if s > 0:   z_mk = (s - 1) / np.sqrt(var_s)
                elif s < 0: z_mk = (s + 1) / np.sqrt(var_s)
                else:       z_mk = 0.0
                p_mk = 2 * (1 - norm_dist.cdf(abs(z_mk)))

                slopes = [(x[j]-x[i])/(j-i) for i in range(n-1) for j in range(i+1, n)]
                sens_slope = float(np.median(slopes))
                trend_total = sens_slope * n

            result_banner(p_mk < .05, z_mk, p_mk)

            stat_grid([
                {"label": "MK S",        "value": f"{s:+.0f}"},
                {"label": "Z-statistic", "value": f"{z_mk:.4f}"},
                {"label": "p-value",     "value": f"{p_mk:.4f}"},
                {"label": "Sen's slope", "value": f"{sens_slope:+.4f} µg/m³/day"},
                {"label": "Total trend", "value": f"{trend_total:+.1f} µg/m³ over {n} days"},
            ])

            if p_mk < .05:
                direction = "INCREASING ↑" if s > 0 else "DECREASING ↓"
                kind = "danger" if s > 0 else "success"
                callout(
                    f"<strong>{direction} trend detected.</strong> "
                    f"Sen's slope = {sens_slope:+.4f} µg/m³/day. "
                    f"Over {n} days: total {'rise' if s>0 else 'fall'} of {abs(trend_total):.1f} µg/m³. "
                    "Consistent with winter inversion intensification as temperatures drop. "
                    "<strong>Policy:</strong> Escalating (not static) emergency protocols are required "
                    "as winter deepens from November to February.",
                    kind,
                )
            else:
                callout("No statistically significant monotonic trend detected.", "info")

            # Chart
            t_idx = np.arange(n)
            intercept = float(np.median(x - sens_slope * t_idx))
            trend_line = sens_slope * t_idx + intercept

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_ts.index, y=daily_ts.values,
                mode="lines", name="Daily PM2.5",
                line=dict(color="#38bdf8", width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=daily_ts.index, y=trend_line,
                mode="lines", name=f"Sen's Slope = {sens_slope:+.4f}",
                line=dict(color="#f97316", width=2.5, dash="dash"),
            ))
            fig.add_hline(y=15, line_dash="dot", line_color="#fbbf24",
                          annotation_text="WHO guideline")
            fig.update_layout(**theme(
                title=dict(
                    text=f"Mann-Kendall Trend — Z={z_mk:.2f}, p={p_mk:.4f}, "
                         f"Sen's slope={sens_slope:+.3f} µg/m³/day",
                    font=dict(size=12),
                ),
                yaxis_title="PM2.5 (µg/m³)", height=380,
            ))
            st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 4: TWO-SAMPLE KS TEST
    # ═══════════════════════════════════════════════════════════════════════
    with tab4:
        section_title("Test 3.8.4 — Two-Sample Kolmogorov-Smirnov Test")
        hypothesis_box(
            h0="PM2.5 in the two seasons are drawn from the same distribution",
            h1="The two seasons have different PM2.5 distributions",
            alpha=0.05,
            test_name="Two-Sample KS Test (Seasonal)",
        )

        if "season" not in df.columns or "pm25" not in df.columns:
            callout("Required columns not found.", "warning")
        else:
            avail = sorted(df["season"].unique())
            if len(avail) < 2:
                callout(f"Only one season found ({avail}). KS requires ≥ 2 seasons.", "warning")
            else:
                col1, col2 = st.columns(2)
                s1 = col1.selectbox("Season 1", avail, index=0)
                s2 = col2.selectbox("Season 2", avail, index=min(1, len(avail)-1))

                s1_data = df[df["season"] == s1]["pm25"].dropna().values
                s2_data = df[df["season"] == s2]["pm25"].dropna().values

                ks_stat, ks_p = ks_2samp(s1_data, s2_data)

                # Overlap coefficient
                x_max = np.percentile(np.concatenate([s1_data, s2_data]), 99)
                bins = np.linspace(0, x_max, 200)
                h1_, _ = np.histogram(s1_data, bins=bins, density=True)
                h2_, _ = np.histogram(s2_data, bins=bins, density=True)
                overlap = float(np.sum(np.minimum(h1_, h2_)) * (bins[1] - bins[0]))

                result_banner(ks_p < .05, ks_stat, ks_p)

                stat_grid([
                    {"label": f"{s1} mean",    "value": f"{s1_data.mean():.1f} µg/m³"},
                    {"label": f"{s2} mean",    "value": f"{s2_data.mean():.1f} µg/m³"},
                    {"label": "KS Statistic",  "value": f"{ks_stat:.4f}"},
                    {"label": "p-value",       "value": f"{ks_p:.2e}"},
                    {"label": "Dist. Overlap", "value": f"{overlap:.3f}"},
                    {"label": "Ratio",         "value": f"{max(s1_data.mean(),s2_data.mean())/min(s1_data.mean(),s2_data.mean()):.2f}×"},
                ])

                if ks_p < .05:
                    callout(
                        f"✅ <strong>Distributions are significantly different</strong> "
                        f"(KS = {ks_stat:.4f}, p = {ks_p:.2e}). "
                        f"Overlap = {overlap:.3f} — only {overlap*100:.0f}% of probability mass "
                        "overlaps between seasons. "
                        "<strong>Policy implication:</strong> Season-specific distributional parameters "
                        "(not pooled statistics) must be used for health risk and permit thresholds.",
                        "success",
                    )

                # ECDF comparison
                fig = go.Figure()
                for data, name, color in [(s1_data, s1, "#38bdf8"), (s2_data, s2, "#f97316")]:
                    sorted_d = np.sort(data)
                    ecdf = np.arange(1, len(sorted_d)+1) / len(sorted_d)
                    fig.add_trace(go.Scatter(
                        x=sorted_d, y=ecdf, mode="lines", name=name,
                        line=dict(color=color, width=2.5),
                    ))

                # Mark KS gap
                all_x = np.sort(np.concatenate([s1_data, s2_data]))
                e1 = np.searchsorted(np.sort(s1_data), all_x, side="right") / len(s1_data)
                e2 = np.searchsorted(np.sort(s2_data), all_x, side="right") / len(s2_data)
                ks_idx = int(np.argmax(np.abs(e1 - e2)))
                fig.add_shape(
                    type="line",
                    x0=all_x[ks_idx], x1=all_x[ks_idx],
                    y0=min(e1[ks_idx], e2[ks_idx]),
                    y1=max(e1[ks_idx], e2[ks_idx]),
                    line=dict(color="#f87171", width=3, dash="dot"),
                )
                fig.update_layout(**theme(
                    title=dict(text=f"ECDF Comparison: {s1} vs {s2}<br>"
                               f"<sup>KS = max gap between curves = {ks_stat:.4f}</sup>",
                               font=dict(size=13)),
                    xaxis_title="PM2.5 (µg/m³)",
                    yaxis_title="Cumulative Probability",
                    xaxis_range=[0, x_max],
                    height=400,
                ))
                st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 5: DIURNAL ANOVA
    # ═══════════════════════════════════════════════════════════════════════
    with tab5:
        section_title("Test 3.8.5 — One-Way ANOVA: Time-of-Day (Diurnal Cycle)")
        hypothesis_box(
            h0="Mean PM2.5 is equal across all four time-of-day periods",
            h1="At least one time-of-day period has significantly different mean PM2.5",
            alpha=0.05,
            test_name="One-Way ANOVA — Diurnal Analysis",
        )

        if "hour" not in df.columns or "pm25" not in df.columns:
            callout("hour or pm25 column not found.", "warning")
        else:
            def time_of_day(h):
                if   6 <= h <= 11:  return "Morning (06–11)"
                elif 12 <= h <= 17: return "Afternoon (12–17)"
                elif 18 <= h <= 23: return "Evening (18–23)"
                else:               return "Night (00–05)"

            df2 = df.copy()
            df2["tod"] = df2["hour"].map(time_of_day)
            tod_order = ["Night (00–05)","Morning (06–11)","Afternoon (12–17)","Evening (18–23)"]
            tod_groups = [df2[df2["tod"] == g]["pm25"].dropna().values for g in tod_order]
            tod_groups = [g for g in tod_groups if len(g) > 0]
            tod_labels = [t for t, g in zip(tod_order, tod_groups) if len(g) > 0]

            # Levene
            lev_s, lev_p = levene(*tod_groups)
            with st.expander("Assumption: Levene's Test (equal variances)"):
                stat_grid([
                    {"label": "Levene Stat", "value": f"{lev_s:.4f}"},
                    {"label": "p-value",     "value": f"{lev_p:.4f}"},
                    {"label": "Result",      "value": "Equal variances ✅" if lev_p>.05 else "Unequal ❌ (noted)"},
                ])

            f_tod, p_tod = f_oneway(*tod_groups)
            grand_mean   = np.concatenate(tod_groups).mean()
            ss_between   = sum(len(g)*(np.mean(g)-grand_mean)**2 for g in tod_groups)
            ss_total     = sum((v-grand_mean)**2 for g in tod_groups for v in g)
            eta_sq_tod   = ss_between / ss_total if ss_total > 0 else 0

            result_banner(p_tod < .05, f_tod, p_tod,
                          f"η² = {eta_sq_tod:.4f}")

            means_tod = {label: np.mean(g) for label, g in zip(tod_labels, tod_groups)}
            peak = max(means_tod, key=means_tod.get)
            low  = min(means_tod, key=means_tod.get)

            stat_grid([
                {"label": "F-Statistic",  "value": f"{f_tod:.4f}"},
                {"label": "p-value",      "value": f"{p_tod:.2e}"},
                {"label": "η²",           "value": f"{eta_sq_tod:.4f}"},
                {"label": "Peak period",  "value": peak},
                {"label": "Peak mean",    "value": f"{means_tod[peak]:.1f} µg/m³"},
                {"label": "Lowest period","value": low},
            ])

            if p_tod < .05:
                callout(
                    f"✅ PM2.5 differs significantly across time of day "
                    f"(F = {f_tod:.2f}, p = {p_tod:.2e}, η² = {eta_sq_tod:.3f}). "
                    f"Peak: <strong>{peak}</strong> ({means_tod[peak]:.1f} µg/m³). "
                    "Nighttime inversions trap pollutants near ground level; morning rush hours "
                    "generate traffic emissions. "
                    "<strong>Policy:</strong> School start time adjustments and timed traffic "
                    "restrictions have maximum health impact.",
                    "success",
                )

            # Tukey HSD
            if p_tod < .05:
                with st.expander("Tukey HSD Post-hoc"):
                    tod_col = df2[["tod","pm25"]].dropna()
                    tukey_tod = pairwise_tukeyhsd(tod_col["pm25"], tod_col["tod"], alpha=.05)
                    tukey_df = pd.DataFrame(
                        data=tukey_tod._results_table.data[1:],
                        columns=tukey_tod._results_table.data[0],
                    )
                    st.dataframe(tukey_df, use_container_width=True)

            # Box plot per TOD
            fig = go.Figure()
            for i, (label, group) in enumerate(zip(tod_labels, tod_groups)):
                fig.add_trace(go.Box(
                    y=group, name=label,
                    marker_color=CITY_PALETTE[i % len(CITY_PALETTE)],
                    boxmean=True,
                ))
            fig.update_layout(**theme(
                title=dict(
                    text=f"PM2.5 by Time of Day<br><sup>F={f_tod:.2f}, p={p_tod:.2e}, η²={eta_sq_tod:.3f}</sup>",
                    font=dict(size=13),
                ),
                yaxis_title="PM2.5 (µg/m³)",
                height=380, showlegend=False,
            ))
            st.plotly_chart(fig, use_container_width=True)

            # 24-hour diurnal curve
            section_title("Full 24-Hour Diurnal PM2.5 Cycle")
            hourly_mean = df.groupby("hour")["pm25"].mean()
            hourly_std  = df.groupby("hour")["pm25"].std()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=hourly_mean.index, y=hourly_mean.values,
                mode="lines+markers",
                line=dict(color="#f97316", width=2.5),
                marker=dict(size=6),
                name="Hourly mean PM2.5",
            ))
            fig2.add_trace(go.Scatter(
                x=list(hourly_mean.index) + list(hourly_mean.index[::-1]),
                y=list(hourly_mean + hourly_std) + list((hourly_mean - hourly_std)[::-1]),
                fill="toself",
                fillcolor="rgba(249,115,22,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="±1 SD",
            ))
            for vx, label, color in [(6,"Dawn","#38bdf8"),(12,"Noon","#fbbf24"),(18,"Dusk","#8b949e")]:
                fig2.add_vline(x=vx, line_dash="dash", line_color=color,
                               annotation_text=label, annotation_font_size=9)
            fig2.update_layout(**theme(
                title=dict(text="24-Hour Diurnal PM2.5 Cycle (mean ± 1 SD)",
                           font=dict(size=13)),
                xaxis_title="Hour of Day (0–23)",
                yaxis_title="Mean PM2.5 (µg/m³)",
                xaxis=dict(tickmode="array", tickvals=list(range(0, 24, 2))),
                height=380,
            ))
            st.plotly_chart(fig2, use_container_width=True)
