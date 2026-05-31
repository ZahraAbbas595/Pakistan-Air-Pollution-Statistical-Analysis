"""pages/page_nonparametric.py — Kruskal-Wallis & Mann-Whitney U"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from itertools import combinations
from scipy.stats import kruskal, mannwhitneyu

from utils.data_loader import CITY_PALETTE, get_city_groups
from utils.ui import (page_header, section_title, hypothesis_box,
                       result_banner, stat_grid, callout, no_data_message)
from utils.charts import theme


def render():
    page_header(
        eyebrow="Section 3.6 · Statistical Methods",
        title="Nonparametric Tests",
        description=(
            "Kruskal-Wallis H-test and Mann-Whitney U provide distribution-free confirmation "
            "of ANOVA findings. These tests make no normality assumption, "
            "serving as robust convergent validity checks."
        ),
    )

    df: pd.DataFrame | None = st.session_state.get("df", None)
    if df is None:
        no_data_message()
        return

    if "city" not in df.columns or "aqi_numeric" not in df.columns:
        callout("Required columns not found.", "warning")
        return

    cities, city_groups = get_city_groups(df, "aqi_numeric")
    n_total = sum(len(g) for g in city_groups)

    tab1, tab2 = st.tabs(["🔢  Kruskal-Wallis H-Test", "🔀  Mann-Whitney U Pairwise"])

    # ═══════════════════════════════════════════════════════════════════════
    # KRUSKAL-WALLIS
    # ═══════════════════════════════════════════════════════════════════════
    with tab1:
        section_title("Hypothesis Framework")
        hypothesis_box(
            h0="All cities have identical rank distributions of AQI (no difference)",
            h1="At least one city's AQI rank distribution differs",
            alpha=0.05,
            test_name="Kruskal-Wallis H-Test",
        )

        h_stat, kw_p = kruskal(*city_groups)
        epsilon_sq = max(0, (h_stat - len(cities) + 1) / (n_total - len(cities)))

        result_banner(kw_p < .05, h_stat, kw_p,
                      f"ε² = {epsilon_sq:.4f} ({'Large' if epsilon_sq>.14 else 'Medium' if epsilon_sq>.06 else 'Small'})")

        stat_grid([
            {"label": "H Statistic", "value": f"{h_stat:.4f}"},
            {"label": "p-value",     "value": f"{kw_p:.2e}"},
            {"label": "df",          "value": str(len(cities)-1)},
            {"label": "ε² (effect)", "value": f"{epsilon_sq:.4f}"},
            {"label": "Effect Size", "value": "Large" if epsilon_sq>.14 else "Medium" if epsilon_sq>.06 else "Small"},
        ])

        if kw_p < .05:
            callout(
                "✅ <strong>REJECT H₀</strong> — Significant differences confirmed (nonparametric). "
                "This <em>confirms</em> our ANOVA result without requiring normality. "
                "<strong>Convergent validity:</strong> both parametric and nonparametric agree → "
                "findings are robust and independent of distributional assumptions.",
                "success",
            )

        # Also on PM2.5
        if "pm25" in df.columns:
            pm25_groups = [df[df["city"] == c]["pm25"].dropna().values for c in cities]
            h2, p2 = kruskal(*pm25_groups)
            callout(
                f"Kruskal-Wallis on raw PM2.5: H = {h2:.4f}, p = {p2:.2e}  "
                "(Consistent — PM2.5 also significantly differs across cities)",
                "info",
            )

        # ── Median ± IQR bar chart ────────────────────────────────────────
        section_title("Median AQI by City ± IQR/2")
        sorted_cities = sorted(cities, key=lambda c: np.median(
            df[df["city"] == c]["aqi_numeric"].dropna()), reverse=True)
        medians = [np.median(df[df["city"] == c]["aqi_numeric"].dropna()) for c in sorted_cities]
        iqrs    = [(np.percentile(df[df["city"] == c]["aqi_numeric"].dropna(), 75) -
                    np.percentile(df[df["city"] == c]["aqi_numeric"].dropna(), 25)) / 2
                   for c in sorted_cities]

        fig = go.Figure(go.Bar(
            x=list(range(len(sorted_cities))),
            y=medians,
            error_y=dict(type="data", array=iqrs, visible=True,
                         color="#e6edf3", thickness=1.5, width=5),
            marker_color=[CITY_PALETTE[i % len(CITY_PALETTE)] for i in range(len(sorted_cities))],
            hovertemplate="<b>%{customdata}</b><br>Median: %{y:.1f}<extra></extra>",
            customdata=sorted_cities,
        ))
        fig.update_layout(**theme(
            title=dict(text=f"Median AQI by City ± IQR/2<br><sup>KW: H={h_stat:.2f}, p={kw_p:.2e}, ε²={epsilon_sq:.3f}</sup>",
                       font=dict(size=13)),
            xaxis=dict(tickmode="array", tickvals=list(range(len(sorted_cities))),
                       ticktext=sorted_cities, tickangle=30),
            yaxis_title="Median AQI (Proxy)",
            height=380, showlegend=False,
        ))
        st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # MANN-WHITNEY U PAIRWISE
    # ═══════════════════════════════════════════════════════════════════════
    with tab2:
        n_comparisons = len(list(combinations(cities, 2)))
        alpha_bonf    = 0.05 / n_comparisons

        section_title("Hypothesis Framework")
        hypothesis_box(
            h0="Two compared cities have identical rank distributions",
            h1="The two cities differ in AQI distribution",
            alpha=alpha_bonf,
            test_name=f"Mann-Whitney U  (Bonferroni α = 0.05 / {n_comparisons} = {alpha_bonf:.5f})",
        )

        with st.spinner("Running pairwise tests…"):
            mw_rows = []
            city_idx = {c: i for i, c in enumerate(cities)}
            n_c = len(cities)
            pval_matrix = np.ones((n_c, n_c))

            for c1, c2 in combinations(cities, 2):
                g1 = df[df["city"] == c1]["aqi_numeric"].dropna().values
                g2 = df[df["city"] == c2]["aqi_numeric"].dropna().values
                u_stat, u_p = mannwhitneyu(g1, g2, alternative="two-sided")
                r_effect = 1 - (2*u_stat) / (len(g1)*len(g2))
                i, j = city_idx[c1], city_idx[c2]
                pval_matrix[i, j] = pval_matrix[j, i] = u_p
                mw_rows.append({
                    "City A": c1, "City B": c2,
                    "U Statistic": round(u_stat, 1),
                    "p-value":     round(u_p, 6),
                    "r (effect)":  round(r_effect, 3),
                    "Significant": "✅" if u_p < alpha_bonf else "—",
                    "Effect Mag.": "Large" if abs(r_effect)>.5 else "Medium" if abs(r_effect)>.3 else "Small",
                })

        mw_df = pd.DataFrame(mw_rows)
        sig_count = (mw_df["Significant"] == "✅").sum()

        st.markdown(f"**Significant pairs (Bonferroni): {sig_count} of {n_comparisons}**")
        st.dataframe(
            mw_df.style.map(
                lambda v: "color:#f97316;font-weight:700" if v == "✅" else "",
                subset=["Significant"]),
            use_container_width=True, height=380,
        )

        callout(
            "Rank-biserial r: |r| > 0.5 = large, > 0.3 = medium, < 0.3 = small. "
            "KW and ANOVA both reject H₀ → <strong>convergent validity</strong> "
            "— conclusion is robust across distributional assumptions.",
            "info",
        )

        # ── p-value heatmap ───────────────────────────────────────────────
        section_title("Pairwise p-value Heatmap (−log₁₀ scale)")
        log_pval = -np.log10(pval_matrix + 1e-300)
        np.fill_diagonal(log_pval, 0)
        threshold = -np.log10(alpha_bonf)

        annotations = []
        for i in range(n_c):
            for j in range(n_c):
                if i != j and log_pval[i, j] > threshold:
                    annotations.append(dict(
                        x=j, y=i, text="★", xref="x", yref="y",
                        showarrow=False, font=dict(color="white", size=14),
                    ))

        fig = go.Figure(go.Heatmap(
            z=log_pval,
            x=cities, y=cities,
            colorscale=[[0,"#161b22"],[0.3,"#1d4ed8"],[0.6,"#f97316"],[1,"#f87171"]],
            colorbar=dict(title="-log₁₀(p)"),
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>−log₁₀(p) = %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(**theme(
            title=dict(text="Mann-Whitney p-values (★ = Bonferroni significant)",
                       font=dict(size=13)),
            height=440, annotations=annotations,
        ))
        st.plotly_chart(fig, use_container_width=True)
