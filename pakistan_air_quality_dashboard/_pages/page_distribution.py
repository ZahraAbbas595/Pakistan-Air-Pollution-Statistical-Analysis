"""pages/page_distribution.py — PM2.5 distribution fitting"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import norm, lognorm, gamma, weibull_min, kstest

from utils.ui import (page_header, section_title, hypothesis_box,
                       result_banner, stat_grid, callout, no_data_message)
from utils.charts import pm25_histogram, theme


DISTRIBUTIONS = {
    "Normal":    norm,
    "Lognormal": lognorm,
    "Gamma":     gamma,
    "Weibull":   weibull_min,
}

DIST_COLORS = {
    "Normal":    "#f87171",
    "Lognormal": "#4ade80",
    "Gamma":     "#fbbf24",
    "Weibull":   "#c084fc",
}


def render():
    page_header(
        eyebrow="Section 3.3 · Statistical Methods",
        title="Probability Distribution Fitting",
        description=(
            "Identifies which statistical distribution best models PM2.5 exposure. "
            "Uses AIC model selection + Kolmogorov-Smirnov goodness-of-fit tests. "
            "Essential for health risk quantification and alert threshold calibration."
        ),
    )

    df: pd.DataFrame | None = st.session_state.get("df", None)
    if df is None:
        no_data_message()
        return

    if "pm25" not in df.columns:
        callout("PM2.5 column not found in dataset.", "warning")
        return

    pm25_data = df["pm25"].dropna().values
    pm25_data = pm25_data[pm25_data > 0]

    # ── Hypothesis framework ─────────────────────────────────────────────────
    section_title("Hypothesis Framework")
    hypothesis_box(
        h0="The observed PM2.5 data follows the candidate distribution",
        h1="The data does NOT follow the candidate distribution",
        alpha=0.05,
        test_name="Kolmogorov-Smirnov Goodness-of-Fit",
    )

    # ── Sample stats ─────────────────────────────────────────────────────────
    section_title("PM2.5 Sample Statistics")
    skew = pd.Series(pm25_data).skew()
    kurt = pd.Series(pm25_data).kurt()
    stat_grid([
        {"label": "N",       "value": f"{len(pm25_data):,}"},
        {"label": "Mean",    "value": f"{pm25_data.mean():.2f} µg/m³"},
        {"label": "Median",  "value": f"{np.median(pm25_data):.2f} µg/m³"},
        {"label": "SD",      "value": f"{pm25_data.std():.2f}"},
        {"label": "Skewness","value": f"{skew:.3f}"},
        {"label": "Kurtosis","value": f"{kurt:.3f}"},
    ])
    if skew > 1:
        callout(
            f"<strong>Right-skewed (skew = {skew:.2f}).</strong> "
            "Lognormal or Gamma expected to fit better than Normal.",
            "info",
        )

    # ── Fit distributions ────────────────────────────────────────────────────
    with st.spinner("Fitting distributions…"):
        fit_results = {}
        for name, dist_obj in DISTRIBUTIONS.items():
            try:
                params   = dist_obj.fit(pm25_data)
                ks_stat, ks_p = kstest(pm25_data, dist_obj.cdf, args=params)
                ll = np.sum(dist_obj.logpdf(pm25_data, *params))
                aic = 2 * len(params) - 2 * ll
                bic = len(params) * np.log(len(pm25_data)) - 2 * ll
                fit_results[name] = {
                    "params": params, "ks_stat": ks_stat,
                    "ks_p": ks_p, "log_lik": ll, "aic": aic, "bic": bic,
                }
            except Exception:
                pass

    best_dist = min(fit_results, key=lambda k: fit_results[k]["aic"])

    # ── Results table ────────────────────────────────────────────────────────
    section_title("Goodness-of-Fit Results")
    rows = []
    for name, res in fit_results.items():
        rows.append({
            "Distribution": name,
            "K-S Statistic": round(res["ks_stat"], 4),
            "K-S p-value":   round(res["ks_p"], 4),
            "Log-Likelihood":round(res["log_lik"], 2),
            "AIC":           round(res["aic"], 2),
            "BIC":           round(res["bic"], 2),
            "K-S Decision":  "Fail to Reject ✅" if res["ks_p"] > .05 else "Reject ❌",
            "Best?":         "⭐ BEST" if name == best_dist else "",
        })
    res_df = pd.DataFrame(rows).sort_values("AIC")
    st.dataframe(
        res_df.style.map(
            lambda v: "font-weight:700;color:#fbbf24" if v == "⭐ BEST" else "",
            subset=["Best?"]
        ).format({"AIC": "{:.2f}", "BIC": "{:.2f}"}),
        use_container_width=True,
    )

    callout(
        f"<strong>Best fit by AIC: {best_dist}</strong> — "
        + (
            "This confirms the right-skewed, positive nature of pollution data. "
            "Using a Normal distribution would <em>underestimate</em> the probability of "
            "extreme pollution events. Health risk thresholds must use the fitted tail percentiles."
            if best_dist in ["Lognormal", "Gamma"]
            else "The Normal distribution adequately models this variable."
        ),
        "success",
    )

    # ── Main histogram + fitted curves ───────────────────────────────────────
    section_title("Fitted Distributions vs Observed Data")
    x_fit = np.linspace(pm25_data.min(), np.percentile(pm25_data, 99), 500)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pm25_data,
        nbinsx=70,
        histnorm="probability density",
        marker_color="#38bdf8",
        opacity=0.55,
        name="Observed PM2.5",
        marker_line=dict(color="#161b22", width=0.3),
    ))
    for name, res in fit_results.items():
        pdf_vals = DISTRIBUTIONS[name].pdf(x_fit, *res["params"])
        lw = 3.5 if name == best_dist else 1.8
        fig.add_trace(go.Scatter(
            x=x_fit, y=pdf_vals, mode="lines",
            name=f"{name} (AIC={res['aic']:.0f})",
            line=dict(color=DIST_COLORS[name], width=lw,
                      dash="solid" if name == best_dist else "dash"),
        ))
    fig.update_layout(**theme(
        title=dict(text="Fitted Distributions vs Observed PM2.5", font=dict(size=13)),
        xaxis_title="PM2.5 (µg/m³)",
        yaxis_title="Probability Density",
        height=420,
        legend=dict(yanchor="top", y=.98, xanchor="right", x=.99),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ── Q-Q plots ────────────────────────────────────────────────────────────
    section_title("Q-Q Plots — Quantile Comparison")
    callout(
        "Points lying on the red diagonal indicate a good fit. "
        "Deviations at the tails show where the distribution under- or over-estimates extreme values.",
        "info",
    )

    cols = st.columns(2)
    for i, (name, res) in enumerate(fit_results.items()):
        n_pts = min(1500, len(pm25_data))
        sample = np.sort(np.random.choice(pm25_data, n_pts, replace=False))
        probs  = (np.arange(1, n_pts+1) - .5) / n_pts
        try:
            theoretical = DISTRIBUTIONS[name].ppf(probs, *res["params"])
            mn = min(theoretical.min(), sample.min())
            mx = max(theoretical.max(), sample.max())

            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=np.sort(theoretical), y=sample,
                mode="markers",
                marker=dict(color=DIST_COLORS[name], size=3, opacity=.5),
                name="Quantiles",
            ))
            fig_qq.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx],
                mode="lines",
                line=dict(color="#f87171", width=2),
                name="Perfect fit",
            ))
            fig_qq.update_layout(**theme(
                title=dict(text=f"Q-Q: {name}  AIC={res['aic']:.0f}", font=dict(size=12)),
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                height=320, showlegend=False,
            ))
            with cols[i % 2]:
                st.plotly_chart(fig_qq, use_container_width=True)
        except Exception:
            pass

    # ── Per-city distribution ─────────────────────────────────────────────────
    if "city" in df.columns:
        section_title("PM2.5 Distribution by City")
        fig3 = go.Figure()
        from utils.data_loader import CITY_PALETTE
        for i, city in enumerate(sorted(df["city"].unique())):
            data_c = df[df["city"] == city]["pm25"].dropna().values
            data_c = data_c[data_c > 0]
            if len(data_c) < 10:
                continue
            try:
                params_c = lognorm.fit(data_c)
                pdf_c = lognorm.pdf(x_fit, *params_c)
                fig3.add_trace(go.Scatter(
                    x=x_fit, y=pdf_c, mode="lines", name=city,
                    line=dict(color=CITY_PALETTE[i % len(CITY_PALETTE)], width=2),
                ))
            except Exception:
                pass
        fig3.update_layout(**theme(
            title=dict(text="City-Specific Lognormal Fits", font=dict(size=13)),
            xaxis_title="PM2.5 (µg/m³)",
            yaxis_title="Probability Density",
            height=400,
        ))
        st.plotly_chart(fig3, use_container_width=True)
        callout(
            "Rightward shifts indicate more polluted cities. "
            "Wider curves reflect greater daily variability (erratic industrial emissions).",
            "info",
        )
