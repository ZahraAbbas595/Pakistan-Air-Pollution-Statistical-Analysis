"""pages/page_regression.py — Multiple Linear Regression"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

from utils.ui import (page_header, section_title, hypothesis_box,
                       result_banner, stat_grid, callout, no_data_message)
from utils.charts import scatter_plot, theme


def render():
    page_header(
        eyebrow="Section 3.4 · Statistical Methods",
        title="Multiple Linear Regression",
        description=(
            "Quantifies which meteorological and pollutant variables most strongly predict PM2.5. "
            "Full diagnostic suite: VIF, Breusch-Pagan, Jarque-Bera, Q-Q, Scale-Location."
        ),
    )

    df: pd.DataFrame | None = st.session_state.get("df", None)
    if df is None:
        no_data_message()
        return

    potential = ["pm10","no2","so2","co","temperature","humidity","wind_speed"]
    predictors = [c for c in potential if c in df.columns]
    outcome_col = "pm25" if "pm25" in df.columns else ("aqi_numeric" if "aqi_numeric" in df.columns else None)

    if not outcome_col or len(predictors) < 2:
        callout("Insufficient columns for regression.", "warning")
        return

    # ── Controls ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Regression Settings**")
        sel_preds = st.multiselect("Include predictors", predictors, default=predictors)

    if len(sel_preds) < 2:
        callout("Select at least 2 predictors.", "warning")
        return

    reg_df = df[[outcome_col] + sel_preds].dropna()

    # ── Hypothesis ───────────────────────────────────────────────────────────
    section_title("Hypothesis Framework")
    hypothesis_box(
        h0="β₁ = β₂ = … = βₖ = 0  — None of the predictors explain PM2.5",
        h1="At least one βᵢ ≠ 0  — The model has explanatory power",
        alpha=0.05,
        test_name="Multiple OLS Regression (Overall F-test)",
    )

    # ── VIF ──────────────────────────────────────────────────────────────────
    section_title("Multicollinearity — Variance Inflation Factors")
    X_vif = sm.add_constant(reg_df[sel_preds])
    vif_vals = []
    for p in sel_preds:
        try:
            r2 = sm.OLS(reg_df[p], X_vif.drop(columns=p)).fit().rsquared
            vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        except Exception:
            vif = float("nan")
        vif_vals.append(vif)

    vif_df = pd.DataFrame({"Variable": sel_preds, "VIF": vif_vals}).sort_values("VIF", ascending=False)
    vif_colors = ["#f87171" if v > 10 else "#fbbf24" if v > 5 else "#4ade80" for v in vif_df["VIF"]]

    fig_vif = go.Figure(go.Bar(
        x=vif_df["VIF"], y=vif_df["Variable"], orientation="h",
        marker_color=vif_colors,
        text=[f"{v:.2f}" for v in vif_df["VIF"]], textposition="outside",
    ))
    fig_vif.add_vline(x=10, line_dash="dash", line_color="#f87171",
                      annotation_text="VIF = 10 (critical)")
    fig_vif.add_vline(x=5, line_dash="dot", line_color="#fbbf24",
                      annotation_text="VIF = 5 (moderate)")
    fig_vif.update_layout(**theme(
        title=dict(text="Variance Inflation Factors", font=dict(size=13)),
        xaxis_title="VIF", height=320,
    ))
    st.plotly_chart(fig_vif, use_container_width=True)

    keep_preds = vif_df[vif_df["VIF"] <= 10]["Variable"].tolist()
    if not keep_preds:
        keep_preds = sel_preds
    removed = [p for p in sel_preds if p not in keep_preds]
    if removed:
        callout(f"Predictors removed (VIF > 10): <strong>{', '.join(removed)}</strong>", "warning")
    else:
        callout("All predictors within acceptable VIF range ✅. Coefficients are interpretable as independent effects.", "success")

    # ── Fit model ────────────────────────────────────────────────────────────
    X = sm.add_constant(reg_df[keep_preds])
    y = reg_df[outcome_col]
    model = sm.OLS(y, X).fit()

    # ── Model summary ────────────────────────────────────────────────────────
    section_title("Model Summary")
    result_banner(model.f_pvalue < .05, model.fvalue, model.f_pvalue,
                  f"R² = {model.rsquared:.4f}  |  Adj R² = {model.rsquared_adj:.4f}")

    stat_grid([
        {"label": "R²",             "value": f"{model.rsquared:.4f}"},
        {"label": "Adj R²",         "value": f"{model.rsquared_adj:.4f}"},
        {"label": "F-Statistic",    "value": f"{model.fvalue:.4f}"},
        {"label": "p-value (F)",    "value": f"{model.f_pvalue:.2e}"},
        {"label": "AIC",            "value": f"{model.aic:.1f}"},
        {"label": "BIC",            "value": f"{model.bic:.1f}"},
        {"label": "N",              "value": f"{int(model.nobs):,}"},
    ])

    callout(
        f"The model explains <strong>{model.rsquared*100:.1f}%</strong> of PM2.5 variance. "
        f"The remaining {(1-model.rsquared)*100:.1f}% reflects unmeasured factors "
        "(wind direction, land use, traffic counts).",
        "info",
    )

    # ── Coefficient plot ──────────────────────────────────────────────────────
    section_title("Coefficients ± 95% CI")
    coefs = model.params.drop("const")
    ci_l  = model.conf_int().drop("const")[0]
    ci_u  = model.conf_int().drop("const")[1]
    pvals = model.pvalues.drop("const")

    colors = ["#f97316" if c > 0 else "#38bdf8" for c in coefs.values]
    sig    = ["⭐" if p < .05 else "" for p in pvals.values]

    fig_coef = go.Figure()
    fig_coef.add_trace(go.Bar(
        x=coefs.values, y=coefs.index, orientation="h",
        marker_color=colors, opacity=.8,
        error_x=dict(
            type="data",
            arrayminus=(coefs - ci_l).values,
            array=(ci_u - coefs).values,
            visible=True, color="#e6edf3", thickness=1.5, width=6,
        ),
        text=[f"β={v:.4f} {s}" for v, s in zip(coefs.values, sig)],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig_coef.add_vline(x=0, line_color="#8b949e", line_dash="dash")
    fig_coef.update_layout(**theme(
        title=dict(text="Regression Coefficients ± 95% CI<br><sup>Bars crossing 0 = not significant at α=0.05 (⭐ = significant)</sup>",
                   font=dict(size=12)),
        xaxis_title="Coefficient Value",
        height=360, showlegend=False,
    ))
    st.plotly_chart(fig_coef, use_container_width=True)

    # ── Significant predictors table ──────────────────────────────────────────
    with st.expander("Significant Predictors Detail", expanded=True):
        coef_table = pd.DataFrame({
            "β": model.params.drop("const").round(4),
            "SE": model.bse.drop("const").round(4),
            "t": model.tvalues.drop("const").round(4),
            "p-value": model.pvalues.drop("const").apply(lambda v: f"{v:.4e}"),
            "CI Lower": ci_l.round(4),
            "CI Upper": ci_u.round(4),
            "Significant": ["✅" if p < .05 else "—" for p in model.pvalues.drop("const")],
        })
        st.dataframe(coef_table.style.map(
            lambda v: "color:#f97316;font-weight:700" if v == "✅" else "",
            subset=["Significant"]), use_container_width=True)

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    section_title("Regression Diagnostics")
    residuals   = model.resid
    fitted_vals = model.fittedvalues
    std_resid   = residuals / residuals.std()

    diag_tab1, diag_tab2, diag_tab3, diag_tab4 = st.tabs([
        "① Residuals vs Fitted",
        "② Q-Q Plot",
        "③ Scale-Location",
        "④ Formal Tests",
    ])

    with diag_tab1:
        fig_r = scatter_plot(fitted_vals, residuals, "Fitted Values", "Residuals",
                             "Residuals vs Fitted — want: random horizontal scatter")
        fig_r.add_hline(y=0, line_color="#f87171", line_dash="dash")
        st.plotly_chart(fig_r, use_container_width=True)
        callout("A flat cloud of points around zero indicates homoscedasticity and linearity.", "info")

    with diag_tab2:
        from scipy.stats import norm as norm_dist
        probs = (np.arange(1, len(residuals)+1) - .5) / len(residuals)
        theoretical = norm_dist.ppf(probs)
        sorted_res  = np.sort(residuals.values)
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theoretical, y=sorted_res, mode="markers",
            marker=dict(color="#38bdf8", size=3, opacity=.5), name="Residuals",
        ))
        mn, mx = theoretical.min(), theoretical.max()
        fig_qq.add_trace(go.Scatter(
            x=[mn, mx], y=[mn * sorted_res.std() + sorted_res.mean(),
                            mx * sorted_res.std() + sorted_res.mean()],
            mode="lines", line=dict(color="#f87171", width=2), name="Reference",
        ))
        fig_qq.update_layout(**theme(
            title=dict(text="Normal Q-Q Plot of Residuals", font=dict(size=13)),
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=380, showlegend=False,
        ))
        st.plotly_chart(fig_qq, use_container_width=True)

    with diag_tab3:
        sqrt_abs = np.sqrt(np.abs(std_resid))
        fig_sl = scatter_plot(fitted_vals, sqrt_abs, "Fitted Values",
                              "√|Standardized Residuals|",
                              "Scale-Location — flat line = homoscedastic")
        st.plotly_chart(fig_sl, use_container_width=True)

    with diag_tab4:
        section_title("Formal Assumption Tests")
        rows2 = []

        # Shapiro-Wilk
        sw_sample = residuals.sample(min(5000, len(residuals)), random_state=42)
        sw_stat, sw_p = shapiro(sw_sample)
        rows2.append({"Test": "Shapiro-Wilk (normality)", "Statistic": round(sw_stat, 4),
                      "p-value": f"{sw_p:.4f}",
                      "Decision": "FAIL TO REJECT H₀ ✅" if sw_p > .05 else "REJECT H₀ ❌",
                      "Conclusion": "Residuals normal" if sw_p > .05 else "Non-normal (CLT covers large n)"})

        # Breusch-Pagan
        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X)
            rows2.append({"Test": "Breusch-Pagan (homoscedasticity)",
                          "Statistic": round(float(bp_stat), 4), "p-value": f"{bp_p:.4f}",
                          "Decision": "FAIL TO REJECT H₀ ✅" if bp_p > .05 else "REJECT H₀ ❌",
                          "Conclusion": "Homoscedastic" if bp_p > .05 else "Heteroscedastic → use HC3 SE"})
        except Exception:
            pass

        # Jarque-Bera
        try:
            jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(residuals)
            rows2.append({"Test": "Jarque-Bera (normality joint)", "Statistic": round(float(jb_stat), 4),
                          "p-value": f"{jb_p:.4f}",
                          "Decision": "FAIL TO REJECT H₀ ✅" if jb_p > .05 else "REJECT H₀ ❌",
                          "Conclusion": f"Skew={jb_skew:.3f}, Excess Kurt={jb_kurt:.3f}"})
        except Exception:
            pass

        if rows2:
            st.dataframe(pd.DataFrame(rows2), use_container_width=True)
