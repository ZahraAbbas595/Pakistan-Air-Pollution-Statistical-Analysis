"""pages/page_timeseries.py — STL Decomposition + ARIMA Forecasting"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

from utils.ui import (page_header, section_title, hypothesis_box,
                       result_banner, stat_grid, callout, no_data_message)
from utils.charts import ts_line, theme, hex_to_rgba as _hex_rgba


def render():
    page_header(
        eyebrow="Section 3.5 · Statistical Methods",
        title="Time Series Analysis & ARIMA",
        description=(
            "STL seasonal decomposition separates trend, seasonal, and residual components. "
            "ADF + KPSS stationarity tests guide ARIMA order selection. "
            "Ljung-Box test confirms white-noise residuals."
        ),
    )

    df: pd.DataFrame | None = st.session_state.get("df", None)
    if df is None:
        no_data_message()
        return

    if "date" not in df.columns:
        callout("Date column not found.", "warning")
        return

    ts_var = "pm25" if "pm25" in df.columns else ("aqi_numeric" if "aqi_numeric" in df.columns else None)
    if ts_var is None:
        callout("Neither pm25 nor aqi_numeric found.", "warning")
        return

    ts_label = "PM2.5 (µg/m³)" if ts_var == "pm25" else "AQI Proxy"

    # ── City selector ─────────────────────────────────────────────────────────
    cities = sorted(df["city"].unique()) if "city" in df.columns else []
    if cities:
        default_city = df.groupby("city")[ts_var].count().idxmax()
        sel_city = st.selectbox("Select city for analysis", cities,
                                index=cities.index(default_city))
        city_df = df[df["city"] == sel_city].copy()
    else:
        city_df = df.copy()
        sel_city = "All"

    ts_raw = city_df.set_index("date")[ts_var].resample("D").mean().dropna()
    freq_label = "Daily"
    if len(ts_raw) < 20:
        ts_raw = city_df.set_index("date")[ts_var].resample("ME").mean().dropna()
        freq_label = "Monthly"

    st.markdown(f"**City:** {sel_city} · **Frequency:** {freq_label} · **n = {len(ts_raw):,}**")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 STL Decomposition",
        "🔎 Stationarity Tests",
        "🤖 ARIMA Model",
        "🔮 Forecast",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # STL DECOMPOSITION
    # ═══════════════════════════════════════════════════════════════════════
    with tab1:
        section_title("STL Seasonal Decomposition")
        callout(
            "STL (Seasonal-Trend decomposition using LOESS) separates observed data into "
            "<strong>Trend</strong> (long-run trajectory), <strong>Seasonal</strong> (recurring cycles), "
            "and <strong>Residual</strong> (random noise).",
            "info",
        )

        if len(ts_raw) < 20:
            callout("Insufficient data for STL (need ≥ 20 observations).", "warning")
        else:
            period = 7 if freq_label == "Daily" else 12
            try:
                with st.spinner("Running STL decomposition…"):
                    stl = STL(ts_raw, period=period, robust=True)
                    result = stl.fit()

                components = {
                    "Observed":  ts_raw.values,
                    "Trend":     result.trend,
                    "Seasonal":  result.seasonal,
                    "Residual":  result.resid,
                }
                colors = ["#38bdf8", "#f97316", "#4ade80", "#c084fc"]
                titles = [
                    "Observed (raw data)",
                    "Trend component",
                    f"Seasonal component (period={period})",
                    "Residual (unexplained)",
                ]

                for (name, vals), color, t in zip(components.items(), colors, titles):
                    fig = go.Figure(go.Scatter(
                        x=ts_raw.index, y=vals, mode="lines",
                        line=dict(color=color, width=1.8), name=name,
                    ))
                    if name == "Trend":
                        fig.add_traces(go.Scatter(
                            x=ts_raw.index, y=vals,
                            mode="none", fill="tozeroy",
                            fillcolor=_hex_rgba(color, 0.13),
                        ))
                    if name in ("Observed", "Trend"):
                        fig.add_hline(y=15, line_dash="dot", line_color="#fbbf24",
                                      annotation_text="WHO guideline", annotation_font_size=9)
                    fig.update_layout(**theme(
                        title=dict(text=t, font=dict(size=12)),
                        height=200, margin=dict(t=40, b=30, l=60, r=20),
                        yaxis_title=ts_label,
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                trend_dir = "↑ upward" if result.trend[-1] > result.trend[0] else "↓ downward"
                callout(
                    f"<strong>Trend:</strong> {trend_dir} movement in {ts_label} over the study period. "
                    f"<strong>Seasonal:</strong> Reveals {'weekly' if freq_label=='Daily' else 'annual'} cycles — "
                    "winter peaks confirm temperature inversion effects. "
                    "<strong>Residual:</strong> Should be white noise if decomposition is adequate.",
                    "success",
                )
            except Exception as e:
                callout(f"STL error: {e}", "danger")

    # ═══════════════════════════════════════════════════════════════════════
    # STATIONARITY TESTS
    # ═══════════════════════════════════════════════════════════════════════
    with tab2:
        section_title("Augmented Dickey-Fuller Test")
        hypothesis_box(
            h0="Series has a unit root (non-stationary)",
            h1="Series is stationary (no unit root)",
            alpha=0.05,
            test_name="ADF Test",
        )

        try:
            adf = adfuller(ts_raw, autolag="AIC")
            adf_stat = adf_p = 0.0
            if isinstance(adf, tuple) and len(adf) >= 2:
                adf_stat, adf_p = float(adf[0]), float(adf[1])
            adf_stationary = adf_p < .05
            result_banner(adf_stationary, adf_stat, adf_p)
            stat_grid([
                {"label": "ADF Statistic", "value": f"{adf_stat:.4f}"},
                {"label": "p-value",       "value": f"{adf_p:.4f}"},
                {"label": "Conclusion",    "value": "Stationary ✅" if adf_stationary else "Non-stationary ❌"},
            ])
            if not adf_stationary:
                callout("Non-stationary — 1st differencing will be applied (d = 1).", "warning")
        except Exception as e:
            callout(f"ADF test error: {e}", "danger")
            adf_stationary = True

        section_title("KPSS Test (complementary)")
        hypothesis_box(
            h0="Series is trend-stationary",
            h1="Series is NOT trend-stationary",
            alpha=0.05,
            test_name="KPSS Test",
        )

        try:
            kpss_stat, kpss_p, kpss_lags, _ = kpss(ts_raw, regression="c", nlags="auto")
            kpss_stationary = kpss_p > .05
            result_banner(not kpss_stationary, kpss_stat, kpss_p)
            stat_grid([
                {"label": "KPSS Statistic", "value": f"{kpss_stat:.4f}"},
                {"label": "p-value",        "value": f"{kpss_p:.4f}"},
                {"label": "Conclusion",     "value": "Stationary ✅" if kpss_stationary else "Non-stationary ❌"},
            ])
        except Exception:
            kpss_stationary = adf_stationary

        d_order = 0 if adf_stationary else 1
        ts_model = ts_raw.copy() if d_order == 0 else ts_raw.diff().dropna()

        if d_order == 0:
            callout("Both tests agree: series is <strong>stationary</strong> → d = 0", "success")
        else:
            callout("Series non-stationary → 1st differencing applied → d = 1", "warning")

        # ACF/PACF via matplotlib (stored as image)
        section_title("ACF & PACF Plots")
        try:
            lags = min(30, len(ts_model)//3)
            fig_acf, axes = plt.subplots(1, 2, figsize=(12, 3),
                                          facecolor="#161b22")
            for ax in axes:
                ax.set_facecolor("#1c2128")
                ax.tick_params(colors="#8b949e")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#30363d")

            plot_acf(ts_model, lags=lags, ax=axes[0], alpha=.05, color="#38bdf8",
                     vlines_kwargs={"color":"#38bdf8"})
            axes[0].set_title("ACF → guides MA order (q)", color="#e6edf3", fontsize=11)

            plot_pacf(ts_model, lags=lags, ax=axes[1], alpha=.05, method="ywm",
                      color="#f97316", vlines_kwargs={"color":"#f97316"})
            axes[1].set_title("PACF → guides AR order (p)", color="#e6edf3", fontsize=11)

            plt.tight_layout()
            st.pyplot(fig_acf)
            plt.close()
        except Exception as e:
            callout(f"ACF/PACF error: {e}", "warning")
        
        st.session_state["_ts_raw"] = ts_raw
        st.session_state["_d_order"] = d_order
        st.session_state["_ts_model"] = ts_model
        st.session_state["_ts_label"] = ts_label
        st.session_state["_freq_label"] = freq_label

    # ═══════════════════════════════════════════════════════════════════════
    # ARIMA MODEL SELECTION
    # ═══════════════════════════════════════════════════════════════════════
    with tab3:
        section_title("ARIMA Model Selection (AIC)")
        ts_raw_  = st.session_state.get("_ts_raw", ts_raw)
        d_order_ = st.session_state.get("_d_order", 0 if adf_stationary else 1)

        candidates = [
            (1,d_order_,1),(2,d_order_,1),(1,d_order_,2),
            (2,d_order_,2),(1,d_order_,0),(0,d_order_,1),(3,d_order_,1),
        ]

        with st.spinner("Fitting ARIMA candidates…"):
            results = []
            best_aic, best_order, best_model = np.inf, None, None
            for order in candidates:
                try:
                    m = ARIMA(ts_raw_, order=order).fit()
                    results.append({
                        "Order": str(order), "AIC": round(m.aic, 2),
                        "BIC": round(m.bic, 2), "Log-Lik": round(m.llf, 2),
                        "Best?": "",
                    })
                    if m.aic < best_aic:
                        best_aic, best_order, best_model = m.aic, order, m
                except Exception:
                    pass

            if best_order:
                for r in results:
                    if r["Order"] == str(best_order):
                        r["Best?"] = "⭐"

        if results:
            aic_df = pd.DataFrame(results).sort_values("AIC")
            st.dataframe(aic_df.style.map(
                lambda v: "font-weight:700;color:#fbbf24" if v == "⭐" else "",
                subset=["Best?"]), use_container_width=True)

        if best_model:
            st.session_state["_best_model"] = best_model
            st.session_state["_best_order"] = best_order

            callout(
                f"<strong>Best ARIMA order: {best_order}</strong>  AIC = {best_aic:.2f}",
                "success",
            )

            # Ljung-Box test
            section_title("Ljung-Box Residual White Noise Test")
            hypothesis_box(
                h0="Residuals are white noise (no remaining autocorrelation)",
                h1="Residuals have remaining autocorrelation (model inadequate)",
                alpha=0.05,
                test_name="Ljung-Box Test",
            )

            try:
                lb = acorr_ljungbox(best_model.resid.dropna(), lags=[10, 20], return_df=True)
                st.dataframe(lb.style.format({"lb_stat":"{:.4f}","lb_pvalue":"{:.4f}"}),
                             use_container_width=True)

                if all(lb["lb_pvalue"] > .05):
                    callout(
                        "✅ FAIL TO REJECT H₀ — residuals are white noise. "
                        "The ARIMA model adequately captured all temporal structure. "
                        "Forecast intervals will be reliable.",
                        "success",
                    )
                else:
                    callout("❌ Autocorrelation remains — consider higher p or q.", "warning")
            except Exception as e:
                callout(f"Ljung-Box error: {e}", "warning")

    # ═══════════════════════════════════════════════════════════════════════
    # FORECAST
    # ═══════════════════════════════════════════════════════════════════════
    with tab4:
        best_model_ = st.session_state.get("_best_model", None)
        best_order_ = st.session_state.get("_best_order", None)
        ts_raw_     = st.session_state.get("_ts_raw", ts_raw)
        freq_label_ = st.session_state.get("_freq_label", freq_label)
        ts_label_   = st.session_state.get("_ts_label", ts_label)

        if best_model_ is None:
            callout("Run the ARIMA model tab first.", "warning")
            return

        n_forecast = st.slider("Forecast horizon (days/months)", 7, 60, 30)

        try:
            fc = best_model_.get_forecast(steps=n_forecast)
            fc_mean = fc.predicted_mean
            fc_ci   = fc.conf_int(alpha=.05)

            freq_alias = "D" if freq_label_ == "Daily" else "ME"
            fc_idx = pd.date_range(ts_raw_.index[-1], periods=n_forecast+1, freq=freq_alias)[1:]
            fc_mean.index = fc_idx
            fc_ci.index   = fc_idx

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts_raw_.index, y=ts_raw_.values,
                mode="lines", name="Observed",
                line=dict(color="#38bdf8", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=fc_mean.index, y=fc_mean.values,
                mode="lines", name=f"ARIMA{best_order_} Forecast",
                line=dict(color="#f97316", width=2.5, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=list(fc_ci.index) + list(fc_ci.index[::-1]),
                y=list(fc_ci.iloc[:,0]) + list(fc_ci.iloc[:,1][::-1]),
                fill="toself", fillcolor="rgba(249,115,22,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% Prediction Interval",
            ))
            fig.add_hline(y=15, line_dash="dot", line_color="#fbbf24",
                          annotation_text="WHO guideline")
            fig.add_vline(x=str(ts_raw_.index[-1]), line_dash="dash",
                          line_color="#8b949e")
            fig.update_layout(**theme(
                title=dict(text=f"ARIMA{best_order_} Forecast — {n_forecast} {freq_label_.lower()} ahead",
                           font=dict(size=13)),
                yaxis_title=ts_label_,
                height=440,
            ))
            st.plotly_chart(fig, use_container_width=True)
            callout(
                "Prediction intervals widen with forecast horizon — uncertainty grows. "
                "If the model forecasts PM2.5 exceeding 150 µg/m³, "
                "pre-emptive health advisories can be issued proactively rather than reactively.",
                "info",
            )
        except Exception as e:
            callout(f"Forecast error: {e}", "danger")
