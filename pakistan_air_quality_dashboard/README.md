# 🌫️ Pakistan Air Quality Dashboard

**STAT222 · Advanced Statistics · BSDS-02**  
A professional multi-page Streamlit dashboard for the rigorous statistical analysis of air quality across Pakistan's 10 major cities.

---

## 📋 Overview

This dashboard is a fully interactive companion to the Jupyter notebook analysis. It implements all 10 formal hypothesis tests and 7 statistical methods — with live computation, professional Plotly visualisations, and real-time controls.

| Method | Section | Status |
|--------|---------|--------|
| One-Way ANOVA + Tukey HSD | 3.1 | ✅ |
| Two-Way ANOVA (City × Season) | 3.2 | ✅ |
| Probability Distribution Fitting | 3.3 | ✅ |
| Multiple Linear Regression (OLS) | 3.4 | ✅ |
| Time Series (STL + ARIMA) | 3.5 | ✅ |
| Kruskal-Wallis + Mann-Whitney U | 3.6 | ✅ |
| Welch's T, Chi-Square, Mann-Kendall, KS, Diurnal ANOVA | 3.8 | ✅ |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Load the dataset

Upload `pakistan_air_quality_final_clean.csv` via the upload widget on the **Overview** page, or place it in the same directory as `app.py` before launching.

> **Dataset source:** [Kaggle — Pakistan Air Quality & Weather (10 cities)](https://www.kaggle.com/datasets/ahsanneural/pakistan-air-quality-and-weather-10-cities)

---

## 📁 Project Structure

```
pakistan_air_quality_dashboard/
│
├── app.py                          # Entry point & navigation
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── .streamlit/
│   └── config.toml                 # Dark theme configuration
│
├── assets/
│   └── style.css                   # Global CSS — dark analytical theme
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py              # Data loading, caching, preprocessing
│   ├── charts.py                   # Reusable Plotly chart builders
│   └── ui.py                       # HTML/CSS UI components
│
└── pages/
    ├── __init__.py
    ├── page_home.py                # Overview, dataset upload, top-level metrics
    ├── page_eda.py                 # Exploratory Data Analysis
    ├── page_anova.py               # One-Way & Two-Way ANOVA
    ├── page_distribution.py        # Distribution Fitting (Normal/Log/Gamma/Weibull)
    ├── page_regression.py          # Multiple OLS Regression + diagnostics
    ├── page_timeseries.py          # STL Decomposition + ARIMA Forecasting
    ├── page_nonparametric.py       # Kruskal-Wallis + Mann-Whitney U
    ├── page_extra_tests.py         # 5 extra hypothesis tests
    └── page_summary.py             # Summary, policy recommendations, references
```

---

## 🎨 Design System

- **Theme:** Dark analytical — GitHub-inspired slate `#0d1117` with ember orange `#f97316` accent
- **Typography:** Sora (display) + JetBrains Mono (code/stats)
- **Charts:** Plotly with custom dark theme — consistent colour palette across all pages
- **AQI colours:** Green → Yellow → Orange → Red → Purple → Slate

---

## 📊 Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | Upload dataset, top-level metrics, AQI distribution, city ranking |
| 📊 EDA | Descriptive stats, distributions, correlations, temporal trends, outlier detection |
| 🔬 ANOVA | One-Way + Two-Way ANOVA, Tukey HSD, interaction plots |
| 📐 Distribution | Normal/Lognormal/Gamma/Weibull fitting, AIC comparison, Q-Q plots |
| 📈 Regression | OLS model, VIF, coefficient plot, 4-panel diagnostics |
| ⏱️ Time Series | STL decomposition, ADF/KPSS tests, ACF/PACF, ARIMA selection, forecast |
| 🔢 Nonparametric | Kruskal-Wallis, Mann-Whitney pairwise, Bonferroni correction, p-value heatmap |
| 🧪 Extra Tests | Welch's T, Chi-Square, Mann-Kendall, Two-Sample KS, Diurnal ANOVA |
| ✅ Summary | Master hypothesis table, key findings, 5 policy recommendations, WHO exceedance |

---

## 💡 Features

- **Live computation** — all statistical tests computed in real time on the uploaded dataset
- **Interactive controls** — city selector, variable selector, forecast horizon slider
- **Full assumption checking** — every test includes normality, variance, and fit tests
- **Hypothesis framework** — H₀/H₁ framing displayed before every test
- **Effect sizes** — η², ε², Cramér's V, Cohen's d reported throughout
- **Real-world interpretation** — every result includes a Pakistan-specific policy conclusion
- **Responsive layout** — wide mode, 1440px max container width

---

## 🗂️ Expected Dataset Columns

The app auto-detects columns but works best with:

| Column | Description |
|--------|-------------|
| `timestamp` or `date` | Datetime (hourly preferred) |
| `city` | City name |
| `aqi_category` | AQI category (Good → Hazardous) |
| `pm2_5` or `pm25` | PM2.5 concentration (µg/m³) |
| `pm10` | PM10 concentration (µg/m³) |
| `temperature` | Temperature (°C) |
| `humidity` | Relative humidity (%) |
| `wind_speed` | Wind speed (m/s) |
| `no2`, `so2`, `co` | Pollutant concentrations |
| `season` | Season label (auto-derived if absent) |
| `is_weekend` | 0/1 flag (auto-derived if absent) |
| `hour` | Hour of day (auto-derived from datetime) |

---

## 📚 References

- IQAir (2025). *2024 World Air Quality Report*
- Ahmad et al. (2020). *Science of the Total Environment*, 742, 140571
- Khan et al. (2022). *Atmospheric Environment*, 275, 119003
- Box, Jenkins, Reinsel & Ljung (2015). *Time Series Analysis* (5th ed.)
- WHO (2021). *Global Air Quality Guidelines*

---

*Python 3 · pandas · scipy · statsmodels · plotly · streamlit · matplotlib*
