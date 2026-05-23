# 🌍 Pakistan Air Pollution Statistical Analysis

A comprehensive statistical analysis project examining air pollution trends across 10 major Pakistani cities using regression modeling, hypothesis testing, probability distributions, and time series forecasting.

This project was developed for **STAT222 – Advanced Statistics** and analyzes over 21,000 hourly environmental observations to study AQI variation, PM2.5 behavior, seasonal trends, and pollutant relationships across Pakistan.

---

## 📌 Project Overview

Pakistan consistently ranks among the most polluted countries in the world. This project applies advanced statistical techniques to analyze air quality patterns and identify significant environmental trends.

The analysis focuses on:

- AQI differences across cities
- Seasonal pollution variation
- PM2.5 prediction modeling
- Probability distribution fitting
- ARIMA forecasting
- Hypothesis testing
- Environmental trend analysis

---

## ✨ Statistical Methods Used

### Parametric Methods
- One-Way ANOVA
- Two-Way ANOVA
- Multiple Linear Regression
- Welch’s T-Test
- ARIMA Forecasting

### Nonparametric Methods
- Kruskal-Wallis Test
- Mann-Whitney U Test
- Mann-Kendall Trend Test
- Kolmogorov-Smirnov Test
- Chi-Square Test

### Distribution Analysis
- Normal Distribution
- Lognormal Distribution
- Gamma Distribution
- Weibull Distribution

---

## 📂 Project Files

```bash
Pakistan-Air-Pollution-Statistical-Analysis/
│
├── dataset.csv
├── statistical_analysis.py
├── Semester_Final_Project_Report.pdf
└── README.md
```

---

## 📊 Dataset Information

- **Dataset:** Pakistan Air Quality & Weather Dataset
- **Observations:** 21,840 hourly records
- **Cities:** 10 major Pakistani cities
- **Time Period:** November 2025 – February 2026

### Cities Included
- Lahore
- Faisalabad
- Gujranwala
- Karachi
- Multan
- Peshawar
- Islamabad
- Rawalpindi
- Quetta
- Hyderabad

---

## 📌 Variables Included

### Pollutants
- PM2.5
- PM10
- NO2
- SO2
- CO

### Meteorological Variables
- Temperature
- Humidity
- Wind Speed

### Additional Features
- AQI Category
- Season
- Timestamp
- City

---

## 📈 Key Findings

- Lahore, Faisalabad, and Gujranwala showed the highest pollution levels.
- PM2.5 followed a Lognormal distribution more closely than a Normal distribution.
- PM10 was the strongest predictor of PM2.5 concentration.
- Wind speed showed a significant negative relationship with pollution levels.
- ARIMA(1,0,0) produced the best forecasting performance.

---

## 🚀 How to Run

### 1️⃣ Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
```

---

### 2️⃣ Run the Python File

```bash
python statistical_analysis.py
```

---

## 📉 Statistical Analysis Included

- Exploratory Data Analysis (EDA)
- Correlation Analysis
- One-Way ANOVA
- Two-Way ANOVA
- Multiple Linear Regression
- Time Series Forecasting (ARIMA)
- Distribution Fitting
- Residual Diagnostics
- Hypothesis Testing
- Nonparametric Analysis

---

## 📚 Learning Outcomes

This project demonstrates practical implementation of:

- Advanced statistical inference
- Environmental data analysis
- Time series forecasting
- Regression diagnostics
- Probability distribution fitting
- Statistical hypothesis testing
- Data visualization

---

## ⚠️ Limitations

- Dataset covers only Autumn and Winter seasons
- Single monitoring station per city
- Results are specific to the observed winter pollution period

---

## 👨‍💻 Author

- Zahra Abbas
- Linkedin: https://www.linkedin.com/in/zahra-abbas-9a665037a/
---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
