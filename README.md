# 🌫️ Pakistan Air Pollution Statistical Analysis Dashboard

## 📌 Overview

This project presents a comprehensive statistical analysis of air pollution patterns across major cities in Pakistan using advanced statistical techniques, data visualization, and an interactive Streamlit dashboard.

The study investigates air quality trends, pollutant distributions, city-wise differences, temporal patterns, and environmental relationships through a combination of classical statistics, regression modeling, and time-series forecasting.

The project was developed as part of an Advanced Statistics course and transformed into a fully deployed interactive analytics dashboard.

---

## 🌐 Live Dashboard

🔗 **Live App:** https://air-pollution-analysis.streamlit.app/

Users can explore the complete analysis interactively without installing any software.

---

## 📊 Dataset

The dataset contains:

* Air Quality Indicators
* PM2.5 Concentrations
* PM10 Concentrations
* NO₂ Levels
* SO₂ Levels
* CO Levels
* Temperature
* Humidity
* Wind Speed
* Timestamp Information
* Multiple Pakistani Cities

### Study Scope

* 21,840+ observations
* 10 major cities
* Multi-month pollution monitoring period

---

## 🎯 Objectives

This project aims to:

* Analyze air pollution trends across Pakistan
* Compare pollution levels among cities
* Evaluate statistical differences in air quality
* Model relationships between pollutants and environmental variables
* Forecast future pollution patterns
* Build an interactive analytical dashboard for exploration and decision-making

---

# 📈 Statistical Techniques Applied

## 1️⃣ Exploratory Data Analysis (EDA)

* Missing value assessment
* Descriptive statistics
* Distribution analysis
* Outlier investigation
* Correlation analysis

---

## 2️⃣ One-Way ANOVA

Used to determine whether significant differences exist between pollution levels across different cities.

### Research Question

Do mean pollution levels differ significantly among Pakistani cities?

---

## 3️⃣ Two-Way ANOVA

Used to evaluate:

* City effects
* Seasonal effects
* City × Season interaction effects

---

## 4️⃣ Distribution Fitting

Various probability distributions were evaluated to determine the best fit for air pollution measurements.

Distributions explored include:

* Normal Distribution
* Lognormal Distribution
* Gamma Distribution
* Weibull Distribution

---

## 5️⃣ Multiple Linear Regression

Regression modeling was performed to investigate the relationship between:

### Target Variable

* PM2.5

### Predictor Variables

* Temperature
* Humidity
* Wind Speed
* NO₂
* SO₂
* CO

---

## 6️⃣ Time Series Analysis

Advanced temporal analysis was conducted to identify pollution trends over time.

### Methods

* Trend Analysis
* Rolling Statistics
* Seasonal Decomposition
* ARIMA Modeling

---

## 7️⃣ ARIMA Forecasting

Autoregressive Integrated Moving Average (ARIMA) models were used to forecast future PM2.5 concentrations.

---

## 8️⃣ Nonparametric Statistical Testing

Robust statistical testing was performed using:

* Kruskal-Wallis Test
* Mann-Whitney U Test
* Kolmogorov-Smirnov Test

These methods provide reliable conclusions when normality assumptions are violated.

---

# 🖥️ Interactive Streamlit Dashboard

The project was transformed into a fully interactive Streamlit application featuring:

### 🏠 Overview

* Project introduction
* Dataset summary
* Key statistics

### 📊 Exploratory Analysis

* Interactive visualizations
* Distribution exploration

### 🔬 ANOVA Analysis

* One-Way ANOVA Results
* Two-Way ANOVA Results

### 📐 Distribution Fitting

* Distribution comparisons
* Goodness-of-fit analysis

### 📈 Regression Modeling

* Model performance
* Statistical interpretation

### ⏱️ Time Series & ARIMA

* Trend visualization
* Forecasting outputs

### 🔢 Nonparametric Analysis

* Robust statistical testing

### 🧪 Additional Hypothesis Tests

* Extended statistical validation

### ✅ Summary & Conclusions

* Key findings
* Final recommendations

---

# 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Statistics

* SciPy
* StatsModels

### Visualization

* Plotly
* Matplotlib
* Seaborn

### Dashboard Development

* Streamlit

### Deployment

* GitHub
* Streamlit Community Cloud

---

# 📂 Project Structure

```text
Pakistan-Air-Pollution-Statistical-Analysis/

│
├── dataset.csv
├── python_code.ipynb
├── report.docx
│
├── pakistan_air_quality_dashboard/
│   ├── app.py
│   ├── assets/
│   ├── utils/
│   ├── _pages/
│   └── requirements.txt
│
└── README.md
```

---

# 🚀 How To Run Locally

Clone the repository:

```bash
git clone <repository-url>
```

Navigate into the dashboard directory:

```bash
cd pakistan_air_quality_dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# 📌 Key Insights

* Significant differences in pollution levels exist among cities.
* Air quality varies substantially across seasons.
* PM2.5 exhibits strong relationships with multiple environmental variables.
* Time-series analysis reveals persistent temporal patterns.
* Statistical evidence supports targeted environmental interventions.

---

# 👩‍💻 Author

**Zahra Abbas**

Computer Engineering Student | Data Analytics Enthusiast | Machine Learning Learner

### Connect With Me

* GitHub: https://github.com/ZahraAbbas595
* LinkedIn: www.linkedin.com/in/zahra-abbas-9a665037a
* Fiverr: https://www.fiverr.com/users/zahraabbas959

---

## ⭐ If you found this project useful, consider giving it a star!
