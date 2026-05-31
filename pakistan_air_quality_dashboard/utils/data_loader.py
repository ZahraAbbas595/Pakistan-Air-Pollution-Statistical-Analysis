"""
utils/data_loader.py
Central data loading, caching, and preprocessing — mirrors the notebook pipeline.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

# ── AQI constants ─────────────────────────────────────────────────────────────
AQI_CATEGORY_ORDER = [
    "Good", "Moderate", "Unhealthy for Sensitive Groups",
    "Unhealthy", "Very Unhealthy", "Hazardous",
]

AQI_MIDPOINTS = {
    "Good": 25, "Moderate": 75,
    "Unhealthy for Sensitive Groups": 125,
    "Unhealthy": 175, "Very Unhealthy": 250, "Hazardous": 400,
}

CAT_COLORS = {
    "Good": "#4ade80",
    "Moderate": "#fbbf24",
    "Unhealthy for Sensitive Groups": "#fb923c",
    "Unhealthy": "#f87171",
    "Very Unhealthy": "#c084fc",
    "Hazardous": "#94a3b8",
}

SHORT_FORM_MAP = {
    "Usg": "Unhealthy for Sensitive Groups",
    "Unhealthy For Sensitive Groups": "Unhealthy for Sensitive Groups",
    "Sensitive": "Unhealthy for Sensitive Groups",
}

CITY_PALETTE = [
    "#f97316", "#38bdf8", "#4ade80", "#fbbf24", "#c084fc",
    "#f472b6", "#34d399", "#fb7185", "#60a5fa", "#a78bfa",
]

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

POSSIBLE_FILES = [
    BASE_DIR / "dataset.csv",
    BASE_DIR / "pakistan_air_quality_final_clean.csv",
    BASE_DIR / "air_quality_pakistan.csv",
    BASE_DIR / "Pakistan_Air_Quality_Weather.csv",
    BASE_DIR / "data.csv",
]


@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None):
    """Load and preprocess the dataset. Returns clean DataFrame or None."""
    df_raw = None

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
    else:
        for fname in POSSIBLE_FILES:
            if os.path.exists(fname):
                df_raw = pd.read_csv(fname)
                break

    if df_raw is None:
        return None

    return _preprocess(df_raw)


def _preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Standardize column names
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    )

    # Parse datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "date" in df.columns:
            df.drop(columns=["date"], inplace=True)
        df.rename(columns={"timestamp": "date"}, inplace=True)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Detect key columns
    aqi_col  = (next((c for c in df.columns if "aqi" in c and "category" in c), None)
                or next((c for c in df.columns if "aqi" in c), None))
    pm25_col = next((c for c in df.columns if "pm2" in c or "pm_2" in c), None)
    pm10_col = next((c for c in df.columns if "pm10" in c or "pm_10" in c), None)
    city_col = next((c for c in df.columns if "city" in c or "location" in c or "station" in c), None)
    temp_col = next((c for c in df.columns if "temp" in c), None)
    hum_col  = next((c for c in df.columns if "humid" in c), None)
    wind_col = ("wind_speed" if "wind_speed" in df.columns
                else next((c for c in df.columns if "wind_speed" in c), None)
                or next((c for c in df.columns if "wind" in c), None))
    no2_col  = next((c for c in df.columns if "no2" in c or "nitrogen" in c), None)
    so2_col  = next((c for c in df.columns if "so2" in c or "sulph" in c), None)
    co_col   = next((c for c in df.columns if "carbon" in c or c == "co"), None)

    # Standardize city names
    if city_col:
        df[city_col] = df[city_col].str.strip().str.title()
        df.rename(columns={city_col: "city"}, inplace=True)

    # Rename to standard names
    rename_map = {}
    for old, new in [
        (pm25_col, "pm25"), (pm10_col, "pm10"), (temp_col, "temperature"),
        (hum_col, "humidity"), (wind_col, "wind_speed"),
        (no2_col, "no2"), (so2_col, "so2"), (co_col, "co"),
        (aqi_col, "aqi_category"),
    ]:
        if old and old != new:
            rename_map[old] = new
    df.rename(columns=rename_map, inplace=True, errors="ignore")

    # Drop missing AQI rows
    if "aqi_category" in df.columns:
        df.dropna(subset=["aqi_category"], inplace=True)

    # Clip negative values
    for col in ["pm25", "pm10"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # AQI category encoding
    if "aqi_category" in df.columns:
        df["aqi_category"] = (
            df["aqi_category"].astype(str).str.strip().str.title()
            .replace(SHORT_FORM_MAP)
        )
        df["aqi_category"] = pd.Categorical(
            df["aqi_category"], categories=AQI_CATEGORY_ORDER, ordered=True
        )
        df["aqi_numeric"] = df["aqi_category"].map(AQI_MIDPOINTS).astype(float)

    # Season feature
    if "season" not in df.columns and "date" in df.columns:
        def _season(m):
            if m in [12, 1, 2]: return "Winter"
            elif m in [3, 4, 5]: return "Spring"
            elif m in [6, 7, 8]: return "Monsoon"
            else: return "Autumn/Post-Monsoon"
        df["season"] = df["date"].dt.month.map(_season)

    # Derived time columns
    if "date" in df.columns and hasattr(df["date"], "dt"):
        df["year"]  = df["date"].dt.year
        df["month"] = df["date"].dt.month
        if "hour" not in df.columns:
            df["hour"] = df["date"].dt.hour

    # Weekend flag
    if "date" in df.columns and "is_weekend" not in df.columns:
        df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    return df


def get_focus_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in
        ["aqi_numeric", "pm25", "pm10", "no2", "so2", "co",
         "temperature", "humidity", "wind_speed"]
        if c in df.columns]


def get_city_list(df: pd.DataFrame) -> list[str]:
    if "city" not in df.columns:
        return []
    return sorted(df["city"].unique())


def get_city_groups(df: pd.DataFrame, col: str = "aqi_numeric") -> tuple[list, list]:
    cities = get_city_list(df)
    groups = [df[df["city"] == c][col].dropna().values for c in cities]
    return cities, groups
