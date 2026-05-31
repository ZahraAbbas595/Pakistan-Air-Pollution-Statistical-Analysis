"""
Pakistan Air Quality Crisis -- STAT222 Advanced Statistics Dashboard
Single-page app with manual routing (avoids Streamlit MPA auto-discovery).
"""

import sys, os
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st

st.set_page_config(
    page_title="Pakistan AQ Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from pathlib import Path

BASE_DIR = Path(__file__).parent

css_file = BASE_DIR / "assets" / "style.css"

with open(css_file, encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from _pages.page_home          import render as home_render
from _pages.page_eda           import render as eda_render
from _pages.page_anova         import render as anova_render
from _pages.page_distribution  import render as dist_render
from _pages.page_regression    import render as reg_render
from _pages.page_timeseries    import render as ts_render
from _pages.page_nonparametric import render as np_render
from _pages.page_extra_tests   import render as extra_render
from _pages.page_summary       import render as summary_render

PAGES = {
    "🏠  Overview":               home_render,
    "📊  Exploratory Analysis":   eda_render,
    "🔬  ANOVA Tests":            anova_render,
    "📐  Distribution Fitting":   dist_render,
    "📈  Regression Model":       reg_render,
    "⏱️   Time Series & ARIMA":   ts_render,
    "🔢  Nonparametric Tests":    np_render,
    "🧪  Extra Hypothesis Tests": extra_render,
    "✅  Summary & Conclusions":  summary_render,
}

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-icon">🌫️</div>
        <div class="sidebar-title">Pakistan AQ<br><span>Dashboard</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='nav-label'>NAVIGATION</div>", unsafe_allow_html=True)

    selection = st.radio(
        label="nav",
        options=list(PAGES.keys()),
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div class='sidebar-meta'>
        <div><b>Course</b> · STAT222</div>
        <div><b>Class</b> · BSDS-02</div>
        <div><b>Dataset</b> · 21,840 obs · 10 cities</div>
        <div><b>Period</b> · Nov 2025 – Feb 2026</div>
    </div>
    """, unsafe_allow_html=True)

PAGES[selection]()
