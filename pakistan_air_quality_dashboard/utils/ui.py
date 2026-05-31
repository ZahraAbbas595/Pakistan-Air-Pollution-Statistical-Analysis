"""
utils/ui.py
Reusable HTML UI building blocks for the dashboard.
"""

import streamlit as st


def page_header(eyebrow: str, title: str, description: str = ""):
    st.markdown(f"""
    <div class="page-header">
        <div class="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        {"<p>" + description + "</p>" if description else ""}
    </div>
    """, unsafe_allow_html=True)


def hypothesis_box(h0: str, h1: str, alpha: float = 0.05, test_name: str = ""):
    name_html = f"<div style='font-weight:700;color:#38bdf8;margin-bottom:10px;font-size:.8rem;letter-spacing:.08em;text-transform:uppercase'>{test_name}</div>" if test_name else ""
    st.markdown(f"""
    <div class="hypothesis-box">
        {name_html}
        <div class="h-row"><span class="h-label">H₀</span><span class="h-text">{h0}</span></div>
        <div class="h-row"><span class="h-label">H₁</span><span class="h-text">{h1}</span></div>
        <div class="alpha">Significance level α = {alpha}</div>
    </div>
    """, unsafe_allow_html=True)


def result_banner(reject: bool, f_val=None, p_val=None, extra: str = ""):
    if reject:
        badge = '<span class="result-badge reject">⚡ REJECT H₀</span>'
        color = "#f97316"
    else:
        badge = '<span class="result-badge fail">✓ FAIL TO REJECT H₀</span>'
        color = "#4ade80"

    stats_html = ""
    if f_val is not None:
        stats_html += f"<code style='color:#38bdf8;font-size:.82rem'> F = {f_val:.4f}</code> "
    if p_val is not None:
        stats_html += f"<code style='color:#fbbf24;font-size:.82rem'> p = {p_val:.2e}</code> "
    if extra:
        stats_html += f"<span style='color:#8b949e;font-size:.82rem'> {extra}</span>"

    st.markdown(
        f"<div style='margin:10px 0'>{badge} {stats_html}</div>",
        unsafe_allow_html=True,
    )


def section_title(text: str, icon: str = ""):
    prefix = f"{icon} " if icon else ""
    st.markdown(
        f'<div class="section-title">{prefix}{text}</div>',
        unsafe_allow_html=True,
    )


def metric_row(metrics: list[dict]):
    """
    metrics: list of dicts with keys: label, value, sub, color (optional)
    color: 'danger' | 'warning' | 'good' | '' (default white)
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            color_class = m.get("color", "")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{m['label']}</div>
                <div class="metric-value {color_class}">{m['value']}</div>
                <div class="metric-sub">{m.get('sub', '')}</div>
            </div>
            """, unsafe_allow_html=True)


def stat_grid(items: list[dict]):
    """items: list of {label, value}"""
    cells = "".join(
        f"""<div class="stat-cell">
              <div class="sc-label">{it['label']}</div>
              <div class="sc-value">{it['value']}</div>
           </div>"""
        for it in items
    )
    st.markdown(f'<div class="stat-result-grid">{cells}</div>', unsafe_allow_html=True)


def callout(text: str, kind: str = "info"):
    """kind: info | warning | success | danger"""
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def no_data_message():
    callout(
        "<strong>No dataset loaded.</strong> Upload your CSV using the uploader below "
        "or place the file in the app directory and restart.",
        "warning",
    )


def data_uploader():
    st.markdown("#### Upload Dataset")
    return st.file_uploader(
        "Upload pakistan_air_quality_final_clean.csv (or compatible format)",
        type=["csv"],
        label_visibility="collapsed",
    )
