"""
Macro Regime Dashboard — Streamlit app
Run locally:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import the model (must be in the same folder)
from macro_regime_model import (
    build_us_data, build_uk_data,
    classify_regimes, compute_dynamic_allocation,
    ASSET_CLASSES, BASE_ALLOCATIONS, EXPECTED_RETURNS,
    REGIME_COLORS, SHORT_LABELS,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Macro Regime Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme tweaks ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid;
        margin-bottom: 8px;
    }
    h1 { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_COLORS = {
    0: '#2ecc71',
    1: '#e74c3c',
    2: '#f39c12',
    3: '#3498db',
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("---")

    country = st.selectbox("Country", ["US", "UK", "Both"])

    st.markdown("#### Regime Thresholds")
    st.caption("Override the auto-calculated medians")
    use_custom = st.toggle("Use custom thresholds", value=False)

    custom_cpi = st.slider("Inflation threshold (%)", 0.5, 8.0, 2.5, 0.1,
                           disabled=not use_custom)
    custom_gdp = st.slider("GDP growth threshold (%)", -2.0, 6.0, 2.0, 0.1,
                           disabled=not use_custom)

    st.markdown("---")
    st.markdown("#### Date Range")
    year_range = st.slider("Years", 1990, 2025, (1990, 2025))

    st.markdown("---")
    st.caption("Data: BLS, BEA, ONS, Bank of England  \n"
               "Model: GMM regime classifier  \n"
               "Allocations: theoretical / illustrative")


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    us_raw = build_us_data()
    uk_raw = build_uk_data()
    us_df, _, _ = compute_dynamic_allocation(classify_regimes(us_raw))
    uk_df, _, _ = compute_dynamic_allocation(classify_regimes(uk_raw))
    return us_df, uk_df


us_df_full, uk_df_full = load_data()


def apply_custom_thresholds(df, cpi_thresh, gdp_thresh):
    """Re-classify with user-supplied thresholds."""
    df = df.copy()
    high_infl = df['cpi_smooth'] >= cpi_thresh
    high_grow = df['gdp_smooth'] >= gdp_thresh
    df['regime'] = np.where(~high_infl &  high_grow, 0,
                   np.where( high_infl & ~high_grow, 1,
                   np.where( high_infl &  high_grow, 2, 3))).astype(int)
    # Re-run probabilities (simplified: use hard assignment)
    for r in range(4):
        df[f'prob_{r}'] = (df['regime'] == r).astype(float)
    df, _, _ = compute_dynamic_allocation(df)
    return df


def filter_by_year(df):
    start = pd.Timestamp(f"{year_range[0]}-01-01")
    end   = pd.Timestamp(f"{year_range[1]}-12-31")
    return df[(df.index >= start) & (df.index <= end)]


# Apply custom thresholds if toggled
if use_custom:
    us_df_full = apply_custom_thresholds(us_df_full, custom_cpi, custom_gdp)
    uk_df_full = apply_custom_thresholds(uk_df_full, custom_cpi, custom_gdp)

# Filter date range
us_df = filter_by_year(us_df_full)
uk_df = filter_by_year(uk_df_full)

# Pick which countries to show
dfs = {}
if country in ("US", "Both"):
    dfs["US"] = us_df
if country in ("UK", "Both"):
    dfs["UK"] = uk_df


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Macroeconomic Regime Dashboard")
st.caption("US & UK  |  1990–2025  |  GMM-based regime classification + dynamic asset allocation")
st.markdown("---")


# ── Current Regime Cards ──────────────────────────────────────────────────────
st.subheader("Current Macro Regime (2025 Q1)")
cols = st.columns(len(dfs) * 4)

for ci, (lbl, df) in enumerate(dfs.items()):
    last = df.iloc[-1]
    rid  = int(last['regime'])
    color = PLOTLY_COLORS[rid]
    base = ci * 4

    with cols[base]:
        st.metric(f"🌍 {lbl} Regime", SHORT_LABELS[rid])
    with cols[base+1]:
        st.metric("📈 CPI YoY", f"{last['cpi_yoy']:.1f}%")
    with cols[base+2]:
        st.metric("📉 GDP Growth", f"{last['gdp_growth']:.1f}%")
    with cols[base+3]:
        st.metric("🎯 Exp. Return", f"{last['expected_port_return']:.1f}% pa")

st.markdown("---")


# ── Regime Probability Gauge ──────────────────────────────────────────────────
st.subheader("Regime Posterior Probabilities")
prob_cols = st.columns(len(dfs))

for ci, (lbl, df) in enumerate(dfs.items()):
    last = df.iloc[-1]
    probs = [last[f'prob_{r}'] for r in range(4)]
    names = [SHORT_LABELS[r] for r in range(4)]
    colors_list = [PLOTLY_COLORS[r] for r in range(4)]

    fig = go.Figure(go.Bar(
        x=names, y=[p * 100 for p in probs],
        marker_color=colors_list,
        text=[f"{p*100:.0f}%" for p in probs],
        textposition='outside',
    ))
    fig.update_layout(
        title=f"{lbl} — Q1 2025",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 110],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=280,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    prob_cols[ci].plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ── Macro Time Series ─────────────────────────────────────────────────────────
st.subheader("Macro Time Series")
ts_cols = st.columns(len(dfs))

for ci, (lbl, df) in enumerate(dfs.items()):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['cpi_yoy'],
                             name='CPI YoY %', line=dict(color='#e74c3c', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['gdp_growth'],
                             name='GDP Growth %', line=dict(color='#2ecc71', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['policy_rate'],
                             name='Policy Rate %',
                             line=dict(color='#f39c12', width=1.5, dash='dot')))
    fig.add_hline(y=0, line_color='#444455', line_width=1)
    fig.update_layout(
        title=f"{lbl} Macro Indicators",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', height=320,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(t=50, b=20, l=10, r=10),
        hovermode='x unified',
    )
    fig.update_xaxes(gridcolor='#222233')
    fig.update_yaxes(gridcolor='#222233')
    ts_cols[ci].plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ── Regime Timeline ───────────────────────────────────────────────────────────
st.subheader("Regime Timeline")

for lbl, df in dfs.items():
    fig = go.Figure()
    for r in range(4):
        mask = df['regime'] == r
        # Add invisible scatter just for the legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(color=PLOTLY_COLORS[r], size=10, symbol='square'),
            name=SHORT_LABELS[r], showlegend=True,
        ))
    # Shaded regions per quarter
    for i, (dt, row) in enumerate(df.iterrows()):
        fig.add_vrect(
            x0=dt, x1=dt + pd.DateOffset(months=3),
            fillcolor=PLOTLY_COLORS[int(row['regime'])],
            opacity=0.6, line_width=0,
        )
    fig.update_layout(
        title=f"{lbl} Regime History",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', height=160,
        yaxis=dict(visible=False),
        margin=dict(t=40, b=20, l=10, r=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ── Regime Scatter ────────────────────────────────────────────────────────────
st.subheader("Regime Classification — Scatter")
sc_cols = st.columns(len(dfs))

for ci, (lbl, df) in enumerate(dfs.items()):
    fig = go.Figure()
    for r in range(4):
        mask = df['regime'] == r
        sub  = df[mask]
        fig.add_trace(go.Scatter(
            x=sub['cpi_smooth'], y=sub['gdp_smooth'],
            mode='markers',
            marker=dict(color=PLOTLY_COLORS[r], size=7, opacity=0.8),
            name=SHORT_LABELS[r],
            text=sub.index.strftime('%Y Q%q'),
            hovertemplate='%{text}<br>CPI: %{x:.1f}%<br>GDP: %{y:.1f}%<extra></extra>',
        ))
    # Threshold lines
    thresh_cpi = df['inf_threshold'].iloc[0] if not use_custom else custom_cpi
    thresh_gdp = df['gdp_threshold'].iloc[0] if not use_custom else custom_gdp
    fig.add_vline(x=thresh_cpi, line_dash='dash', line_color='white', opacity=0.4)
    fig.add_hline(y=thresh_gdp, line_dash='dash', line_color='white', opacity=0.4)
    fig.update_layout(
        title=f"{lbl} Regime Scatter",
        xaxis_title='CPI YoY % (smoothed)',
        yaxis_title='Real GDP Growth % (ann.)',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', height=380,
        margin=dict(t=40, b=30, l=40, r=10),
    )
    fig.update_xaxes(gridcolor='#222233')
    fig.update_yaxes(gridcolor='#222233')
    sc_cols[ci].plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ── Dynamic Allocation ────────────────────────────────────────────────────────
st.subheader("Dynamic Asset Allocation")
alloc_tab1, alloc_tab2 = st.tabs(["📊 Current Allocation", "📈 Allocation Over Time"])

with alloc_tab1:
    al_cols = st.columns(len(dfs))
    for ci, (lbl, df) in enumerate(dfs.items()):
        last = df.iloc[-1]
        alloc_cols = [c for c in df.columns if c.startswith('w_')]
        weights = last[alloc_cols].values
        colors  = px.colors.qualitative.Set2[:len(ASSET_CLASSES)]

        fig = go.Figure(go.Bar(
            x=weights, y=ASSET_CLASSES,
            orientation='h',
            marker_color=colors,
            text=[f"{w:.1f}%" for w in weights],
            textposition='outside',
        ))
        rid = int(last['regime'])
        fig.update_layout(
            title=f"{lbl} — 2025 Q1  [{SHORT_LABELS[rid]}]",
            xaxis_title='Allocation (%)',
            xaxis_range=[0, 60],
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=380,
            margin=dict(t=50, b=20, l=10, r=60),
        )
        fig.update_xaxes(gridcolor='#222233')
        al_cols[ci].plotly_chart(fig, use_container_width=True)

with alloc_tab2:
    for lbl, df in dfs.items():
        alloc_cols = [c for c in df.columns if c.startswith('w_')]
        colors = px.colors.qualitative.Set2[:len(ASSET_CLASSES)]
        fig = go.Figure()
        for i, (col, ac) in enumerate(zip(alloc_cols, ASSET_CLASSES)):
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                name=ac,
                stackgroup='one',
                fillcolor=colors[i],
                line=dict(color=colors[i], width=0.5),
                hovertemplate=f'{ac}: %{{y:.1f}}%<extra></extra>',
            ))
        fig.update_layout(
            title=f"{lbl} Dynamic Allocation Over Time",
            yaxis_title='Allocation (%)',
            yaxis_range=[0, 100],
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=380,
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        font=dict(size=10)),
            hovermode='x unified',
            margin=dict(t=50, b=20, l=40, r=10),
        )
        fig.update_xaxes(gridcolor='#222233')
        fig.update_yaxes(gridcolor='#222233')
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ── Regime Reference Table ────────────────────────────────────────────────────
st.subheader("Regime Reference — Target Allocations")

rows = []
for r in range(4):
    row = {'Regime': SHORT_LABELS[r]}
    for ac, w in zip(ASSET_CLASSES, BASE_ALLOCATIONS[r]):
        row[ac] = f"{w}%"
    rows.append(row)

ref_df = pd.DataFrame(rows).set_index('Regime')

def color_regime(val):
    idx = list(SHORT_LABELS.values()).index(val) if val in SHORT_LABELS.values() else -1
    return ''

st.dataframe(
    ref_df,
    use_container_width=True,
    height=180,
)

st.caption("Allocations are theoretical / illustrative and do not constitute investment advice.")
