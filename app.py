"""
GlassDeck — Crypto Price Distribution Calculator
Quantitative price distribution modeling for BTC / ETH
Models: GBM · Jump-Diffusion · Heston SV · Full Model
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GlassDeck · Price Distribution",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME / CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #04080f; color: #c8d6ef; }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #080d18;
        border-right: 1px solid rgba(99,179,255,0.08);
    }
    [data-testid="stSidebar"] .stMarkdown p { color: #4a6080; font-size: 11px; letter-spacing: 0.08em; }

    /* Headers */
    h1, h2, h3 { color: #e8edf8 !important; font-weight: 700 !important; letter-spacing: -0.02em; }
    h1 { font-size: 1.6rem !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(12,18,30,0.99), rgba(6,10,18,0.99));
        border: 1px solid rgba(99,179,255,0.08);
        border-radius: 12px;
        padding: 1rem;
    }
    [data-testid="stMetric"] label { color: #3a5070 !important; font-size: 11px !important; letter-spacing: 0.08em; text-transform: uppercase; }
    [data-testid="stMetricValue"] { color: #e8edf8 !important; font-size: 1.3rem !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] svg { display: none; }

    /* Sliders */
    .stSlider [data-baseweb="slider"] { padding: 0.5rem 0; }

    /* Selectbox / radio */
    .stSelectbox label, .stRadio label { color: #4a6080 !important; font-size: 11px !important; letter-spacing: 0.06em; text-transform: uppercase; }

    /* Section headers custom */
    .section-head {
        font-size: 10px; font-weight: 600; color: #2a3850;
        letter-spacing: 0.12em; text-transform: uppercase;
        border-bottom: 1px solid rgba(99,179,255,0.06);
        padding-bottom: 6px; margin-bottom: 12px; margin-top: 8px;
    }
    .logo-row {
        display: flex; align-items: center; gap: 10px; margin-bottom: 20px;
    }
    .logo-box {
        width: 30px; height: 30px; border-radius: 8px;
        background: linear-gradient(135deg, #63b3ff, #1d4ed8);
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; font-weight: 900;
    }
    .logo-text { font-size: 18px; font-weight: 800; color: #e8edf8; letter-spacing: -0.02em; }
    .logo-accent { color: #63b3ff; }
    .badge {
        display: inline-block; font-size: 10px; padding: 2px 10px;
        border-radius: 6px; font-weight: 600; letter-spacing: 0.05em;
    }
    .badge-green  { background: rgba(30,158,117,0.15); color: #1d9e75; border: 1px solid rgba(30,158,117,0.25); }
    .badge-red    { background: rgba(244,63,94,0.12);  color: #f43f5e; border: 1px solid rgba(244,63,94,0.22); }
    .badge-blue   { background: rgba(99,179,255,0.10); color: #63b3ff; border: 1px solid rgba(99,179,255,0.22); }
    .badge-amber  { background: rgba(251,146,60,0.12); color: #fb923c; border: 1px solid rgba(251,146,60,0.22); }
    .interp-box {
        background: rgba(10,16,28,0.8); border: 1px solid rgba(99,179,255,0.07);
        border-radius: 10px; padding: 14px 16px; font-size: 13px;
        line-height: 1.7; color: #5a7090;
    }
    .interp-box b { color: #8aa0c0; }
    .prob-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 12px; border-radius: 8px; margin-bottom: 4px;
        background: rgba(10,16,28,0.6); border: 1px solid rgba(99,179,255,0.05);
        font-size: 12px;
    }
    .prob-label { color: #3a5070; }
    .stButton > button {
        background: rgba(99,179,255,0.1) !important;
        border: 1px solid rgba(99,179,255,0.25) !important;
        color: #63b3ff !important; font-weight: 600 !important;
        border-radius: 8px !important; font-size: 13px !important;
        transition: all 0.15s !important;
    }
    .stButton > button:hover { background: rgba(99,179,255,0.18) !important; }
    div[data-testid="stHorizontalBlock"] { gap: 10px; }
    .stInfo { background: rgba(99,179,255,0.06) !important; border: 1px solid rgba(99,179,255,0.15) !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#4a6080", size=11),
    margin=dict(l=48, r=16, t=32, b=36),
    xaxis=dict(
        gridcolor="rgba(99,179,255,0.05)",
        linecolor="rgba(99,179,255,0.07)",
        tickfont=dict(size=10, color="#2a3850"),
        zerolinecolor="rgba(99,179,255,0.06)",
    ),
    yaxis=dict(
        gridcolor="rgba(99,179,255,0.05)",
        linecolor="rgba(99,179,255,0.07)",
        tickfont=dict(size=10, color="#2a3850"),
        zerolinecolor="rgba(99,179,255,0.06)",
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#4a6080"),
        borderwidth=0,
    ),
    hoverlabel=dict(
        bgcolor="rgba(4,8,15,0.97)",
        bordercolor="rgba(99,179,255,0.2)",
        font=dict(family="DM Mono, monospace", size=11, color="#c8d6ef"),
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# MATH CORE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_montecarlo(
    S0, sigma, mu, T_years, n_paths, n_steps,
    model,
    lam, mu_j, sig_j,
    xi, kappa, rho,
    signal_adj,
):
    """
    Monte Carlo simulation supporting 4 models:
      gbm   — Geometric Brownian Motion
      jump  — Merton Jump-Diffusion
      heston — Heston Stochastic Volatility
      full  — Heston + Jumps (most realistic)
    Returns array of terminal prices.
    """
    rng   = np.random.default_rng(42)
    dt    = T_years / n_steps
    mu_eff = mu + signal_adj
    paths  = np.full(n_paths, S0, dtype=np.float64)
    v      = np.full(n_paths, sigma**2)

    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)

        if model in ("heston", "full"):
            theta = sigma**2
            v = np.maximum(
                v + kappa * (theta - v) * dt + xi * np.sqrt(np.maximum(v, 0) * dt) * z2,
                1e-8,
            )
            sig_eff = np.sqrt(v)
        else:
            sig_eff = sigma

        drift = (mu_eff - 0.5 * sig_eff**2) * dt
        diff  = sig_eff * np.sqrt(dt) * z1
        paths *= np.exp(drift + diff)

        if model in ("jump", "full"):
            n_jumps = rng.poisson(lam * dt, n_paths)
            mask    = n_jumps > 0
            if mask.any():
                jump_sizes = rng.normal(mu_j, sig_j, mask.sum())
                paths[mask] *= np.exp(jump_sizes)

    return paths


def compute_signal_adjustment(funding, ls_ratio, exchange_inflow, fear_greed):
    """
    Convert observable market signals → annualized drift adjustment.
    Based on empirical relationships:
      - Persistent positive funding → mean-reversion drag
      - Crowded L/S → contrarian pull
      - Exchange inflows → selling pressure
      - Fear & Greed extremes → mild sentiment tilt
    """
    adj  = -funding * 3 * 365 * 0.30    # funding rate × 3 periods/day × 365 days
    adj += -(ls_ratio - 1.0) * 0.15     # crowded longs pull downward
    adj += -exchange_inflow * 0.25       # inflows = selling pressure
    adj += (fear_greed - 50) / 50 * 0.08
    return float(adj)


def compute_stats(paths, S0):
    sorted_paths = np.sort(paths)
    N = len(sorted_paths)
    mean   = float(np.mean(paths))
    median = float(np.median(paths))
    std    = float(np.std(paths))
    skew   = float(scipy_stats.skew(paths))
    kurt   = float(scipy_stats.kurtosis(paths, fisher=True))
    p01    = float(np.percentile(paths, 1))
    p05    = float(np.percentile(paths, 5))
    p10    = float(np.percentile(paths, 10))
    p25    = float(np.percentile(paths, 25))
    p75    = float(np.percentile(paths, 75))
    p90    = float(np.percentile(paths, 90))
    p95    = float(np.percentile(paths, 95))
    p99    = float(np.percentile(paths, 99))
    cvar5  = float(np.mean(sorted_paths[:max(1, int(N * 0.05))]))
    cvar1  = float(np.mean(sorted_paths[:max(1, int(N * 0.01))]))
    var5   = p05
    var1   = p01
    prob_up   = float(np.mean(paths > S0))
    prob_up10 = float(np.mean(paths > S0 * 1.10))
    prob_up20 = float(np.mean(paths > S0 * 1.20))
    prob_up50 = float(np.mean(paths > S0 * 1.50))
    prob_up2x = float(np.mean(paths > S0 * 2.00))
    prob_dn10 = float(np.mean(paths < S0 * 0.90))
    prob_dn20 = float(np.mean(paths < S0 * 0.80))
    prob_dn40 = float(np.mean(paths < S0 * 0.60))
    prob_flat = float(np.mean((paths > S0 * 0.90) & (paths < S0 * 1.10)))
    return dict(
        mean=mean, median=median, std=std, skew=skew, kurt=kurt,
        p01=p01, p05=p05, p10=p10, p25=p25, p75=p75, p90=p90, p95=p95, p99=p99,
        cvar5=cvar5, cvar1=cvar1, var5=var5, var1=var1,
        prob_up=prob_up, prob_up10=prob_up10, prob_up20=prob_up20,
        prob_up50=prob_up50, prob_up2x=prob_up2x,
        prob_dn10=prob_dn10, prob_dn20=prob_dn20, prob_dn40=prob_dn40,
        prob_flat=prob_flat,
    )


def volatility_cone(S0, sigma, mu, max_days=365):
    """
    Analytical vol cone under GBM — useful as baseline even for advanced models.
    Returns dict of arrays for each horizon.
    """
    days = np.array([1, 3, 7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365])
    days = days[days <= max_days + 1]
    T    = days / 365
    med  = S0 * np.exp((mu - sigma**2 / 2) * T)
    s1u  = S0 * np.exp((mu - sigma**2 / 2) * T + sigma * np.sqrt(T))
    s1d  = S0 * np.exp((mu - sigma**2 / 2) * T - sigma * np.sqrt(T))
    s2u  = S0 * np.exp((mu - sigma**2 / 2) * T + 2 * sigma * np.sqrt(T))
    s2d  = S0 * np.exp((mu - sigma**2 / 2) * T - 2 * sigma * np.sqrt(T))
    return dict(days=days, med=med, s1u=s1u, s1d=s1d, s2u=s2u, s2d=s2d)


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def chart_distribution(paths, S0, asset, dte):
    fig = go.Figure()
    n_bins = 80
    hist, edges = np.histogram(paths, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    colors  = ["rgba(244,63,94,0.75)" if c < S0 else "rgba(30,158,117,0.75)" for c in centers]

    fig.add_trace(go.Bar(
        x=centers, y=hist,
        marker_color=colors,
        marker_line_width=0,
        name="Simulated paths",
        hovertemplate="Price: $%{x:,.0f}<br>Paths: %{y}<extra></extra>",
    ))

    for pct, label, color in [
        (np.percentile(paths, 5),  "VaR 5%",  "rgba(244,63,94,0.8)"),
        (np.percentile(paths, 25), "P25",      "rgba(251,146,60,0.6)"),
        (np.median(paths),         "Median",   "rgba(99,179,255,0.9)"),
        (np.mean(paths),           "Mean",     "rgba(167,139,250,0.9)"),
        (np.percentile(paths, 75), "P75",      "rgba(30,158,117,0.7)"),
        (np.percentile(paths, 95), "P95",      "rgba(30,158,117,0.9)"),
    ]:
        fig.add_vline(x=pct, line_dash="dot", line_color=color, line_width=1.5,
                      annotation_text=label, annotation_font_size=9,
                      annotation_font_color=color, annotation_position="top")

    fig.add_vline(x=S0, line_dash="solid", line_color="rgba(255,255,255,0.25)",
                  line_width=1.5, annotation_text=f"Current {asset}",
                  annotation_font_size=9, annotation_font_color="rgba(255,255,255,0.5)")

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"{asset} price distribution — {dte}d horizon", font=dict(size=13, color="#8aa0c0"), x=0),
        xaxis_title="Price ($)",
        yaxis_title="Path count",
        showlegend=False,
        height=320,
        bargap=0.02,
    )
    return fig


def chart_cone(cone, S0, asset):
    days = cone["days"]
    fig  = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([cone["s2u"], cone["s2d"][::-1]]),
        fill="toself", fillcolor="rgba(55,138,221,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="95% range (2σ)",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([cone["s1u"], cone["s1d"][::-1]]),
        fill="toself", fillcolor="rgba(55,138,221,0.18)",
        line=dict(color="rgba(0,0,0,0)"), name="68% range (1σ)",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=cone["s2u"], mode="lines",
        line=dict(color="rgba(55,138,221,0.25)", width=1, dash="dot"),
        name="2σ upper", hovertemplate="Day %{x}: $%{y:,.0f}<extra>2σ upper</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=cone["s2d"], mode="lines",
        line=dict(color="rgba(55,138,221,0.25)", width=1, dash="dot"),
        name="2σ lower", hovertemplate="Day %{x}: $%{y:,.0f}<extra>2σ lower</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=cone["s1u"], mode="lines",
        line=dict(color="rgba(55,138,221,0.5)", width=1),
        name="1σ upper", hovertemplate="Day %{x}: $%{y:,.0f}<extra>1σ upper</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=cone["s1d"], mode="lines",
        line=dict(color="rgba(55,138,221,0.5)", width=1),
        name="1σ lower", hovertemplate="Day %{x}: $%{y:,.0f}<extra>1σ lower</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=cone["med"], mode="lines",
        line=dict(color="#378ADD", width=2),
        name="Expected path", hovertemplate="Day %{x}: $%{y:,.0f}<extra>Expected</extra>",
    ))
    fig.add_hline(y=S0, line_dash="dot", line_color="rgba(251,146,60,0.5)",
                  line_width=1, annotation_text=f"Spot {asset}",
                  annotation_font_size=9, annotation_font_color="rgba(251,146,60,0.6)")

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Volatility cone — expected price range over time", font=dict(size=13, color="#8aa0c0"), x=0),
        xaxis_title="Days forward",
        yaxis_title="Price ($)",
        yaxis_tickformat="$,.0f",
        height=300,
    )
    return fig


def chart_path_sample(S0, sigma, mu, T_years, n_steps, signal_adj, model,
                       lam, mu_j, sig_j, xi, kappa, rho, n_display=80):
    """Show a sample of individual price paths."""
    rng = np.random.default_rng(99)
    dt  = T_years / n_steps
    mu_eff = mu + signal_adj
    times   = np.linspace(0, T_years * 365, n_steps + 1)
    all_paths = np.zeros((n_display, n_steps + 1))
    all_paths[:, 0] = S0
    v = np.full(n_display, sigma**2)

    for step in range(n_steps):
        z1 = rng.standard_normal(n_display)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_display)
        if model in ("heston", "full"):
            theta = sigma**2
            v = np.maximum(v + kappa*(theta-v)*dt + xi*np.sqrt(np.maximum(v,0)*dt)*z2, 1e-8)
            sig_eff = np.sqrt(v)
        else:
            sig_eff = sigma
        drift = (mu_eff - 0.5 * sig_eff**2) * dt
        diff  = sig_eff * np.sqrt(dt) * z1
        all_paths[:, step+1] = all_paths[:, step] * np.exp(drift + diff)
        if model in ("jump", "full"):
            mask = rng.random(n_display) < lam * dt
            if mask.any():
                jump_sizes = rng.normal(mu_j, sig_j, mask.sum())
                all_paths[mask, step+1] *= np.exp(jump_sizes)

    fig = go.Figure()
    final_prices = all_paths[:, -1]
    for i in range(n_display):
        color = "rgba(30,158,117,0.12)" if final_prices[i] >= S0 else "rgba(244,63,94,0.12)"
        fig.add_trace(go.Scatter(
            x=times, y=all_paths[i],
            mode="lines", line=dict(color=color, width=0.8),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_hline(y=S0, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1)
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"{n_display} simulated price paths", font=dict(size=13, color="#8aa0c0"), x=0),
        xaxis_title="Days forward",
        yaxis_title="Price ($)",
        yaxis_tickformat="$,.0f",
        height=280,
    )
    return fig


def chart_return_distribution(paths, S0):
    """Log-return distribution vs fitted normal."""
    log_returns = np.log(paths / S0)
    mu_r, std_r = scipy_stats.norm.fit(log_returns)
    x_range = np.linspace(log_returns.min(), log_returns.max(), 300)
    normal_pdf = scipy_stats.norm.pdf(x_range, mu_r, std_r)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=log_returns, nbinsx=60, name="Simulated",
        histnorm="probability density",
        marker_color="rgba(99,179,255,0.4)",
        marker_line_width=0,
        hovertemplate="Return: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_pdf, mode="lines",
        name="Normal fit", line=dict(color="rgba(251,146,60,0.8)", width=2, dash="dot"),
        hovertemplate="Return: %{x:.3f}<br>Normal PDF: %{y:.4f}<extra></extra>",
    ))
    for pct, label, color in [
        (np.percentile(log_returns, 1),  "1%",  "rgba(244,63,94,0.7)"),
        (np.percentile(log_returns, 5),  "5%",  "rgba(244,63,94,0.5)"),
        (np.percentile(log_returns, 99), "99%", "rgba(30,158,117,0.7)"),
    ]:
        fig.add_vline(x=pct, line_dash="dot", line_color=color, line_width=1,
                      annotation_text=label, annotation_font_size=9,
                      annotation_font_color=color)
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Log-return distribution vs normal", font=dict(size=13, color="#8aa0c0"), x=0),
        xaxis_title="Log return (ln S_T / S_0)",
        yaxis_title="Density",
        height=260,
    )
    return fig


def chart_var_tearsheet(paths, S0):
    """Risk tearsheet — VaR / CVaR bar chart at multiple confidence levels."""
    levels = [0.50, 0.75, 0.90, 0.95, 0.99]
    labels = ["50%", "75%", "90%", "95%", "99%"]
    vars   = [float(np.percentile(paths, (1 - l) * 100)) for l in levels]
    cvars  = []
    sorted_p = np.sort(paths)
    N = len(sorted_p)
    for l in levels:
        n = max(1, int(N * (1 - l)))
        cvars.append(float(np.mean(sorted_p[:n])))
    var_pct  = [(v / S0 - 1) * 100 for v in vars]
    cvar_pct = [(c / S0 - 1) * 100 for c in cvars]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=var_pct,
        name="VaR (loss floor)", marker_color="rgba(244,63,94,0.65)",
        hovertemplate="VaR %{x}: %{y:.1f}% ($%{customdata:,.0f})<extra></extra>",
        customdata=vars,
    ))
    fig.add_trace(go.Bar(
        x=labels, y=cvar_pct,
        name="CVaR (expected shortfall)", marker_color="rgba(244,63,94,0.30)",
        hovertemplate="CVaR %{x}: %{y:.1f}% ($%{customdata:,.0f})<extra></extra>",
        customdata=cvars,
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="VaR & CVaR at multiple confidence levels", font=dict(size=13, color="#8aa0c0"), x=0),
        xaxis_title="Confidence level",
        yaxis_title="Return (%)",
        barmode="group",
        height=260,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

def interpret_signals(funding, ls_ratio, inflow, fg, model, st_dict, S0):
    msgs = []
    if funding > 0.05:
        msgs.append(("amber", "Funding rate is elevated — longs overextended, mean-reversion drag applied to drift."))
    elif funding < -0.01:
        msgs.append(("green", "Negative funding — shorts are paying longs, potential short-squeeze risk."))
    else:
        msgs.append(("blue", "Funding rate is neutral — no significant positioning bias."))

    if ls_ratio > 2.0:
        msgs.append(("red", f"L/S ratio very high ({ls_ratio:.2f}) — crowded longs. Contrarian downside risk."))
    elif ls_ratio > 1.5:
        msgs.append(("amber", f"L/S ratio elevated ({ls_ratio:.2f}) — mild long crowding."))
    elif ls_ratio < 0.7:
        msgs.append(("green", f"L/S ratio low ({ls_ratio:.2f}) — crowded shorts. Squeeze risk to upside."))
    else:
        msgs.append(("blue", f"L/S ratio balanced ({ls_ratio:.2f})."))

    if inflow > 0.3:
        msgs.append(("red", "High exchange inflows — selling pressure. Jump intensity adjusted upward."))
    elif inflow < -0.3:
        msgs.append(("green", "Exchange outflows — accumulation signal. Reduced selling pressure."))

    if fg > 80:
        msgs.append(("red", f"Fear & Greed at {fg:.0f} (Extreme Greed) — historically precedes corrections."))
    elif fg < 25:
        msgs.append(("green", f"Fear & Greed at {fg:.0f} (Extreme Fear) — historically a buying opportunity."))
    elif fg > 65:
        msgs.append(("amber", f"Fear & Greed at {fg:.0f} (Greed zone) — elevated, watch for reversal."))
    else:
        msgs.append(("blue", f"Fear & Greed neutral at {fg:.0f}."))

    skew_note = (
        "Distribution is right-skewed (fat upside tail)." if st_dict["skew"] > 0.3
        else "Distribution is left-skewed (fat downside tail) — crash risk elevated."
        if st_dict["skew"] < -0.3 else "Distribution is roughly symmetric."
    )
    kurt_note = (
        f"Excess kurtosis {st_dict['kurt']:.2f} — significant fat tails vs normal distribution."
        if abs(st_dict["kurt"]) > 1
        else f"Kurtosis near normal ({st_dict['kurt']:.2f})."
    )
    msgs.append(("blue", f"Shape: {skew_note} {kurt_note}"))

    model_notes = {
        "gbm":    "GBM assumes constant volatility and no discontinuous jumps. Underestimates tail risk in crypto.",
        "jump":   "Jump-Diffusion adds Poisson-distributed price shocks. Better for modeling liquidation cascades.",
        "heston": "Heston SV adds stochastic volatility with mean-reversion. Best for options pricing calibration.",
        "full":   "Full Model (Heston + Jumps) — highest fidelity. Recommended for tail risk and options strategies.",
    }
    msgs.append(("blue", f"Model note: {model_notes[model]}"))
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def build_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="logo-row">
          <div class="logo-box">◈</div>
          <span class="logo-text">GLASS<span class="logo-accent">DECK</span></span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-head">Asset</p>', unsafe_allow_html=True)
        asset = st.radio("", ["BTC", "ETH"], horizontal=True, label_visibility="collapsed")

        defaults = {
            "BTC": dict(spot=95000, iv=80, mu=50, lam=3.0, muj=-12, sigj=20, xi=1.20, kap=2.0, rho=-0.70),
            "ETH": dict(spot=3800,  iv=95, mu=40, lam=4.0, muj=-14, sigj=22, xi=1.40, kap=2.5, rho=-0.65),
        }[asset]

        st.markdown('<p class="section-head">Model</p>', unsafe_allow_html=True)
        model = st.selectbox(
            "", ["GBM (baseline)", "Jump-Diffusion", "Heston SV", "Full Model (Heston + Jumps)"],
            label_visibility="collapsed"
        )
        model_key = {"GBM (baseline)": "gbm", "Jump-Diffusion": "jump",
                     "Heston SV": "heston", "Full Model (Heston + Jumps)": "full"}[model]

        st.markdown('<p class="section-head">Market inputs</p>', unsafe_allow_html=True)
        spot     = st.number_input("Spot price ($)", value=defaults["spot"], step=100, format="%d")
        iv_pct   = st.slider("Implied volatility (%, annualized)", 20, 200, defaults["iv"], 1)
        mu_pct   = st.slider("Drift / trend bias (%/yr)", -150, 300, defaults["mu"], 5)
        dte      = st.slider("Forecast horizon (days)", 1, 365, 30, 1)
        n_paths  = st.select_slider("Simulation paths", [1000, 5000, 10000, 50000], value=10000)

        st.markdown('<p class="section-head">Positioning signals</p>', unsafe_allow_html=True)
        funding  = st.slider("Funding rate (%/8h)", -0.10, 0.20, 0.010, 0.001, format="%.3f")
        ls_ratio = st.slider("Long/Short ratio", 0.40, 3.00, 1.40, 0.05)
        inflow   = st.slider("Exchange inflow pressure (−1 outflow → +1 inflow)", -1.0, 1.0, 0.0, 0.05)
        fg       = st.slider("Fear & Greed index (0–100)", 0, 100, 55, 1)

        show_jump = model_key in ("jump", "full")
        show_sv   = model_key in ("heston", "full")

        lam, muj, sigj = defaults["lam"], defaults["muj"] / 100, defaults["sigj"] / 100
        xi, kap, rho   = defaults["xi"], defaults["kap"], defaults["rho"]

        if show_jump:
            st.markdown('<p class="section-head">Jump parameters</p>', unsafe_allow_html=True)
            lam  = st.slider("Jump intensity λ (events/year)", 0.0, 20.0, defaults["lam"], 0.5)
            muj  = st.slider("Avg jump size μ_J (%)", -50, 30, defaults["muj"], 1) / 100
            sigj = st.slider("Jump volatility σ_J (%)", 1, 80, defaults["sigj"], 1) / 100

        if show_sv:
            st.markdown('<p class="section-head">Heston parameters</p>', unsafe_allow_html=True)
            xi  = st.slider("Vol-of-vol ξ", 0.10, 3.00, defaults["xi"], 0.05)
            kap = st.slider("Mean reversion κ", 0.10, 10.00, defaults["kap"], 0.10)
            rho = st.slider("Spot-vol correlation ρ", -0.99, 0.99, defaults["rho"], 0.01)

        run = st.button("▶  Run simulation", use_container_width=True)

    return dict(
        asset=asset, model=model_key, spot=spot,
        sigma=iv_pct / 100, mu=mu_pct / 100, dte=dte, n_paths=n_paths,
        funding=funding, ls_ratio=ls_ratio, inflow=inflow, fg=fg,
        lam=lam, muj=muj, sigj=sigj, xi=xi, kap=kap, rho=rho, run=run,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = build_sidebar()

    # ── Header ──────────────────────────────────────────────────────────────
    col_title, col_badge = st.columns([5, 1])
    with col_title:
        st.markdown(
            f"## {p['asset']} Price Distribution  "
            f"<span style='font-size:14px;color:#2a3850;font-weight:400'>"
            f"Monte Carlo · {p['model'].upper()} · {p['n_paths']:,} paths · {p['dte']}d horizon"
            f"</span>",
            unsafe_allow_html=True,
        )
    with col_badge:
        model_labels = {
            "gbm": ("GBM", "badge-blue"),
            "jump": ("Jump-Diff", "badge-amber"),
            "heston": ("Heston SV", "badge-amber"),
            "full": ("Full Model", "badge-green"),
        }
        lbl, cls = model_labels[p["model"]]
        st.markdown(f'<br><span class="badge {cls}">{lbl}</span>', unsafe_allow_html=True)

    if not p["run"] and "last_paths" not in st.session_state:
        st.info(
            "Set your parameters in the sidebar and click **▶ Run simulation** to generate the price distribution.",
            icon="📊",
        )
        st.markdown('<p class="section-head" style="margin-top:1.5rem">Quick reference</p>', unsafe_allow_html=True)
        ref_cols = st.columns(3)
        refs = [
            ("GBM", "Constant vol, no jumps. Fast baseline. Underestimates crypto tail risk."),
            ("Jump-Diffusion", "Adds Poisson crash/spike events. Better for liquidation cascades."),
            ("Heston SV", "Stochastic volatility. Best for options chain calibration."),
            ("Full Model", "Heston + Jumps. Highest fidelity. Best for risk management."),
            ("Signal adj.", "Funding, L/S ratio, exchange inflows and Fear & Greed shift the drift."),
            ("CVaR 5%", "Average loss in the worst 5% of outcomes — the tail risk you actually care about."),
        ]
        for i, (t, d) in enumerate(refs):
            with ref_cols[i % 3]:
                st.markdown(
                    f'<div style="background:rgba(10,16,28,0.6);border:1px solid rgba(99,179,255,0.07);border-radius:10px;padding:12px 14px;margin-bottom:10px;">'
                    f'<div style="font-size:12px;font-weight:600;color:#4a6080;margin-bottom:5px;">{t}</div>'
                    f'<div style="font-size:12px;color:#2a3850;line-height:1.5;">{d}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        return

    # ── Run simulation ───────────────────────────────────────────────────────
    if p["run"]:
        signal_adj = compute_signal_adjustment(p["funding"], p["ls_ratio"], p["inflow"], p["fg"])
        n_steps    = max(int(p["dte"]), 1)
        with st.spinner("Running Monte Carlo simulation..."):
            paths = run_montecarlo(
                S0=p["spot"], sigma=p["sigma"], mu=p["mu"],
                T_years=p["dte"] / 365, n_paths=p["n_paths"], n_steps=n_steps,
                model=p["model"],
                lam=p["lam"], mu_j=p["muj"], sig_j=p["sigj"],
                xi=p["xi"], kappa=p["kap"], rho=p["rho"],
                signal_adj=signal_adj,
            )
        st.session_state["last_paths"]  = paths
        st.session_state["last_params"] = p
        st.session_state["signal_adj"]  = signal_adj
    else:
        paths      = st.session_state["last_paths"]
        p          = st.session_state["last_params"]
        signal_adj = st.session_state["signal_adj"]

    st_dict = compute_stats(paths, p["spot"])
    cone    = volatility_cone(p["spot"], p["sigma"], p["mu"] + signal_adj, max_days=p["dte"] + 1)

    fm = lambda v: f"${v:,.0f}"
    pct_chg = lambda v: f"{(v / p['spot'] - 1) * 100:+.1f}%"
    col_green = "#1d9e75"
    col_red   = "#f43f5e"

    # ── Metric cards ─────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Mean price", fm(st_dict["mean"]),
                  delta=pct_chg(st_dict["mean"]))
    with m2:
        st.metric("Median price", fm(st_dict["median"]),
                  delta=pct_chg(st_dict["median"]))
    with m3:
        st.metric("95th percentile", fm(st_dict["p95"]),
                  delta=pct_chg(st_dict["p95"]))
    with m4:
        st.metric("VaR 5% (floor)", fm(st_dict["p05"]),
                  delta=pct_chg(st_dict["p05"]))
    with m5:
        st.metric("CVaR 5% (tail)", fm(st_dict["cvar5"]),
                  delta=pct_chg(st_dict["cvar5"]))
    with m6:
        up_prob = st_dict["prob_up"] * 100
        st.metric("P(above spot)", f"{up_prob:.1f}%",
                  delta=f"{'Bullish' if up_prob > 50 else 'Bearish'} bias")

    st.markdown("---")

    # ── Distribution + paths ─────────────────────────────────────────────────
    chart_col, paths_col = st.columns([3, 2])
    with chart_col:
        st.plotly_chart(
            chart_distribution(paths, p["spot"], p["asset"], p["dte"]),
            use_container_width=True, config={"displayModeBar": False},
        )
    with paths_col:
        n_steps = max(int(p["dte"]), 1)
        st.plotly_chart(
            chart_path_sample(
                p["spot"], p["sigma"], p["mu"], p["dte"] / 365,
                n_steps, signal_adj, p["model"],
                p["lam"], p["muj"], p["sigj"], p["xi"], p["kap"], p["rho"], n_display=80,
            ),
            use_container_width=True, config={"displayModeBar": False},
        )

    # ── Volatility cone + return dist ────────────────────────────────────────
    cone_col, ret_col = st.columns([3, 2])
    with cone_col:
        st.plotly_chart(chart_cone(cone, p["spot"], p["asset"]),
                        use_container_width=True, config={"displayModeBar": False})
    with ret_col:
        st.plotly_chart(chart_return_distribution(paths, p["spot"]),
                        use_container_width=True, config={"displayModeBar": False})

    # ── VaR tearsheet ────────────────────────────────────────────────────────
    st.plotly_chart(chart_var_tearsheet(paths, p["spot"]),
                    use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # ── Probability table + distribution stats ───────────────────────────────
    prob_col, stat_col = st.columns(2)

    with prob_col:
        st.markdown('<p class="section-head">Probability targets</p>', unsafe_allow_html=True)
        probs = [
            (f"P(price > spot)", f"{st_dict['prob_up']*100:.1f}%",
             col_green if st_dict["prob_up"] > 0.5 else col_red),
            (f"P(> +10%) — ${p['spot']*1.1:,.0f}", f"{st_dict['prob_up10']*100:.1f}%", col_green),
            (f"P(> +20%) — ${p['spot']*1.2:,.0f}", f"{st_dict['prob_up20']*100:.1f}%", col_green),
            (f"P(> +50%) — ${p['spot']*1.5:,.0f}", f"{st_dict['prob_up50']*100:.1f}%", col_green),
            (f"P(2× = ${p['spot']*2:,.0f})", f"{st_dict['prob_up2x']*100:.1f}%", col_green),
            ("", "", ""),
            (f"P(< −10%) — ${p['spot']*0.9:,.0f}", f"{st_dict['prob_dn10']*100:.1f}%", col_red),
            (f"P(< −20%) — ${p['spot']*0.8:,.0f}", f"{st_dict['prob_dn20']*100:.1f}%", col_red),
            (f"P(< −40%) — ${p['spot']*0.6:,.0f}", f"{st_dict['prob_dn40']*100:.1f}%", col_red),
            (f"P(stays ±10%)", f"{st_dict['prob_flat']*100:.1f}%", "#63b3ff"),
        ]
        for label, val, color in probs:
            if not label:
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                continue
            st.markdown(
                f'<div class="prob-row">'
                f'<span class="prob-label">{label}</span>'
                f'<span style="font-weight:600;color:{color};font-size:13px">{val}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with stat_col:
        st.markdown('<p class="section-head">Distribution statistics</p>', unsafe_allow_html=True)
        stats_display = [
            ("Mean",            fm(st_dict["mean"])),
            ("Median",          fm(st_dict["median"])),
            ("Std deviation",   fm(st_dict["std"])),
            ("Skewness",        f"{st_dict['skew']:.3f}"),
            ("Excess kurtosis", f"{st_dict['kurt']:.3f}"),
            ("P01 / P99",       f"{fm(st_dict['p01'])} / {fm(st_dict['p99'])}"),
            ("P05 / P95",       f"{fm(st_dict['p05'])} / {fm(st_dict['p95'])}"),
            ("P10 / P90",       f"{fm(st_dict['p10'])} / {fm(st_dict['p90'])}"),
            ("P25 / P75",       f"{fm(st_dict['p25'])} / {fm(st_dict['p75'])}"),
            ("VaR 5%",          f"{fm(st_dict['var5'])} ({pct_chg(st_dict['var5'])})"),
            ("CVaR 5%",         f"{fm(st_dict['cvar5'])} ({pct_chg(st_dict['cvar5'])})"),
            ("VaR 1%",          f"{fm(st_dict['var1'])} ({pct_chg(st_dict['var1'])})"),
            ("CVaR 1%",         f"{fm(st_dict['cvar1'])} ({pct_chg(st_dict['cvar1'])})"),
            ("Signal adj. drift", f"{signal_adj*100:+.2f}%/yr"),
        ]
        for label, val in stats_display:
            st.markdown(
                f'<div class="prob-row">'
                f'<span class="prob-label">{label}</span>'
                f'<span style="font-size:12px;font-weight:500;color:#8aa0c0">{val}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Signal interpretation ─────────────────────────────────────────────────
    st.markdown('<p class="section-head">Model interpretation & signal read</p>', unsafe_allow_html=True)
    msgs = interpret_signals(p["funding"], p["ls_ratio"], p["inflow"], p["fg"], p["model"], st_dict, p["spot"])
    badge_map = {
        "green": "badge-green", "red": "badge-red",
        "amber": "badge-amber", "blue": "badge-blue",
    }
    interp_html = "<br>".join(
        f'<span class="badge {badge_map[color]}">{color.upper()}</span> {msg}'
        for color, msg in msgs
    )
    st.markdown(f'<div class="interp-box">{interp_html}</div>', unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-head">Export</p>', unsafe_allow_html=True)
    export_col1, export_col2, _ = st.columns([1, 1, 3])

    df_paths = pd.DataFrame({"terminal_price": paths})
    with export_col1:
        st.download_button(
            "⬇ Download paths (CSV)",
            df_paths.to_csv(index=False).encode(),
            file_name=f"{p['asset']}_paths_{p['dte']}d.csv",
            mime="text/csv",
            use_container_width=True,
        )

    df_stats = pd.DataFrame([{
        "asset": p["asset"], "model": p["model"], "spot": p["spot"],
        "iv_pct": p["sigma"]*100, "dte": p["dte"], "n_paths": p["n_paths"],
        **{k: round(v, 4) for k, v in st_dict.items()},
    }])
    with export_col2:
        st.download_button(
            "⬇ Download stats (CSV)",
            df_stats.to_csv(index=False).encode(),
            file_name=f"{p['asset']}_stats_{p['dte']}d.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown(
        '<p style="font-size:10px;color:#1a2a3a;margin-top:24px;">'
        'GlassDeck · Quantitative price distribution modeling · '
        'Not financial advice · Results are probabilistic, not predictive.'
        '</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
