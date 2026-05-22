"""
GlassDeck — Crypto Price Distribution Calculator
Clean white + blue professional theme
Models: GBM · Jump-Diffusion · Heston SV · Full Model
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GlassDeck · Price Distribution",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME — clean white + blue accent
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    .stApp { background-color: #f8fafc; color: #1e293b; }
    .main .block-container { padding-top: 1.75rem; padding-bottom: 2.5rem; max-width: 1400px; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
        box-shadow: 2px 0 12px rgba(15,23,42,0.05);
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #94a3b8; font-size: 10px;
        letter-spacing: 0.1em; font-weight: 600;
    }

    /* Typography */
    h1, h2, h3 { color: #0f172a !important; font-weight: 700 !important; letter-spacing: -0.025em; }
    h1 { font-size: 1.6rem !important; }
    p { color: #475569; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 4px rgba(15,23,42,0.06);
    }
    [data-testid="stMetric"] label {
        color: #94a3b8 !important; font-size: 10px !important;
        letter-spacing: 0.09em; text-transform: uppercase; font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #0f172a !important; font-size: 1.35rem !important; font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] { font-size: 12px !important; font-weight: 500 !important; }
    [data-testid="stMetricDelta"] svg { display: none; }

    /* Sliders */
    .stSlider [data-baseweb="slider"] { padding: 0.4rem 0; }
    [data-testid="stSlider"] label { color: #475569 !important; font-size: 12px !important; font-weight: 500; }

    /* Select / Radio */
    .stSelectbox label, .stRadio label {
        color: #64748b !important; font-size: 10px !important;
        letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600;
    }
    .stSelectbox [data-baseweb="select"] > div {
        background: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    .stRadio [data-baseweb="radio"] { gap: 10px; }

    /* Number input */
    [data-testid="stNumberInput"] input {
        background: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #0f172a !important;
        font-weight: 600;
    }

    /* Section dividers */
    .section-head {
        font-size: 10px; font-weight: 700; color: #94a3b8;
        letter-spacing: 0.14em; text-transform: uppercase;
        border-bottom: 1.5px solid #e2e8f0;
        padding-bottom: 7px; margin-bottom: 14px; margin-top: 12px;
    }

    /* Logo */
    .logo-row { display: flex; align-items: center; gap: 10px; margin-bottom: 24px; padding-bottom: 20px; border-bottom: 1px solid #f1f5f9; }
    .logo-box {
        width: 34px; height: 34px; border-radius: 9px;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        display: flex; align-items: center; justify-content: center;
        font-size: 17px; color: #fff;
        box-shadow: 0 2px 8px rgba(59,130,246,0.35);
    }
    .logo-text  { font-size: 17px; font-weight: 800; color: #0f172a; letter-spacing: -0.03em; line-height: 1.2; }
    .logo-accent { color: #3b82f6; }
    .logo-sub   { font-size: 9px; color: #94a3b8; letter-spacing: 0.12em; text-transform: uppercase; }

    /* Badges */
    .badge { display: inline-block; font-size: 10px; padding: 3px 10px; border-radius: 6px; font-weight: 600; letter-spacing: 0.04em; }
    .badge-green { background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }
    .badge-red   { background: #fff1f2; color: #be123c; border: 1px solid #fecdd3; }
    .badge-blue  { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
    .badge-amber { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }

    /* Model badge in header */
    .model-badge {
        display: inline-block; font-size: 11px; padding: 4px 14px;
        border-radius: 20px; font-weight: 600; letter-spacing: 0.05em;
        background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe;
    }

    /* Interpretation box */
    .interp-box {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px; font-size: 13px;
        line-height: 1.8; color: #475569;
    }
    .interp-box b { color: #1e293b; }

    /* Probability rows */
    .prob-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 9px 14px; border-radius: 8px; margin-bottom: 5px;
        background: #f8fafc; border: 1px solid #e2e8f0; font-size: 12px;
        transition: background 0.12s, border-color 0.12s;
    }
    .prob-row:hover { background: #eff6ff; border-color: #bfdbfe; }
    .prob-label { color: #64748b; font-weight: 500; }

    /* Stat rows */
    .stat-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 14px; border-radius: 8px; margin-bottom: 5px;
        background: #f8fafc; border: 1px solid #e2e8f0; font-size: 12px;
    }
    .stat-label { color: #64748b; font-weight: 500; }
    .stat-val   { color: #1e293b; font-weight: 600; font-size: 12px; }

    /* Quick-ref cards */
    .ref-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.05);
    }
    .ref-card-title { font-size: 12px; font-weight: 700; color: #1e293b; margin-bottom: 5px; }
    .ref-card-body  { font-size: 12px; color: #64748b; line-height: 1.55; }

    /* Run button */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        border: none !important; color: #ffffff !important;
        font-weight: 600 !important; border-radius: 9px !important;
        font-size: 13px !important; letter-spacing: 0.02em !important;
        padding: 0.6rem 1.25rem !important;
        box-shadow: 0 2px 8px rgba(59,130,246,0.30) !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        box-shadow: 0 4px 16px rgba(59,130,246,0.40) !important;
        transform: translateY(-1px) !important;
    }

    /* Horizontal rule */
    hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }

    /* Info box */
    [data-testid="stInfo"] {
        background: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 10px !important;
        color: #1e40af !important;
    }

    /* Download buttons */
    [data-testid="stDownloadButton"] button {
        background: #ffffff !important; border: 1px solid #cbd5e1 !important;
        color: #475569 !important; border-radius: 8px !important; font-weight: 500 !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: #eff6ff !important; border-color: #93c5fd !important; color: #1d4ed8 !important;
    }

    /* Chart container cards */
    .chart-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 14px; padding: 4px 4px 0;
        box-shadow: 0 1px 4px rgba(15,23,42,0.06);
        margin-bottom: 4px;
    }

    /* Header accent bar */
    .header-bar {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 0 12px 12px 0;
        padding: 16px 22px; margin-bottom: 22px;
        box-shadow: 0 1px 4px rgba(15,23,42,0.06);
        display: flex; align-items: center; justify-content: space-between;
    }
    .header-title { font-size: 20px; font-weight: 800; color: #0f172a; letter-spacing: -0.03em; }
    .header-sub   { font-size: 12px; color: #94a3b8; margin-top: 3px; }

    /* Footer */
    .footer { font-size: 10px; color: #cbd5e1; margin-top: 28px; text-align: center; padding-top: 16px; border-top: 1px solid #f1f5f9; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY LIGHT THEME
# ─────────────────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(family="Inter, -apple-system, sans-serif", color="#64748b", size=11),
    margin=dict(l=52, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor="#f1f5f9",
        linecolor="#e2e8f0",
        tickfont=dict(size=10, color="#94a3b8"),
        zerolinecolor="#e2e8f0",
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor="#f1f5f9",
        linecolor="#e2e8f0",
        tickfont=dict(size=10, color="#94a3b8"),
        zerolinecolor="#e2e8f0",
        showgrid=True,
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e2e8f0",
        borderwidth=1,
        font=dict(size=11, color="#64748b"),
    ),
    hoverlabel=dict(
        bgcolor="#ffffff",
        bordercolor="#e2e8f0",
        font=dict(family="Inter, sans-serif", size=12, color="#1e293b"),
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# MATH CORE (unchanged — only theme changes)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_montecarlo(S0, sigma, mu, T_years, n_paths, n_steps, model,
                   lam, mu_j, sig_j, xi, kappa, rho, signal_adj):
    rng    = np.random.default_rng(42)
    dt     = T_years / n_steps
    mu_eff = mu + signal_adj
    paths  = np.full(n_paths, S0, dtype=np.float64)
    v      = np.full(n_paths, sigma**2)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
        if model in ("heston", "full"):
            theta = sigma**2
            v = np.maximum(v + kappa*(theta-v)*dt + xi*np.sqrt(np.maximum(v,0)*dt)*z2, 1e-8)
            sig_eff = np.sqrt(v)
        else:
            sig_eff = sigma
        paths *= np.exp((mu_eff - 0.5*sig_eff**2)*dt + sig_eff*np.sqrt(dt)*z1)
        if model in ("jump", "full"):
            mask = rng.random(n_paths) < lam * dt
            if mask.any():
                paths[mask] *= np.exp(rng.normal(mu_j, sig_j, mask.sum()))
    return paths


def compute_signal_adjustment(funding, ls_ratio, exchange_inflow, fear_greed):
    adj  = -funding * 3 * 365 * 0.30
    adj += -(ls_ratio - 1.0) * 0.15
    adj += -exchange_inflow * 0.25
    adj += (fear_greed - 50) / 50 * 0.08
    return float(adj)


def compute_stats(paths, S0):
    sp = np.sort(paths); N = len(sp)
    mean   = float(np.mean(paths));   median = float(np.median(paths))
    std    = float(np.std(paths));    skew   = float(scipy_stats.skew(paths))
    kurt   = float(scipy_stats.kurtosis(paths, fisher=True))
    pcts   = {f"p{p:02d}": float(np.percentile(paths, p)) for p in [1,5,10,25,75,90,95,99]}
    cvar5  = float(np.mean(sp[:max(1,int(N*0.05))]));  cvar1 = float(np.mean(sp[:max(1,int(N*0.01))]))
    return dict(
        mean=mean, median=median, std=std, skew=skew, kurt=kurt,
        **pcts,
        var5=pcts["p05"], var1=pcts["p01"], cvar5=cvar5, cvar1=cvar1,
        prob_up=float(np.mean(paths>S0)),
        prob_up10=float(np.mean(paths>S0*1.1)),  prob_up20=float(np.mean(paths>S0*1.2)),
        prob_up50=float(np.mean(paths>S0*1.5)),  prob_up2x=float(np.mean(paths>S0*2.0)),
        prob_dn10=float(np.mean(paths<S0*0.9)),  prob_dn20=float(np.mean(paths<S0*0.8)),
        prob_dn40=float(np.mean(paths<S0*0.6)),
        prob_flat=float(np.mean((paths>S0*0.9)&(paths<S0*1.1))),
    )


def volatility_cone(S0, sigma, mu, max_days=365):
    days = np.array([1,3,7,14,21,30,45,60,90,120,180,270,365])
    days = days[days <= max_days+1]; T = days/365
    d    = (mu - sigma**2/2)*T
    sq   = sigma*np.sqrt(T)
    return dict(days=days,
        med=S0*np.exp(d), s1u=S0*np.exp(d+sq),   s1d=S0*np.exp(d-sq),
        s2u=S0*np.exp(d+2*sq), s2d=S0*np.exp(d-2*sq))


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS — light palette
# ─────────────────────────────────────────────────────────────────────────────

BLUE       = "#3b82f6"
BLUE_LIGHT = "#93c5fd"
BLUE_FILL  = "rgba(59,130,246,0.08)"
BLUE_FILL2 = "rgba(59,130,246,0.18)"
GREEN      = "#16a34a"
GREEN_BAR  = "rgba(22,163,74,0.70)"
RED        = "#dc2626"
RED_BAR    = "rgba(220,38,38,0.65)"
AMBER      = "#d97706"
PURPLE     = "#7c3aed"
SLATE      = "#64748b"


def chart_distribution(paths, S0, asset, dte):
    hist, edges = np.histogram(paths, bins=80)
    centers = 0.5*(edges[:-1]+edges[1:])
    colors  = [RED_BAR if c < S0 else GREEN_BAR for c in centers]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centers, y=hist, marker_color=colors, marker_line_width=0,
        name="Paths", hovertemplate="$%{x:,.0f}<br>%{y} paths<extra></extra>",
    ))
    for val, lbl, col in [
        (np.percentile(paths,5),  "VaR 5%", RED),
        (np.percentile(paths,25), "P25",    AMBER),
        (np.median(paths),        "Median", BLUE),
        (np.mean(paths),          "Mean",   PURPLE),
        (np.percentile(paths,75), "P75",    GREEN),
        (np.percentile(paths,95), "P95",    "#059669"),
    ]:
        fig.add_vline(x=val, line_dash="dot", line_color=col, line_width=1.5,
                      annotation_text=lbl, annotation_font_size=9,
                      annotation_font_color=col, annotation_position="top")
    fig.add_vline(x=S0, line_dash="solid", line_color="#94a3b8", line_width=2,
                  annotation_text=f"Spot {asset}",
                  annotation_font_size=9, annotation_font_color="#64748b")
    fig.update_layout(**CHART_LAYOUT,
        title=dict(text=f"{asset} simulated price distribution — {dte}d", font=dict(size=13,color="#1e293b",weight=600), x=0),
        xaxis_title="Price ($)", yaxis_title="Path count",
        showlegend=False, height=320, bargap=0.02)
    return fig


def chart_cone(cone, S0, asset):
    days = cone["days"]
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([cone["s2u"], cone["s2d"][::-1]]),
        fill="toself", fillcolor="rgba(59,130,246,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="95% range (2σ)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([cone["s1u"], cone["s1d"][::-1]]),
        fill="toself", fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="68% range (1σ)", hoverinfo="skip"))
    for key, col, dash, w, nm in [
        ("s2u", BLUE_LIGHT, "dot", 1, "2σ upper"),
        ("s2d", BLUE_LIGHT, "dot", 1, "2σ lower"),
        ("s1u", BLUE,       "solid",1.2,"1σ upper"),
        ("s1d", BLUE,       "solid",1.2,"1σ lower"),
        ("med", "#1d4ed8",  "solid",2.2,"Expected"),
    ]:
        fig.add_trace(go.Scatter(
            x=days, y=cone[key], mode="lines",
            line=dict(color=col, width=w, dash=dash), name=nm,
            hovertemplate=f"Day %{{x}}: $%{{y:,.0f}}<extra>{nm}</extra>"))
    fig.add_hline(y=S0, line_dash="dot", line_color="#94a3b8", line_width=1.5,
                  annotation_text=f"Spot {asset}", annotation_font_size=9,
                  annotation_font_color="#94a3b8")
    fig.update_layout(**CHART_LAYOUT,
        title=dict(text="Volatility cone — expected price range", font=dict(size=13,color="#1e293b"), x=0),
        xaxis_title="Days forward", yaxis_title="Price ($)",
        yaxis_tickformat="$,.0f", height=300)
    return fig


def chart_paths(S0, sigma, mu, T_years, n_steps, signal_adj, model,
                lam, mu_j, sig_j, xi, kappa, rho, n_display=80):
    rng    = np.random.default_rng(99)
    dt     = T_years / n_steps
    mu_eff = mu + signal_adj
    times  = np.linspace(0, T_years*365, n_steps+1)
    paths  = np.zeros((n_display, n_steps+1)); paths[:,0] = S0
    v      = np.full(n_display, sigma**2)
    for s in range(n_steps):
        z1 = rng.standard_normal(n_display)
        z2 = rho*z1 + np.sqrt(1-rho**2)*rng.standard_normal(n_display)
        if model in ("heston","full"):
            v = np.maximum(v + kappa*(sigma**2-v)*dt + xi*np.sqrt(np.maximum(v,0)*dt)*z2, 1e-8)
            se = np.sqrt(v)
        else: se = sigma
        paths[:,s+1] = paths[:,s]*np.exp((mu_eff-0.5*se**2)*dt + se*np.sqrt(dt)*z1)
        if model in ("jump","full"):
            mask = rng.random(n_display)<lam*dt
            if mask.any(): paths[mask,s+1] *= np.exp(rng.normal(mu_j,sig_j,mask.sum()))
    fig = go.Figure()
    finals = paths[:,-1]
    for i in range(n_display):
        col = "rgba(22,163,74,0.10)" if finals[i]>=S0 else "rgba(220,38,38,0.10)"
        fig.add_trace(go.Scatter(x=times, y=paths[i], mode="lines",
            line=dict(color=col, width=0.8), showlegend=False, hoverinfo="skip"))
    fig.add_hline(y=S0, line_dash="dot", line_color="#94a3b8", line_width=1.5)
    fig.update_layout(**CHART_LAYOUT,
        title=dict(text=f"{n_display} simulated paths", font=dict(size=13,color="#1e293b"), x=0),
        xaxis_title="Days forward", yaxis_title="Price ($)",
        yaxis_tickformat="$,.0f", height=280)
    return fig


def chart_returns(paths, S0):
    lr   = np.log(paths/S0)
    mu_r, std_r = scipy_stats.norm.fit(lr)
    x    = np.linspace(lr.min(), lr.max(), 300)
    fig  = go.Figure()
    fig.add_trace(go.Histogram(
        x=lr, nbinsx=60, histnorm="probability density",
        name="Simulated", marker_color="rgba(59,130,246,0.45)",
        marker_line_color="#3b82f6", marker_line_width=0.5,
        hovertemplate="Return: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=scipy_stats.norm.pdf(x, mu_r, std_r), mode="lines",
        name="Normal fit", line=dict(color=AMBER, width=2, dash="dot"),
        hovertemplate="Return: %{x:.3f}<br>Normal: %{y:.4f}<extra></extra>"))
    for p, lbl, col in [(1,"1%",RED),(5,"5%","#f97316"),(99,"99%",GREEN)]:
        pv = np.percentile(lr, p)
        fig.add_vline(x=pv, line_dash="dot", line_color=col, line_width=1.2,
                      annotation_text=lbl, annotation_font_size=9, annotation_font_color=col)
    fig.update_layout(**CHART_LAYOUT,
        title=dict(text="Log-return distribution vs normal", font=dict(size=13,color="#1e293b"), x=0),
        xaxis_title="Log return", yaxis_title="Density", height=265)
    return fig


def chart_var(paths, S0):
    levels = [0.50,0.75,0.90,0.95,0.99]
    labels = ["50%","75%","90%","95%","99%"]
    sp = np.sort(paths); N = len(sp)
    vars_  = [float(np.percentile(paths,(1-l)*100)) for l in levels]
    cvars  = [float(np.mean(sp[:max(1,int(N*(1-l)))])) for l in levels]
    vp     = [(v/S0-1)*100 for v in vars_]
    cp     = [(c/S0-1)*100 for c in cvars]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=vp, name="VaR",
        marker_color="rgba(220,38,38,0.70)",
        hovertemplate="VaR %{x}: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=cp, name="CVaR",
        marker_color="rgba(220,38,38,0.30)",
        hovertemplate="CVaR %{x}: %{y:.1f}%<extra></extra>"))
    fig.update_layout(**CHART_LAYOUT,
        title=dict(text="VaR & CVaR across confidence levels", font=dict(size=13,color="#1e293b"), x=0),
        xaxis_title="Confidence level", yaxis_title="Return (%)",
        barmode="group", height=265)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

def interpret_signals(funding, ls_ratio, inflow, fg, model, st_d, S0):
    msgs = []
    if funding > 0.05:
        msgs.append(("amber","Funding elevated — long crowding detected, mean-reversion drag applied."))
    elif funding < -0.01:
        msgs.append(("green","Negative funding — shorts dominant, potential short-squeeze risk."))
    else:
        msgs.append(("blue","Funding rate neutral — no significant positioning bias detected."))

    if ls_ratio > 2.0:
        msgs.append(("red",f"L/S ratio very high ({ls_ratio:.2f}) — crowded longs. Contrarian downside risk elevated."))
    elif ls_ratio > 1.5:
        msgs.append(("amber",f"L/S ratio elevated ({ls_ratio:.2f}) — mild long crowding."))
    elif ls_ratio < 0.7:
        msgs.append(("green",f"L/S ratio low ({ls_ratio:.2f}) — crowded shorts, squeeze risk to upside."))
    else:
        msgs.append(("blue",f"L/S ratio balanced ({ls_ratio:.2f}) — no significant crowding."))

    if inflow > 0.3:
        msgs.append(("red","Exchange inflows elevated — selling pressure expected near-term."))
    elif inflow < -0.3:
        msgs.append(("green","Exchange outflows — accumulation signal, reduced selling pressure."))

    if fg > 80:
        msgs.append(("red",f"Fear & Greed {fg:.0f} (Extreme Greed) — historically precedes corrections."))
    elif fg < 25:
        msgs.append(("green",f"Fear & Greed {fg:.0f} (Extreme Fear) — historically a buying opportunity."))
    elif fg > 65:
        msgs.append(("amber",f"Fear & Greed {fg:.0f} (Greed) — elevated, watch for reversal."))
    else:
        msgs.append(("blue",f"Fear & Greed neutral at {fg:.0f}."))

    skew_note = ("Right-skewed — fat upside tail." if st_d["skew"]>0.3
                 else "Left-skewed — fat downside tail, crash risk elevated." if st_d["skew"]<-0.3
                 else "Roughly symmetric distribution.")
    msgs.append(("blue",f"Shape: {skew_note} Excess kurtosis {st_d['kurt']:.2f} "
                 f"({'fat tails vs normal' if abs(st_d['kurt'])>1 else 'near-normal tails'})."))

    model_notes = {
        "gbm":    "GBM: constant volatility baseline. Underestimates crypto tail risk.",
        "jump":   "Jump-Diffusion: adds crash/spike events — better for liquidation scenarios.",
        "heston": "Heston SV: stochastic volatility — best for options chain calibration.",
        "full":   "Full Model (Heston + Jumps): highest fidelity — recommended for risk management.",
    }
    msgs.append(("blue", model_notes[model]))
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def build_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="logo-row">
          <div class="logo-box">📊</div>
          <div>
            <div class="logo-text">GLASS<span class="logo-accent">DECK</span></div>
            <div class="logo-sub">Price Distribution</div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<p class="section-head">Asset</p>', unsafe_allow_html=True)
        asset = st.radio("", ["BTC","ETH"], horizontal=True, label_visibility="collapsed")
        defs  = {"BTC": dict(spot=95000,iv=80,mu=50,lam=3.0,muj=-12,sigj=20,xi=1.20,kap=2.0,rho=-0.70),
                 "ETH": dict(spot=3800, iv=95,mu=40,lam=4.0,muj=-14,sigj=22,xi=1.40,kap=2.5,rho=-0.65)}[asset]

        st.markdown('<p class="section-head">Model</p>', unsafe_allow_html=True)
        model = st.selectbox("", ["GBM (baseline)","Jump-Diffusion","Heston SV","Full Model (Heston + Jumps)"],
                             label_visibility="collapsed")
        mkey  = {"GBM (baseline)":"gbm","Jump-Diffusion":"jump",
                 "Heston SV":"heston","Full Model (Heston + Jumps)":"full"}[model]

        st.markdown('<p class="section-head">Market inputs</p>', unsafe_allow_html=True)
        spot   = st.number_input("Spot price ($)", value=defs["spot"], step=100, format="%d")
        iv_pct = st.slider("Implied volatility (%, annualized)", 20, 200, defs["iv"], 1)
        mu_pct = st.slider("Drift / trend bias (%/yr)", -150, 300, defs["mu"], 5)
        dte    = st.slider("Forecast horizon (days)", 1, 365, 30, 1)
        n_paths= st.select_slider("Simulation paths", [1000,5000,10000,50000], value=10000)

        st.markdown('<p class="section-head">Positioning signals</p>', unsafe_allow_html=True)
        funding= st.slider("Funding rate (%/8h)", -0.10, 0.20, 0.010, 0.001, format="%.3f")
        ls     = st.slider("Long/Short ratio", 0.40, 3.00, 1.40, 0.05)
        inflow = st.slider("Exchange inflow (−1 outflow → +1 inflow)", -1.0, 1.0, 0.0, 0.05)
        fg     = st.slider("Fear & Greed index", 0, 100, 55, 1)

        lam,muj,sigj = defs["lam"], defs["muj"]/100, defs["sigj"]/100
        xi,kap,rho   = defs["xi"], defs["kap"], defs["rho"]

        if mkey in ("jump","full"):
            st.markdown('<p class="section-head">Jump parameters</p>', unsafe_allow_html=True)
            lam  = st.slider("Jump intensity λ (events/yr)", 0.0, 20.0, defs["lam"], 0.5)
            muj  = st.slider("Avg jump size μ_J (%)", -50, 30, defs["muj"], 1) / 100
            sigj = st.slider("Jump volatility σ_J (%)", 1, 80, defs["sigj"], 1) / 100

        if mkey in ("heston","full"):
            st.markdown('<p class="section-head">Heston parameters</p>', unsafe_allow_html=True)
            xi  = st.slider("Vol-of-vol ξ", 0.10, 3.00, defs["xi"], 0.05)
            kap = st.slider("Mean reversion κ", 0.10, 10.00, defs["kap"], 0.10)
            rho = st.slider("Spot-vol correlation ρ", -0.99, 0.99, defs["rho"], 0.01)

        st.markdown("---")
        run = st.button("▶  Run simulation", use_container_width=True)

    return dict(asset=asset, model=mkey, spot=spot,
                sigma=iv_pct/100, mu=mu_pct/100, dte=dte, n_paths=n_paths,
                funding=funding, ls_ratio=ls, inflow=inflow, fg=fg,
                lam=lam, muj=muj, sigj=sigj, xi=xi, kap=kap, rho=rho, run=run)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = build_sidebar()

    # Header bar
    model_labels = {"gbm":"GBM","jump":"Jump-Diffusion","heston":"Heston SV","full":"Full Model"}
    st.markdown(f"""
    <div class="header-bar">
      <div>
        <div class="header-title">{p['asset']} Price Distribution</div>
        <div class="header-sub">Monte Carlo &nbsp;·&nbsp; {model_labels[p['model']]} &nbsp;·&nbsp;
          {p['n_paths']:,} paths &nbsp;·&nbsp; {p['dte']}d horizon</div>
      </div>
      <span class="model-badge">{model_labels[p['model']]}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Empty state ──────────────────────────────────────────────────────────
    if not p["run"] and "last_paths" not in st.session_state:
        st.info("Set your parameters in the sidebar and click **▶ Run simulation** to generate the price distribution.", icon="📊")
        st.markdown('<p class="section-head" style="margin-top:1.5rem">Quick reference</p>', unsafe_allow_html=True)
        refs = [
            ("GBM","Constant volatility baseline. Fast and analytically tractable. Underestimates crypto tail risk."),
            ("Jump-Diffusion","Adds Poisson crash/spike events. Better for modeling liquidation cascades."),
            ("Heston SV","Stochastic volatility with mean-reversion. Best for options chain calibration."),
            ("Full Model","Heston + Jumps combined. Highest fidelity. Recommended for risk management."),
            ("Signal adjustment","Funding, L/S ratio, exchange inflows and Fear & Greed all shift the drift."),
            ("CVaR 5%","Average loss in the worst 5% of outcomes — the tail risk you actually care about."),
        ]
        cols = st.columns(3)
        for i,(t,d) in enumerate(refs):
            with cols[i%3]:
                st.markdown(f'<div class="ref-card"><div class="ref-card-title">{t}</div><div class="ref-card-body">{d}</div></div>',
                            unsafe_allow_html=True)
        return

    # ── Run / restore ─────────────────────────────────────────────────────────
    if p["run"]:
        sadj = compute_signal_adjustment(p["funding"],p["ls_ratio"],p["inflow"],p["fg"])
        with st.spinner("Running Monte Carlo simulation..."):
            paths = run_montecarlo(p["spot"],p["sigma"],p["mu"],p["dte"]/365,
                                   p["n_paths"],max(int(p["dte"]),1),p["model"],
                                   p["lam"],p["muj"],p["sigj"],p["xi"],p["kap"],p["rho"],sadj)
        st.session_state.update({"last_paths":paths,"last_params":p,"signal_adj":sadj})
    else:
        paths = st.session_state["last_paths"]
        p     = st.session_state["last_params"]
        sadj  = st.session_state["signal_adj"]

    st_d = compute_stats(paths, p["spot"])
    cone = volatility_cone(p["spot"],p["sigma"],p["mu"]+sadj,max_days=p["dte"]+1)
    fm   = lambda v: f"${v:,.0f}"
    dpct = lambda v: f"{(v/p['spot']-1)*100:+.1f}%"

    # ── Metric cards ──────────────────────────────────────────────────────────
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    with m1: st.metric("Mean price",      fm(st_d["mean"]),  delta=dpct(st_d["mean"]))
    with m2: st.metric("Median price",    fm(st_d["median"]),delta=dpct(st_d["median"]))
    with m3: st.metric("95th percentile", fm(st_d["p95"]),   delta=dpct(st_d["p95"]))
    with m4: st.metric("VaR 5% (floor)",  fm(st_d["p05"]),   delta=dpct(st_d["p05"]))
    with m5: st.metric("CVaR 5% (tail)",  fm(st_d["cvar5"]), delta=dpct(st_d["cvar5"]))
    with m6:
        up = st_d["prob_up"]*100
        st.metric("P(above spot)", f"{up:.1f}%", delta="Bullish bias" if up>50 else "Bearish bias")

    st.markdown("---")

    # ── Distribution + paths ──────────────────────────────────────────────────
    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_distribution(paths,p["spot"],p["asset"],p["dte"]),
                        use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_paths(p["spot"],p["sigma"],p["mu"],p["dte"]/365,
                                    max(int(p["dte"]),1),sadj,p["model"],
                                    p["lam"],p["muj"],p["sigj"],p["xi"],p["kap"],p["rho"]),
                        use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Cone + return dist ────────────────────────────────────────────────────
    c3, c4 = st.columns([3,2])
    with c3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_cone(cone,p["spot"],p["asset"]),
                        use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_returns(paths,p["spot"]),
                        use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── VaR tearsheet ─────────────────────────────────────────────────────────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(chart_var(paths,p["spot"]),
                    use_container_width=True, config={"displayModeBar":False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Prob table + stats ────────────────────────────────────────────────────
    prob_col, stat_col = st.columns(2)

    with prob_col:
        st.markdown('<p class="section-head">Probability targets</p>', unsafe_allow_html=True)
        probs = [
            (f"P(price > spot)",         f"{st_d['prob_up']*100:.1f}%",   GREEN if st_d["prob_up"]>0.5 else RED),
            (f"P(> +10%)  ${p['spot']*1.1:,.0f}",f"{st_d['prob_up10']*100:.1f}%",GREEN),
            (f"P(> +20%)  ${p['spot']*1.2:,.0f}",f"{st_d['prob_up20']*100:.1f}%",GREEN),
            (f"P(> +50%)  ${p['spot']*1.5:,.0f}",f"{st_d['prob_up50']*100:.1f}%",GREEN),
            (f"P(2×)  ${p['spot']*2:,.0f}",      f"{st_d['prob_up2x']*100:.1f}%",GREEN),
            ("","",""),
            (f"P(< −10%)  ${p['spot']*0.9:,.0f}",f"{st_d['prob_dn10']*100:.1f}%",RED),
            (f"P(< −20%)  ${p['spot']*0.8:,.0f}",f"{st_d['prob_dn20']*100:.1f}%",RED),
            (f"P(< −40%)  ${p['spot']*0.6:,.0f}",f"{st_d['prob_dn40']*100:.1f}%",RED),
            (f"P(stays ±10%)",                    f"{st_d['prob_flat']*100:.1f}%",BLUE),
        ]
        for lbl,val,col in probs:
            if not lbl:
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True); continue
            st.markdown(f'<div class="prob-row"><span class="prob-label">{lbl}</span>'
                        f'<span style="font-weight:700;color:{col};font-size:13px">{val}</span></div>',
                        unsafe_allow_html=True)

    with stat_col:
        st.markdown('<p class="section-head">Distribution statistics</p>', unsafe_allow_html=True)
        for lbl,val in [
            ("Mean",              fm(st_d["mean"])),
            ("Median",            fm(st_d["median"])),
            ("Std deviation",     fm(st_d["std"])),
            ("Skewness",          f"{st_d['skew']:.3f}"),
            ("Excess kurtosis",   f"{st_d['kurt']:.3f}"),
            ("P01 / P99",         f"{fm(st_d['p01'])} / {fm(st_d['p99'])}"),
            ("P05 / P95",         f"{fm(st_d['p05'])} / {fm(st_d['p95'])}"),
            ("P10 / P90",         f"{fm(st_d['p10'])} / {fm(st_d['p90'])}"),
            ("P25 / P75",         f"{fm(st_d['p25'])} / {fm(st_d['p75'])}"),
            ("VaR 5%",            f"{fm(st_d['var5'])} ({dpct(st_d['var5'])})"),
            ("CVaR 5%",           f"{fm(st_d['cvar5'])} ({dpct(st_d['cvar5'])})"),
            ("VaR 1%",            f"{fm(st_d['var1'])} ({dpct(st_d['var1'])})"),
            ("CVaR 1%",           f"{fm(st_d['cvar1'])} ({dpct(st_d['cvar1'])})"),
            ("Signal adj. drift", f"{sadj*100:+.2f}%/yr"),
        ]:
            st.markdown(f'<div class="stat-row"><span class="stat-label">{lbl}</span>'
                        f'<span class="stat-val">{val}</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Interpretation ────────────────────────────────────────────────────────
    st.markdown('<p class="section-head">Model interpretation & signal read</p>', unsafe_allow_html=True)
    msgs = interpret_signals(p["funding"],p["ls_ratio"],p["inflow"],p["fg"],p["model"],st_d,p["spot"])
    bmap = {"green":"badge-green","red":"badge-red","amber":"badge-amber","blue":"badge-blue"}
    html = "<br>".join(f'<span class="badge {bmap[c]}">{c.upper()}</span>&nbsp; {m}' for c,m in msgs)
    st.markdown(f'<div class="interp-box">{html}</div>', unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-head">Export results</p>', unsafe_allow_html=True)
    e1,e2,_ = st.columns([1,1,3])
    with e1:
        st.download_button("⬇ Paths (CSV)",
            pd.DataFrame({"terminal_price":paths}).to_csv(index=False).encode(),
            f"{p['asset']}_paths_{p['dte']}d.csv","text/csv",use_container_width=True)
    with e2:
        st.download_button("⬇ Stats (CSV)",
            pd.DataFrame([{"asset":p["asset"],"model":p["model"],"spot":p["spot"],
                "iv_pct":p["sigma"]*100,"dte":p["dte"],"n_paths":p["n_paths"],
                **{k:round(v,4) for k,v in st_d.items()}}]).to_csv(index=False).encode(),
            f"{p['asset']}_stats_{p['dte']}d.csv","text/csv",use_container_width=True)

    st.markdown('<div class="footer">GlassDeck &nbsp;·&nbsp; Quantitative price distribution modeling &nbsp;·&nbsp; Not financial advice &nbsp;·&nbsp; Results are probabilistic, not predictive.</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
