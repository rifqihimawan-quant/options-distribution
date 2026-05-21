# GlassDeck — Crypto Price Distribution Calculator

Quantitative price distribution modeling for BTC / ETH using Monte Carlo simulation.

## Models supported

| Model | Description |
|-------|-------------|
| **GBM** | Geometric Brownian Motion — constant vol baseline |
| **Jump-Diffusion** | Merton (1976) — adds Poisson crash/spike events |
| **Heston SV** | Stochastic volatility with mean-reversion + spot-vol correlation |
| **Full Model** | Heston + Jumps — highest fidelity, best for tail risk |

## Variables

**Market inputs**
- Spot price, implied volatility, drift/trend bias, forecast horizon

**Jump-Diffusion parameters**
- λ (jump intensity), μ_J (avg jump size), σ_J (jump volatility)

**Heston parameters**
- ξ (vol-of-vol), κ (mean reversion), ρ (spot-vol correlation)

**Positioning signals** → adjust drift + tail probabilities
- Funding rate, Long/Short ratio, Exchange inflow pressure, Fear & Greed index

## Output

- Price distribution histogram with VaR/CVaR/percentile markers
- 80 sample Monte Carlo paths
- Volatility cone (±1σ / ±2σ) over chosen horizon
- Log-return distribution vs normal
- VaR & CVaR tearsheet at 5 confidence levels
- Full probability table (P>+10%, P<-20%, etc.)
- Distribution statistics (skewness, kurtosis, all percentiles)
- Signal interpretation panel
- CSV export (paths + stats)

---

## Run locally

```bash
# 1. Clone or download this folder

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Deploy to Streamlit Community Cloud (free)

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Click **New app** → select your repo → set main file to `app.py`
4. Click **Deploy** — live URL in ~2 minutes

No server setup needed. Free tier supports public apps.

---

## Deploy to other platforms

**Railway:**
```bash
# railway.toml is not needed — Streamlit auto-detects requirements.txt
railway login
railway init
railway up
# Set PORT environment variable in Railway dashboard if needed
```

**Heroku:**
```
# Add Procfile:
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

---

## Integrate with live Coinglass data

Replace the hardcoded defaults in `build_sidebar()` with live API calls:

```python
import requests

@st.cache_data(ttl=30)
def fetch_live_iv(asset):
    """Fetch DVOL (Deribit Volatility Index) from Coinglass proxy"""
    r = requests.get(f"http://your-proxy/api/dvol?asset={asset}", timeout=5)
    return r.json().get("dvol", 80)  # fallback to 80

@st.cache_data(ttl=30)
def fetch_live_funding(asset):
    r = requests.get(f"http://your-proxy/api/funding?asset={asset}", timeout=5)
    return r.json().get("rate", 0.01)
```

Then in `build_sidebar()`:
```python
iv_pct = st.slider("IV %", 20, 200, fetch_live_iv(asset), 1)
funding = st.slider("Funding", -0.1, 0.2, fetch_live_funding(asset), 0.001)
```

---

## Mathematical notes

**GBM:**  dS = μS dt + σS dW

**Jump-Diffusion (Merton):**  dS = μS dt + σS dW + S dJ
where J is a compound Poisson process with intensity λ, jump size ~ N(μ_J, σ_J²)

**Heston:**  dS = μS dt + √v S dW_S
dv = κ(θ − v) dt + ξ√v dW_v
corr(dW_S, dW_v) = ρ dt

**Signal drift adjustment:**
adj = −f × 3 × 365 × 0.3  (funding drag)
    − (LS − 1) × 0.15      (crowding drag)
    − I × 0.25             (inflow drag)
    + (FG − 50)/50 × 0.08  (sentiment tilt)

All outputs are probabilistic estimates, not financial advice.
