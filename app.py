import streamlit as st
import pandas as pd
import numpy as np
import requests
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="FX Macro Bias App", layout="wide")

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


@st.cache_data(ttl=60 * 60)
def fred_series(series_id: str, start_date: str = "1970-01-01") -> pd.Series:
    """Fetch a FRED series (date->value) as a pandas Series."""
    api_key = st.secrets.get("FRED_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing FRED_API_KEY. Add it in Streamlit Cloud → App Settings → Secrets.")

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }
    r = requests.get(FRED_BASE, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])

    dates = []
    vals = []
    for o in obs:
        v = o.get("value", ".")
        if v == ".":
            continue
        dates.append(pd.to_datetime(o["date"]))
        vals.append(float(v))

    s = pd.Series(vals, index=pd.DatetimeIndex(dates)).sort_index()
    s.name = series_id
    return s


@st.cache_data(ttl=60 * 30)
def yahoo_fx(pair: str, start="2000-01-01") -> pd.Series:
    """
    Fetch FX from Stooq free CSV feed (no extra libs).
    Stooq symbols:
      - GBPUSD -> gbpusd
      - GBPJPY -> gbpjpy
    """
    symbol = {"GBPUSD": "gbpusd", "GBPJPY": "gbpjpy"}[pair]
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    df = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    if "Close" not in df.columns:
        raise RuntimeError("Stooq data missing Close column.")
    s = df["Close"].dropna()
    s = s[s.index >= pd.to_datetime(start)]
    s.name = pair
    if s.empty:
        raise RuntimeError("No FX data returned (check Stooq availability).")
    return s



def to_monthly_last(s: pd.Series) -> pd.Series:
    return s.resample("MS").last().ffill()


def compute_bias(pair: str, fx_m: pd.Series, vix_m: pd.Series, spx_m: pd.Series):
    """
    Starter bias model:
    - Risk regime from VIX + SPX:
        Risk-off if (VIX > 6M median) OR (SPX 3M return < 0)
    - Bias:
        GBPJPY: Risk-off -> avoid longs / short bias; Risk-on -> carry/trend-friendly
        GBPUSD: Risk-off -> USD bid risk; Risk-on -> neutral/trend-friendly
    """
    df = pd.DataFrame(index=fx_m.index)
    df[pair] = fx_m
    df["VIX"] = vix_m.reindex(df.index).ffill()
    df["SPX"] = spx_m.reindex(df.index).ffill()

    df = df.dropna()

    df["spx_ret_3m"] = df["SPX"].pct_change(3) * 100
    df["vix_med_6m"] = df["VIX"].rolling(6).median()
    df["risk_off"] = (df["VIX"] > df["vix_med_6m"]) | (df["spx_ret_3m"] < 0)

    latest = df.iloc[-1]

    if pair == "GBPJPY":
        if bool(latest["risk_off"]):
            return "SHORT / Avoid longs", "Risk-off: VIX elevated and/or SPX negative → carry can unwind fast", df
        return "LONG bias ok", "Risk-on: carry/trend conditions more reliable", df

    # GBPUSD
    if bool(latest["risk_off"]):
        return "SHORT/Neutral GBPUSD", "Risk-off often supports USD → GBPUSD headwind", df
    return "Neutral / follow price trend", "Risk not flashing red → let trend/structure decide", df


# ---------------- UI ----------------
st.title("FX Macro Bias App (VIX + S&P risk regime)")

pair = st.selectbox("Select pair", ["GBPJPY", "GBPUSD"], index=0)

c1, c2, c3 = st.columns(3)
with c1:
    start_date = st.selectbox("History start (Yahoo FX)", ["2000-01-01", "1990-01-01", "2010-01-01"], index=0)
with c2:
    trend_threshold = st.slider("Trend highlight threshold (12M move %)", 10, 40, 20)
with c3:
    show_raw = st.checkbox("Show raw data table", value=False)

st.divider()

# --------------- Load Data ---------------
with st.spinner("Loading data (FRED + Yahoo)..."):
    vix = fred_series("VIXCLS", "1990-01-01")
    spx = fred_series("SP500", "1990-01-01")
    fx = yahoo_fx(pair, start=start_date)

# Monthly (macro-friendly)
fx_m = to_monthly_last(fx)
vix_m = to_monthly_last(vix)
spx_m = to_monthly_last(spx)

bias, reason, df = compute_bias(pair, fx_m, vix_m, spx_m)

# --------------- Output ---------------
st.subheader("Live bias")
st.write(f"**Pair:** {pair}")
st.write(f"**Bias:** {bias}")
st.write(f"**Why:** {reason}")

# --------------- Charts ---------------
st.subheader("Trend timeline (12M % change, shaded = clear trends)")
df_plot = df.copy()
df_plot["ret_12m"] = df_plot[pair].pct_change(12) * 100
thresh = trend_threshold

strong = df_plot["ret_12m"].abs() >= thresh
periods = []
run_start = None

for t, flag in strong.items():
    if flag and run_start is None:
        run_start = t
    if (not flag) and run_start is not None:
        run_end = df_plot.index[df_plot.index.get_loc(t) - 1]
        if (run_end.to_period("M") - run_start.to_period("M")).n + 1 >= 6:
            periods.append((run_start, run_end))
        run_start = None

if run_start is not None:
    run_end = df_plot.index[-1]
    if (run_end.to_period("M") - run_start.to_period("M")).n + 1 >= 6:
        periods.append((run_start, run_end))

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["ret_12m"], name="12M % change"))
fig.add_hline(y=0)

for s, e in periods:
    fig.add_vrect(x0=s, x1=e, fillcolor="LightSalmon", opacity=0.25, line_width=0)

fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

cA, cB = st.columns(2)

with cA:
    st.subheader(f"{pair} (monthly)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_plot.index, y=df_plot[pair], name=pair))
    fig2.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

with cB:
    st.subheader("Risk proxies")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_plot.index, y=df_plot["VIX"], name="VIX"))
    fig3.add_trace(go.Scatter(x=df_plot.index, y=df_plot["SPX"], name="SP500"))
    fig3.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig3, use_container_width=True)

if show_raw:
    st.subheader("Raw monthly dataset")
    st.dataframe(df_plot, use_container_width=True)

st.caption("Next step after this is deployed: add rate differentials + your historical hit-rate regime table for true probability-based bias.")
