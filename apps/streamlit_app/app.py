import streamlit as st
import plotly.express as px

from core.data.yahoo import fetch_adj_close
from core.features.returns import pct_returns
from core.models.em_pca import estimate_em_pca
from core.models.kalman import kf_real_data_em

st.set_page_config(page_title="Kalman Portfolio Lab", layout="wide")

st.title("Kalman Portfolio Lab (EM-PCA + Kalman Filter)")
st.caption("MVP: fetch data, compute returns, preview series. Next: EM-PCA + Kalman model tabs.")

# --- session state init ---
if "px_assets" not in st.session_state:
    st.session_state.px_assets = None
if "px_factors" not in st.session_state:
    st.session_state.px_factors = None
if "r_assets" not in st.session_state:
    st.session_state.r_assets = None

with st.sidebar:
    st.header("Inputs")
    tickers = st.text_area("Assets (comma separated)", "AAPL,MSFT,GOOGL").split(",")
    tickers = [t.strip() for t in tickers if t.strip()]

    factors = st.text_area("Factors (comma separated)", "^IXIC,GC=F,CL=F").split(",")
    factors = [t.strip() for t in factors if t.strip()]

    start = st.date_input("Start", value=None)
    end = st.date_input("End", value=None)

    load_data = st.button("Load data")

# --- data loading ---
if load_data:
    if start is None or end is None:
        st.error("Pick start and end dates.")
        st.stop()

    st.session_state.px_assets = fetch_adj_close(tickers, str(start), str(end))
    st.session_state.px_factors = fetch_adj_close(factors, str(start), str(end))
    st.session_state.r_assets = pct_returns(st.session_state.px_assets).dropna()

px_assets = st.session_state.px_assets
px_factors = st.session_state.px_factors
r_assets = st.session_state.r_assets

# --- show data if loaded ---
if px_assets is not None and r_assets is not None:
    st.subheader("Prices")
    st.dataframe(px_assets.tail())

    st.subheader("Returns (assets)")
    st.dataframe(r_assets.tail())

    fig = px.line(r_assets, title="Asset returns")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Click **Load data** to fetch prices and compute returns.")

st.divider()
st.subheader("EM-PCA")

m = st.sidebar.number_input("m (latent factors)", min_value=1, max_value=10, value=3)

if st.button("Run EM-PCA"):
    if r_assets is None:
        st.error("Load data first.")
        st.stop()

    # safer: only use tickers that actually exist in r_assets.columns
    cols = [t for t in tickers if t in r_assets.columns]
    missing = [t for t in tickers if t not in r_assets.columns]
    if not cols:
        st.error("None of your tickers exist in the returns dataframe. Check ticker symbols.")
        st.stop()
    if missing:
        st.warning(f"Missing tickers (not found in returns): {missing}")

    Y = r_assets[cols].T.to_numpy()  # (n, T)
    res = estimate_em_pca(Y, int(m), seed=0)
    st.write(f"Converged in {res.n_iter} iterations. SSE={res.sse:.4f}")
    st.dataframe(res.C)

st.divider()
st.subheader("Kalman Filter")

k = st.sidebar.number_input("k (factors)", min_value=1, max_value=10, value=3)
p = st.sidebar.number_input("p (lag order)", min_value=0, max_value=5, value=1)

# Compute and display m for user feedback
m_kf = int(k) * (int(p) + 1)
if r_assets is not None:
    cols = [t for t in tickers if t in r_assets.columns]
    if cols:
        Y_preview = r_assets[cols].T.to_numpy()
        n_preview, T_preview = Y_preview.shape
        max_m_preview = min(n_preview, T_preview)
        st.caption(f"Kalman latent dimension m = k*(p+1) = {m_kf}. Must be <= min(n,T) = {max_m_preview}.")
    else:
        st.caption(f"Kalman latent dimension m = k*(p+1) = {m_kf}.")
else:
    st.caption(f"Kalman latent dimension m = k*(p+1) = {m_kf}. Must be <= min(n,T).")

if st.button("Run Kalman (EM-PCA C_est)"):
    if r_assets is None:
        st.error("Load data first.")
        st.stop()

    cols = [t for t in tickers if t in r_assets.columns]
    missing = [t for t in tickers if t not in r_assets.columns]
    if not cols:
        st.error("None of your tickers exist in the returns dataframe. Check ticker symbols.")
        st.stop()
    if missing:
        st.warning(f"Missing tickers (not found in returns): {missing}")

    Y = r_assets[cols].T.to_numpy()  # (n, T)
    n, T = Y.shape
    max_m = min(n, T)

    if m_kf > max_m:
        st.error(
            f"Invalid settings: m={m_kf} but min(n,T)={max_m}. "
            f"Add more tickers (n), choose a longer date range (T), or reduce k/p."
        )
        st.stop()

    out = kf_real_data_em(Y, k=int(k), p=int(p))
    st.json({"y_actual": out.y_actual, "y_estimate": out.y_estimate, "e_c_est": out.e_c_est})
