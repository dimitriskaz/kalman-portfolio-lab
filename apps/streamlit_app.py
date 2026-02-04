# apps/streamlit_app.py
from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from core.yahoo_real import load_yahoo_returns, to_matrices
from core.kf_real import kalman_with_C_est  # Approach 1 (MATLAB translated)
from core.approach2_real import run_approach2_em_pca_on_yahoo  # Approach 2 (paper-faithful)

# Curated options for multi-select (Yahoo Finance symbols)
ASSET_OPTIONS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "BRK-B", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "DIS", "BAC", "ADBE", "XOM", "CRM", "CSCO",
    "PEP", "KO", "NFLX", "COST", "AVGO", "TMO", "ABT", "ACN", "MCD", "DHR",
    "NEE", "LIN", "WFC", "PM", "TXN", "UNH", "RTX", "HON", "ORCL", "AMD",
]
FACTOR_OPTIONS = [
    "^GSPC",   # S&P 500
    "^VIX",    # VIX
    "^DJI",    # Dow Jones
    "^IXIC",   # Nasdaq
    "^TNX",    # 10Y Treasury
    "^IRX",    # 13-week T-bill
    "SPY",     # S&P 500 ETF
    "QQQ",     # Nasdaq 100 ETF
]


def plot_portfolio(idx: pd.Index, Y_true: np.ndarray, Y_pred: np.ndarray, assets: list[str], approach_name: str):
    T = min(Y_true.shape[1], Y_pred.shape[1], len(idx))
    plot_idx = idx[:T]

    # Equal-weight portfolio returns
    port_true = Y_true[:, :T].mean(axis=0)
    port_pred = Y_pred[:, :T].mean(axis=0)

    df_port = pd.DataFrame({"Actual Returns": port_true, "Predicted Returns": port_pred}, index=plot_idx)
    st.plotly_chart(px.line(df_port, title="Portfolio Forecast (Equal-Weight)", labels={"value": "Log Returns", "index": "Date"}),
                    use_container_width=True)

    # Single-asset view: dropdown to pick ticker
    selected_ticker = st.selectbox(
        "Select ticker for Actual vs Predicted",
        options=assets,
        key=f"ticker_select_{approach_name}",
    )
    asset_ix = assets.index(selected_ticker)
    df_asset = pd.DataFrame(
        {"Actual Returns": Y_true[asset_ix, :T], "Predicted Returns": Y_pred[asset_ix, :T]},
        index=plot_idx,
    )
    st.plotly_chart(
        px.line(df_asset, title=f"{selected_ticker} — Actual vs Predicted Returns", labels={"value": "Log Returns", "index": "Date"}),
        use_container_width=True,
    )

    # Prediction error per time step
    err = np.linalg.norm(Y_true[:, :T] - Y_pred[:, :T], axis=0)
    st.plotly_chart(px.line(pd.Series(err, index=plot_idx, name="Forecast Error"),
                            title="Prediction Accuracy Over Time", labels={"value": "Error (lower = better)", "index": "Date"}),
                    use_container_width=True)


st.set_page_config(page_title="Kalman Portfolio Lab", layout="wide")
st.title("Kalman Portfolio Lab")
st.caption("Compare observable-factor (CVX) vs latent-factor (EM-PCA) Dynamic Factor Models on real market data.")

# ----- Main frame: parameters used by BOTH models (data + run) -----
with st.sidebar:
    st.header("Data (shared)")
    st.caption("Used by both approaches: same assets and date range.")
    assets = st.multiselect(
        "Assets",
        options=ASSET_OPTIONS,
        default=["AAPL", "MSFT", "GOOGL"],
        key="assets",
        help="Select one or more tickers.",
    )
    factors = st.multiselect(
        "Observable factors (for CVX)",
        options=FACTOR_OPTIONS,
        default=["^GSPC", "^VIX"],
        key="factors",
        help="Select market indices/factors for the CVX model.",
    )

    _end_default = datetime.now().strftime("%Y-%m-%d")
    _start_default = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    start = st.text_input("Start date (YYYY-MM-DD)", _start_default, key="start")
    end = st.text_input("End date (YYYY-MM-DD)", _end_default, key="end")

# Tabs: each holds model-specific parameters + results
tab1, tab2 = st.tabs(["Observable Factors (CVX)", "Latent Factors (EM-PCA)"])

with tab1:
    st.subheader("Observable Factors (CVX)")
    st.caption("Uses your chosen market factors (e.g. indices) and convex optimization to estimate loadings, then a Kalman filter to forecast returns.")
    with st.expander("Model parameters", expanded=True):
        p1 = st.slider("Lag order p", 0, 10, 1, help="Number of lags in the factor dynamics.", key="p1")
        noise_scale1 = st.number_input("Noise scale", min_value=0.01, value=0.5, step=0.05, help="Scaling for measurement noise covariance.", key="noise_scale1")

    run_cvx_btn = st.button("Run this model", type="primary", key="run_cvx")

    res = st.session_state.get("portfolio_results")
    if res:
        st.divider()
        st.markdown("**Results**")
        plot_portfolio(res["idx1"], res["out1"]["Y_true"], res["out1"]["Y_pred"], res["assets"], "Observable Factors")

with tab2:
    st.subheader("Latent Factors (EM-PCA)")
    st.caption("Discovers hidden factors from returns via EM-PCA, then uses a Kalman filter to forecast. No market indices required.")
    with st.expander("Model parameters", expanded=True):
        k2 = st.number_input("k (latent base factors)", min_value=1, value=1, step=1, help="Number of latent factors. Must satisfy k×(p+1) ≤ number of assets.", key="k2")
        p2 = st.slider("p (lag order)", 0, 10, 1, help="Lag order for the latent state dynamics.", key="p2")
        noise_scale2 = st.number_input("Noise scale", min_value=0.01, value=0.5, step=0.05, help="Scaling for measurement noise covariance.", key="noise_scale2")
        em_iters = st.slider("EM iterations", 20, 400, 250, help="Max iterations for the EM-PCA algorithm.", key="em_iters")
        em_tol = st.number_input("EM tolerance", min_value=1e-10, value=1e-6, format="%.1e", help="Convergence tolerance for EM.", key="em_tol")

    run_em_btn = st.button("Run this model", type="primary", key="run_em")

    res2 = st.session_state.get("portfolio_results")
    if res2:
        st.divider()
        st.markdown("**Results**")
        if res2["out2"] is None:
            m2 = int(k2) * (int(p2) + 1)
            st.error(f"Invalid params: m = k×(p+1) = {m2} > number of assets ({res2['n']}). Reduce k or p in the parameters above.")
        else:
            plot_portfolio(res2["idx2"], res2["out2"]["Y_true"], res2["out2"]["Y_pred"], res2["assets"], "Latent Factors")
            st.caption(f"EM-PCA fit: MSE = {float(res2['out2']['mse'][0]):.6g} | latent dimension m = {int(res2['out2']['m'][0])}")


def _load_data(assets: list[str], factors: list[str], start: str, end: str):
    asset_rets, factor_rets = load_yahoo_returns(assets, factors, start, end)
    Y, u, idx = to_matrices(asset_rets, factor_rets)
    n = len(assets)
    k_obs = len(factors)
    return Y, u, idx, n, k_obs


# Run CVX model only (tab1 button)
if run_cvx_btn:
    _assets = assets or ["AAPL", "MSFT", "GOOGL"]
    _factors = factors or ["^GSPC", "^VIX"]
    Y, u, idx, n, k_obs = _load_data(_assets, _factors, start, end)
    out1 = kalman_with_C_est(Y=Y, u=u, k=k_obs, p=p1, noise_scale=noise_scale1)
    prev = st.session_state.get("portfolio_results") or {}
    st.session_state["portfolio_results"] = {
        "assets": _assets,
        "idx1": idx[:-1],
        "out1": out1,
        "idx2": idx,
        "out2": prev.get("out2"),
        "n": n,
        "k_obs": k_obs,
    }

# Run EM-PCA model only (tab2 button)
if run_em_btn:
    _assets = assets or ["AAPL", "MSFT", "GOOGL"]
    _factors = factors or ["^GSPC", "^VIX"]
    Y, u, idx, n, k_obs = _load_data(_assets, _factors, start, end)
    out2 = None
    if int(k2) * (int(p2) + 1) <= n:
        out2 = run_approach2_em_pca_on_yahoo(
            Y=Y, k=int(k2), p=int(p2), noise_scale=noise_scale2,
            em_iterations=int(em_iters), em_tol=float(em_tol), seed=0,
        )
    prev = st.session_state.get("portfolio_results") or {}
    st.session_state["portfolio_results"] = {
        "assets": _assets,
        "idx1": idx[:-1],
        "out1": prev.get("out1"),
        "idx2": idx,
        "out2": out2,
        "n": n,
        "k_obs": k_obs,
    }

# Run both with defaults on first load
run_with_defaults = "portfolio_results" not in st.session_state
if run_with_defaults:
    assets = ["AAPL", "MSFT", "GOOGL"]
    factors = ["^GSPC", "^VIX"]
    _end = datetime.now().strftime("%Y-%m-%d")
    _start = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    Y, u, idx, n, k_obs = _load_data(assets, factors, _start, _end)
    out1 = kalman_with_C_est(Y=Y, u=u, k=k_obs, p=1, noise_scale=0.5)
    out2 = run_approach2_em_pca_on_yahoo(Y=Y, k=1, p=1, noise_scale=0.5, em_iterations=250, em_tol=1e-6, seed=0)
    st.session_state["portfolio_results"] = {
        "assets": assets,
        "idx1": idx[:-1],
        "out1": out1,
        "idx2": idx,
        "out2": out2,
        "n": n,
        "k_obs": k_obs,
    }

