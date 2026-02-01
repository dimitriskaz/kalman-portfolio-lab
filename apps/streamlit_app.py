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


def plot_portfolio(idx: pd.Index, Y_true: np.ndarray, Y_pred: np.ndarray, assets: list[str], approach_name: str):
    T = min(Y_true.shape[1], Y_pred.shape[1], len(idx))
    plot_idx = idx[:T]

    # Equal-weight portfolio returns
    port_true = Y_true[:, :T].mean(axis=0)
    port_pred = Y_pred[:, :T].mean(axis=0)

    df_port = pd.DataFrame({"Actual Returns": port_true, "Predicted Returns": port_pred}, index=plot_idx)
    st.plotly_chart(px.line(df_port, title=f"Portfolio Forecast (Equal-Weight)", labels={"value": "Log Returns", "index": "Date"}),
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
st.title("🎯 Kalman Portfolio Lab — Forecast Returns with Two Approaches")
st.caption("Compare observable-factor (CVX) vs latent-factor (EM-PCA) Dynamic Factor Models on real market data")

with st.sidebar:
    st.header("Tickers")
    assets_str = st.text_input("Assets (comma)", "AAPL,MSFT,GOOGL", key="assets")
    factors_str = st.text_input("Observable Factors (comma)", "^GSPC,^VIX", key="factors")

    st.header("Date range")
    _end_default = datetime.now().strftime("%Y-%m-%d")
    _start_default = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    start = st.text_input("Start (YYYY-MM-DD)", _start_default, key="start")
    end = st.text_input("End (YYYY-MM-DD)", _end_default, key="end")

    st.header("Observable Factors (CVX) settings")
    p1 = st.slider("Lag order p", 0, 10, 1, key="p1")
    noise_scale1 = st.number_input("Noise scale", min_value=0.01, value=0.5, step=0.05, key="noise_scale1")

    st.header("Latent Factors (EM-PCA) settings")
    k2 = st.number_input("k (latent base factors)", min_value=1, value=1, step=1, key="k2")
    p2 = st.slider("p (lag order)", 0, 10, 1, key="p2")
    noise_scale2 = st.number_input("Noise scale", min_value=0.01, value=0.5, step=0.05, key="noise_scale2")
    em_iters = st.slider("EM iterations", 20, 400, 250, key="em_iters")
    em_tol = st.number_input("EM tolerance", min_value=1e-10, value=1e-6, format="%.1e", key="em_tol")

    run_btn = st.button("Run models", key="run_btn")


if run_btn:
    assets = [x.strip().upper() for x in assets_str.split(",") if x.strip()]
    factors = [x.strip().upper() for x in factors_str.split(",") if x.strip()]
    asset_rets, factor_rets = load_yahoo_returns(assets, factors, start, end)
    Y, u, idx = to_matrices(asset_rets, factor_rets)
    n = len(assets)
    k_obs = len(factors)

    out1 = kalman_with_C_est(Y=Y, u=u, k=k_obs, p=p1, noise_scale=noise_scale1)
    out2 = None
    if int(k2) * (int(p2) + 1) <= n:
        out2 = run_approach2_em_pca_on_yahoo(
            Y=Y, k=int(k2), p=int(p2), noise_scale=noise_scale2,
            em_iterations=int(em_iters), em_tol=float(em_tol), seed=0,
        )
    st.session_state["portfolio_results"] = {
        "assets": assets,
        "idx1": idx[:-1],
        "out1": out1,
        "idx2": idx,
        "out2": out2,
        "n": n,
        "k_obs": k_obs,
    }

res = st.session_state.get("portfolio_results")
if res:
    tab1, tab2 = st.tabs(["📊 Observable Factors (CVX)", "🔬 Latent Factors (EM-PCA)"])
    assets = res["assets"]

    with tab1:
        st.caption("Uses your chosen market factors (e.g. indices) and convex optimization to estimate loadings, then Kalman filter to forecast returns.")
        plot_portfolio(res["idx1"], res["out1"]["Y_true"], res["out1"]["Y_pred"], assets, "Observable Factors")

    with tab2:
        st.caption("Discovers hidden factors from returns via EM-PCA, then uses Kalman filter to forecast. No market indices required.")
        if res["out2"] is None:
            m2 = int(k2) * (int(p2) + 1)
            st.error(f"Invalid params: m = k×(p+1) = {m2} > number of assets ({res['n']}). Reduce k or p.")
        else:
            plot_portfolio(res["idx2"], res["out2"]["Y_true"], res["out2"]["Y_pred"], assets, "Latent Factors")
            st.caption(f"EM-PCA fit: MSE = {float(res['out2']['mse'][0]):.6g} | latent dimension m = {int(res['out2']['m'][0])}")
