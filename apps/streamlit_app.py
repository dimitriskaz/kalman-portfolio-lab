# apps/streamlit_app.py
from __future__ import annotations

from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from core.yahoo_real import load_yahoo_returns, to_matrices
from core.kf_real import kalman_with_C_est

st.set_page_config(page_title="Approach 1 — Yahoo", layout="wide")
st.title("Approach 1 (Paper MATLAB → Python) on Yahoo Finance Data")

# Initialize session state
if "dfm_results" not in st.session_state:
    st.session_state.dfm_results = None

with st.sidebar:
    st.header("Tickers")
    assets_str = st.text_input("Assets (comma)", "AAPL,MSFT,GOOGL")
    factors_str = st.text_input("Factors (comma)", "^GSPC,^VIX")

    st.header("Date range")
    default_end = date.today()
    default_start = default_end - timedelta(days=120)
    start = st.date_input("Start", value=default_start)
    end = st.date_input("End", value=default_end)

    st.header("Model parameters")
    p = st.slider("Lag order p", 0, 10, 1)
    noise_scale = st.number_input("Noise scale (E diag factor)", min_value=0.01, value=0.5, step=0.05)

    run_btn = st.button("Run")

if run_btn:
    assets = [x.strip().upper() for x in assets_str.split(",") if x.strip()]
    factors = [x.strip().upper() for x in factors_str.split(",") if x.strip()]

    # Load real returns
    asset_rets, factor_rets = load_yahoo_returns(assets, factors, str(start), str(end))
    Y, u, idx = to_matrices(asset_rets, factor_rets)

    n = len(assets)
    k = len(factors)

    st.subheader("Data shapes")
    st.write({"Y (n x T)": Y.shape, "u (k x T)": u.shape, "n": n, "k": k, "p": p})

    out = kalman_with_C_est(Y=Y, u=u, k=k, p=p, noise_scale=noise_scale)

    C_est = out["C_est"]
    Y_pred = out["Y_pred"]  # (n x T-1)
    Y_true = out["Y_true"]  # (n x T-1)

    # Store results in session state
    st.session_state.dfm_results = {
        "assets": assets,
        "C_est": C_est,
        "Y_pred": Y_pred,
        "Y_true": Y_true,
        "plot_idx": idx[:-1],  # because Y_pred is T-1
    }

# Display results from session state
if st.session_state.dfm_results is not None:
    results = st.session_state.dfm_results
    assets = results["assets"]
    C_est = results["C_est"]
    Y_pred = results["Y_pred"]
    Y_true = results["Y_true"]
    plot_idx = results["plot_idx"]

    st.subheader("Estimated Factor Loading Matrix C_est")
    st.dataframe(pd.DataFrame(C_est, index=assets))

    # -----------------------------
    # 1) Single asset plot
    # -----------------------------
    st.subheader("Single Asset")
    
    # Dropdown to select asset
    selected_asset = st.selectbox("Select asset to plot", assets, index=0, key="asset_selector")
    a0 = assets.index(selected_asset)
    
    df_asset = pd.DataFrame(
        {"true": Y_true[a0, :], "pred": Y_pred[a0, :]},
        index=plot_idx,
    )
    st.plotly_chart(px.line(df_asset, title=f"{selected_asset}: predicted vs true"), use_container_width=True)

    # -----------------------------
    # 2) Portfolio return (equal-weighted)
    # -----------------------------
    port_true_eq = Y_true.mean(axis=0)
    port_pred_eq = Y_pred.mean(axis=0)

    df_port_eq = pd.DataFrame(
        {"true_port_eq": port_true_eq, "pred_port_eq": port_pred_eq},
        index=plot_idx,
    )
    st.subheader("Portfolio (Equal-Weighted) — Predicted vs True")
    st.plotly_chart(px.line(df_port_eq, title="Equal-weighted portfolio return"), use_container_width=True)

    # -----------------------------
    # 3) Portfolio magnitude (MATLAB-style norm)
    # -----------------------------
    port_true_norm = np.linalg.norm(Y_true, axis=0)
    port_pred_norm = np.linalg.norm(Y_pred, axis=0)

    df_port_norm = pd.DataFrame(
        {"true_port_norm": port_true_norm, "pred_port_norm": port_pred_norm},
        index=plot_idx,
    )
    st.subheader("Portfolio (L2 Norm, MATLAB-style) — Predicted vs True")
    st.plotly_chart(px.line(df_port_norm, title="Portfolio L2 norm of returns"), use_container_width=True)

    # -----------------------------
    # 4) Normalized error series (Euclidean per time step, paper-style)
    # -----------------------------
    err = np.linalg.norm(Y_true - Y_pred, axis=0)
    err_series = pd.Series(err, index=plot_idx, name="normalized_error")
    st.subheader("Normalized Error (per time step)")
    st.plotly_chart(px.line(err_series, title="Normalized error over time"), use_container_width=True)
