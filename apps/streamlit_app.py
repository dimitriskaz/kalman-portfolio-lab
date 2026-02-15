# apps/streamlit_app.py
from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from core.yahoo_real import load_yahoo_returns, to_matrices
from core.kf_real import kalman_with_C_est  # Approach 1
from core.approach2_real import run_approach2_em_pca_on_yahoo  # Approach 2

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

    # Equal-weight portfolio: cumulative log returns
    port_true = Y_true[:, :T].mean(axis=0)
    port_pred = Y_pred[:, :T].mean(axis=0)
    cum_true = np.cumsum(port_true)
    cum_pred = np.cumsum(port_pred)

    df_port = pd.DataFrame({"Actual (cumulative)": cum_true, "Predicted (cumulative)": cum_pred}, index=plot_idx)
    st.plotly_chart(px.line(df_port, title="Portfolio Forecast (Equal-Weight) — Cumulative Log Returns", labels={"value": "Cumulative Log Return", "index": "Date"}),
                    use_container_width=True)

    # Single-asset view: dropdown to pick ticker
    selected_ticker = st.selectbox(
        "Select ticker for Actual vs Predicted",
        options=assets,
        key=f"ticker_select_{approach_name}",
    )
    asset_ix = assets.index(selected_ticker)
    cum_true_asset = np.cumsum(Y_true[asset_ix, :T])
    cum_pred_asset = np.cumsum(Y_pred[asset_ix, :T])
    df_asset = pd.DataFrame(
        {"Actual (cumulative)": cum_true_asset, "Predicted (cumulative)": cum_pred_asset},
        index=plot_idx,
    )
    st.plotly_chart(
        px.line(df_asset, title=f"{selected_ticker} — Actual vs Predicted Cumulative Log Returns", labels={"value": "Cumulative Log Return", "index": "Date"}),
        use_container_width=True,
    )

    # Prediction error per time step
    err = np.linalg.norm(Y_true[:, :T] - Y_pred[:, :T], axis=0)
    st.plotly_chart(px.line(pd.Series(err, index=plot_idx, name="Forecast Error"),
                            title="Prediction Accuracy Over Time", labels={"value": "Error (lower = better)", "index": "Date"}),
                    use_container_width=True)


def plot_c_heatmap(
    C_est: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    explanation: str,
) -> None:
    """Plot C_est loading matrix as a heatmap with explanation."""
    df = pd.DataFrame(C_est, index=row_labels, columns=col_labels)
    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            x=col_labels,
            y=row_labels,
            colorscale="RdBu_r",
            zmid=0,
            colorbar={"title": "Loading"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Factor / State",
        yaxis_title="Asset",
        height=400,
        xaxis={"tickangle": -45},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(explanation)


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
        help="Asset tickers whose returns you want to forecast. Both models use these as the target series.",
    )
    factors = st.multiselect(
        "Observable factors (for CVX)",
        options=FACTOR_OPTIONS,
        default=["^GSPC", "^VIX"],
        key="factors",
        help="Market indices or ETFs used as observable factors in the CVX model. Asset returns are modeled as linear combinations of these factor returns.",
    )

    _end_default = datetime.now().strftime("%Y-%m-%d")
    _start_default = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    start = st.text_input("Start date (YYYY-MM-DD)", _start_default, key="start", help="Start of the historical window for fetching price data.")
    end = st.text_input("End date (YYYY-MM-DD)", _end_default, key="end", help="End of the historical window for fetching price data.")

# Tabs: each holds model-specific parameters + results
tab1, tab2 = st.tabs(["Observable Factors (CVX)", "Latent Factors (EM-PCA)"])

with tab1:
    st.subheader("Observable Factors (CVX)")
    st.caption("Uses your chosen market factors (e.g. indices) and convex optimization to estimate loadings, then a Kalman filter to forecast returns.")
    with st.expander("Model parameters", expanded=True):
        p1 = st.slider(
            "Lag order p",
            0, 10, 1,
            key="p1",
            help="Number of lagged factor values in the state dynamics. p=0 uses only current factors; p=1 adds one lag (shift-register). Higher p captures more history but increases state dimension m = k×(p+1).",
        )
        noise_scale1 = st.number_input(
            "Noise scale",
            min_value=0.01, value=0.5, step=0.05,
            key="noise_scale1",
            help="Scales the measurement noise covariance E. Higher values assume more noise in the observed returns relative to the factor model; lower values trust the factor structure more.",
        )

    run_cvx_btn = st.button("Run this model", type="primary", key="run_cvx")

    res = st.session_state.get("portfolio_results")
    if res:
        st.divider()
        st.markdown("**Results**")
        plot_portfolio(res["idx1"], res["out1"]["Y_true"], res["out1"]["Y_pred"], res["assets"], "Observable Factors")

        # C_est heatmap (CVX)
        C_est = res["out1"]["C_est"]
        k_obs = res.get("k_obs", C_est.shape[1])
        factors_used = res.get("factors", [f"Factor {i+1}" for i in range(k_obs)])
        col_labels = []
        for j in range(C_est.shape[1]):
            f_idx = j % k_obs
            lag = j // k_obs
            f = factors_used[f_idx] if f_idx < len(factors_used) else f"Factor {f_idx+1}"
            col_labels.append(f"{f} (lag {lag})" if lag > 0 else f)
        st.markdown("**Estimated loadings (C_est)**")
        plot_c_heatmap(
            C_est,
            res["assets"],
            col_labels,
            "CVX: Asset loadings on observable factors",
            "Each cell shows how much an asset's return responds to a factor (or its lag). "
            "Positive (red) = asset moves with the factor; negative (blue) = moves opposite. "
            "Estimated via convex optimization from historical returns.",
        )

with tab2:
    st.subheader("Latent Factors (EM-PCA)")
    st.caption("Discovers hidden factors from returns via EM-PCA, then uses a Kalman filter to forecast. No market indices required.")
    with st.expander("Model parameters", expanded=True):
        k2 = st.number_input(
            "k (latent base factors)",
            min_value=1, value=1, step=1,
            key="k2",
            help="Number of latent factors discovered from returns via EM-PCA. Must satisfy k×(p+1) ≤ number of assets. Higher k captures more structure but requires more assets.",
        )
        p2 = st.slider(
            "p (lag order)",
            0, 10, 1,
            key="p2",
            help="Number of lagged values in the latent state dynamics. Same role as in CVX: p=0 uses only current latent factors; higher p adds lagged terms.",
        )
        noise_scale2 = st.number_input(
            "Noise scale",
            min_value=0.01, value=0.5, step=0.05,
            key="noise_scale2",
            help="Scales the measurement noise covariance. Higher = more uncertainty in how returns relate to latent factors.",
        )
        em_iters = st.slider(
            "EM iterations",
            20, 400, 250,
            key="em_iters",
            help="Maximum iterations for the EM-PCA algorithm. Increase if the fit has not converged by the end.",
        )
        em_tol = st.number_input(
            "EM tolerance",
            min_value=1e-10, value=1e-6, format="%.1e",
            key="em_tol",
            help="EM stops when the change in log-likelihood falls below this value. Smaller = stricter convergence.",
        )

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

            # C_est heatmap (EM-PCA)
            C_est = res2["out2"]["C_est"]
            k_val = int(res2["out2"]["k"][0])
            col_labels = []
            for j in range(C_est.shape[1]):
                f_idx = j % k_val
                lag = j // k_val
                col_labels.append(f"Latent {f_idx+1} (lag {lag})" if lag > 0 else f"Latent {f_idx+1}")
            st.markdown("**Estimated loadings (C_est)**")
            plot_c_heatmap(
                C_est,
                res2["assets"],
                col_labels,
                "EM-PCA: Asset loadings on latent factors",
                "Each cell shows how much an asset's return loads on a latent factor discovered by EM-PCA. "
                "These factors are not directly observable (unlike market indices). "
                "Positive (red) = asset co-moves with that latent factor; negative (blue) = opposite. "
                "The model learns these loadings from the return data alone.",
            )


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
        "factors": _factors,
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
        "factors": _factors,
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
        "factors": factors,
        "idx1": idx[:-1],
        "out1": out1,
        "idx2": idx,
        "out2": out2,
        "n": n,
        "k_obs": k_obs,
    }

