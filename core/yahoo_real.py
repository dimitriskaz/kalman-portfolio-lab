from __future__ import annotations

import numpy as np
import pandas as pd

from core.data import download_adj_close, log_returns, align_on_intersection


def load_yahoo_returns(
    asset_tickers: list[str],
    factor_tickers: list[str],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      asset_rets_df: (T x n)
      factor_rets_df: (T x k)
    aligned on same dates.
    """
    asset_prices = download_adj_close(asset_tickers, start, end)
    factor_prices = download_adj_close(factor_tickers, start, end)

    asset_rets = log_returns(asset_prices)
    factor_rets = log_returns(factor_prices)

    asset_rets, factor_rets = align_on_intersection(asset_rets, factor_rets)

    # drop any remaining NaNs after alignment
    df = asset_rets.join(factor_rets, how="inner", lsuffix="_a", rsuffix="_f").dropna()
    asset_rets = df[asset_rets.columns]
    factor_rets = df[factor_rets.columns]

    if len(asset_rets) < 50:
        raise RuntimeError("Not enough overlapping return observations. Try a longer date range.")

    return asset_rets, factor_rets


def to_matrices(asset_rets: pd.DataFrame, factor_rets: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Convert to paper-style matrices:
      Y: (n x T)
      u: (k x T)
    """
    Y = asset_rets.to_numpy(dtype=float).T
    u = factor_rets.to_numpy(dtype=float).T
    return Y, u, asset_rets.index
