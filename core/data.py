from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def download_adj_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    tickers = [t.strip() for t in tickers if t.strip()]
    if not tickers:
        raise ValueError("Tickers list is empty.")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("Yahoo response missing 'Adj Close'.")
        prices = raw["Adj Close"].copy()
    else:
        if "Adj Close" not in raw.columns:
            raise RuntimeError("Yahoo response missing 'Adj Close'.")
        prices = raw[["Adj Close"]].copy()
        prices.columns = [tickers[0]]

    prices = prices.sort_index().dropna(how="all")
    if prices.empty:
        raise RuntimeError("No price data returned from Yahoo.")
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.shape[0] < 2:
        raise ValueError("Not enough observations to compute returns.")
    rets = np.log(prices / prices.shift(1))
    rets = rets.dropna(how="any")
    if rets.empty:
        raise RuntimeError("Returns are empty after NaN removal.")
    return rets


def align_on_intersection(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        raise RuntimeError("No overlapping dates between datasets.")
    return a.loc[idx].copy(), b.loc[idx].copy()
