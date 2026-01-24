# core/features/returns.py
import pandas as pd

def pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().dropna()
    return rets

