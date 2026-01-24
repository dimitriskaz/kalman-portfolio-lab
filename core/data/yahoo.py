# core/data/yahoo.py
import pandas as pd
import yfinance as yf

def fetch_adj_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"]
    else:
        px = data[["Close"]].rename(columns={"Close": tickers[0]})
    px = px.dropna(how="all")
    return px

