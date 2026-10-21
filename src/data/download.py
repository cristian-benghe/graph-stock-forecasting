"""Download and cache raw stock data from Yahoo Finance."""

import pandas as pd
import yfinance as yf
from pathlib import Path


def download_stock_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    save_dir: str = "data/raw",
) -> pd.DataFrame:
    """Download adjusted close and volume for a list of tickers.

    Returns a MultiIndex DataFrame with columns (ticker, field).
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    cache_file = save_path / "prices.parquet"
    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"Downloading {len(tickers)} tickers from {start_date} to {end_date}")
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        threads=True,
    )

    df = raw[["Close", "Volume"]].copy()
    df.columns.names = ["field", "ticker"]
    df = df.swaplevel(axis=1).sort_index(axis=1)

    df.to_parquet(cache_file)
    print(f"Saved to {cache_file} ({len(df)} trading days)")
    return df
