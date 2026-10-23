"""Download and cache raw stock data from Yahoo Finance.

Yahoo downloads fail intermittently (rate limits, cache lock contention), and a
failed ticker comes back as an all-NaN column rather than an exception. Silently
propagating that produces NaN training losses much later, so this module retries
per-ticker failures and validates coverage before returning.
"""

import time
import pandas as pd
import yfinance as yf
from pathlib import Path


class DataCoverageError(RuntimeError):
    """Raised when downloaded data has gaps that would corrupt training."""


def _download_batch(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        threads=False,  # threaded downloads trigger sqlite cache lock errors
        progress=False,
    )
    df = raw[["Close", "Volume"]].copy()
    df.columns.names = ["field", "ticker"]
    return df.swaplevel(axis=1).sort_index(axis=1)


def _failed_tickers(df: pd.DataFrame, tickers: list[str]) -> list[str]:
    """Tickers whose Close column is entirely missing or entirely NaN."""
    closes = df.xs("Close", axis=1, level="field")
    failed = []
    for t in tickers:
        if t not in closes.columns or closes[t].notna().sum() == 0:
            failed.append(t)
    return failed


def download_stock_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    save_dir: str = "data/raw",
    max_retries: int = 3,
    force: bool = False,
) -> pd.DataFrame:
    """Download adjusted close and volume, retrying tickers that come back empty.

    Returns a MultiIndex DataFrame with columns (ticker, field).

    Raises:
        DataCoverageError: if any ticker still has no data after all retries.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    cache_file = save_path / "prices.parquet"

    if cache_file.exists() and not force:
        df = pd.read_parquet(cache_file)
        still_failed = _failed_tickers(df, tickers)
        if not still_failed:
            print(f"Loaded cached data from {cache_file}")
            return df
        print(f"Cache is missing {still_failed}; re-downloading")

    print(f"Downloading {len(tickers)} tickers from {start_date} to {end_date}")
    df = _download_batch(tickers, start_date, end_date)

    for attempt in range(1, max_retries + 1):
        failed = _failed_tickers(df, tickers)
        if not failed:
            break
        print(f"Retry {attempt}/{max_retries} for {failed}")
        time.sleep(2 * attempt)  # back off; failures are usually rate limits
        retry_df = _download_batch(failed, start_date, end_date)
        for t in failed:
            if t in retry_df.columns.get_level_values("ticker"):
                for field in ["Close", "Volume"]:
                    if (t, field) in retry_df.columns:
                        df[(t, field)] = retry_df[(t, field)]
        df = df.sort_index(axis=1)

    failed = _failed_tickers(df, tickers)
    if failed:
        raise DataCoverageError(
            f"No data for {failed} after {max_retries} retries. "
            "Check the ticker symbols or rerun later."
        )

    df.to_parquet(cache_file)
    print(f"Saved to {cache_file} ({len(df)} trading days)")
    return df


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker first/last valid date and observation count."""
    closes = df.xs("Close", axis=1, level="field")
    rows = []
    for t in closes.columns:
        s = closes[t].dropna()
        rows.append({
            "ticker": t,
            "first_valid": s.index[0].date() if len(s) else None,
            "last_valid": s.index[-1].date() if len(s) else None,
            "n_obs": len(s),
        })
    return pd.DataFrame(rows).sort_values("first_valid").reset_index(drop=True)


def common_start_date(df: pd.DataFrame) -> pd.Timestamp:
    """Earliest date on which every ticker has data."""
    closes = df.xs("Close", axis=1, level="field")
    return max(closes[t].dropna().index[0] for t in closes.columns)


def trim_to_common_window(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the date range where all tickers have data."""
    start = common_start_date(df)
    trimmed = df.loc[start:]
    closes = trimmed.xs("Close", axis=1, level="field")
    remaining = int(closes.isna().sum().sum())
    if remaining:
        # Sporadic mid-series holes (holidays, halts) - forward fill is safe
        # because it only ever reuses a past observation.
        trimmed = trimmed.ffill()
        print(f"Forward-filled {remaining} sporadic missing values")
    return trimmed
