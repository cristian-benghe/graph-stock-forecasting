"""Feature engineering: log returns, rolling volatility, normalized volume."""

import numpy as np
import pandas as pd


def compute_features(
    prices_df: pd.DataFrame,
    volatility_window: int = 20,
) -> dict[str, pd.DataFrame]:
    """Compute per-ticker features from raw price/volume data.

    Returns dict with keys 'log_return', 'volatility', 'volume',
    each a DataFrame of shape (time, num_tickers).
    """
    tickers = prices_df.columns.get_level_values("ticker").unique()

    closes = prices_df.xs("Close", axis=1, level="field")[tickers]
    volumes = prices_df.xs("Volume", axis=1, level="field")[tickers]

    log_returns = np.log(closes / closes.shift(1))

    volatility = log_returns.rolling(window=volatility_window).std()

    vol_mean = volumes.rolling(window=volatility_window).mean()
    vol_std = volumes.rolling(window=volatility_window).std()
    norm_volume = (volumes - vol_mean) / (vol_std + 1e-8)

    first_valid = max(
        log_returns.first_valid_index(),
        volatility.first_valid_index(),
        norm_volume.first_valid_index(),
    )
    start_idx = log_returns.index.get_loc(first_valid)
    safe_start = max(start_idx, volatility_window + 1)

    log_returns = log_returns.iloc[safe_start:]
    volatility = volatility.iloc[safe_start:]
    norm_volume = norm_volume.iloc[safe_start:]

    return {
        "log_return": log_returns,
        "volatility": volatility,
        "volume": norm_volume,
    }


def build_tensor(
    features: dict[str, pd.DataFrame],
) -> np.ndarray:
    """Stack features into a single array of shape (time, num_nodes, num_features)."""
    arrays = [features[k].values for k in ["log_return", "volatility", "volume"]]
    return np.stack(arrays, axis=-1)
