"""Download and preprocess stock data. Run this first.

Trims the panel to the window where every ticker has data. Two of the
agriculture names did not exist in 2015 (Nutrien was formed by the 2018
Agrium/PotashCorp merger, Corteva was spun off from DowDuPont in 2019), so the
usable common window starts later than the configured start date.
"""

import sys
sys.path.insert(0, ".")

import yaml
import numpy as np
from pathlib import Path
from src.data.download import (
    download_stock_data,
    coverage_report,
    trim_to_common_window,
)
from src.data.features import compute_features, build_tensor


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    tickers = cfg["data"]["semiconductors"] + cfg["data"]["agriculture"]
    print(f"Tickers ({len(tickers)}): {tickers}\n")

    df = download_stock_data(
        tickers=tickers,
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
    )

    print("\nCoverage before trimming:")
    print(coverage_report(df).to_string(index=False))

    df = trim_to_common_window(df)
    print(f"\nCommon window: {df.index[0].date()} to {df.index[-1].date()} "
          f"({len(df)} trading days)")

    features = compute_features(df, volatility_window=cfg["features"]["volatility_window"])
    tensor = build_tensor(features)

    if np.isnan(tensor).any():
        n = int(np.isnan(tensor).sum())
        raise ValueError(
            f"Feature tensor still contains {n} NaNs after trimming. "
            "Inspect the coverage report above."
        )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", tensor)

    ticker_list = list(features["log_return"].columns)
    (out_dir / "tickers.txt").write_text("\n".join(ticker_list))

    dates = features["log_return"].index
    (out_dir / "dates.txt").write_text(
        "\n".join(d.strftime("%Y-%m-%d") for d in dates)
    )

    print(f"\nFeature tensor: {tensor.shape} "
          f"(time={tensor.shape[0]}, nodes={tensor.shape[1]}, features={tensor.shape[2]})")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"NaNs: {int(np.isnan(tensor).sum())}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
