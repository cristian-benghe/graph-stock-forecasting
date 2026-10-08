"""Download and preprocess stock data. Run this first."""

import sys
sys.path.insert(0, ".")

import yaml
import numpy as np
from pathlib import Path
from src.data.download import download_stock_data
from src.data.features import compute_features, build_tensor


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    tickers = cfg["data"]["semiconductors"] + cfg["data"]["agriculture"]
    print(f"Tickers ({len(tickers)}): {tickers}")

    df = download_stock_data(
        tickers=tickers,
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
    )

    features = compute_features(df, volatility_window=cfg["features"]["volatility_window"])
    tensor = build_tensor(features)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", tensor)

    print(f"Feature tensor shape: {tensor.shape}")
    print(f"  (time_steps={tensor.shape[0]}, nodes={tensor.shape[1]}, features={tensor.shape[2]})")
    print(f"Saved to {out_dir / 'features.npy'}")

    # Save ticker order for later graph visualization
    ticker_list = list(features["log_return"].columns)
    with open(out_dir / "tickers.txt", "w") as f:
        f.write("\n".join(ticker_list))


if __name__ == "__main__":
    main()
