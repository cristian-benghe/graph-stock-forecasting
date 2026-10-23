"""Error analysis: where do the models fail?

Addresses the proposal's error-analysis plan by conditioning test-set error on
realised volatility and reporting it per company, for the learned-graph MTGNN
against the zero-forecast reference.
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import chronological_split
from src.evaluation.metrics import directional_accuracy, mae
from src.models.mtgnn import MTGNN
from src.training.trainer import predict, set_seed, train_model


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cpu")
    data = np.load("data/processed/features.npy")
    tickers = Path("data/processed/tickers.txt").read_text().split()
    dates = Path("data/processed/dates.txt").read_text().split()

    m = cfg["mtgnn"]
    seed = cfg["evaluation"]["seeds"][0]
    set_seed(seed)

    train_ds, val_ds, test_ds = chronological_split(
        data,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        window_size=cfg["features"]["window_size"],
        horizon=cfg["features"]["horizon"],
    )
    train_loader = DataLoader(train_ds, batch_size=m["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=m["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=m["batch_size"])

    model = MTGNN(
        num_nodes=data.shape[1], in_features=data.shape[2],
        hidden_channels=m["hidden_channels"], num_layers=m["num_layers"],
        node_dim=m["node_dim"], temporal_kernel=m["temporal_kernel"],
        top_k=m["top_k"], dropout=m["dropout"],
    )
    print("Training MTGNN for error analysis...")
    out = train_model(
        model, train_loader, val_loader,
        epochs=m["epochs"], learning_rate=m["learning_rate"],
        weight_decay=m["weight_decay"], device=device, verbose=False,
    )
    y_true, y_pred = predict(out["model"], test_loader, device)

    # Align the volatility feature to the same targets the test loader produced
    T = len(data)
    val_end = int(T * (cfg["split"]["train_ratio"] + cfg["split"]["val_ratio"]))
    w, h = cfg["features"]["window_size"], cfg["features"]["horizon"]
    offset = max(0, val_end - w - h)
    seg_vol = data[offset:, :, 1]
    seg_dates = dates[offset:]
    idxs = [i + w + h - 1 for i in range(len(seg_vol) - w - h + 1)]
    vol = seg_vol[idxs]
    tgt_dates = [seg_dates[i] for i in idxs]

    assert vol.shape == y_true.shape, (vol.shape, y_true.shape)

    abs_err = np.abs(y_true - y_pred)
    zero_err = np.abs(y_true)

    summary = {
        "test_period": {"start": tgt_dates[0], "end": tgt_dates[-1], "n_days": len(tgt_dates)},
        "overall": {
            "mtgnn_mae": float(abs_err.mean()),
            "zero_mae": float(zero_err.mean()),
        },
    }

    # --- Error by volatility quartile ---
    q = np.quantile(vol, [0.25, 0.5, 0.75])
    bucket = np.digitize(vol, q)
    print("\n=== MAE by realised-volatility quartile ===")
    print(f"{'quartile':10s} {'n':>7s} {'MTGNN MAE':>12s} {'Zero MAE':>12s} {'MTGNN DA':>10s}")
    by_vol = []
    for b, name in enumerate(["Q1 (calm)", "Q2", "Q3", "Q4 (volatile)"]):
        msk = bucket == b
        row = {
            "quartile": name,
            "n": int(msk.sum()),
            "mtgnn_mae": float(abs_err[msk].mean()),
            "zero_mae": float(zero_err[msk].mean()),
            "directional_accuracy": directional_accuracy(y_true[msk], y_pred[msk]),
            "mean_volatility": float(vol[msk].mean()),
        }
        by_vol.append(row)
        print(f"{name:10s} {row['n']:7d} {row['mtgnn_mae']:12.6f} "
              f"{row['zero_mae']:12.6f} {row['directional_accuracy']:10.4f}")
    summary["by_volatility_quartile"] = by_vol

    # --- Error per company ---
    print("\n=== Per-company test error ===")
    print(f"{'ticker':8s} {'MTGNN MAE':>12s} {'Zero MAE':>12s} {'ratio':>8s} {'DA':>8s}")
    per_node = []
    for i, t in enumerate(tickers):
        row = {
            "ticker": t,
            "mtgnn_mae": float(abs_err[:, i].mean()),
            "zero_mae": float(zero_err[:, i].mean()),
            "directional_accuracy": directional_accuracy(y_true[:, i], y_pred[:, i]),
        }
        row["ratio"] = row["mtgnn_mae"] / row["zero_mae"] if row["zero_mae"] else float("nan")
        per_node.append(row)
    for row in sorted(per_node, key=lambda r: -r["ratio"]):
        print(f"{row['ticker']:8s} {row['mtgnn_mae']:12.6f} {row['zero_mae']:12.6f} "
              f"{row['ratio']:8.3f} {row['directional_accuracy']:8.4f}")
    summary["per_company"] = per_node

    n_better = sum(1 for r in per_node if r["ratio"] < 1.0)
    n_above_chance = sum(1 for r in per_node if r["directional_accuracy"] > 0.5)
    summary["n_companies_beating_zero"] = n_better
    summary["n_companies_above_chance_direction"] = n_above_chance
    print(f"\nCompanies where MTGNN beats the zero forecast: {n_better}/{len(tickers)}")
    print(f"Companies with directional accuracy > 0.50: {n_above_chance}/{len(tickers)}")

    # --- Figure ---
    figdir = Path("outputs/figures"); figdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    x = np.arange(len(by_vol))
    axes[0].bar(x - 0.2, [r["mtgnn_mae"] for r in by_vol], 0.4, label="MTGNN", color="#1F4E79")
    axes[0].bar(x + 0.2, [r["zero_mae"] for r in by_vol], 0.4, label="Zero forecast", color="#888888")
    axes[0].set_xticks(x, [r["quartile"] for r in by_vol], fontsize=9)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Test MAE by realised-volatility quartile")
    axes[0].legend(fontsize=8)

    order = np.argsort([r["mtgnn_mae"] for r in per_node])
    axes[1].barh(np.arange(len(per_node)),
                 [per_node[i]["mtgnn_mae"] for i in order], color="#1F4E79")
    axes[1].set_yticks(np.arange(len(per_node)),
                       [per_node[i]["ticker"] for i in order], fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("MAE")
    axes[1].set_title("Per-company test MAE (MTGNN)")

    fig.tight_layout()
    fig.savefig(figdir / "error_analysis.png", dpi=180)
    plt.close(fig)
    print(f"\nwrote {figdir/'error_analysis.png'}")

    out_path = Path("outputs/error_analysis.json")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
