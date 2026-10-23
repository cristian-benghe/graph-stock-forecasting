"""Optional financial evaluation: a simple long-short rule per model.

Each day, rank companies by predicted return, go long the top k and short the
bottom k with equal weights, dollar-neutral. The same rule is applied to every
model's predictions so the comparison reflects signal quality rather than
differing trading logic.

Transaction costs are reported because a daily-rebalanced 20-name strategy
turns over its book constantly; a gross result that ignores costs is not a
claim about tradability.
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.significance_tests import load_predictions, pooled
from src.evaluation.metrics import sharpe_ratio

TOP_K = 5
COST_PER_SIDE_BPS = 5.0  # 5 basis points per unit traded, each side


def long_short_returns(y_true, y_pred, top_k=TOP_K, cost_bps=0.0):
    """Daily returns of a dollar-neutral top-k/bottom-k rule."""
    T, N = y_true.shape
    weights = np.zeros((T, N))
    for t in range(T):
        order = np.argsort(y_pred[t])
        shorts, longs = order[:top_k], order[-top_k:]
        weights[t, longs] = 1.0 / top_k
        weights[t, shorts] = -1.0 / top_k

    gross = (weights * y_true).sum(axis=1)
    if cost_bps <= 0:
        return gross, weights

    # Turnover: absolute weight change between consecutive days
    prev = np.vstack([np.zeros((1, N)), weights[:-1]])
    turnover = np.abs(weights - prev).sum(axis=1)
    return gross - turnover * (cost_bps / 10_000.0), weights


def max_drawdown(cumulative: np.ndarray) -> float:
    peak = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - peak))


def is_degenerate(y_pred) -> bool:
    """True when the forecast carries no cross-sectional ranking information.

    If predictions are identical across companies on a given day, the long-short
    ranking is decided entirely by argsort tie-breaking, i.e. by the order
    companies happen to appear in the data file. The resulting basket is a fixed
    arbitrary bet rather than a strategy, so its Sharpe ratio says nothing about
    the forecast and must not be read as performance.
    """
    return bool(np.allclose(y_pred.std(axis=1), 0.0))


def evaluate_strategy(y_true, y_pred, label):
    gross, weights = long_short_returns(y_true, y_pred, cost_bps=0.0)
    net, _ = long_short_returns(y_true, y_pred, cost_bps=COST_PER_SIDE_BPS)

    prev = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
    avg_turnover = float(np.abs(weights - prev).sum(axis=1).mean())

    out = {
        "model": label,
        "gross_sharpe": sharpe_ratio(gross),
        "net_sharpe": sharpe_ratio(net),
        "gross_cumulative_return": float(gross.sum()),
        "net_cumulative_return": float(net.sum()),
        "gross_mean_daily": float(gross.mean()),
        "max_drawdown_net": max_drawdown(np.cumsum(net)),
        "hit_rate": float((gross > 0).mean()),
        "avg_daily_turnover": avg_turnover,
        "n_days": int(len(gross)),
        "degenerate_ranking": is_degenerate(y_pred),
        "constant_basket": bool(np.allclose(weights, weights[0])),
    }
    return out, net


def main():
    if not Path("outputs/predictions.npz").exists():
        raise SystemExit("outputs/predictions.npz not found; run run_experiments.py first")

    models = load_predictions()
    order = ["Zero", "Persistence", "TrainMean", "VAR", "LSTM",
             "MTGNN-Identity", "MTGNN-CorrGraph", "CMA-ES", "MTGNN"]
    names = [n for n in order if n in models] + [n for n in models if n not in order]

    print(f"Long-short top-{TOP_K}/bottom-{TOP_K}, dollar-neutral, "
          f"daily rebalance, {COST_PER_SIDE_BPS:.0f} bps per side\n")
    print(f"{'Model':20s} {'Sharpe(gross)':>14s} {'Sharpe(net)':>12s} "
          f"{'CumRet(net)':>12s} {'MaxDD':>9s} {'Turnover':>9s}  note")

    rows, curves = [], {}
    for n in names:
        y_true, y_pred = pooled(models, n)
        row, net = evaluate_strategy(y_true, y_pred, n)
        rows.append(row)
        curves[n] = np.cumsum(net)
        note = ""
        if row["degenerate_ranking"]:
            note = "DEGENERATE: ranking is argsort tie-break, not a forecast"
        elif row["constant_basket"]:
            note = "constant basket (static bet)"
        print(f"{n:20s} {row['gross_sharpe']:>14.3f} {row['net_sharpe']:>12.3f} "
              f"{row['net_cumulative_return']:>12.4f} {row['max_drawdown_net']:>9.4f} "
              f"{row['avg_daily_turnover']:>9.3f}  {note}")

    degenerate = [r["model"] for r in rows if r["degenerate_ranking"]]
    if degenerate:
        print(f"\nWARNING: {degenerate} produce identical predictions across "
              "companies, so their Sharpe ratios reflect an arbitrary fixed basket "
              "chosen by data-file ordering. Do not interpret them as skill.")

    figdir = Path("report/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for n, c in curves.items():
        style = dict(lw=2.0) if n == "MTGNN" else dict(lw=1.2, alpha=0.8)
        if n in ("Zero", "Persistence", "TrainMean"):
            style["ls"] = "--"
        ax.plot(c, label=n, **style)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("test-period trading day")
    ax.set_ylabel("cumulative net log return")
    ax.set_title(f"Long-short top-{TOP_K} strategy, net of {COST_PER_SIDE_BPS:.0f} bps per side")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figdir / "portfolio.png", dpi=180)
    plt.close(fig)
    print(f"\nwrote {figdir/'portfolio.png'}")

    out = Path("outputs/portfolio.json")
    out.write_text(json.dumps({
        "top_k": TOP_K,
        "cost_per_side_bps": COST_PER_SIDE_BPS,
        "results": rows,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
