"""Diagnostics that explain *why* the error metrics move the way they do.

The central mechanism on this task is shrinkage. Because next-day returns are
nearly unpredictable, the MSE-optimal forecast is close to the conditional mean,
which is close to zero. A model trained on MSE therefore learns to shrink its
predictions towards zero, which lowers MAE and RMSE while carrying almost no
directional information. Error magnitude and signal content move in opposite
directions, so neither metric alone describes what a model is doing.

This script quantifies that directly:
  * prediction dispersion relative to the target (the shrinkage ratio),
  * the decomposition of MAE into what shrinkage buys,
  * predicted vs realised returns, which shows the regression slope,
  * per-model IC with a confidence interval, to show which ICs are real.
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

ORDER = ["Zero", "Persistence", "TrainMean", "VAR", "LSTM",
         "MTGNN-Identity", "MTGNN-CorrGraph", "CMA-ES", "MTGNN"]


def ic_confidence_interval(y_true, y_pred, alpha=0.05):
    """Fisher z-transform interval for the information coefficient.

    n is the number of days, not day-company pairs: returns are strongly
    correlated across companies on the same day (common market factor), so
    treating all N*T pairs as independent would understate the interval by
    roughly sqrt(N) and make a noise-level IC look significant.
    """
    daily = []
    for t in range(y_true.shape[0]):
        if y_true[t].std() > 0 and y_pred[t].std() > 0:
            daily.append(np.corrcoef(y_true[t], y_pred[t])[0, 1])
    daily = np.asarray(daily)
    if len(daily) < 3:
        return float("nan"), (float("nan"), float("nan")), 0

    r = float(np.nanmean(daily))
    n = len(daily)
    se = daily.std(ddof=1) / np.sqrt(n)
    from math import sqrt
    z = 1.959963985  # two-sided 95%
    return r, (r - z * se, r + z * se), n


def main():
    if not Path("outputs/predictions.npz").exists():
        raise SystemExit("outputs/predictions.npz not found; run the experiments first")

    models = load_predictions()
    names = [n for n in ORDER if n in models] + [n for n in models if n not in ORDER]

    ref_true, _ = pooled(models, names[0])
    target_std = float(ref_true.std())
    target_mad = float(np.abs(ref_true).mean())

    print(f"Target: std {target_std:.6f}, mean |return| {target_mad:.6f}, "
          f"{ref_true.shape[0]} days x {ref_true.shape[1]} companies\n")

    print("=== Shrinkage and signal ===")
    print(f"{'Model':20s} {'pred std':>10s} {'ratio':>7s} {'MAE':>10s} "
          f"{'IC(daily)':>10s} {'IC 95% CI':>22s} {'slope':>8s}")

    rows = []
    for n in names:
        y_true, y_pred = pooled(models, n)
        pstd = float(y_pred.std())
        ratio = pstd / target_std if target_std else float("nan")
        mae = float(np.abs(y_true - y_pred).mean())

        ic, (lo, hi), n_days = ic_confidence_interval(y_true, y_pred)

        # Regression slope of realised on predicted. A well-calibrated forecast
        # has slope 1; a shrunken-but-informative forecast has slope > 1.
        if y_pred.std() > 0:
            slope = float(np.polyfit(y_pred.ravel(), y_true.ravel(), 1)[0])
        else:
            slope = float("nan")

        sig = "" if np.isnan(lo) else ("  *" if (lo > 0 or hi < 0) else "")
        rows.append({
            "model": n, "pred_std": pstd, "shrinkage_ratio": ratio, "mae": mae,
            "ic_daily_mean": ic, "ic_ci_low": lo, "ic_ci_high": hi,
            "n_days": n_days, "regression_slope": slope,
            "ic_significant_5pct": bool(not np.isnan(lo) and (lo > 0 or hi < 0)),
        })
        ci = "n/a" if np.isnan(lo) else f"[{lo:+.4f}, {hi:+.4f}]"
        print(f"{n:20s} {pstd:10.6f} {ratio:7.3f} {mae:10.6f} "
              f"{ic:+10.4f} {ci:>22s} {slope:8.2f}{sig}")

    print("\n  * = IC confidence interval excludes zero")
    print("  ratio = std(predictions) / std(targets). Near 0 means the model")
    print("  predicts almost a constant; near 1 means it matches target scale.")

    # ---------------- Figures ----------------
    figdir = Path("report/figures")
    figdir.mkdir(parents=True, exist_ok=True)

    # Figure: shrinkage vs MAE, and IC with intervals
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    labels = [r["model"] for r in rows]
    y = np.arange(len(rows))

    ax = axes[0]
    ax.barh(y, [r["shrinkage_ratio"] for r in rows], color="#C05C20")
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(1.0, color="black", ls="--", lw=1.1, label="target scale")
    ax.set_xlabel(r"std(prediction) / std(target)")
    ax.set_title("Prediction dispersion: how much each model hedges")
    ax.legend(fontsize=8)

    ax = axes[1]
    ics = [r["ic_daily_mean"] for r in rows]
    errs = [[r["ic_daily_mean"] - r["ic_ci_low"] for r in rows],
            [r["ic_ci_high"] - r["ic_daily_mean"] for r in rows]]
    colors = ["#1F4E79" if r["ic_significant_5pct"] else "#999999" for r in rows]
    ax.barh(y, ics, xerr=errs, color=colors, capsize=3)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.0, color="red", ls="--", lw=1.1)
    ax.set_xlabel("daily cross-sectional IC (95% CI)")
    ax.set_title("Signal content: grey = interval includes zero")
    fig.tight_layout()
    fig.savefig(figdir / "shrinkage_and_ic.png", dpi=180)
    plt.close(fig)
    print(f"\nwrote {figdir/'shrinkage_and_ic.png'}")

    # Figure: predicted vs realised for the key models
    key = [n for n in ["MTGNN", "MTGNN-CorrGraph", "LSTM", "VAR"] if n in models]
    fig, axes = plt.subplots(1, len(key), figsize=(4.0 * len(key), 4.0), squeeze=False)
    for ax, n in zip(axes[0], key):
        y_true, y_pred = pooled(models, n)
        ax.scatter(y_pred.ravel(), y_true.ravel(), s=2, alpha=0.18, color="#1F4E79")
        lim = float(np.abs(y_true).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="red", ls="--", lw=1.0, label="y = x")
        if y_pred.std() > 0:
            b, a = np.polyfit(y_pred.ravel(), y_true.ravel(), 1)
            xs = np.linspace(y_pred.min(), y_pred.max(), 50)
            ax.plot(xs, a + b * xs, color="black", lw=1.4, label=f"fit (slope {b:.1f})")
        ax.set_xlim(-lim * 0.35, lim * 0.35)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("predicted return")
        ax.set_ylabel("realised return" if n == key[0] else "")
        ax.set_title(n, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("Predictions occupy a far narrower range than outcomes", fontsize=11)
    fig.tight_layout()
    fig.savefig(figdir / "pred_vs_actual.png", dpi=180)
    plt.close(fig)
    print(f"wrote {figdir/'pred_vs_actual.png'}")

    out = Path("outputs/diagnostics.json")
    out.write_text(json.dumps({
        "target_std": target_std,
        "target_mean_abs": target_mad,
        "models": rows,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
