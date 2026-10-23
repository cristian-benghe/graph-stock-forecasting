"""Diebold-Mariano tests on saved predictions.

With model differences this small, point estimates alone cannot establish
whether any gap is real. The Diebold-Mariano test (Diebold and Mariano, 1995)
compares two forecasts' loss differentials and tests whether the mean
differential is zero, using a Newey-West HAC variance estimate to allow for
serial correlation in the loss series.
"""

import sys
sys.path.insert(0, ".")

import json
from itertools import combinations
from pathlib import Path

import numpy as np

REFERENCE = "Zero"


def newey_west_variance(d: np.ndarray, lag: int | None = None) -> float:
    """HAC long-run variance of the loss differential series."""
    n = len(d)
    if lag is None:
        lag = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    d = d - d.mean()
    gamma0 = float(np.dot(d, d) / n)
    total = gamma0
    for k in range(1, lag + 1):
        gk = float(np.dot(d[k:], d[:-k]) / n)
        total += 2.0 * (1.0 - k / (lag + 1)) * gk
    return max(total, 1e-24)


def diebold_mariano(y_true, pred_a, pred_b, loss="abs"):
    """Test H0: forecasts a and b have equal expected loss.

    Losses are averaged across companies per day, so the differential is a
    time series and the HAC correction addresses autocorrelation in it.
    A negative statistic means a has lower loss than b.
    """
    if loss == "abs":
        la = np.abs(y_true - pred_a)
        lb = np.abs(y_true - pred_b)
    else:
        la = (y_true - pred_a) ** 2
        lb = (y_true - pred_b) ** 2

    d = (la - lb).mean(axis=1)  # daily mean loss differential
    n = len(d)
    var = newey_west_variance(d)
    stat = float(d.mean() / np.sqrt(var / n))

    # Normal approximation; two-sided
    from math import erfc, sqrt
    p = float(erfc(abs(stat) / sqrt(2.0)))
    return {"statistic": stat, "p_value": p, "mean_diff": float(d.mean()), "n_days": n}


def load_predictions(path="outputs/predictions.npz"):
    blob = np.load(path)
    models = {}
    for key in blob.files:
        name, seed, kind = key.split("|")
        models.setdefault(name, {}).setdefault(seed, {})[kind] = blob[key]
    return models


def pooled(models, name):
    """Average predictions across seeds; targets are identical across seeds."""
    seeds = sorted(models[name])
    preds = np.mean([models[name][s]["pred"] for s in seeds], axis=0)
    true = models[name][seeds[0]]["true"]
    for s in seeds:
        assert np.allclose(models[name][s]["true"], true), f"{name}: targets differ across seeds"
    return true, preds


def main():
    path = Path("outputs/predictions.npz")
    if not path.exists():
        raise SystemExit("outputs/predictions.npz not found; run run_experiments.py first")

    models = load_predictions(path)
    names = list(models)
    print(f"Loaded predictions for {len(names)} models: {names}\n")

    # Verify every model was scored on the same targets
    ref_true, _ = pooled(models, names[0])
    for n in names:
        t, _ = pooled(models, n)
        assert t.shape == ref_true.shape, f"{n} target shape {t.shape} != {ref_true.shape}"
        assert np.allclose(t, ref_true), f"{n} was scored on different targets"
    print(f"All models scored on identical targets: {ref_true.shape[0]} days x "
          f"{ref_true.shape[1]} companies\n")

    results = {}

    print(f"=== Diebold-Mariano vs the {REFERENCE} reference (MAE loss) ===")
    print(f"{'Model':22s} {'DM stat':>10s} {'p-value':>10s}  verdict")
    if REFERENCE in models:
        _, ref_pred = pooled(models, REFERENCE)
        for n in names:
            if n == REFERENCE:
                continue
            _, p = pooled(models, n)
            r = diebold_mariano(ref_true, p, ref_pred)
            verdict = ("better" if r["statistic"] < 0 else "worse") if r["p_value"] < 0.05 \
                else "no significant difference"
            results[f"{n}_vs_{REFERENCE}"] = r
            print(f"{n:22s} {r['statistic']:>10.3f} {r['p_value']:>10.4f}  {verdict}")

    print("\n=== Pairwise comparisons among the graph variants (MAE loss) ===")
    graph_models = [n for n in
                    ["MTGNN", "MTGNN-CorrGraph", "MTGNN-Identity", "CMA-ES"]
                    if n in models]
    print(f"{'Comparison':46s} {'DM stat':>10s} {'p-value':>10s}")
    for a, b in combinations(graph_models, 2):
        _, pa = pooled(models, a)
        _, pb = pooled(models, b)
        r = diebold_mariano(ref_true, pa, pb)
        results[f"{a}_vs_{b}"] = r
        print(f"{a + ' vs ' + b:46s} {r['statistic']:>10.3f} {r['p_value']:>10.4f}")

    n_sig = sum(1 for r in results.values() if r["p_value"] < 0.05)
    print(f"\nSignificant at the 5% level: {n_sig}/{len(results)} comparisons")

    out = Path("outputs/significance_tests.json")
    out.write_text(json.dumps(
        {"n_comparisons": len(results), "n_significant_5pct": n_sig, "tests": results},
        indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
