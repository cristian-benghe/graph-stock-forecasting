"""Evaluation metrics and multi-seed aggregation."""

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with the correct sign.

    Exact zeros in the target are dropped, since a zero return has no direction
    to get right. A predictor that outputs only zeros makes no directional call
    at all, so its accuracy is undefined (NaN) rather than 0; scoring it as 0
    would wrongly suggest it is worse than chance.
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    if np.all(y_pred == 0):
        return float("nan")
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cross-sectional Pearson correlation between predictions and outcomes.

    The standard signal-quality measure in quantitative finance; more
    informative than MAE when the signal is weak relative to the noise.
    """
    yt, yp = y_true.ravel(), y_pred.ravel()
    if yt.std() == 0 or yp.std() == 0:
        return 0.0
    return float(np.corrcoef(yt, yp)[0, 1])


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio of a return series (zero risk-free rate)."""
    r = np.asarray(returns, dtype=float).ravel()
    if r.size == 0 or r.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std())


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "information_coefficient": information_coefficient(y_true, y_pred),
    }


def aggregate_seeds(results: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and std for each metric across seed runs."""
    if not results:
        return {}
    return {
        key: {
            "mean": float(np.mean([r[key] for r in results])),
            "std": float(np.std([r[key] for r in results])),
        }
        for key in results[0]
    }


def format_mean_std(agg: dict[str, dict[str, float]], key: str, decimals: int = 5) -> str:
    if key not in agg:
        return "n/a"
    return f"{agg[key]['mean']:.{decimals}f} ± {agg[key]['std']:.{decimals}f}"
