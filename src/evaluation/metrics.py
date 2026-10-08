"""Evaluation metrics for forecasting models."""

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of times the predicted direction (up/down) matches actual."""
    correct = np.sign(y_true) == np.sign(y_pred)
    return np.mean(correct)


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, annualize: int = 252) -> float:
    """Annualized Sharpe ratio from daily returns."""
    excess = returns - risk_free / annualize
    if excess.std() == 0:
        return 0.0
    return np.sqrt(annualize) * excess.mean() / excess.std()


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all metrics and return as a dict."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }
