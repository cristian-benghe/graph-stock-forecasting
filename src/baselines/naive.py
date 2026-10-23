"""Naive reference predictors.

Daily equity returns are close to unpredictable, so a model that always predicts
zero achieves a very low MAE while carrying no information at all. Reporting
these alongside the learned models prevents a misreading where a low error is
taken as evidence of skill.
"""

import numpy as np


def zero_forecast(y_true: np.ndarray) -> np.ndarray:
    """Always predict a zero return (the unconditional-mean forecast)."""
    return np.zeros_like(y_true)


def persistence_forecast(returns: np.ndarray, window_size: int, horizon: int = 1) -> np.ndarray:
    """Predict tomorrow's return equals today's (random-walk-with-drift style).

    Aligned to the same targets the sliding-window datasets produce.
    """
    preds = []
    for idx in range(len(returns) - window_size - horizon + 1):
        preds.append(returns[idx + window_size - 1])
    return np.asarray(preds)


def train_mean_forecast(
    train_returns: np.ndarray, n_samples: int
) -> np.ndarray:
    """Predict each node's training-period mean return, repeated n_samples times."""
    mu = train_returns.mean(axis=0)
    return np.tile(mu, (n_samples, 1))
