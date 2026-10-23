"""VAR baseline: Vector Autoregression.

Captures linear dependencies between every pair of series without an explicit
graph. Fitted once on the training window; predictions use a rolling history so
only information available before each prediction time is used.
"""

import numpy as np
from statsmodels.tsa.api import VAR as StatsVAR


class VARForecaster:
    def __init__(self, max_lags: int = 5):
        self.max_lags = max_lags
        self.fitted = None

    def fit(self, train_returns: np.ndarray) -> "VARForecaster":
        """Fit on training log returns, shape (time, nodes).

        AIC frequently selects zero lags on daily equity returns, meaning it
        finds no linear cross-series predictability. That is a legitimate result
        rather than a failure, and it reduces VAR to a constant intercept
        forecast, so the zero-lag case is handled explicitly.
        """
        self.fitted = StatsVAR(train_returns).fit(maxlags=self.max_lags, ic="aic")
        self.selected_lags = int(self.fitted.k_ar)
        self.lag_order = max(self.selected_lags, 1)
        # Intercept row, one value per series: the zero-lag forecast
        self.intercept = np.asarray(self.fitted.params[0], dtype=float).ravel()
        return self

    def predict_windows(
        self, series: np.ndarray, window_size: int, horizon: int = 1
    ) -> np.ndarray:
        """Predict for each sliding window, matching the dataset's targets.

        For window starting at idx, inputs are series[idx : idx+window_size] and
        the target is series[idx+window_size+horizon-1], so the forecast must be
        built only from the window itself.
        """
        if self.fitted is None:
            raise RuntimeError("Call fit() before predict_windows()")

        n = len(series) - window_size - horizon + 1

        if self.selected_lags == 0:
            # Constant intercept forecast, identical for every window
            return np.tile(self.intercept, (n, 1))

        preds = []
        for idx in range(n):
            hist = series[idx : idx + window_size]
            recent = hist[-self.lag_order :]
            preds.append(self.fitted.forecast(recent, steps=horizon)[horizon - 1])
        return np.asarray(preds)
