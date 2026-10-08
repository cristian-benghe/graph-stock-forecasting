"""VAR baseline: Vector Autoregression using statsmodels."""

import numpy as np
from statsmodels.tsa.api import VAR as StatsVAR


class VARForecaster:
    """Wrapper around statsmodels VAR for consistent evaluation interface."""

    def __init__(self, max_lags: int = 5):
        self.max_lags = max_lags
        self.model = None
        self.fitted = None

    def fit(self, train_returns: np.ndarray):
        """Fit VAR on training log-returns. Shape: (time, num_nodes)."""
        model = StatsVAR(train_returns)
        self.fitted = model.fit(maxlags=self.max_lags, ic="aic")
        print(f"VAR fitted with {self.fitted.k_ar} lags")

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Predict next step given recent history. Returns (num_nodes,)."""
        lag_order = self.fitted.k_ar
        recent = history[-lag_order:]
        return self.fitted.forecast(recent, steps=1)[0]

    def predict_rolling(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Generate predictions for the full test set using rolling windows."""
        predictions = []
        for t in range(window_size, len(data)):
            history = data[t - window_size : t]
            pred = self.predict(history)
            predictions.append(pred)
        return np.array(predictions)
