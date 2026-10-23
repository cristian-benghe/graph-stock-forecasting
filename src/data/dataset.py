"""Sliding-window dataset with chronological splits and train-only normalization."""

import numpy as np
import torch
from torch.utils.data import Dataset


class Normalizer:
    """Per-node, per-feature standardization fitted on training data only.

    Fitting on the full series would leak test-period statistics into training,
    which inflates results. Targets are left in raw log-return units so MAE and
    RMSE stay interpretable.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray) -> "Normalizer":
        # data: (time, nodes, features)
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std


class StockGraphDataset(Dataset):
    """Each sample is a window of normalized features and the next raw return.

    x: (window, nodes, features) normalized
    y: (nodes,) raw next-step log returns
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        window_size: int = 20,
        horizon: int = 1,
    ):
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.window_size = window_size
        self.horizon = horizon
        self.num_samples = len(features) - window_size - horizon + 1
        if self.num_samples <= 0:
            raise ValueError(
                f"Need more than {window_size + horizon} rows, got {len(features)}"
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        x = self.features[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size + self.horizon - 1]
        return torch.from_numpy(x), torch.from_numpy(y)


def chronological_split(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    window_size: int = 20,
    horizon: int = 1,
    return_normalizer: bool = False,
):
    """Split chronologically, fit the normalizer on train only.

    Val and test segments are extended backwards by `window_size` so the first
    prediction in each segment has a full history. The overlap only ever reuses
    past observations as inputs, so no target is seen more than once.
    """
    T = len(data)
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    norm = Normalizer().fit(data[:train_end])

    segments = {
        "train": data[:train_end],
        "val": data[max(0, train_end - window_size - horizon) : val_end],
        "test": data[max(0, val_end - window_size - horizon) :],
    }

    datasets = {}
    for name, seg in segments.items():
        datasets[name] = StockGraphDataset(
            features=norm.transform(seg),
            targets=seg[:, :, 0],  # raw log returns
            window_size=window_size,
            horizon=horizon,
        )

    out = (datasets["train"], datasets["val"], datasets["test"])
    return (*out, norm) if return_normalizer else out
