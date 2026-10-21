"""Sliding-window PyTorch dataset for multivariate time-series forecasting."""

import numpy as np
import torch
from torch.utils.data import Dataset


class StockGraphDataset(Dataset):
    """Sliding window dataset.

    Each sample is:
        x: (window_size, num_nodes, num_features) - historical observations
        y: (num_nodes,) - next-day log returns for all nodes
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 20,
        horizon: int = 1,
    ):
        self.data = data.astype(np.float32)
        self.window_size = window_size
        self.horizon = horizon
        self.num_samples = len(data) - window_size - horizon + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size + self.horizon - 1, :, 0]
        return torch.from_numpy(x), torch.from_numpy(y)


def chronological_split(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    window_size: int = 20,
    horizon: int = 1,
) -> tuple[StockGraphDataset, StockGraphDataset, StockGraphDataset]:
    """Split data chronologically into train/val/test datasets."""
    T = len(data)
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end - window_size : val_end]
    test_data = data[val_end - window_size :]

    return (
        StockGraphDataset(train_data, window_size, horizon),
        StockGraphDataset(val_data, window_size, horizon),
        StockGraphDataset(test_data, window_size, horizon),
    )
