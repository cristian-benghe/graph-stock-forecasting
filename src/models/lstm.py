"""LSTM baseline: treats each node independently (no graph structure)."""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """Multi-input LSTM that processes all nodes but without inter-node message passing."""

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, num_nodes, in_features)
        Returns:
            predictions: (batch, num_nodes)
        """
        B, T, N, F = x.shape

        # Process each node's time series independently
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (B*N, hidden)
        preds = self.output_proj(last_hidden)  # (B*N, 1)
        return preds.reshape(B, N)
