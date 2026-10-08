"""MTGNN: Connecting the Dots (Wu et al., 2020).

Learns a directed adjacency matrix from trainable node embeddings,
then applies graph convolution + temporal convolution for forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLearner(nn.Module):
    """Learn a sparse directed adjacency matrix from node embeddings."""

    def __init__(self, num_nodes: int, node_dim: int, top_k: int = 10):
        super().__init__()
        self.emb_s = nn.Embedding(num_nodes, node_dim)
        self.emb_t = nn.Embedding(num_nodes, node_dim)
        self.top_k = top_k

    def forward(self) -> torch.Tensor:
        s = torch.tanh(self.emb_s.weight)
        t = torch.tanh(self.emb_t.weight)
        adj = F.relu(s @ t.T)

        # Sparsify: keep only top-k per row
        if self.top_k < adj.size(0):
            topk_vals, topk_idx = torch.topk(adj, self.top_k, dim=1)
            mask = torch.zeros_like(adj)
            mask.scatter_(1, topk_idx, 1.0)
            adj = adj * mask

        # Row-normalize
        row_sum = adj.sum(dim=1, keepdim=True) + 1e-8
        return adj / row_sum


class MixHopGraphConv(nn.Module):
    """Mix-hop graph convolution: aggregate at multiple hop distances."""

    def __init__(self, in_channels: int, out_channels: int, num_hops: int = 2):
        super().__init__()
        self.num_hops = num_hops
        self.linear = nn.Linear(in_channels * (num_hops + 1), out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_nodes, channels)
        outputs = [x]
        h = x
        for _ in range(self.num_hops):
            h = torch.einsum("ij,bjc->bic", adj, h)
            outputs.append(h)
        return self.linear(torch.cat(outputs, dim=-1))


class DilatedTemporalConv(nn.Module):
    """Dilated causal temporal convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        out = self.conv(x)
        out = out[..., :x.size(-1)]  # causal: trim future
        return F.relu(self.bn(out))


class MTGNNLayer(nn.Module):
    """Single MTGNN layer: temporal conv -> graph conv -> residual."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, num_hops: int = 2):
        super().__init__()
        self.temporal = DilatedTemporalConv(channels, channels, kernel_size, dilation)
        self.graph = MixHopGraphConv(channels, channels, num_hops)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_nodes, time, channels)
        B, N, T, C = x.shape
        residual = x

        # Temporal conv: reshape to (B*N, C, T)
        h = x.reshape(B * N, T, C).permute(0, 2, 1)
        h = self.temporal(h)
        h = h.permute(0, 2, 1).reshape(B, N, T, C)

        # Graph conv at each time step
        out = []
        for t in range(T):
            out.append(self.graph(h[:, :, t, :], adj))
        h = torch.stack(out, dim=2)

        return self.norm(h + residual)


class MTGNN(nn.Module):
    """Full MTGNN model for multivariate time-series forecasting."""

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_channels: int = 32,
        num_layers: int = 3,
        node_dim: int = 40,
        temporal_kernel: int = 3,
        top_k: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.graph_learner = GraphLearner(num_nodes, node_dim, top_k)
        self.input_proj = nn.Linear(in_features, hidden_channels)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.layers.append(MTGNNLayer(hidden_channels, temporal_kernel, dilation))

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, num_nodes, in_features)
        Returns:
            predictions: (batch, num_nodes) - predicted next-day returns
        """
        B, T, N, F = x.shape
        adj = self.graph_learner()

        # Project input features
        h = self.input_proj(x)  # (B, T, N, C)
        h = h.permute(0, 2, 1, 3)  # (B, N, T, C)

        for layer in self.layers:
            h = self.dropout(layer(h, adj))

        # Use last time step for prediction
        h = h[:, :, -1, :]  # (B, N, C)
        return self.output_proj(h).squeeze(-1)  # (B, N)

    def get_learned_graph(self) -> torch.Tensor:
        """Return the current learned adjacency matrix."""
        with torch.no_grad():
            return self.graph_learner().cpu()
