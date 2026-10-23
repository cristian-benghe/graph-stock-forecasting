"""MTGNN: Connecting the Dots (Wu et al., 2020).

Learns a directed adjacency matrix from trainable node embeddings, then applies
mix-hop graph convolution and dilated temporal convolution for forecasting.

`forward` accepts an optional external adjacency so the same backbone can be
driven by a learned graph (MTGNN), a fixed correlation graph (ablation), or a
CMA-ES candidate, keeping the comparison to a single varying factor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparsify_topk(adj: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep the top-k entries per row, then row-normalize to sum to 1."""
    if 0 < top_k < adj.size(0):
        topk_idx = torch.topk(adj, top_k, dim=1).indices
        mask = torch.zeros_like(adj)
        mask.scatter_(1, topk_idx, 1.0)
        adj = adj * mask
    return adj / (adj.sum(dim=1, keepdim=True) + 1e-8)


class GraphLearner(nn.Module):
    """Learn a sparse directed adjacency matrix from node embeddings.

    Uses the asymmetric formulation from the MTGNN paper: subtracting the
    transpose makes the relation directed, so an edge j->i does not imply i->j.
    """

    def __init__(self, num_nodes: int, node_dim: int, top_k: int = 10, alpha: float = 3.0):
        super().__init__()
        self.emb_s = nn.Embedding(num_nodes, node_dim)
        self.emb_t = nn.Embedding(num_nodes, node_dim)
        self.lin_s = nn.Linear(node_dim, node_dim)
        self.lin_t = nn.Linear(node_dim, node_dim)
        self.top_k = top_k
        self.alpha = alpha

    def forward(self) -> torch.Tensor:
        s = torch.tanh(self.alpha * self.lin_s(self.emb_s.weight))
        t = torch.tanh(self.alpha * self.lin_t(self.emb_t.weight))
        # Asymmetric: s@t.T - t@s.T is antisymmetric, so relu keeps one direction
        adj = F.relu(torch.tanh(self.alpha * (s @ t.T - t @ s.T)))
        return sparsify_topk(adj, self.top_k)


class MixHopGraphConv(nn.Module):
    """Mix-hop graph convolution over a (batch, nodes, time, channels) tensor.

    Aggregates at hop distances 0..num_hops and mixes them with a linear layer.
    """

    def __init__(self, in_channels: int, out_channels: int, num_hops: int = 2):
        super().__init__()
        self.num_hops = num_hops
        self.linear = nn.Linear(in_channels * (num_hops + 1), out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        outputs = [x]
        h = x
        for _ in range(self.num_hops):
            # Propagate across nodes at every time step at once
            h = torch.einsum("ij,bjtc->bitc", adj, h)
            outputs.append(h)
        return self.linear(torch.cat(outputs, dim=-1))


class DilatedTemporalConv(nn.Module):
    """Dilated causal temporal convolution with a gated activation."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.filt = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.gate = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time). Left-pad only, so no future leakage.
        x = F.pad(x, (self.pad, 0))
        return torch.tanh(self.filt(x)) * torch.sigmoid(self.gate(x))


class MTGNNLayer(nn.Module):
    """Temporal conv -> graph conv -> residual, with layer norm."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, num_hops: int = 2):
        super().__init__()
        self.temporal = DilatedTemporalConv(channels, kernel_size, dilation)
        self.graph = MixHopGraphConv(channels, channels, num_hops)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (batch, nodes, time, channels)
        B, N, T, C = x.shape
        residual = x

        h = x.reshape(B * N, T, C).transpose(1, 2)
        h = self.temporal(h)
        h = h.transpose(1, 2).reshape(B, N, T, C)

        h = self.graph(h, adj)
        return self.norm(h + residual)


class MTGNN(nn.Module):
    """MTGNN for multivariate time-series forecasting."""

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
        num_hops: int = 2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.graph_learner = GraphLearner(num_nodes, node_dim, top_k)
        self.input_proj = nn.Linear(in_features, hidden_channels)
        self.layers = nn.ModuleList([
            MTGNNLayer(hidden_channels, temporal_kernel, 2 ** i, num_hops)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, window, nodes, features)
            adj: optional (nodes, nodes) adjacency. If None, uses the learned graph.
        Returns:
            (batch, nodes) predicted next-step returns
        """
        if adj is None:
            adj = self.graph_learner()

        h = self.input_proj(x)          # (B, T, N, C)
        h = h.permute(0, 2, 1, 3)       # (B, N, T, C)

        for layer in self.layers:
            h = self.dropout(layer(h, adj))

        h = h[:, :, -1, :]              # last time step
        return self.output_proj(h).squeeze(-1)

    def forward_with_graph(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Explicit alias used by the CMA-ES search and the ablation."""
        return self.forward(x, adj=adj)

    @torch.no_grad()
    def get_learned_graph(self) -> torch.Tensor:
        return self.graph_learner().detach().cpu()
