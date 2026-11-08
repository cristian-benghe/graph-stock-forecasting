"""MLP baseline: no graph, no convolution, no recurrence.

The identity-graph ablation removes message passing but keeps MTGNN's dilated
temporal convolution stack, so it does not isolate the contribution of the
architecture as a whole. This flattens each node's window into one vector and
applies a per-node MLP, which is the simplest model that can express a
nonlinear function of the same inputs. Comparing against it separates "does
the graph help" from "does any of this architecture help".

Weights are shared across nodes, matching MTGNN, so the comparison is not
confounded by parameter count per node.
"""

import torch
import torch.nn as nn


class MLPForecaster(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        window_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        in_dim = window_size * in_features

        layers = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_size
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, adj=None) -> torch.Tensor:
        """
        Args:
            x: (batch, window, nodes, features)
            adj: accepted and ignored, so this model can share the training harness.
        Returns:
            (batch, nodes)
        """
        B, T, N, F = x.shape
        # (B, N, T*F): each node's own window, flattened
        h = x.permute(0, 2, 1, 3).reshape(B, N, T * F)
        return self.net(h).squeeze(-1)
