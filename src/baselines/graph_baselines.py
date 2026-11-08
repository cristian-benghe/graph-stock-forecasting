"""Fixed-graph baselines requested as comparators for the learned graph.

Three constructions, all built from training data only and all sparsified to the
same top-$k$ as MTGNN so that only the graph varies:

  * `knn_adjacency`     - k-nearest neighbours under the correlation distance
                          d_ij = sqrt(2(1 - rho_ij)), the standard metric for
                          financial correlation networks (Mantegna, 1999).
  * `random_adjacency`  - a seeded random k-regular graph. This is the control
                          that separates "the learned graph is informative"
                          from "any sparse graph of this density performs the
                          same", which the identity ablation alone cannot show.
  * `correlation_adjacency` (in correlation_graph.py) - top-k by |rho|.

Note that kNN-on-distance and top-k-on-|rho| are not the same graph: a strongly
*negatively* correlated pair is near-maximally distant under d_ij but ranks
highly under |rho|. Reporting both separates "co-moving" from "related in
magnitude".
"""

import numpy as np
import torch

from src.models.mtgnn import sparsify_topk


def correlation_distance(train_returns: np.ndarray) -> np.ndarray:
    """d_ij = sqrt(2(1 - rho_ij)); 0 for identical series, 2 for anti-correlated."""
    corr = np.nan_to_num(np.corrcoef(train_returns, rowvar=False), nan=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    return np.sqrt(2.0 * (1.0 - corr))


def knn_adjacency(
    train_returns: np.ndarray,
    k: int = 10,
    weighted: bool = True,
) -> torch.Tensor:
    """kNN graph under the correlation distance, row-normalized.

    Args:
        train_returns: (time, nodes) training-period log returns.
        k: neighbours per node.
        weighted: weight edges by 1/(1+d) rather than uniformly, so nearer
            neighbours contribute more. Uniform weights make it a pure
            topology baseline.
    """
    dist = correlation_distance(train_returns)
    n = dist.shape[0]
    np.fill_diagonal(dist, np.inf)  # never select self

    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        nn = np.argsort(dist[i])[:k]
        adj[i, nn] = 1.0 / (1.0 + dist[i, nn]) if weighted else 1.0

    return sparsify_topk(torch.tensor(adj, dtype=torch.float32), k)


def random_adjacency(num_nodes: int, k: int = 10, seed: int = 42) -> torch.Tensor:
    """Seeded random k-regular directed graph, row-normalized.

    Matches the density of the learned graph while carrying no information, so
    any performance it achieves is attributable to the architecture rather than
    to the graph being meaningful.
    """
    rng = np.random.default_rng(seed)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        choices = [j for j in range(num_nodes) if j != i]
        nn = rng.choice(choices, size=min(k, len(choices)), replace=False)
        adj[i, nn] = 1.0
    return sparsify_topk(torch.tensor(adj, dtype=torch.float32), k)
