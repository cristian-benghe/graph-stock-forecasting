"""Fixed correlation graph for the ablation study.

Removes MTGNN's learned-graph component and substitutes a static graph built
from training-period return correlations. Comparing against this isolates how
much the *learning* of the structure contributes, as opposed to merely having
some sensible graph.
"""

import numpy as np
import torch

from src.models.mtgnn import sparsify_topk


def correlation_adjacency(
    train_returns: np.ndarray,
    top_k: int = 10,
    use_absolute: bool = True,
) -> torch.Tensor:
    """Build a row-normalized top-k graph from return correlations.

    Args:
        train_returns: (time, nodes) training-period log returns. Training data
            only, otherwise the ablation leaks test information.
        top_k: neighbours kept per node, matching the MTGNN sparsity level.
        use_absolute: treat strong negative correlation as a strong relation.
    """
    corr = np.corrcoef(train_returns, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = np.abs(corr) if use_absolute else np.clip(corr, 0.0, None)
    np.fill_diagonal(corr, 0.0)
    return sparsify_topk(torch.tensor(corr, dtype=torch.float32), top_k)


def identity_adjacency(num_nodes: int) -> torch.Tensor:
    """No-message-passing control: each node sees only itself."""
    return torch.eye(num_nodes, dtype=torch.float32)
