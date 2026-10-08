"""CMA-ES baseline: derivative-free search for graph adjacency weights.

Uses the CMA-ES evolutionary strategy to optimize the adjacency matrix
that a frozen GNN backbone uses for message passing, instead of learning
the graph end-to-end with gradients.
"""

import numpy as np
import cma
import torch
from torch.utils.data import DataLoader


def adjacency_from_vector(vec: np.ndarray, num_nodes: int, top_k: int = 10) -> torch.Tensor:
    """Reshape a flat vector into a sparse adjacency matrix."""
    adj = torch.tensor(vec, dtype=torch.float32).reshape(num_nodes, num_nodes)
    adj = torch.relu(adj)
    adj.fill_diagonal_(0.0)

    if top_k < num_nodes:
        topk_vals, topk_idx = torch.topk(adj, top_k, dim=1)
        mask = torch.zeros_like(adj)
        mask.scatter_(1, topk_idx, 1.0)
        adj = adj * mask

    row_sum = adj.sum(dim=1, keepdim=True) + 1e-8
    return adj / row_sum


def evaluate_adjacency(
    adj: torch.Tensor,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate a fixed adjacency matrix with the given model. Returns val MSE."""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            # Override the model's learned graph with the CMA-ES candidate
            preds = model.forward_with_graph(x, adj.to(device))
            total_loss += ((preds - y) ** 2).sum().item()
            count += y.numel()

    return total_loss / count


def run_cmaes_search(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_nodes: int,
    top_k: int = 10,
    sigma0: float = 0.5,
    max_iterations: int = 500,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, float]:
    """Search for the best adjacency matrix using CMA-ES.

    Returns (best_adjacency, best_val_loss).
    """
    dim = num_nodes * num_nodes

    def objective(vec):
        adj = adjacency_from_vector(vec, num_nodes, top_k)
        return evaluate_adjacency(adj, model, val_loader, device)

    es = cma.CMAEvolutionStrategy(np.zeros(dim), sigma0, {
        "maxiter": max_iterations,
        "popsize": 20,
        "verb_disp": 50,
    })

    es.optimize(objective)
    best_vec = es.result.xbest
    best_adj = adjacency_from_vector(best_vec, num_nodes, top_k)

    return best_adj, es.result.fbest
