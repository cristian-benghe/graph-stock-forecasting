"""CMA-ES: derivative-free search over graph structure.

To isolate the *optimizer* rather than the parameterization, CMA-ES searches the
same node-embedding formulation MTGNN learns by gradient. A candidate is a flat
vector holding two (num_nodes x node_dim) embedding matrices, decoded into a
sparse directed adjacency with the identical function MTGNN uses.

A small node_dim keeps the search space tractable: the proposal notes CMA-ES
becomes expensive when too many graph parameters are optimized at once, and a
full dense 20x20 matrix (400 dims) is far less sample-efficient than the
embedding form (2*20*4 = 160 dims).
"""

import time

import cma
import numpy as np
import torch

from src.models.mtgnn import sparsify_topk


def decode_adjacency(
    vec: np.ndarray,
    num_nodes: int,
    node_dim: int,
    top_k: int = 10,
    alpha: float = 3.0,
) -> torch.Tensor:
    """Decode a flat CMA-ES vector into a sparse directed adjacency matrix."""
    split = num_nodes * node_dim
    s = torch.tensor(vec[:split], dtype=torch.float32).reshape(num_nodes, node_dim)
    t = torch.tensor(vec[split:], dtype=torch.float32).reshape(num_nodes, node_dim)
    s, t = torch.tanh(alpha * s), torch.tanh(alpha * t)
    adj = torch.relu(torch.tanh(alpha * (s @ t.T - t @ s.T)))
    return sparsify_topk(adj, top_k)


def search_dimension(num_nodes: int, node_dim: int) -> int:
    return 2 * num_nodes * node_dim


@torch.no_grad()
def _val_mae(model, adj, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x, adj=adj.to(device))
        total += (out - y).abs().sum().item()
        n += y.numel()
    return total / n


def run_cmaes_search(
    model,
    val_loader,
    num_nodes: int,
    node_dim: int = 4,
    top_k: int = 10,
    sigma0: float = 0.5,
    max_evaluations: int = 2000,
    population_size: int = 12,
    seed: int = 42,
    device=torch.device("cpu"),
    verbose: bool = True,
) -> dict:
    """Search graph embeddings with CMA-ES against a frozen backbone.

    The backbone is held fixed during the search, so this measures how well
    CMA-ES can find a graph for an already-trained forecaster. Returns the best
    adjacency, its validation MAE, and the wall-clock and evaluation cost (the
    proposal asks for a computational-cost comparison, not just accuracy).
    """
    dim = search_dimension(num_nodes, node_dim)
    for p in model.parameters():
        p.requires_grad_(False)

    es = cma.CMAEvolutionStrategy(
        np.zeros(dim),
        sigma0,
        {
            "popsize": population_size,
            "maxfevals": max_evaluations,
            "seed": seed,
            "verbose": -9,
        },
    )

    start = time.time()
    evaluations = 0
    history = []

    while not es.stop():
        candidates = es.ask()
        fitness = []
        for c in candidates:
            adj = decode_adjacency(c, num_nodes, node_dim, top_k)
            fitness.append(_val_mae(model, adj, val_loader, device))
            evaluations += 1
        es.tell(candidates, fitness)
        history.append(float(min(fitness)))
        if verbose and len(history) % 20 == 0:
            print(f"    gen {len(history):4d} | evals {evaluations:5d} | "
                  f"best val MAE {es.result.fbest:.6f}")

    elapsed = time.time() - start
    best_adj = decode_adjacency(es.result.xbest, num_nodes, node_dim, top_k)

    for p in model.parameters():
        p.requires_grad_(True)

    return {
        "adjacency": best_adj,
        "best_val_mae": float(es.result.fbest),
        "evaluations": evaluations,
        "generations": len(history),
        "seconds": elapsed,
        "history": history,
        "search_dim": dim,
    }
