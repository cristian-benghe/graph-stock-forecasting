"""Analyze and visualize learned adjacency matrices.

Answers the proposal's graph-analysis question: do the strongest learned edges
fall within an industry, or across the semiconductor/agriculture boundary?
"""

import sys
sys.path.insert(0, ".")

import json

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml


def load_graph(path):
    blob = torch.load(path, weights_only=False)
    return blob["adjacency"].numpy(), blob["tickers"]


def sector_map(cfg):
    return (
        {t: "Semiconductor" for t in cfg["data"]["semiconductors"]}
        | {t: "Agriculture" for t in cfg["data"]["agriculture"]}
    )


def edge_statistics(adj, tickers, sectors):
    """Share of total edge weight that stays within a sector vs crosses sectors.

    The random baseline matters for interpretation: with 10 nodes per sector and
    self-loops excluded, a graph wiring neighbours uniformly at random puts
    9/19 = 47.4% of its weight within-sector, so only a materially higher share
    indicates the model actually recovered industry structure.
    """
    n = len(tickers)
    within = cross = 0.0
    per_node = {}

    for i in range(n):
        wi = ci = 0.0
        for j in range(n):
            if i == j:
                continue
            w = float(adj[i, j])
            if sectors[tickers[i]] == sectors[tickers[j]]:
                wi += w
            else:
                ci += w
        within += wi
        cross += ci
        total = wi + ci
        per_node[tickers[i]] = (wi / total) if total > 0 else float("nan")

    total = within + cross
    return {
        "within_sector_weight": within,
        "cross_sector_weight": cross,
        "within_sector_share": (within / total) if total else float("nan"),
        "random_baseline_share": 9 / 19,
        "per_node_within_share": per_node,
    }


def top_edges(adj, tickers, sectors, k=15):
    n = len(tickers)
    edges = [
        (float(adj[i, j]), tickers[j], tickers[i],
         "within" if sectors[tickers[i]] == sectors[tickers[j]] else "cross")
        for i in range(n) for j in range(n) if i != j and adj[i, j] > 0
    ]
    edges.sort(reverse=True)
    return edges[:k]


def sector_ordering(tickers, cfg):
    """Permutation grouping semiconductors first, then agriculture.

    The downloaded panel stores tickers alphabetically, so plotting the raw
    matrix interleaves the two industries and any sector separator line drawn at
    index 10 would be meaningless. Reordering makes block structure, if present,
    visible on the diagonal.
    """
    semis = [t for t in cfg["data"]["semiconductors"] if t in tickers]
    agri = [t for t in cfg["data"]["agriculture"] if t in tickers]
    ordered = semis + agri
    missing = [t for t in tickers if t not in ordered]
    ordered += missing
    idx = [tickers.index(t) for t in ordered]
    return idx, ordered, len(semis)


def plot_heatmap(adj, tickers, cfg, title, out_path):
    idx, labels, n_semi = sector_ordering(tickers, cfg)
    adj = adj[np.ix_(idx, idx)]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(adj, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    # Boundary between the two industries, valid only after reordering
    ax.axhline(n_semi - 0.5, color="red", lw=1.5)
    ax.axvline(n_semi - 0.5, color="red", lw=1.5)
    ax.set_xlabel("source node $j$   (information flows $j \\rightarrow i$)")
    ax.set_ylabel("target node $i$")
    ax.set_title(f"{title}\nnodes ordered: semiconductors (first {n_semi}), "
                 f"then agriculture", fontsize=10)
    fig.colorbar(im, ax=ax, label="edge weight", fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    sectors = sector_map(cfg)
    figdir = Path("report/figures")
    figdir.mkdir(parents=True, exist_ok=True)

    graphs = {
        "learned": ("outputs/learned_graph.pt", "MTGNN learned graph"),
        "correlation": ("outputs/correlation_graph.pt", "Fixed correlation graph"),
        "cmaes": ("outputs/cmaes_graph.pt", "CMA-ES searched graph"),
    }

    summary = {}
    for key, (path, title) in graphs.items():
        if not Path(path).exists():
            print(f"skip {key}: {path} not found")
            continue
        adj, tickers = load_graph(path)
        print(f"\n=== {title} ===")
        stats = edge_statistics(adj, tickers, sectors)
        print(f"  within-sector weight share: {stats['within_sector_share']:.3f} "
              f"(random baseline {stats['random_baseline_share']:.3f})")
        nnz = int((adj > 0).sum())
        print(f"  non-zero edges: {nnz} / {adj.size} "
              f"({100*nnz/adj.size:.1f}% density)")

        edges = top_edges(adj, tickers, sectors)
        print("  strongest edges (source -> target, weight, type):")
        for w, src, dst, kind in edges[:10]:
            print(f"    {src:6s} -> {dst:6s}  {w:.4f}  {kind}")

        plot_heatmap(adj, tickers, cfg, title, figdir / f"graph_{key}.png")
        summary[key] = {
            "within_sector_share": stats["within_sector_share"],
            "random_baseline_share": stats["random_baseline_share"],
            "n_nonzero_edges": nnz,
            "density": nnz / adj.size,
            "top_edges": [
                {"source": s, "target": d, "weight": w, "type": k}
                for w, s, d, k in edges
            ],
            "per_node_within_share": stats["per_node_within_share"],
        }

    out = Path("outputs/graph_analysis.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
