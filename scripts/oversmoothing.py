"""Measure oversmoothing, and test whether depth or graph density causes it.

Oversmoothing is the collapse of node representations towards a common vector
under repeated graph convolution. It matters here because the configuration is
aggressive: top_k=10 of 20 nodes means every node aggregates over half the
graph, and 3 layers x 2 mix-hop steps gives 6 effective propagation hops, so a
random walk on this graph mixes almost immediately.

Two metrics, both computed on the test set after training:

  * Mean pairwise cosine similarity between node representations at each layer.
    Approaching 1.0 means the representations have become indistinguishable.
  * Dirichlet energy, sum over edges of squared representation differences,
    normalized by representation norm. Decaying towards 0 with depth is the
    standard signature of oversmoothing.

Both are reported for the learned graph and for the identity graph. The identity
graph performs no mixing, so it acts as a control: any collapse it shows is
caused by the layers themselves rather than by message passing.
"""

import sys
sys.path.insert(0, ".")

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.baselines.correlation_graph import identity_adjacency
from src.data.dataset import chronological_split
from src.evaluation.metrics import evaluate_all
from src.models.mtgnn import MTGNN
from src.training.trainer import predict, set_seed, train_model


def pairwise_cosine(h: torch.Tensor) -> float:
    """Mean off-diagonal cosine similarity between node representations.

    h: (batch, nodes, channels) -> averaged over the batch.
    """
    hn = torch.nn.functional.normalize(h, dim=-1)
    sim = hn @ hn.transpose(1, 2)            # (B, N, N)
    N = sim.size(1)
    off = ~torch.eye(N, dtype=torch.bool, device=sim.device)
    return float(sim[:, off].mean())


def dirichlet_energy(h: torch.Tensor, adj: torch.Tensor) -> float:
    """Normalized Dirichlet energy: edge-weighted representation disagreement."""
    # h: (B, N, C); adj: (N, N)
    diff = h.unsqueeze(2) - h.unsqueeze(1)          # (B, N, N, C)
    sq = (diff ** 2).sum(-1)                        # (B, N, N)
    energy = (adj.unsqueeze(0) * sq).sum((1, 2))    # (B,)
    norm = (h ** 2).sum((1, 2)) + 1e-12
    return float((energy / norm).mean())


@torch.no_grad()
def layerwise_metrics(model: MTGNN, x: torch.Tensor, adj: torch.Tensor) -> list[dict]:
    """Replay the forward pass, recording metrics after each layer."""
    model.eval()
    h = model.input_proj(x).permute(0, 2, 1, 3)  # (B, N, T, C)

    rows = []
    # Layer 0 = after input projection, before any graph convolution
    rows.append({
        "layer": 0,
        "cosine": pairwise_cosine(h[:, :, -1, :]),
        "dirichlet": dirichlet_energy(h[:, :, -1, :], adj),
    })
    for i, layer in enumerate(model.layers, start=1):
        h = layer(h, adj)
        rows.append({
            "layer": i,
            "cosine": pairwise_cosine(h[:, :, -1, :]),
            "dirichlet": dirichlet_energy(h[:, :, -1, :], adj),
        })
    return rows


def build(cfg, data, num_layers=None, top_k=None):
    m = cfg["mtgnn"]
    return MTGNN(
        num_nodes=data.shape[1], in_features=data.shape[2],
        hidden_channels=m["hidden_channels"],
        num_layers=num_layers if num_layers is not None else m["num_layers"],
        node_dim=m["node_dim"], temporal_kernel=m["temporal_kernel"],
        top_k=top_k if top_k is not None else m["top_k"],
        dropout=m["dropout"],
    )


def train_and_eval(cfg, data, device, seed, num_layers=None, top_k=None, adj=None):
    set_seed(seed)
    m = cfg["mtgnn"]
    tr, va, te = chronological_split(
        data, train_ratio=cfg["split"]["train_ratio"], val_ratio=cfg["split"]["val_ratio"],
        window_size=cfg["features"]["window_size"], horizon=cfg["features"]["horizon"],
    )
    loaders = [DataLoader(d, batch_size=m["batch_size"], shuffle=(i == 0))
               for i, d in enumerate([tr, va, te])]

    model = build(cfg, data, num_layers, top_k)
    out = train_model(
        model, loaders[0], loaders[1], epochs=m["epochs"],
        learning_rate=m["learning_rate"], weight_decay=m["weight_decay"],
        adj=adj.to(device) if adj is not None else None,
        device=device, verbose=False,
    )
    y_true, y_pred = predict(out["model"], loaders[2], device,
                             adj=adj.to(device) if adj is not None else None)
    met = evaluate_all(y_true, y_pred)
    met["pred_std"] = float(y_pred.std())
    return met, out["model"], loaders[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--out", default="outputs/oversmoothing.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    device = torch.device("cpu")
    data = np.load("data/processed/features.npy")
    N = data.shape[1]
    results = {}

    # ---- 1. Layerwise collapse, learned graph vs identity control ----
    print("=== Layerwise representation collapse (seed "
          f"{args.seeds[0]}, {cfg['mtgnn']['num_layers']} layers, "
          f"top_k={cfg['mtgnn']['top_k']}) ===")
    met, model, test_loader = train_and_eval(cfg, data, device, args.seeds[0])
    xb = torch.cat([x for x, _ in test_loader]).to(device)

    learned_adj = model.graph_learner()
    eye = identity_adjacency(N).to(device)

    layer_rows = {}
    for name, adj in [("learned", learned_adj), ("identity", eye)]:
        rows = layerwise_metrics(model, xb, adj)
        layer_rows[name] = rows
        print(f"\n  {name} graph:")
        print(f"    {'layer':>6s} {'mean cos sim':>13s} {'Dirichlet':>12s}")
        for r in rows:
            print(f"    {r['layer']:>6d} {r['cosine']:>13.4f} {r['dirichlet']:>12.4f}")
    results["layerwise"] = layer_rows

    # ---- 2. Depth sweep: does performance degrade with more propagation? ----
    print("\n=== Depth sweep (learned graph) ===")
    print(f"  {'layers':>6s} {'hops':>5s} {'MAE':>22s} {'pred std':>10s}")
    depth = {}
    for L in [1, 2, 3, 4]:
        runs = [train_and_eval(cfg, data, device, s, num_layers=L)[0] for s in args.seeds]
        mae = np.mean([r["mae"] for r in runs])
        sd = np.std([r["mae"] for r in runs])
        ps = np.mean([r["pred_std"] for r in runs])
        depth[L] = {"mae_mean": float(mae), "mae_std": float(sd),
                    "pred_std": float(ps), "n_seeds": len(runs)}
        print(f"  {L:>6d} {2*L:>5d} {mae:>12.6f} +/- {sd:<8.6f} {ps:>10.6f}")
    results["depth_sweep"] = depth

    # ---- 3. Sparsity sweep: is half-the-graph aggregation the problem? ----
    print("\n=== Graph sparsity sweep (3 layers) ===")
    print(f"  {'top_k':>6s} {'% of graph':>11s} {'MAE':>22s} {'pred std':>10s}")
    spars = {}
    for k in [2, 3, 5, 10, 19]:
        runs = [train_and_eval(cfg, data, device, s, top_k=k)[0] for s in args.seeds]
        mae = np.mean([r["mae"] for r in runs])
        sd = np.std([r["mae"] for r in runs])
        ps = np.mean([r["pred_std"] for r in runs])
        spars[k] = {"mae_mean": float(mae), "mae_std": float(sd),
                    "pred_std": float(ps), "n_seeds": len(runs)}
        print(f"  {k:>6d} {k/(N-1):>10.0%} {mae:>12.6f} +/- {sd:<8.6f} {ps:>10.6f}")
    results["sparsity_sweep"] = spars

    # ---- Figure ----
    figdir = Path("report/figures"); figdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    for name, rows in layer_rows.items():
        ax.plot([r["layer"] for r in rows], [r["cosine"] for r in rows],
                marker="o", label=f"{name} graph")
    ax.axhline(1.0, color="red", ls="--", lw=1.1, label="total collapse")
    ax.set_xlabel("layer"); ax.set_ylabel("mean pairwise cosine similarity")
    ax.set_title("Representation collapse with depth")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8)

    ax = axes[1]
    Ls = sorted(depth)
    ax.errorbar(Ls, [depth[L]["mae_mean"] for L in Ls],
                yerr=[depth[L]["mae_std"] for L in Ls], marker="o", capsize=3,
                color="#1F4E79")
    ax.axhline(0.020869, color="grey", ls="--", lw=1.1, label="zero forecast")
    ax.set_xlabel("number of layers"); ax.set_ylabel("test MAE")
    ax.set_xticks(Ls); ax.set_title("Depth sweep"); ax.legend(fontsize=8)

    ax = axes[2]
    ks = sorted(spars)
    ax.errorbar(ks, [spars[k]["mae_mean"] for k in ks],
                yerr=[spars[k]["mae_std"] for k in ks], marker="s", capsize=3,
                color="#C05C20")
    ax.axhline(0.020869, color="grey", ls="--", lw=1.1, label="zero forecast")
    ax.set_xlabel("top-$k$ neighbours per node"); ax.set_ylabel("test MAE")
    ax.set_xticks(ks); ax.set_title("Graph sparsity sweep"); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(figdir / "oversmoothing.png", dpi=180)
    plt.close(fig)
    print(f"\nwrote {figdir/'oversmoothing.png'}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
