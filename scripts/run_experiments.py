"""Run every model on identical splits and write results to JSON.

All models share one chronological split, one normalizer fitted on training data
only, and the same seed list. Test data is touched exactly once per run, after
checkpoint selection on validation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.baselines.cmaes_search import run_cmaes_search, decode_adjacency
from src.baselines.correlation_graph import correlation_adjacency, identity_adjacency
from src.baselines.naive import persistence_forecast, zero_forecast
from src.baselines.var import VARForecaster
from src.data.dataset import chronological_split
from src.evaluation.metrics import aggregate_seeds, evaluate_all
from src.models.lstm import LSTMForecaster
from src.models.mtgnn import MTGNN
from src.training.trainer import predict, set_seed, train_model


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_loaders(data, cfg, batch_size):
    train_ds, val_ds, test_ds, norm = chronological_split(
        data,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        window_size=cfg["features"]["window_size"],
        horizon=cfg["features"]["horizon"],
        return_normalizer=True,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
        (train_ds, val_ds, test_ds),
        norm,
    )


def run_mtgnn(data, cfg, seed, device, adj=None, label="MTGNN"):
    """Train MTGNN. adj=None learns the graph; otherwise the graph is fixed."""
    set_seed(seed)
    m = cfg["mtgnn"]
    train_loader, val_loader, test_loader, _, _ = build_loaders(data, cfg, m["batch_size"])

    model = MTGNN(
        num_nodes=data.shape[1],
        in_features=data.shape[2],
        hidden_channels=m["hidden_channels"],
        num_layers=m["num_layers"],
        node_dim=m["node_dim"],
        temporal_kernel=m["temporal_kernel"],
        top_k=m["top_k"],
        dropout=m["dropout"],
    )

    t0 = time.time()
    out = train_model(
        model, train_loader, val_loader,
        epochs=m["epochs"],
        learning_rate=m["learning_rate"],
        weight_decay=m["weight_decay"],
        adj=adj.to(device) if adj is not None else None,
        device=device,
        verbose=True,
    )
    train_seconds = time.time() - t0

    y_true, y_pred = predict(
        out["model"], test_loader, device,
        adj=adj.to(device) if adj is not None else None,
    )
    metrics = evaluate_all(y_true, y_pred)
    metrics["train_seconds"] = train_seconds
    metrics["best_epoch"] = out["best_epoch"]

    print(f"  [{label} seed {seed}] test MAE {metrics['mae']:.6f} "
          f"DA {metrics['directional_accuracy']:.4f} IC {metrics['information_coefficient']:+.4f}")
    return metrics, out["model"], (y_true, y_pred)


def run_lstm(data, cfg, seed, device):
    set_seed(seed)
    l = cfg["lstm"]
    train_loader, val_loader, test_loader, _, _ = build_loaders(data, cfg, l["batch_size"])

    model = LSTMForecaster(
        num_nodes=data.shape[1],
        in_features=data.shape[2],
        hidden_size=l["hidden_size"],
        num_layers=l["num_layers"],
        dropout=l["dropout"],
    )

    t0 = time.time()
    out = train_model(
        model, train_loader, val_loader,
        epochs=l["epochs"],
        learning_rate=l["learning_rate"],
        device=device,
        verbose=True,
    )
    train_seconds = time.time() - t0

    y_true, y_pred = predict(out["model"], test_loader, device)
    metrics = evaluate_all(y_true, y_pred)
    metrics["train_seconds"] = train_seconds
    metrics["best_epoch"] = out["best_epoch"]
    print(f"  [LSTM seed {seed}] test MAE {metrics['mae']:.6f} "
          f"DA {metrics['directional_accuracy']:.4f} IC {metrics['information_coefficient']:+.4f}")
    return metrics


def run_var(data, cfg):
    """Deterministic, so a single run suffices."""
    T = len(data)
    train_end = int(T * cfg["split"]["train_ratio"])
    val_end = int(T * (cfg["split"]["train_ratio"] + cfg["split"]["val_ratio"]))
    w = cfg["features"]["window_size"]
    h = cfg["features"]["horizon"]

    returns = data[:, :, 0]
    test_seg = returns[max(0, val_end - w - h):]

    t0 = time.time()
    var = VARForecaster(max_lags=5).fit(returns[:train_end])
    y_pred = var.predict_windows(test_seg, window_size=w, horizon=h)
    elapsed = time.time() - t0

    y_true = np.asarray([
        test_seg[i + w + h - 1] for i in range(len(test_seg) - w - h + 1)
    ])
    metrics = evaluate_all(y_true, y_pred)
    metrics["train_seconds"] = elapsed
    metrics["lag_order"] = var.lag_order
    print(f"  [VAR] test MAE {metrics['mae']:.6f} "
          f"DA {metrics['directional_accuracy']:.4f} IC {metrics['information_coefficient']:+.4f} "
          f"(lags={var.lag_order})")
    return metrics


def run_naive(data, cfg):
    T = len(data)
    val_end = int(T * (cfg["split"]["train_ratio"] + cfg["split"]["val_ratio"]))
    w, h = cfg["features"]["window_size"], cfg["features"]["horizon"]
    returns = data[:, :, 0]
    test_seg = returns[max(0, val_end - w - h):]

    y_true = np.asarray([
        test_seg[i + w + h - 1] for i in range(len(test_seg) - w - h + 1)
    ])

    out = {}
    out["Zero"] = evaluate_all(y_true, zero_forecast(y_true))
    out["Persistence"] = evaluate_all(y_true, persistence_forecast(test_seg, w, h))
    for k, v in out.items():
        print(f"  [{k}] test MAE {v['mae']:.6f} "
              f"DA {v['directional_accuracy']:.4f} IC {v['information_coefficient']:+.4f}")
    return out


def run_cmaes(data, cfg, seed, device):
    """Pretrain a backbone with the correlation graph, search a graph with
    CMA-ES against it, then retrain from scratch on the found graph."""
    set_seed(seed)
    m, c = cfg["mtgnn"], cfg["cmaes"]
    train_loader, val_loader, test_loader, _, _ = build_loaders(data, cfg, m["batch_size"])

    T = len(data)
    train_end = int(T * cfg["split"]["train_ratio"])
    corr_adj = correlation_adjacency(data[:train_end, :, 0], top_k=m["top_k"])

    print("    pretraining backbone for the CMA-ES search")
    set_seed(seed)
    backbone = MTGNN(
        num_nodes=data.shape[1], in_features=data.shape[2],
        hidden_channels=m["hidden_channels"], num_layers=m["num_layers"],
        node_dim=m["node_dim"], temporal_kernel=m["temporal_kernel"],
        top_k=m["top_k"], dropout=m["dropout"],
    )
    pre = train_model(
        backbone, train_loader, val_loader,
        epochs=c["pretrain_epochs"], learning_rate=m["learning_rate"],
        weight_decay=m["weight_decay"], adj=corr_adj.to(device),
        device=device, verbose=False,
    )

    print("    searching graph with CMA-ES")
    search = run_cmaes_search(
        pre["model"], val_loader,
        num_nodes=data.shape[1],
        node_dim=c["node_dim"],
        top_k=m["top_k"],
        sigma0=c["sigma0"],
        max_evaluations=c["max_evaluations"],
        population_size=c["population_size"],
        seed=seed, device=device, verbose=True,
    )

    print("    retraining backbone on the CMA-ES graph")
    metrics, model, preds = run_mtgnn(
        data, cfg, seed, device, adj=search["adjacency"], label="CMA-ES"
    )
    metrics["search_seconds"] = search["seconds"]
    metrics["search_evaluations"] = search["evaluations"]
    metrics["search_dim"] = search["search_dim"]
    metrics["total_seconds"] = metrics["train_seconds"] + search["seconds"]
    return metrics, search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--out", default="outputs/results.json")
    ap.add_argument("--quick", action="store_true", help="1 seed, few epochs, for smoke tests")
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load("data/processed/features.npy")
    tickers = Path("data/processed/tickers.txt").read_text().split()

    seeds = cfg["evaluation"]["seeds"]
    if args.quick:
        seeds = seeds[:1]
        cfg["mtgnn"]["epochs"] = 3
        cfg["lstm"]["epochs"] = 3
        cfg["cmaes"]["pretrain_epochs"] = 2
        cfg["cmaes"]["max_evaluations"] = 60

    want = set(args.models)
    run_this = lambda name: "all" in want or name in want

    print(f"Device: {device} | data {data.shape} | seeds {seeds}\n")

    results, extras = {}, {}
    T = len(data)
    train_end = int(T * cfg["split"]["train_ratio"])

    if run_this("naive"):
        print("=== Naive baselines ===")
        for k, v in run_naive(data, cfg).items():
            results[k] = {"seeds": [v], "aggregate": aggregate_seeds([v])}

    if run_this("var"):
        print("\n=== VAR ===")
        v = run_var(data, cfg)
        results["VAR"] = {"seeds": [v], "aggregate": aggregate_seeds([v])}

    if run_this("lstm"):
        print("\n=== LSTM ===")
        runs = [run_lstm(data, cfg, s, device) for s in seeds]
        results["LSTM"] = {"seeds": runs, "aggregate": aggregate_seeds(runs)}

    if run_this("mtgnn"):
        print("\n=== MTGNN (learned graph) ===")
        runs = []
        for s in seeds:
            mt, model, _ = run_mtgnn(data, cfg, s, device)
            runs.append(mt)
            if s == seeds[0]:
                adj = model.get_learned_graph()
                Path("outputs").mkdir(exist_ok=True)
                torch.save({"adjacency": adj, "tickers": tickers},
                           "outputs/learned_graph.pt")
        results["MTGNN"] = {"seeds": runs, "aggregate": aggregate_seeds(runs)}

    if run_this("ablation"):
        print("\n=== Ablation: fixed correlation graph ===")
        corr = correlation_adjacency(data[:train_end, :, 0], top_k=cfg["mtgnn"]["top_k"])
        runs = [run_mtgnn(data, cfg, s, device, adj=corr, label="Corr-graph")[0] for s in seeds]
        results["MTGNN-CorrGraph"] = {"seeds": runs, "aggregate": aggregate_seeds(runs)}
        torch.save({"adjacency": corr, "tickers": tickers}, "outputs/correlation_graph.pt")

        print("\n=== Ablation: identity graph (no message passing) ===")
        eye = identity_adjacency(data.shape[1])
        runs = [run_mtgnn(data, cfg, s, device, adj=eye, label="Identity")[0] for s in seeds]
        results["MTGNN-Identity"] = {"seeds": runs, "aggregate": aggregate_seeds(runs)}

    if run_this("cmaes"):
        print("\n=== CMA-ES graph search ===")
        runs, searches = [], []
        for s in seeds:
            mt, search = run_cmaes(data, cfg, s, device)
            runs.append(mt)
            searches.append({k: search[k] for k in
                             ["best_val_mae", "evaluations", "generations", "seconds", "search_dim"]})
            if s == seeds[0]:
                torch.save({"adjacency": search["adjacency"], "tickers": tickers},
                           "outputs/cmaes_graph.pt")
        results["CMA-ES"] = {"seeds": runs, "aggregate": aggregate_seeds(runs)}
        extras["cmaes_search"] = searches

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg,
        "data_shape": list(data.shape),
        "tickers": tickers,
        "seeds": seeds,
        "results": results,
        "extras": extras,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {out_path}")

    print("\n" + "=" * 78)
    print(f"{'Model':22s} {'MAE':>18s} {'Dir.Acc':>16s} {'IC':>16s}")
    print("=" * 78)
    for name, r in results.items():
        a = r["aggregate"]
        print(f"{name:22s} "
              f"{a['mae']['mean']:.6f}±{a['mae']['std']:.6f} "
              f"{a['directional_accuracy']['mean']:.4f}±{a['directional_accuracy']['std']:.4f} "
              f"{a['information_coefficient']['mean']:+.4f}±{a['information_coefficient']['std']:.4f}")


if __name__ == "__main__":
    main()
