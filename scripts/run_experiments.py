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

from src.baselines.cmaes_search import run_cmaes_search
from src.baselines.correlation_graph import correlation_adjacency, identity_adjacency
from src.baselines.naive import (
    persistence_forecast,
    train_mean_forecast,
    zero_forecast,
)
from src.baselines.var import VARForecaster
from src.data.dataset import chronological_split, segment_offset
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


def _store(preds_store, name, seed, y_true, y_pred):
    """Record test predictions so significance tests and the portfolio
    evaluation can run afterwards without retraining anything."""
    preds_store[f"{name}|seed{seed}|true"] = np.asarray(y_true, dtype=np.float32)
    preds_store[f"{name}|seed{seed}|pred"] = np.asarray(y_pred, dtype=np.float32)


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
    return metrics, (y_true, y_pred)


def run_var(data, cfg):
    """Deterministic, so a single run suffices."""
    T = len(data)
    train_end = int(T * cfg["split"]["train_ratio"])
    val_end = int(T * (cfg["split"]["train_ratio"] + cfg["split"]["val_ratio"]))
    w = cfg["features"]["window_size"]
    h = cfg["features"]["horizon"]

    returns = data[:, :, 0]
    test_seg = returns[segment_offset(val_end, w, h):]

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
    metrics["selected_lags"] = var.selected_lags
    print(f"  [VAR] test MAE {metrics['mae']:.6f} "
          f"DA {metrics['directional_accuracy']:.4f} IC {metrics['information_coefficient']:+.4f} "
          f"(AIC-selected lags={var.selected_lags})")
    return metrics, (y_true, y_pred)


def run_naive(data, cfg):
    T = len(data)
    train_end = int(T * cfg["split"]["train_ratio"])
    val_end = int(T * (cfg["split"]["train_ratio"] + cfg["split"]["val_ratio"]))
    w, h = cfg["features"]["window_size"], cfg["features"]["horizon"]
    returns = data[:, :, 0]
    test_seg = returns[segment_offset(val_end, w, h):]

    y_true = np.asarray([
        test_seg[i + w + h - 1] for i in range(len(test_seg) - w - h + 1)
    ])

    # TrainMean is the constant per-company drift forecast. It is included
    # because AIC selects zero lags for VAR, which reduces VAR to exactly this
    # predictor; comparing the two shows whether VAR's apparent directional
    # skill is conditional forecasting or simply the equity drift.
    forecasts = {
        "Zero": zero_forecast(y_true),
        "Persistence": persistence_forecast(test_seg, w, h),
        "TrainMean": train_mean_forecast(returns[:train_end], len(y_true)),
    }
    out = {}
    for k, yp in forecasts.items():
        out[k] = evaluate_all(y_true, yp)
        da = out[k]["directional_accuracy"]
        da_s = "  n/a " if np.isnan(da) else f"{da:.4f}"
        print(f"  [{k}] test MAE {out[k]['mae']:.6f} "
              f"DA {da_s} IC {out[k]['information_coefficient']:+.4f}")
    return out, y_true, forecasts


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
    _t0 = time.time()
    pre = train_model(
        backbone, train_loader, val_loader,
        epochs=c["pretrain_epochs"], learning_rate=m["learning_rate"],
        weight_decay=m["weight_decay"], adj=corr_adj.to(device),
        device=device, verbose=False,
    )
    pre_seconds = time.time() - _t0

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
    metrics["pretrain_seconds"] = pre_seconds
    metrics["total_seconds"] = metrics["train_seconds"] + search["seconds"] + pre_seconds
    return metrics, search, preds


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

    results, extras, preds_store = {}, {}, {}
    T = len(data)
    train_end = int(T * cfg["split"]["train_ratio"])
    Path("outputs").mkdir(exist_ok=True)

    if run_this("naive"):
        print("=== Naive baselines ===")
        naive_metrics, naive_true, naive_forecasts = run_naive(data, cfg)
        for k, v in naive_metrics.items():
            results[k] = {"seeds": [v], "aggregate": aggregate_seeds([v])}
            _store(preds_store, k, 0, naive_true, naive_forecasts[k])

    if run_this("var"):
        print("\n=== VAR ===")
        v, vp = run_var(data, cfg)
        results["VAR"] = {"seeds": [v], "aggregate": aggregate_seeds([v])}
        _store(preds_store, "VAR", 0, *vp)

    if run_this("lstm"):
        print("\n=== LSTM ===")
        runs = []
        for s in seeds:
            mt, p = run_lstm(data, cfg, s, device)
            runs.append(mt)
            _store(preds_store, "LSTM", s, *p)
        results["LSTM"] = {"seeds": runs, "aggregate": aggregate_seeds(runs)}

    if run_this("mtgnn"):
        print("\n=== MTGNN (learned graph) ===")
        runs = []
        for s in seeds:
            mt, model, p = run_mtgnn(data, cfg, s, device)
            runs.append(mt)
            _store(preds_store, "MTGNN", s, *p)
            if s == seeds[0]:
                torch.save({"adjacency": model.get_learned_graph(), "tickers": tickers},
                           "outputs/learned_graph.pt")
        results["MTGNN"] = {"seeds": runs, "aggregate": aggregate_seeds(runs)}

    if run_this("ablation"):
        print("\n=== Ablation: fixed correlation graph ===")
        corr = correlation_adjacency(data[:train_end, :, 0], top_k=cfg["mtgnn"]["top_k"])
        runs = []
        for s in seeds:
            mt, _, p = run_mtgnn(data, cfg, s, device, adj=corr, label="Corr-graph")
            runs.append(mt)
            _store(preds_store, "MTGNN-CorrGraph", s, *p)
        results["MTGNN-CorrGraph"] = {"seeds": runs, "aggregate": aggregate_seeds(runs)}
        torch.save({"adjacency": corr, "tickers": tickers}, "outputs/correlation_graph.pt")

        print("\n=== Ablation: identity graph (no message passing) ===")
        eye = identity_adjacency(data.shape[1])
        runs = []
        for s in seeds:
            mt, _, p = run_mtgnn(data, cfg, s, device, adj=eye, label="Identity")
            runs.append(mt)
            _store(preds_store, "MTGNN-Identity", s, *p)
        results["MTGNN-Identity"] = {"seeds": runs, "aggregate": aggregate_seeds(runs)}

    if run_this("cmaes"):
        print("\n=== CMA-ES graph search ===")
        runs, searches = [], []
        for s in seeds:
            mt, search, p = run_cmaes(data, cfg, s, device)
            runs.append(mt)
            _store(preds_store, "CMA-ES", s, *p)
            searches.append({k: search[k] for k in
                             ["best_val_mae", "evaluations", "generations", "seconds", "search_dim"]})
            if s == seeds[0]:
                torch.save({"adjacency": search["adjacency"], "tickers": tickers},
                           "outputs/cmaes_graph.pt")
        results["CMA-ES"] = {"seeds": runs, "aggregate": aggregate_seeds(runs)}
        extras["cmaes_search"] = searches

    if preds_store:
        np.savez_compressed("outputs/predictions.npz", **preds_store)
        print(f"\nSaved {len(preds_store)//2} prediction sets to outputs/predictions.npz")

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
