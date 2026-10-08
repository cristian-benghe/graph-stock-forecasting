"""Train the MTGNN model."""

import sys
sys.path.insert(0, ".")

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.dataset import chronological_split
from src.models.mtgnn import MTGNN
from src.evaluation.metrics import evaluate_all


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return evaluate_all(targets, preds)


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load("data/processed/features.npy")
    mcfg = cfg["mtgnn"]

    train_ds, val_ds, test_ds = chronological_split(
        data,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        window_size=cfg["features"]["window_size"],
        horizon=cfg["features"]["horizon"],
    )

    train_loader = DataLoader(train_ds, batch_size=mcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=mcfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=mcfg["batch_size"])

    num_nodes = data.shape[1]
    in_features = data.shape[2]

    model = MTGNN(
        num_nodes=num_nodes,
        in_features=in_features,
        hidden_channels=mcfg["hidden_channels"],
        num_layers=mcfg["num_layers"],
        node_dim=mcfg["node_dim"],
        temporal_kernel=mcfg["temporal_kernel"],
        top_k=mcfg["top_k"],
        dropout=mcfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=mcfg["learning_rate"],
        weight_decay=mcfg["weight_decay"],
    )

    best_val_mae = float("inf")
    for epoch in range(1, mcfg["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            torch.save(model.state_dict(), "outputs/mtgnn_best.pt")

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MAE: {val_metrics['mae']:.6f} | "
                f"Val DA: {val_metrics['directional_accuracy']:.3f}"
            )

    model.load_state_dict(torch.load("outputs/mtgnn_best.pt", weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest Results: {test_metrics}")

    adj = model.get_learned_graph()
    torch.save(adj, "outputs/learned_graph.pt")
    print("Learned adjacency matrix saved to outputs/learned_graph.pt")


if __name__ == "__main__":
    from pathlib import Path
    Path("outputs").mkdir(exist_ok=True)
    main()
