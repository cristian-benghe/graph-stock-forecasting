"""Shared training loop.

All neural models go through this so the comparison varies only the model, not
the optimizer, early-stopping rule, or evaluation path.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.evaluation.metrics import evaluate_all


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader: DataLoader, device, adj=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (targets, predictions) stacked over the loader."""
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x, adj=adj) if adj is not None else model(x)
        preds.append(out.cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(preds)


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    patience: int = 15,
    grad_clip: float = 5.0,
    adj=None,
    device=torch.device("cpu"),
    verbose: bool = True,
    log_every: int = 10,
) -> dict:
    """Train with early stopping on validation MAE.

    Selecting the checkpoint on validation (never test) keeps the test set
    untouched for the final numbers.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, adj=adj) if adj is not None else model(x)
            loss = F.mse_loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = total / n
        y_true, y_pred = predict(model, val_loader, device, adj)
        val = evaluate_all(y_true, y_pred)
        scheduler.step(val["mae"])
        history.append({"epoch": epoch, "train_loss": train_loss, **val})

        if val["mae"] < best_val - 1e-9:
            best_val = val["mae"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and epoch % log_every == 0:
            print(
                f"    epoch {epoch:3d} | train {train_loss:.6f} | "
                f"val MAE {val['mae']:.6f} | val DA {val['directional_accuracy']:.4f}"
            )

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"    early stop at epoch {epoch} (best {best_epoch})")
            break

    model.load_state_dict(best_state)
    return {
        "model": model,
        "best_val_mae": best_val,
        "best_epoch": best_epoch,
        "history": history,
    }
