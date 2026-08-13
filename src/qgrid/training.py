"""Training loop for the neural models (hybrid and classical)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_all_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(X_train, y_train, X_test, y_test, batch_size: int):
    tr = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                       torch.tensor(y_train, dtype=torch.long))
    te = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                       torch.tensor(y_test, dtype=torch.long))
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(te, batch_size=batch_size, shuffle=False))


def train_model(model: nn.Module, train_loader, val_loader, cfg: dict,
                log_every: int = 10, verbose: bool = True) -> dict:
    """Returns a history dict with per-epoch train/val loss and val accuracy."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=cfg["lr_decay_every"], gamma=cfg["lr_decay_factor"])
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(cfg["epochs"]):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        sched.step()

        model.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                val_losses.append(loss_fn(logits, yb).item())
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(np.mean(val_losses)))
        history["val_acc"].append(correct / total)

        if verbose and (epoch + 1) % log_every == 0:
            print(f"  epoch {epoch + 1:3d}/{cfg['epochs']}  "
                  f"train {history['train_loss'][-1]:.4f}  "
                  f"val {history['val_loss'][-1]:.4f}  "
                  f"acc {history['val_acc'][-1]:.4f}", flush=True)

    return history
