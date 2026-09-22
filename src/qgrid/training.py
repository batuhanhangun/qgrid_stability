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


def _grad_norm_groups(model: nn.Module) -> dict[str, list]:
    """Parameters grouped for gradient-norm logging: encoder / quantum
    (the VQC, or the classical bottleneck standing in for it) / decoder."""
    groups: dict[str, list] = {"encoder": [], "quantum": [], "decoder": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            groups["encoder"].append(p)
        elif name.startswith("decoder."):
            groups["decoder"].append(p)
        else:
            groups["quantum"].append(p)
    return groups


def _group_grad_norm(params) -> float:
    """L2 norm of the concatenated gradients already present on `params`."""
    with torch.no_grad():
        sq = sum(float(p.grad.detach().pow(2).sum()) for p in params
                 if p.grad is not None)
    return float(np.sqrt(sq))


def train_model(model: nn.Module, train_loader, val_loader, cfg: dict,
                log_every: int = 10, verbose: bool = True,
                log_grad_norms: bool = False) -> dict:
    """Returns a history dict with per-epoch train/val loss and val accuracy.

    With log_grad_norms=True the history also carries per-epoch mean L2
    gradient norms per parameter group (grad_norm_encoder / _quantum /
    _decoder), read off the gradients left by loss.backward().
    """
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=cfg["lr_decay_every"], gamma=cfg["lr_decay_factor"])
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    if log_grad_norms:
        grad_groups = _grad_norm_groups(model)
        for g in grad_groups:
            history[f"grad_norm_{g}"] = []

    for epoch in range(cfg["epochs"]):
        model.train()
        train_losses = []
        if log_grad_norms:
            batch_norms = {g: [] for g in grad_groups}
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if log_grad_norms:
                for g, ps in grad_groups.items():
                    batch_norms[g].append(_group_grad_norm(ps))
            opt.step()
            train_losses.append(loss.item())
        sched.step()
        if log_grad_norms:
            for g in grad_groups:
                history[f"grad_norm_{g}"].append(float(np.mean(batch_norms[g])))

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
