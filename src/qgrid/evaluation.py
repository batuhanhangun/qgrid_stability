"""Evaluation: clean metrics and input-noise robustness."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)

from .data import NOISE_FNS


def predict_torch(model, X: np.ndarray, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    import torch  # lazy: keeps sklearn-only runs torch-free
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32)
            p = torch.softmax(model(xb), dim=1).numpy()
            probs.append(p)
    probs = np.concatenate(probs)
    return probs.argmax(1), probs[:, 1]


def compute_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_clean(predict_fn, X_test, y_test) -> dict:
    y_pred, y_score = predict_fn(X_test)
    return compute_metrics(y_test, y_pred, y_score)


def evaluate_input_noise(predict_fn, X_test, y_test,
                         noise_cfg: dict, seed: int) -> dict:
    """Sweep all configured input-noise types and levels.

    Noise draws use a dedicated RNG seeded by the run seed so robustness
    results are reproducible per seed but vary across seeds (feeding the
    across-seed statistics).
    """
    rng = np.random.default_rng(seed + 10_000)
    results: dict[str, dict] = {}
    level_key = {"gaussian": "gaussian_sigma",
                 "uniform": "uniform_width",
                 "dropout": "dropout_p"}
    for noise_name, cfg_key in level_key.items():
        results[noise_name] = {}
        for level in noise_cfg.get(cfg_key, []):
            Xn = NOISE_FNS[noise_name](X_test, level, rng)
            y_pred, y_score = predict_fn(Xn)
            results[noise_name][str(level)] = compute_metrics(
                y_test, y_pred, y_score)
    return results
