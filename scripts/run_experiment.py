#!/usr/bin/env python
"""Run a single experiment (one model, one seed) from a config.

Usage:
    python scripts/run_experiment.py --config configs/base.yaml \
        --set experiment.seed=3 model.quantum.n_qubits=4

Every run writes results/<run_id>/result.json containing config, parameter
counts, training history, clean metrics, and input-noise robustness metrics.
Aggregation across runs is done separately (scripts/aggregate_results.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qgrid.utils import load_config, save_result, run_id  # noqa: E402
from qgrid.data import load_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--override", default=None,
                    help="Optional sweep-specific YAML overlay")
    ap.add_argument("--set", nargs="*", default=[], dest="sets",
                    help="Dotted-key overrides, e.g. experiment.seed=3")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override, args.sets)
    seed = cfg["experiment"]["seed"]
    print(f"[qgrid] run: {run_id(cfg)}", flush=True)

    split = load_dataset(
        csv_path=cfg["data"]["csv_path"],
        n_samples=cfg["data"]["n_samples"],
        test_size=cfg["data"]["test_size"],
        seed=seed,
        stratify=cfg["data"]["stratify"],
        standardize=cfg["data"]["standardize"],
    )
    print(f"[qgrid] train={len(split.X_train)} test={len(split.X_test)}",
          flush=True)

    model_type = cfg["model"]["type"]
    t0 = time.time()

    if model_type in ("hybrid", "classical_nn"):
        import torch  # noqa: F401
        from qgrid.models.hybrid import (HybridQNN, MatchedClassicalNN,
                                         count_params)
        from qgrid.training import set_all_seeds, make_loaders, train_model
        from qgrid.evaluation import (predict_torch, evaluate_clean,
                                      evaluate_input_noise)

        set_all_seeds(seed)
        cls = HybridQNN if model_type == "hybrid" else MatchedClassicalNN
        model = cls(cfg["model"])
        params = count_params(model)
        print(f"[qgrid] params: {params}", flush=True)

        train_loader, val_loader = make_loaders(
            split.X_train, split.y_train, split.X_test, split.y_test,
            cfg["training"]["batch_size"])
        history = train_model(model, train_loader, val_loader,
                              cfg["training"])

        predict_fn = lambda X: predict_torch(model, X)  # noqa: E731
        clean = evaluate_clean(predict_fn, split.X_test, split.y_test)
        noise = evaluate_input_noise(predict_fn, split.X_test, split.y_test,
                                     cfg["evaluation"]["input_noise"], seed)
        payload = {"params": params, "history": history,
                   "clean": clean, "input_noise": noise}

    else:
        from qgrid.models.baselines import build_sklearn_baseline
        from qgrid.evaluation import (compute_metrics, evaluate_input_noise)
        import numpy as np

        model = build_sklearn_baseline(model_type, seed)
        model.fit(split.X_train, split.y_train)

        def predict_fn(X):
            proba = model.predict_proba(X)
            return np.argmax(proba, axis=1), proba[:, 1]

        y_pred, y_score = predict_fn(split.X_test)
        clean = compute_metrics(split.y_test, y_pred, y_score)
        noise = evaluate_input_noise(predict_fn, split.X_test, split.y_test,
                                     cfg["evaluation"]["input_noise"], seed)
        payload = {"params": None, "history": None,
                   "clean": clean, "input_noise": noise}

    payload["wall_time_sec"] = time.time() - t0
    path = save_result(cfg, payload)
    print(f"[qgrid] clean accuracy: {payload['clean']['accuracy']:.4f}")
    print(f"[qgrid] saved: {path}", flush=True)


if __name__ == "__main__":
    main()
