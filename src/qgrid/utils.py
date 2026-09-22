"""Config loading, deep-merge overrides, and structured result IO."""
from __future__ import annotations

import copy
import json
import os
from datetime import datetime

import yaml


def load_config(base_path: str, override_path: str | None = None,
                cli_overrides: list[str] | None = None) -> dict:
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    if override_path:
        with open(override_path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f))
    for item in cli_overrides or []:
        key, value = item.split("=", 1)
        set_by_path(cfg, key, yaml.safe_load(value))
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def set_by_path(cfg: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def run_id(cfg: dict) -> str:
    m = cfg["model"]
    parts = [cfg["experiment"]["name"], m["type"]]
    if m["type"] in ("hybrid", "classical_nn"):
        q = m["quantum"]
        parts += [f"q{q['n_qubits']}", f"l{q['n_layers']}", q["embedding"]]
        if q.get("noise", {}).get("type", "none") != "none":
            parts += [q["noise"]["type"], f"p{q['noise']['p']}"]
        # Mismatched train/eval channel noise (revision); absent by default.
        ev = q.get("eval_noise", {})
        if ev.get("type", "none") != "none":
            parts += ["eval", ev["type"], f"p{ev['p']}"]
    parts += [f"n{cfg['data']['n_samples'] or 'full'}",
              f"seed{cfg['experiment']['seed']}"]
    return "_".join(str(p) for p in parts)


def env_info() -> dict:
    """Interpreter and package versions, for the result.json `env` block."""
    import platform
    from importlib import metadata
    out = {"python": platform.python_version()}
    for name, dist in [("pennylane", "pennylane"), ("torch", "torch"),
                       ("sklearn", "scikit-learn"), ("xgboost", "xgboost"),
                       ("numpy", "numpy"), ("scipy", "scipy")]:
        try:
            out[name] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def save_result(cfg: dict, payload: dict):
    out_dir = os.path.join(cfg["experiment"]["output_dir"], run_id(cfg))
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "run_id": run_id(cfg),
        "timestamp": datetime.utcnow().isoformat(),
        "config": cfg,
        **payload,
    }
    path = os.path.join(out_dir, "result.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
