#!/usr/bin/env python
"""Generate the full experiment manifest (cross product of sweep axes).

Emits one command line per run into a manifest file. Runs are independent:
execute the manifest sequentially, or distribute its lines over several
processes or machines.

Usage:
    python scripts/run_sweep.py --sweep configs/sweeps/main.yaml \
        --manifest manifest.txt
    # local sequential execution instead of manifest:
    python scripts/run_sweep.py --sweep configs/sweeps/main.yaml --execute
"""
from __future__ import annotations

import argparse
import itertools
import subprocess

import yaml


def expand(sweep: dict) -> list[list[str]]:
    """sweep file format:
    base: configs/base.yaml
    axes:
      experiment.seed: [0, 1, ..., 29]
      model.quantum.n_qubits: [2, 3, 4]
    fixed:
      data.n_samples: 10000
    """
    axes = sweep.get("axes", {})
    fixed = sweep.get("fixed", {})
    keys = list(axes.keys())
    combos = itertools.product(*(axes[k] for k in keys)) if keys else [()]
    # Optional explicit list of per-run override dicts (crossed with axes);
    # sweeps without a `runs` key expand exactly as before.
    explicit = sweep.get("runs") or [{}]

    runs = []
    for combo in combos:
        for extra in explicit:
            sets = [f"{k}={v}" for k, v in fixed.items()]
            sets += [f"{k}={v}" for k, v in extra.items()]
            sets += [f"{k}={v}" for k, v in zip(keys, combo)]
            runs.append(sets)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--manifest", default="manifest.txt")
    ap.add_argument("--execute", action="store_true",
                    help="Run sequentially instead of writing a manifest")
    args = ap.parse_args()

    with open(args.sweep) as f:
        sweep = yaml.safe_load(f)

    base = sweep.get("base", "configs/base.yaml")
    runs = expand(sweep)
    cmds = [
        f"python scripts/run_experiment.py --config {base} --set "
        + " ".join(sets)
        for sets in runs
    ]

    if args.execute:
        for i, cmd in enumerate(cmds, 1):
            print(f"=== [{i}/{len(cmds)}] {cmd}", flush=True)
            subprocess.run(cmd, shell=True, check=True)
    else:
        with open(args.manifest, "w") as f:
            f.write("\n".join(cmds) + "\n")
        print(f"Wrote {len(cmds)} runs to {args.manifest}")


if __name__ == "__main__":
    main()
