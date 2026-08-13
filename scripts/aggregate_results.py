#!/usr/bin/env python
"""Aggregate results/<run_id>/result.json files into paper-ready summaries.

Groups runs by everything except the seed, reports mean ± std per group, and
runs paired Wilcoxon + effect sizes between any two named groups.

Usage:
    python scripts/aggregate_results.py --results results \
        --metric accuracy --out summary.csv
    python scripts/aggregate_results.py --results results \
        --compare hybrid classical_nn --metric accuracy
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qgrid.stats import summarize, paired_comparison  # noqa: E402


def group_key(cfg: dict) -> str:
    m = cfg["model"]
    parts = [m["type"]]
    if m["type"] in ("hybrid", "classical_nn"):
        q = m["quantum"]
        parts += [f"q{q['n_qubits']}", f"l{q['n_layers']}", q["embedding"]]
        noise = q.get("noise", {})
        if noise.get("type", "none") != "none":
            parts += [noise["type"], f"p{noise['p']}"]
    parts.append(f"n{cfg['data']['n_samples'] or 'full'}")
    return "_".join(str(p) for p in parts)


def load_all(results_dir: str) -> list[dict]:
    records = []
    for p in sorted(Path(results_dir).glob("*/result.json")):
        with open(p) as f:
            records.append(json.load(f))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--metric", default="accuracy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None,
                    metavar=("GROUP_A", "GROUP_B"),
                    help="Two group-key substrings to compare (paired by seed)")
    args = ap.parse_args()

    records = load_all(args.results)
    if not records:
        print("No results found.")
        return

    groups: dict[str, dict[int, float]] = defaultdict(dict)
    for r in records:
        key = group_key(r["config"])
        seed = r["config"]["experiment"]["seed"]
        groups[key][seed] = r["clean"][args.metric]

    rows = []
    for key, by_seed in sorted(groups.items()):
        s = summarize(list(by_seed.values()))
        rows.append({"group": key, **s})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved: {args.out}")

    if args.compare:
        ka = [k for k in groups if args.compare[0] in k]
        kb = [k for k in groups if args.compare[1] in k]
        if len(ka) != 1 or len(kb) != 1:
            print(f"\nAmbiguous compare selectors: {ka} vs {kb}")
            return
        a, b = groups[ka[0]], groups[kb[0]]
        common = sorted(set(a) & set(b))
        cmp = paired_comparison([a[s] for s in common], [b[s] for s in common])
        print(f"\nPaired comparison ({ka[0]} vs {kb[0]}, "
              f"{len(common)} common seeds):")
        for k, v in cmp.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
