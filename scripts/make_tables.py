#!/usr/bin/env python
"""Regenerate all tables and statistical tests in the paper from
results/*/result.json.

Usage:
    python scripts/make_tables.py [--results results] [--out paper_assets/tables]

Prints every table to stdout and writes LaTeX versions. Also prints the
headline statistics quoted in the paper: paired Wilcoxon tests, rank-biserial
correlation, Cohen's d, the TOST equivalence procedure, and the Holm-corrected
channel-noise comparisons.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

NAME = {"hybrid": "Hybrid (ours)", "classical_nn": "Classical NN",
        "svm": "SVM (RBF)", "xgboost": "XGBoost",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting"}
ORDER = list(NAME.keys())


def load(results_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*", "result.json"))):
        with open(p) as f:
            recs.append(json.load(f))
    if not recs:
        raise SystemExit(f"No result.json files under {results_dir}")
    return recs


def key(r):
    c = r["config"]
    q = c["model"]["quantum"]
    return dict(exp=c["experiment"]["name"], seed=c["experiment"]["seed"],
                model=c["model"]["type"], qubits=q["n_qubits"],
                layers=q["n_layers"], emb=q["embedding"],
                qnoise=q.get("noise", {}).get("type", "none"),
                qnoise_p=q.get("noise", {}).get("p", 0.0))


def sel(recs, **filt):
    """seed -> record for records matching all filters."""
    out = {}
    for r in recs:
        k = key(r)
        if all(k[a] == v for a, v in filt.items()):
            out[k["seed"]] = r
    return out


def ms(vals, scale=100):
    a = np.asarray(vals) * scale
    return f"{a.mean():.2f} $\\pm$ {a.std(ddof=1):.2f}"


def paired(a_by_seed, b_by_seed):
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    a = np.array([a_by_seed[s] for s in seeds])
    b = np.array([b_by_seed[s] for s in seeds])
    d = a - b
    w, p = stats.wilcoxon(a, b)
    nz = d[d != 0]
    tot = len(nz) * (len(nz) + 1) / 2
    rb = abs(2 * w / tot - 1) * np.sign(d.mean()) if tot else 0.0
    cd = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
    return dict(p=p, rank_biserial=rb, cohens_d=cd,
                mean_diff_pp=d.mean() * 100, n=len(seeds))


def tost(a_by_seed, b_by_seed, margin=0.005):
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    d = np.array([a_by_seed[s] - b_by_seed[s] for s in seeds])
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    p_low = 1 - stats.t.cdf((m + margin) / se, n - 1)
    p_high = stats.t.cdf((m - margin) / se, n - 1)
    ci = stats.t.interval(0.90, n - 1, loc=m, scale=se)
    return dict(p_tost=max(p_low, p_high),
                ci90_pp=(ci[0] * 100, ci[1] * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper_assets/tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = load(args.results)

    acc = lambda d: {s: r["clean"]["accuracy"] for s, r in d.items()}  # noqa

    # ---- Table 1: reproduction subset ----
    print("=" * 70, "\nTABLE 1: reproduction (n=2000, 10 seeds)")
    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             "\\textbf{Model} & \\textbf{Accuracy (\\%)} & \\textbf{F1 (\\%)} \\\\",
             "\\midrule"]
    for mdl in ["hybrid", "classical_nn"]:
        d = sel(recs, exp="subset2k", model=mdl)
        row = (f"{NAME[mdl]} & {ms([r['clean']['accuracy'] for r in d.values()])}"
               f" & {ms([r['clean']['f1'] for r in d.values()])} \\\\")
        print("  ", row)
        lines.append(row)
    cmp1 = paired(acc(sel(recs, exp="subset2k", model="hybrid")),
                  acc(sel(recs, exp="subset2k", model="classical_nn")))
    print(f"   paired: p={cmp1['p']:.3f}, diff={cmp1['mean_diff_pp']:+.2f}pp")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(args.out, "tab_subset.tex"), "w").write("\n".join(lines))

    # ---- Table 2: main ----
    print("=" * 70, "\nTABLE 2: main comparison (n=10000, 30 seeds)")
    hyb = sel(recs, exp="main", model="hybrid")
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "\\textbf{Model} & \\textbf{Accuracy (\\%)} & \\textbf{F1 (\\%)} & "
             "\\textbf{ROC-AUC (\\%)} & \\textbf{$p$ vs.\\ hybrid} \\\\",
             "\\midrule"]
    for mdl in ORDER:
        d = sel(recs, exp="main", model=mdl)
        if mdl == "hybrid":
            pcol = "--"
        else:
            c = paired(acc(hyb), acc(d))
            pcol = f"{c['p']:.3f}" if c["p"] >= 0.001 else "$<$0.001"
            print(f"   vs {mdl}: p={c['p']:.3g}, d={c['cohens_d']:+.2f}, "
                  f"diff={c['mean_diff_pp']:+.2f}pp, rb={c['rank_biserial']:+.3f}")
        row = (f"{NAME[mdl]} & {ms([r['clean']['accuracy'] for r in d.values()])} & "
               f"{ms([r['clean']['f1'] for r in d.values()])} & "
               f"{ms([r['clean']['roc_auc'] for r in d.values()])} & {pcol} \\\\")
        lines.append(row)
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(args.out, "tab_main.tex"), "w").write("\n".join(lines))
    t = tost(acc(hyb), acc(sel(recs, exp="main", model="classical_nn")))
    print(f"   TOST +/-0.5pp: p={t['p_tost']:.2e}; "
          f"90% CI [{t['ci90_pp'][0]:+.2f}, {t['ci90_pp'][1]:+.2f}]pp")

    # ---- Table 3: ablation ----
    print("=" * 70, "\nTABLE 3: ablation (5 seeds/cell)")
    lines = ["\\begin{tabular}{llccc}", "\\toprule",
             "\\textbf{Embedding} & \\textbf{Qubits} & \\textbf{$L=1$} & "
             "\\textbf{$L=2$} & \\textbf{$L=3$} \\\\", "\\midrule"]
    for emb in ["amplitude", "angle"]:
        for q in [2, 3, 4]:
            cells = []
            for L in [1, 2, 3]:
                d = sel(recs, exp="ablation_arch", emb=emb, qubits=q, layers=L)
                cells.append(ms([r["clean"]["accuracy"] for r in d.values()]))
            row = (f"{emb.capitalize() if q == 2 else ''} & {q} & "
                   + " & ".join(cells) + " \\\\")
            print("  ", row)
            lines.append(row)
        if emb == "amplitude":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(args.out, "tab_ablation.tex"), "w").write("\n".join(lines))

    # ---- Table 4: quantum channel noise (Holm) ----
    print("=" * 70, "\nTABLE 4: quantum channel noise")
    conds, pvals = [], []
    for tch in ["depolarizing", "amplitude_damping"]:
        for p_ in [0.01, 0.05]:
            d = sel(recs, exp="qnoise", qnoise=tch, qnoise_p=p_)
            base10 = {s: hyb[s]["clean"]["accuracy"] for s in d}
            c = paired(acc(d), base10)
            conds.append((tch, p_, [r["clean"]["accuracy"] for r in d.values()],
                          c["p"]))
            pvals.append(c["p"])
    holm = np.array(pvals) * (4 - np.argsort(np.argsort(pvals)))
    lines = ["\\begin{tabular}{llcc}", "\\toprule",
             "\\textbf{Channel} & \\textbf{Strength $p$} & "
             "\\textbf{Accuracy (\\%)} & \\textbf{$p_{\\mathrm{unc}}$ vs.\\ "
             "noiseless} \\\\", "\\midrule",
             f"None (baseline, 30 seeds) & -- & "
             f"{ms([r['clean']['accuracy'] for r in hyb.values()])} & -- \\\\"]
    for (tch, p_, vals, pu), ph in zip(conds, holm):
        nm = "Depolarizing" if tch == "depolarizing" else "Amplitude damping"
        row = f"{nm} & {p_} & {ms(vals)} & {pu:.3f} \\\\"
        print(f"   {row}   (Holm-adjusted p={min(ph, 1):.3f})")
        lines.append(row)
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(args.out, "tab_qnoise.tex"), "w").write("\n".join(lines))

    # ---- Table 5: strongest input noise ----
    print("=" * 70, "\nTABLE 5: strongest input-noise levels (30 seeds)")
    strongest = {"gaussian": "0.3", "uniform": "0.3", "dropout": "0.2"}
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "\\textbf{Model} & \\textbf{Gaussian ($\\sigma{=}0.30$)} & "
             "\\textbf{Uniform ($w{=}0.30$)} & \\textbf{Dropout ($p{=}0.20$)} \\\\",
             "\\midrule"]
    for mdl in ORDER:
        d = sel(recs, exp="main", model=mdl)
        cells = [ms([r["input_noise"][nt][lv]["accuracy"]
                     for r in d.values()]) for nt, lv in strongest.items()]
        row = f"{NAME[mdl]} & " + " & ".join(cells) + " \\\\"
        print("  ", row)
        lines.append(row)
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(args.out, "tab_input_noise.tex"), "w").write(
        "\n".join(lines))

    # per-level hybrid vs classical paired tests (Section 5.4 claim)
    print("=" * 70, "\nSec 5.4: hybrid vs classical per noise level")
    hd = sel(recs, exp="main", model="hybrid")
    cd_ = sel(recs, exp="main", model="classical_nn")
    for nt in ["gaussian", "uniform", "dropout"]:
        for lv in sorted(next(iter(hd.values()))["input_noise"][nt]):
            a = {s: r["input_noise"][nt][lv]["accuracy"] for s, r in hd.items()}
            b = {s: r["input_noise"][nt][lv]["accuracy"] for s, r in cd_.items()}
            c = paired(a, b)
            print(f"   {nt:9s} {lv:5s} diff {c['mean_diff_pp']:+.2f}pp "
                  f"p={c['p']:.4f}")

    print("=" * 70, f"\nWrote LaTeX tables to {args.out}/")


if __name__ == "__main__":
    main()
