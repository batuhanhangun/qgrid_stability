#!/usr/bin/env python
"""Regenerate every table and statistic in the paper from results/*/result.json.

Usage:
    python scripts/make_tables.py [--results results] [--out paper_assets/tables]
                                  [--data data/grid_stability.csv]

Prints all tables and the statistics quoted in the text, and writes LaTeX
versions of the tables to --out. The dataset CSV is needed only for the
Section 5.5 analysis (PCA spectrum and MLP on projections).

Conventions
    accuracy  = trace(confusion_matrix) / sum(confusion_matrix)
    F1        = 2TP / (2TP + FP + FN)          (unstable = positive class)
    means     : exact rational arithmetic on counts, x100, 2 dp, ROUND_HALF_UP
    SDs       : float, ddof=1, ROUND_HALF_UP
    paired p  : scipy.stats.wilcoxon on the integer differences in
                correct-prediction counts, paired by seed
    r         = |2W/T - 1| * sign(mean d), T = m(m+1)/2 over non-zero diffs
    Cohen d   = mean(d) / sd(d)
    collapsed = clean accuracy < 0.70
"""
from __future__ import annotations

import argparse
import json
import os
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import stats

NAME = {"hybrid": "Hybrid", "classical_nn": "Classical NN",
        "svm": "SVM (RBF)", "xgboost": "XGBoost",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting"}
ORDER = list(NAME)
CH = {"none": "None", "depolarizing": "Depolarizing",
      "amplitude_damping": "Amplitude damping"}
COLLAPSE = 0.70


# ----------------------------------------------------------------- loading
def load(results_dir):
    recs = []
    for p in sorted(Path(results_dir).glob("*/result.json")):
        with open(p) as f:
            recs.append(json.load(f))
    if not recs:
        print(f"No result.json files under {results_dir}")
    return recs


def key(r):
    c = r["config"]
    q = c["model"]["quantum"]
    ev = q.get("eval_noise", {"type": "none", "p": 0.0})
    return dict(exp=c["experiment"]["name"], seed=c["experiment"]["seed"],
                model=c["model"]["type"], qubits=q["n_qubits"],
                layers=q["n_layers"], emb=q["embedding"],
                qnoise=q["noise"]["type"], qnoise_p=q["noise"]["p"],
                enoise=ev["type"], enoise_p=ev["p"])


def sel(recs, **filt):
    """seed -> record for the records matching all filters."""
    out = {}
    for r in recs:
        k = key(r)
        if all(k[a] == v for a, v in filt.items()):
            assert k["seed"] not in out, ("duplicate seed", filt, k["seed"])
            out[k["seed"]] = r
    return out


class Missing(Exception):
    """Raised when --results lacks an experiment group a table needs."""


def need(recs, **filt):
    """Like sel(), but raises Missing when no run matches."""
    d = sel(recs, **filt)
    if not d:
        raise Missing(" ".join(f"{a}={v}" for a, v in filt.items()))
    return d


# ----------------------------------------------------------------- metrics
def cm_of(r, noise=None, level=None):
    if noise is None:
        return r["clean"]["confusion_matrix"]
    return r["input_noise"][noise][level]["confusion_matrix"]


def correct(cm):
    return cm[0][0] + cm[1][1]


def total(cm):
    return sum(map(sum, cm))


def acc(cm):
    return Fraction(correct(cm), total(cm))


def f1(cm):
    (_, fp), (fn, tp) = cm
    return Fraction(2 * tp, 2 * tp + fp + fn)


def bal_acc(cm):
    (tn, fp), (fn, tp) = cm
    return (Fraction(tn, tn + fp) + Fraction(tp, tp + fn)) / 2


def recall_stable(cm):
    (tn, fp), _ = cm
    return Fraction(tn, tn + fp)


def recall_unstable(cm):
    _, (fn, tp) = cm
    return Fraction(tp, tp + fn)


def rnd(x, nd=2):
    """Round-half-up to nd decimals; exact for Fractions, repr-based for floats."""
    if isinstance(x, Fraction):
        d = Decimal(x.numerator) / Decimal(x.denominator)
    else:
        d = Decimal(repr(float(x)))
    return d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def mean_pct(vals):
    """Exact mean x100, 2 dp (Fractions stay exact; floats are converted exactly)."""
    vals = list(vals)
    m = sum(Fraction(v) for v in vals) / len(vals)
    return rnd(m * 100)


def sd_raw(vals):
    a = np.array([float(v) for v in vals]) * 100
    return a.std(ddof=1)


def sd_pct(vals, nd=2):
    return rnd(sd_raw(vals), nd)


def ms(vals, nd_sd=2):
    vals = list(vals)
    return f"{mean_pct(vals)} $\\pm$ {sd_pct(vals, nd_sd)}"


def ms_txt(vals, nd_sd=2):
    vals = list(vals)
    return f"{mean_pct(vals)} +- {sd_pct(vals, nd_sd)}"


# ----------------------------------------------------------------- tests
def paired(a_by_seed, b_by_seed, metric=cm_of):
    """Wilcoxon on integer differences in correct counts (a - b), paired by seed.

    Returns dict(p, W, r, d, diff_pp, n); diff_pp is the exact difference of
    mean accuracies (a - b) in percentage points as a Fraction.
    """
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    ca = [metric(a_by_seed[s]) for s in seeds]
    cb = [metric(b_by_seed[s]) for s in seeds]
    d = np.array([correct(x) - correct(y) for x, y in zip(ca, cb)], dtype=int)
    acc_d = np.array([float(acc(x) - acc(y)) for x, y in zip(ca, cb)])
    if np.all(d == 0):
        w, p, r = 0.0, 1.0, 0.0
    else:
        w, p = stats.wilcoxon(d)
        m = int(np.count_nonzero(d))
        T = m * (m + 1) / 2
        r = abs(2 * w / T - 1) * np.sign(d.mean())
    cd = acc_d.mean() / acc_d.std(ddof=1) if acc_d.std(ddof=1) > 0 else 0.0
    diff = (sum(acc(x) for x in ca) - sum(acc(y) for y in cb)) / len(seeds) * 100
    return dict(p=float(p), W=float(w), r=float(r), d=float(cd), diff_pp=diff,
                n=len(seeds))


def wilcoxon_float(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 1.0
    return float(stats.wilcoxon(a, b).pvalue)


def holm(pvals):
    """Holm step-down with monotonicity enforcement."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj


def tost_p(d_pp, margin):
    n, m, se = len(d_pp), d_pp.mean(), d_pp.std(ddof=1) / np.sqrt(len(d_pp))
    p_low = stats.t.sf((m + margin) / se, n - 1)
    p_high = stats.t.cdf((m - margin) / se, n - 1)
    return max(p_low, p_high)


def tost(a_by_seed, b_by_seed, margin=0.5):
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    d = np.array([float(acc(cm_of(a_by_seed[s])) - acc(cm_of(b_by_seed[s])))
                  for s in seeds]) * 100
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    ci = stats.t.interval(0.90, n - 1, loc=m, scale=se)
    lo, hi = 0.0, 5.0
    while hi - lo > 1e-6:
        mid = (lo + hi) / 2
        if tost_p(d, mid) < 0.05:
            hi = mid
        else:
            lo = mid
    return dict(p=tost_p(d, margin), ci90=ci, min_margin=hi)


def t_interval(d_pp, conf=0.95):
    n, m, se = len(d_pp), d_pp.mean(), d_pp.std(ddof=1) / np.sqrt(len(d_pp))
    return stats.t.interval(conf, n - 1, loc=m, scale=se)


def fmt_p(p):
    return f"{p:.3f}" if p >= 0.001 else f"{p:.2g}"


def write_tex(out, name, header, rows, colspec):
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\",
             "\\midrule"]
    for row in rows:
        lines.append(row if row == "\\midrule" else row + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(out, name), "w") as f:
        f.write("\n".join(lines) + "\n")


def banner(title):
    print("=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------- tables
def table1(recs, out):
    groups = {(exp, mdl): need(recs, exp=exp, model=mdl)
              for exp in ["subset2k", "subset770"]
              for mdl in ["hybrid", "classical_nn"]}
    banner("TABLE 1: reproduction subset (n=2000, 10 seeds)")
    rows = []
    for exp in ["subset2k", "subset770"]:
        for mdl in ["hybrid", "classical_nn"]:
            d = groups[(exp, mdl)]
            cms = [cm_of(r) for r in d.values()]
            params = sorted({r["params"]["total"] for r in d.values()})
            row = (f"{NAME[mdl]} ({exp}) & {ms(acc(c) for c in cms)} & "
                   f"{ms(f1(c) for c in cms)} & {params[0]}")
            print(f"  {NAME[mdl]:13s} {exp:9s} acc {ms_txt(acc(c) for c in cms)}  "
                  f"F1 {ms_txt(f1(c) for c in cms)}  params {params}")
            rows.append(row)
    c = paired(groups[("subset2k", "hybrid")], groups[("subset2k", "classical_nn")])
    print(f"  subset2k hybrid vs classical: diff {float(c['diff_pp']):+.2f} pp, "
          f"p = {c['p']:.3f}")
    for mdl in ["hybrid", "classical_nn"]:
        c = paired(groups[("subset770", mdl)], groups[("subset2k", mdl)])
        print(f"  subset770 vs subset2k {NAME[mdl]}: diff {float(c['diff_pp']):+.2f} pp, "
              f"p = {c['p']:.3f}")
    write_tex(out, "table1_reproduction.tex",
              "\\textbf{Model} & \\textbf{Accuracy (\\%)} & \\textbf{F1 (\\%)} & "
              "\\textbf{Params}", rows, "lccc")


def table2(recs, out):
    groups = {m: need(recs, exp="main", model=m) for m in ORDER}
    banner("TABLE 2: main comparison (n=10000, 30 seeds)")
    hyb = groups["hybrid"]
    rows = []
    for mdl in ORDER:
        d = groups[mdl]
        cms = [cm_of(r) for r in d.values()]
        aucs = [r["clean"]["roc_auc"] for r in d.values()]
        if mdl == "hybrid":
            pcol = "--"
        else:
            c = paired(hyb, d)
            pcol = fmt_p(c["p"])
        row = (f"{NAME[mdl]} & {ms(acc(x) for x in cms)} & {ms(f1(x) for x in cms)}"
               f" & {ms(aucs)} & {pcol}")
        print(f"  {NAME[mdl]:17s} acc {ms_txt(acc(x) for x in cms)}  "
              f"F1 {ms_txt(f1(x) for x in cms)}  AUC {ms_txt(aucs)}  p {pcol}")
        rows.append(row)
    write_tex(out, "table2_main.tex",
              "\\textbf{Model} & \\textbf{Accuracy (\\%)} & \\textbf{F1 (\\%)} & "
              "\\textbf{ROC-AUC (\\%)} & \\textbf{$p$ vs.\\ hybrid}", rows, "lcccc")

    c = paired(hyb, groups["classical_nn"])
    print(f"  hybrid vs classical: p = {c['p']:.3f}, r = {c['r']:+.3f}, "
          f"d = {c['d']:+.3f}, diff {float(c['diff_pp']):+.2f} pp")
    for mdl in ["svm", "xgboost", "random_forest", "gradient_boosting"]:
        c = paired(hyb, groups[mdl])
        print(f"  hybrid vs {NAME[mdl]:17s}: diff {float(c['diff_pp']):+.2f} pp, "
              f"p = {c['p']:.2g}, d = {c['d']:.2f}")
    t = tost(hyb, groups["classical_nn"])
    print(f"  TOST (margin 0.5 pp): p = {t['p']:.2e}; 90% CI "
          f"[{t['ci90'][0]:+.2f}, {t['ci90'][1]:+.2f}] pp; "
          f"smallest margin with p < 0.05: +-{t['min_margin']:.3f} pp")
    print("  balanced accuracy:")
    for mdl in ORDER:
        cms = [cm_of(r) for r in groups[mdl].values()]
        print(f"    {NAME[mdl]:17s} {ms_txt(bal_acc(x) for x in cms)}")
    print("  recall stable / unstable:")
    for mdl in ["hybrid", "gradient_boosting"]:
        cms = [cm_of(r) for r in groups[mdl].values()]
        print(f"    {NAME[mdl]:17s} {mean_pct(recall_stable(x) for x in cms)} / "
              f"{mean_pct(recall_unstable(x) for x in cms)}")


def ablation_groups(recs):
    """(emb, q, L) -> seed -> record for the VQC; (emb, q) -> seed -> record for
    the classical control (its n_layers == 1 replicas only)."""
    vqc, cls = {}, {}
    for emb in ["amplitude", "angle"]:
        for q in [2, 3, 4]:
            cls[(emb, q)] = need(recs, exp="ablation30c", emb=emb, qubits=q, layers=1)
            for L in [1, 2, 3]:
                vqc[(emb, q, L)] = need(recs, exp="ablation30", emb=emb, qubits=q,
                                        layers=L)
    return vqc, cls


def seed_variance(recs):
    vqc, cls = ablation_groups(recs)
    banner("Seed-variance comparison (15 configurations, angle q2 excluded)")
    sd_h, sd_c = [], []
    for (emb, q, L), d in vqc.items():
        if emb == "angle" and q == 2:
            continue
        sd_h.append(sd_raw(acc(cm_of(r)) for r in d.values()))
        sd_c.append(sd_raw(acc(cm_of(r)) for r in cls[(emb, q)].values()))
    sd_h, sd_c = np.array(sd_h), np.array(sd_c)
    p = wilcoxon_float(sd_h, sd_c)
    print(f"  hybrid SD lower in {int((sd_h < sd_c).sum())} of {len(sd_h)}; "
          f"Wilcoxon p = {p:.3f}; mean SD hybrid {sd_h.mean():.3f} vs "
          f"classical {sd_c.mean():.3f}")


def table3(recs, out):
    vqc, _ = ablation_groups(recs)
    banner("TABLE 3: ablation (30 seeds per cell)")
    rows = []
    for emb in ["amplitude", "angle"]:
        for q in [2, 3, 4]:
            cells = [ms(acc(cm_of(r)) for r in vqc[(emb, q, L)].values())
                     for L in [1, 2, 3]]
            txt = [ms_txt(acc(cm_of(r)) for r in vqc[(emb, q, L)].values())
                   for L in [1, 2, 3]]
            print(f"  {emb:9s} q{q}: " + " | ".join(txt))
            rows.append(f"{emb.capitalize() if q == 2 else ''} & {q} & "
                        + " & ".join(cells))
        if emb == "amplitude":
            rows.append("\\midrule")
    write_tex(out, "table3_ablation.tex",
              "\\textbf{Embedding} & \\textbf{Qubits} & \\textbf{$L=1$} & "
              "\\textbf{$L=2$} & \\textbf{$L=3$}", rows, "llccc")


def table4_rows(recs):
    """Per (emb, q): VQC pooled (90 runs), classical (30 runs), collapsed counts."""
    vqc, cls = ablation_groups(recs)
    out = {}
    for emb in ["amplitude", "angle"]:
        for q in [2, 3, 4]:
            pooled = [acc(cm_of(r)) for L in [1, 2, 3]
                      for r in vqc[(emb, q, L)].values()]
            ca = [acc(cm_of(r)) for r in cls[(emb, q)].values()]
            out[(emb, q)] = dict(
                vqc=pooled, vqc_col=sum(a < COLLAPSE for a in pooled),
                cls=ca, cls_col=sum(a < COLLAPSE for a in ca))
    return out, vqc, cls


def table4(recs, out):
    cells, vqc, cls = table4_rows(recs)
    banner("TABLE 4: VQC (pooled over depths, 90 runs) vs classical control (30 runs)")
    rows = []
    for emb in ["amplitude", "angle"]:
        for q in [2, 3, 4]:
            c = cells[(emb, q)]
            print(f"  {emb:9s} q{q}: VQC {ms_txt(c['vqc'])} collapsed {c['vqc_col']}/90"
                  f" | classical {ms_txt(c['cls'])} collapsed {c['cls_col']}/30")
            rows.append(f"{emb.capitalize() if q == 2 else ''} & {q} & {ms(c['vqc'])} & "
                        f"{c['vqc_col']}/90 & {ms(c['cls'])} & {c['cls_col']}/30")
        if emb == "amplitude":
            rows.append("\\midrule")
    write_tex(out, "table4_bottleneck.tex",
              "\\textbf{Embedding} & \\textbf{Qubits} & \\textbf{VQC (\\%)} & "
              "\\textbf{Collapsed} & \\textbf{Classical (\\%)} & \\textbf{Collapsed}",
              rows, "llcccc")
    print("  VQC vs classical per (embedding, qubits, layers), Holm over 18:")
    keys, comps = [], []
    for (emb, q, L), d in vqc.items():
        keys.append((emb, q, L))
        comps.append(paired(d, cls[(emb, q)]))
    adj = holm([c["p"] for c in comps])
    others = []
    for k, c, h in zip(keys, comps, adj):
        print(f"    {k[0]:9s} q{k[1]} L{k[2]}: diff {float(c['diff_pp']):+.2f} pp, "
              f"p = {c['p']:.4f}, Holm = {h:.4f}")
        if not (k[0] == "angle" and k[1] == 2):
            others.append((abs(float(c["diff_pp"])), h))
    print(f"  among the other 15: max |diff| {max(a for a, _ in others):.2f} pp, "
          f"smallest Holm p = {min(h for _, h in others):.3f}")


def grad_norms(recs):
    vqc, _ = ablation_groups(recs)
    banner("Gradient norms (ablation30, mean over the final 10 epochs)")

    def g(rs, k):
        return np.array([np.mean(r["history"][k][-10:]) for r in rs])

    def report(label, rs):
        if not rs:
            print(f"  {label:34s} (no runs)")
            return
        print(f"  {label:34s} n={len(rs):2d}  quantum {g(rs, 'grad_norm_quantum').mean():.3f}"
              f"  encoder {g(rs, 'grad_norm_encoder').mean():.3f}")

    for L in [1, 2, 3]:
        rs = list(vqc[("angle", 2, L)].values())
        col = [r for r in rs if acc(cm_of(r)) < COLLAPSE]
        ok = [r for r in rs if acc(cm_of(r)) >= COLLAPSE]
        report(f"angle q2 L{L} collapsed", col)
        report(f"angle q2 L{L} non-collapsed", ok)
    for emb, q, L in [("angle", 3, 2), ("amplitude", 2, 1), ("amplitude", 3, 2)]:
        report(f"{emb} q{q} L{L}", list(vqc[(emb, q, L)].values()))


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    return num / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro"))


def table5(results_dir, out):
    models = ["hybrid", "classical_nn"]
    paths = {(m, seed): Path(results_dir) / f"probe_{m}_q3_l2_amplitude_n10000_seed{seed}"
             / "activations.npz" for m in models for seed in range(10)}
    if not all(p.exists() for p in paths.values()):
        raise Missing("exp=probe (activations.npz, hybrid and classical_nn, seeds 0-9)")
    banner("TABLE 5: representational probes (probe runs, seeds 0-9)")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.decomposition import PCA

    def probe(X, y):
        return cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=3).mean()

    def pr(X):
        ev = np.linalg.eigvalsh(np.cov(X.T))
        return ev.sum() ** 2 / (ev ** 2).sum()

    per = {m: dict(acc=[], enc=[], bot=[], pc1=[], pr=[]) for m in models}
    cka = dict(encoder=[], bottleneck=[], logits=[], null=[], random=[])
    agree = []
    rng = np.random.default_rng(0)
    for seed in range(10):
        z = {}
        for m in models:
            z[m] = dict(np.load(paths[(m, seed)]))
            y = z[m]["labels"]
            per[m]["acc"].append((z[m]["logits"].argmax(1) == y).mean())
            per[m]["enc"].append(probe(z[m]["encoder"], y))
            per[m]["bot"].append(probe(z[m]["bottleneck"], y))
            pc1 = PCA(n_components=1).fit_transform(z[m]["bottleneck"])
            per[m]["pc1"].append(probe(pc1, y))
            per[m]["pr"].append(pr(z[m]["bottleneck"]))
        h, c = z["hybrid"], z["classical_nn"]
        for k in ["encoder", "bottleneck", "logits"]:
            cka[k].append(linear_cka(h[k], c[k]))
        perm = rng.permutation(len(h["bottleneck"]))
        W = rng.standard_normal((h["encoder"].shape[1], h["bottleneck"].shape[1]))
        cka["null"].append(linear_cka(h["bottleneck"], c["bottleneck"][perm]))
        cka["random"].append(linear_cka(h["bottleneck"], h["encoder"] @ W))
        agree.append((h["logits"].argmax(1) == c["logits"].argmax(1)).mean())

    rows = []
    for lab, k in [("Model accuracy (%)", "acc"), ("Probe: encoder (%)", "enc"),
                   ("Probe: bottleneck (%)", "bot"), ("Probe: first PC (%)", "pc1")]:
        vals = [f"{np.mean(per[m][k]) * 100:.2f}" for m in models]
        print(f"  {lab:26s} hybrid {vals[0]} / classical {vals[1]}")
        rows.append(f"{lab.replace('%', chr(92) + '%')} & {vals[0]} & {vals[1]}")
    pr_txt = [(f"{np.mean(per[m]['pr']):.2f}", f"{np.std(per[m]['pr'], ddof=1):.2f}")
              for m in models]
    print(f"  participation ratio        hybrid {pr_txt[0][0]} +- {pr_txt[0][1]} / "
          f"classical {pr_txt[1][0]} +- {pr_txt[1][1]}")
    rows.append(f"Participation ratio & {pr_txt[0][0]} $\\pm$ {pr_txt[0][1]} & "
                f"{pr_txt[1][0]} $\\pm$ {pr_txt[1][1]}")
    rows.append("\\midrule")
    for lab, k in [("encoder", "encoder"), ("bottleneck", "bottleneck"),
                   ("logits", "logits"), ("null (permuted rows)", "null"),
                   ("random projection", "random")]:
        a = np.array(cka[k])
        print(f"  CKA {lab:22s} {a.mean():.3f} +- {a.std(ddof=1):.3f}")
        rows.append(f"CKA {lab} & \\multicolumn{{2}}{{c}}{{{a.mean():.3f} $\\pm$ "
                    f"{a.std(ddof=1):.3f}}}")
    print(f"  prediction agreement       {np.mean(agree) * 100:.2f}")
    rows.append("Prediction agreement (\\%) & "
                f"\\multicolumn{{2}}{{c}}{{{np.mean(agree) * 100:.2f}}}")
    write_tex(out, "table5_probes.tex",
              "\\textbf{Quantity} & \\textbf{Hybrid} & \\textbf{Classical NN}",
              rows, "lcc")


def section55(csv_path):
    banner("Section 5.5: dataset PCA spectrum and MLP on projections")
    if not os.path.exists(csv_path):
        print(f"  {csv_path} not found; run scripts/download_data.py first. Skipped.")
        return
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    df = pd.read_csv(csv_path)
    feats = [f"tau{i}" for i in range(1, 5)] + [f"p{i}" for i in range(1, 5)] + \
            [f"g{i}" for i in range(1, 5)]
    X = df[feats].to_numpy(dtype=float)
    y = (df["stabf"].str.strip() == "unstable").astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0,
                                          stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    pca = PCA().fit(Xtr)
    ev = pca.explained_variance_
    print(f"  cumulative explained variance at k=3: "
          f"{pca.explained_variance_ratio_[:3].sum() * 100:.1f}%")
    print(f"  participation ratio of the PCA spectrum: {ev.sum() ** 2 / (ev ** 2).sum():.2f}")
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    for k in [1, 2, 3, 4, 6, 8, 12]:
        clf = MLPClassifier((32,), max_iter=400, random_state=0)
        clf.fit(Ztr[:, :k], ytr)
        print(f"  MLP on k={k:2d} components: test accuracy "
              f"{clf.score(Zte[:, :k], yte) * 100:.2f}%")


def table6(recs, out):
    cells, _, _ = table4_rows(recs)
    banner("TABLE 6: Table 4 by bottleneck width d_in")
    order = sorted(cells, key=lambda k: (k[1] if k[0] == "angle" else 2 ** k[1], k[0]))
    rows = []
    for emb, q in order:
        c = cells[(emb, q)]
        d_in = q if emb == "angle" else 2 ** q
        print(f"  d_in {d_in:2d} ({emb:9s} q{q}): VQC {ms_txt(c['vqc'])} ({c['vqc_col']}/90)"
              f" | classical {ms_txt(c['cls'])} ({c['cls_col']}/30)")
        rows.append(f"{d_in} & {emb.capitalize()} ($q={q}$) & {ms(c['vqc'])} & "
                    f"{c['vqc_col']}/90 & {ms(c['cls'])} & {c['cls_col']}/30")
    write_tex(out, "table6_width.tex",
              "\\textbf{$d_{\\mathrm{in}}$} & \\textbf{Embedding} & \\textbf{VQC (\\%)} & "
              "\\textbf{Collapsed} & \\textbf{Classical (\\%)} & \\textbf{Collapsed}",
              rows, "llcccc")


def table7(recs, out):
    groups = {m: need(recs, exp="main", model=m) for m in ORDER}
    for exp in ["subset2k", "main"]:
        for mdl in ["hybrid", "classical_nn"]:
            need(recs, exp=exp, model=mdl)
    banner("TABLE 7: strongest input-noise levels (main, 30 seeds)")
    strongest = [("gaussian", "0.3"), ("uniform", "0.3"), ("dropout", "0.2")]
    rows = []
    for mdl in ORDER:
        cells = [ms(acc(cm_of(r, nt, lv)) for r in groups[mdl].values())
                 for nt, lv in strongest]
        txt = [ms_txt(acc(cm_of(r, nt, lv)) for r in groups[mdl].values())
               for nt, lv in strongest]
        print(f"  {NAME[mdl]:17s} " + " | ".join(txt))
        rows.append(f"{NAME[mdl]} & " + " & ".join(cells))
    write_tex(out, "table7_input_noise.tex",
              "\\textbf{Model} & \\textbf{Gaussian ($\\sigma{=}0.30$)} & "
              "\\textbf{Uniform ($w{=}0.30$)} & \\textbf{Dropout ($p{=}0.20$)}",
              rows, "lccc")
    print("  hybrid vs classical per input-noise level:")
    hd, cd = groups["hybrid"], groups["classical_nn"]
    nsig = 0
    for nt in ["gaussian", "uniform", "dropout"]:
        levels = sorted(next(iter(hd.values()))["input_noise"][nt], key=float)
        for lv in levels:
            c = paired(hd, cd, metric=lambda r, nt=nt, lv=lv: cm_of(r, nt, lv))
            flag = "  *" if c["p"] < 0.05 else ""
            nsig += c["p"] < 0.05
            print(f"    {nt:9s} {lv:5s} diff {float(c['diff_pp']):+.2f} pp  "
                  f"p = {c['p']:.4f}{flag}")
    print(f"  levels with p < 0.05: {nsig}")
    print("  dropout 0.2, hybrid minus classical (pp), 95% t-interval:")
    for exp in ["subset2k", "main"]:
        h = sel(recs, exp=exp, model="hybrid")
        c = sel(recs, exp=exp, model="classical_nn")
        seeds = sorted(set(h) & set(c))
        d = np.array([float(acc(cm_of(h[s], "dropout", "0.2"))
                            - acc(cm_of(c[s], "dropout", "0.2"))) for s in seeds]) * 100
        lo, hi = t_interval(d)
        extra = ""
        if exp == "subset2k":
            extra = f"; 2.9*sd/sqrt(n) = {2.9 * d.std(ddof=1) / np.sqrt(len(d)):.2f}"
        print(f"    {exp:9s} {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]{extra}")


def table8(recs, out):
    hyb = need(recs, exp="main", model="hybrid")
    for ch in ["depolarizing", "amplitude_damping"]:
        for p_ in [0.01, 0.05]:
            need(recs, exp="qnoise", qnoise=ch, qnoise_p=p_)
    banner("TABLE 8: channel noise (qnoise, 10 seeds) vs main hybrid, Holm over four")
    hcms = [cm_of(r) for r in hyb.values()]
    rows = [f"{CH['none']} (baseline, 30 seeds) & -- & {ms(acc(x) for x in hcms)} & --"]
    print(f"  baseline (main hybrid): {ms_txt(acc(x) for x in hcms)}")
    conds, ps = [], []
    for ch in ["depolarizing", "amplitude_damping"]:
        for p_ in [0.01, 0.05]:
            d = sel(recs, exp="qnoise", qnoise=ch, qnoise_p=p_)
            c = paired(d, {s: hyb[s] for s in d})
            conds.append((ch, p_, [acc(cm_of(r)) for r in d.values()], c["p"]))
            ps.append(c["p"])
    adj = holm(ps)
    for (ch, p_, vals, pu), h in zip(conds, adj):
        print(f"  {CH[ch]:17s} p={p_}: {ms_txt(vals)}  p = {pu:.3f}  Holm = {h:.3f}")
        rows.append(f"{CH[ch]} & {p_} & {ms(vals)} & {pu:.3f}")
    write_tex(out, "table8_channel_noise.tex",
              "\\textbf{Channel} & \\textbf{Strength $p$} & \\textbf{Accuracy (\\%)} & "
              "\\textbf{$p$ vs.\\ noiseless}", rows, "llcc")


def table9(recs, out):
    conds = ["none", "depolarizing", "amplitude_damping"]
    cell = {}
    for tr in conds:
        for ev in conds:
            if tr != "none" and ev == "none":
                continue
            cell[(tr, ev)] = need(recs, exp="noisematrix", qnoise=tr, enoise=ev)
    qd = need(recs, exp="qnoise", qnoise="depolarizing", qnoise_p=0.05)
    banner("TABLE 9: mismatched train/eval channel noise (noisematrix, 10 seeds)")
    rows = []
    for tr in conds:
        parts, txt = [], []
        for ev in conds:
            if (tr, ev) in cell:
                vals = [acc(cm_of(r)) for r in cell[(tr, ev)].values()]
                parts.append(ms(vals))
                txt.append(ms_txt(vals))
            else:
                parts.append("--")
                txt.append("--")
        print(f"  train {CH[tr]:17s}: " + " | ".join(txt))
        rows.append(f"{CH[tr]} & " + " & ".join(parts))
    write_tex(out, "table9_noise_matrix.tex",
              "\\textbf{Train $\\backslash$ Eval} & \\textbf{None} & "
              "\\textbf{Depolarizing} & \\textbf{Amplitude damping}", rows, "lccc")
    comps = [(("none", "depolarizing"), ("none", "none")),
             (("none", "amplitude_damping"), ("none", "none")),
             (("depolarizing", "amplitude_damping"), ("depolarizing", "depolarizing")),
             (("amplitude_damping", "depolarizing"),
              ("amplitude_damping", "amplitude_damping"))]
    for a, b in comps:
        c = paired(cell[a], cell[b])
        print(f"  {a[0]}->{a[1]} vs {b[0]}->{b[1]}: diff {float(c['diff_pp']):+.2f} pp, "
              f"p = {c['p']:.3f}")
    nn = cell[("none", "none")]
    c = paired(qd, nn)
    print(f"  qnoise depolarizing 0.05 vs noisematrix none->none: diff "
          f"{float(c['diff_pp']):+.2f} pp, p = {c['p']:.3f}")

    def gap(r):
        h = r["history"]
        return np.mean(h["val_loss"][-10:]) - np.mean(h["train_loss"][-10:])

    seeds = sorted(set(qd) & set(nn))
    p = wilcoxon_float([gap(qd[s]) for s in seeds], [gap(nn[s]) for s in seeds])
    print(f"  generalization gap, same pair: Wilcoxon p = {p:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper_assets/tables")
    ap.add_argument("--data", default="data/grid_stability.csv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = load(args.results)
    steps = [("Table 1", lambda: table1(recs, args.out)),
             ("Table 2", lambda: table2(recs, args.out)),
             ("seed-variance comparison", lambda: seed_variance(recs)),
             ("Table 3", lambda: table3(recs, args.out)),
             ("Table 4", lambda: table4(recs, args.out)),
             ("gradient norms", lambda: grad_norms(recs)),
             ("Table 5", lambda: table5(args.results, args.out)),
             ("Section 5.5", lambda: section55(args.data)),
             ("Table 6", lambda: table6(recs, args.out)),
             ("Table 7", lambda: table7(recs, args.out)),
             ("Table 8", lambda: table8(recs, args.out)),
             ("Table 9", lambda: table9(recs, args.out))]
    for name, step in steps:
        try:
            step()
        except Missing as e:
            print(f"skipped {name}: no runs in {args.results} for {e}")
    print("=" * 78)
    print(f"Wrote LaTeX tables to {args.out}/")


if __name__ == "__main__":
    main()
