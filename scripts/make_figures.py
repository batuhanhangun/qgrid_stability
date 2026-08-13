#!/usr/bin/env python
"""Regenerate all figures in the paper from results/*/result.json.

Usage:
    python scripts/make_figures.py [--results results] [--out paper_assets/figures]

Produces: ablation.pdf, input_noise.pdf, convergence.pdf, quantum_noise.pdf,
matching Figures 2 to 5 of the paper (Figure 1, the architecture diagram, is
generated from its own TikZ source in the paper repository).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = {"hybrid": "#0072B2", "classical_nn": "#D55E00", "svm": "#009E73",
     "xgboost": "#CC79A7", "random_forest": "#E69F00",
     "gradient_boosting": "#56B4E9"}
NAME = {"hybrid": "Hybrid (ours)", "classical_nn": "Classical NN",
        "svm": "SVM (RBF)", "xgboost": "XGBoost",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting"}
LEGEND_ORDER = list(C.keys())


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


def group_acc(recs, **filt):
    out = {}
    for r in recs:
        k = key(r)
        if all(k[a] == v for a, v in filt.items()):
            out[k["seed"]] = r["clean"]["accuracy"]
    return out


def fig_ablation(recs, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
    for ax, emb in zip(axes, ["amplitude", "angle"]):
        for L, mk in zip([1, 2, 3], ["o", "s", "^"]):
            mus, sds, qs = [], [], [2, 3, 4]
            for q in qs:
                vals = list(group_acc(recs, exp="ablation_arch", emb=emb,
                                      qubits=q, layers=L).values())
                mus.append(np.mean(vals) * 100)
                sds.append(np.std(vals, ddof=1) * 100)
            ax.errorbar(qs, mus, yerr=sds, marker=mk, ms=4, capsize=2,
                        lw=1.2, label=f"$L={L}$")
        ax.axhline(63.8, color="gray", ls=":", lw=1)
        ax.set_title(f"{emb.capitalize()} embedding", fontsize=9)
        ax.set_xlabel("Number of qubits")
        ax.set_xticks([2, 3, 4])
    axes[0].set_ylabel("Test accuracy (%)")
    axes[1].text(4.0, 64.6, "majority class (63.8%)", fontsize=7,
                 color="gray", ha="right", va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(os.path.join(out, "ablation.pdf"))
    plt.close()


def fig_input_noise(recs, out):
    STYLE = {"hybrid": dict(ls="-", lw=1.6, zorder=10, marker="o"),
             "classical_nn": dict(ls="--", lw=1.2, zorder=9, marker="s"),
             "svm": dict(ls="-", lw=1.2, zorder=5, marker="^"),
             "xgboost": dict(ls="-", lw=1.2, zorder=5, marker="D"),
             "random_forest": dict(ls="-", lw=1.2, zorder=4, marker="v"),
             "gradient_boosting": dict(ls="--", lw=1.2, zorder=3, marker="x")}
    draw = ["gradient_boosting", "random_forest", "xgboost", "svm",
            "classical_nn", "hybrid"]
    panels = [("gaussian", r"Gaussian $\sigma$"),
              ("uniform", r"Uniform half-width $w$"),
              ("dropout", r"Dropout $p$")]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)
    for ax, (ntype, xlab) in zip(axes, panels):
        for mdl in draw:
            per_level = {}
            for r in recs:
                k = key(r)
                if k["exp"] != "main" or k["model"] != mdl:
                    continue
                for lvl, met in r["input_noise"][ntype].items():
                    per_level.setdefault(float(lvl), []).append(
                        met["accuracy"])
            lv = sorted(per_level)
            mu = np.array([np.mean(per_level[x]) for x in lv]) * 100
            sd = np.array([np.std(per_level[x], ddof=1) for x in lv]) * 100
            ax.plot(lv, mu, ms=3.5, color=C[mdl], label=NAME[mdl],
                    markerfacecolor="none" if STYLE[mdl]["ls"] == "--"
                    else C[mdl], **STYLE[mdl])
            ax.fill_between(lv, mu - sd, mu + sd, color=C[mdl], alpha=0.10,
                            lw=0, zorder=STYLE[mdl]["zorder"] - 1)
        ax.set_xlabel(xlab)
        ax.set_title(ntype.capitalize(), fontsize=9)
    axes[0].set_ylabel("Test accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    hl = dict(zip(l, h))
    fig.legend([hl[NAME[k]] for k in LEGEND_ORDER],
               [NAME[k] for k in LEGEND_ORDER], frameon=False, fontsize=7.5,
               ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               columnspacing=1.0, handlelength=1.8)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(os.path.join(out, "input_noise.pdf"))
    plt.close()


def fig_convergence(recs, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), sharex=True)
    for mdl in ["hybrid", "classical_nn"]:
        hs = [r for r in recs
              if key(r)["exp"] == "main" and key(r)["model"] == mdl
              and r.get("history")]
        tr = np.array([r["history"]["train_loss"] for r in hs])
        va = np.array([r["history"]["val_loss"] for r in hs])
        ep = np.arange(1, tr.shape[1] + 1)
        for ax, arr in zip(axes, [tr, va]):
            mu, sd = arr.mean(0), arr.std(0, ddof=1)
            ax.plot(ep, mu, color=C[mdl], lw=1.3, label=NAME[mdl])
            ax.fill_between(ep, mu - sd, mu + sd, color=C[mdl], alpha=0.15,
                            lw=0)
    axes[0].set_title("Training loss", fontsize=9)
    axes[1].set_title("Test loss", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "convergence.pdf"))
    plt.close()


def fig_quantum_noise(recs, out):
    base = list(group_acc(recs, exp="main", model="hybrid").values())
    labels = ["Noiseless"]
    mus = [np.mean(base) * 100]
    sds = [np.std(base, ddof=1) * 100]
    for t in ["depolarizing", "amplitude_damping"]:
        for p_ in [0.01, 0.05]:
            vals = list(group_acc(recs, exp="qnoise", qnoise=t,
                                  qnoise_p=p_).values())
            labels.append(f"{'Depol.' if t == 'depolarizing' else 'Amp. damp.'}\np={p_}")
            mus.append(np.mean(vals) * 100)
            sds.append(np.std(vals, ddof=1) * 100)
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    xs = np.arange(len(labels))
    ax.bar(xs, mus, yerr=sds, capsize=3, width=0.6,
           color=["#444444"] + ["#0072B2"] * 4, alpha=0.9)
    ax.set_ylim(94.5, 97)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Test accuracy (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "quantum_noise.pdf"))
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper_assets/figures")
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150,
                         "savefig.bbox": "tight"})
    os.makedirs(args.out, exist_ok=True)
    recs = load(args.results)
    fig_ablation(recs, args.out)
    fig_input_noise(recs, args.out)
    fig_convergence(recs, args.out)
    fig_quantum_noise(recs, args.out)
    print(f"Wrote 4 figures to {args.out}/")


if __name__ == "__main__":
    main()
