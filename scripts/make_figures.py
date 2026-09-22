#!/usr/bin/env python
"""Regenerate the paper's figures from results/*/result.json.

Usage:
    python scripts/make_figures.py [--results results] [--out paper_assets/figures]

Writes convergence.pdf, ablation.pdf, grad_norms.pdf, input_noise.pdf,
noise_matrix.pdf and quantum_noise.pdf. Figure 1 (the architecture schematic)
is not generated from results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
C = dict(zip(["hybrid", "classical_nn", "svm", "xgboost", "random_forest",
              "gradient_boosting"], OKABE_ITO))
NAME = {"hybrid": "Hybrid", "classical_nn": "Classical NN",
        "svm": "SVM (RBF)", "xgboost": "XGBoost",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting"}
CH = {"none": "None", "depolarizing": "Depolarizing",
      "amplitude_damping": "Amplitude damping"}
COLLAPSE = 0.70
MAJORITY = 63.8
PRIOR_ENTROPY = 0.655


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
                enoise=ev["type"])


def sel(recs, **filt):
    out = {}
    for r in recs:
        k = key(r)
        if all(k[a] == v for a, v in filt.items()):
            out[k["seed"]] = r
    return out


class Missing(Exception):
    """Raised when --results lacks an experiment group a figure needs."""


def need(recs, **filt):
    """Like sel(), but raises Missing when no run matches."""
    d = sel(recs, **filt)
    if not d:
        raise Missing(" ".join(f"{a}={v}" for a, v in filt.items()))
    return d


def acc(r, noise=None, level=None):
    cm = (r["clean"] if noise is None
          else r["input_noise"][noise][level])["confusion_matrix"]
    return (cm[0][0] + cm[1][1]) / sum(map(sum, cm))


def accs(d, **kw):
    return np.array([acc(d[s], **kw) for s in sorted(d)]) * 100


# ------------------------------------------------------------ convergence
def fig_convergence(recs, out):
    groups = {m: need(recs, exp="main", model=m) for m in ["hybrid", "classical_nn"]}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), sharex=True)
    for mdl in ["hybrid", "classical_nn"]:
        hs = list(groups[mdl].values())
        tr = np.array([r["history"]["train_loss"] for r in hs])
        va = np.array([r["history"]["val_loss"] for r in hs])
        ep = np.arange(1, tr.shape[1] + 1)
        for ax, arr in zip(axes, [tr, va]):
            mu, sd = arr.mean(0), arr.std(0, ddof=1)
            ax.plot(ep, mu, color=C[mdl], lw=1.3, label=NAME[mdl])
            ax.fill_between(ep, mu - sd, mu + sd, color=C[mdl], alpha=0.15, lw=0)
    axes[0].set_title("Training loss", fontsize=9)
    axes[1].set_title("Test loss", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "convergence.pdf"))
    plt.close()


# ------------------------------------------------------------ ablation
def fig_ablation(recs, out):
    qs, Ls, offs = [2, 3, 4], [1, 2, 3], [-0.22, 0.0, 0.22]
    for emb in ["amplitude", "angle"]:
        for q in qs:
            need(recs, exp="ablation30c", emb=emb, qubits=q, layers=1)
            for L in Ls:
                need(recs, exp="ablation30", emb=emb, qubits=q, layers=L)
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1.25], hspace=0.08,
                                              wspace=0.12))
    rng = np.random.default_rng(0)
    for col, emb in enumerate(["amplitude", "angle"]):
        top, bot = axes[0, col], axes[1, col]
        for ax in (top, bot):
            for q in qs:
                cls = accs(sel(recs, exp="ablation30c", emb=emb, qubits=q, layers=1))
                ax.hlines(cls.mean(), q - 0.34, q + 0.34, color="grey", ls="--",
                          lw=1.1, zorder=1)
            for L, off, colr in zip(Ls, offs, OKABE_ITO):
                for q in qs:
                    v = accs(sel(recs, exp="ablation30", emb=emb, qubits=q, layers=L))
                    x = q + off + rng.uniform(-0.06, 0.06, len(v))
                    ax.scatter(x, v, s=7, color=colr, alpha=0.45, lw=0, zorder=2)
                    ax.errorbar(q + off, v.mean(), yerr=v.std(ddof=1), fmt="o",
                                ms=3.5, color=colr, mec="black", mew=0.4,
                                capsize=2, lw=1, zorder=3)
            ax.axhline(MAJORITY, color="gray", ls=":", lw=1, zorder=1)
        top.set_ylim(94.5, 97.2)
        bot.set_ylim(60.5, 86)
        top.set_title(f"{emb.capitalize()} embedding", fontsize=9)
        top.spines["bottom"].set_visible(False)
        bot.spines["top"].set_visible(False)
        top.tick_params(axis="x", which="both", bottom=False)
        bot.set_xticks(qs)
        bot.set_xlabel("Number of qubits")
        bot.text(4.4, MAJORITY + 0.4, "majority class (63.8%)", fontsize=7,
                 color="gray", ha="right", va="bottom")
        # diagonal break marks
        d = 0.012
        kw = dict(transform=top.transAxes, color="k", clip_on=False, lw=0.8)
        top.plot((-d, +d), (-d, +d), **kw)
        top.plot((1 - d, 1 + d), (-d, +d), **kw)
        kw = dict(transform=bot.transAxes, color="k", clip_on=False, lw=0.8)
        bot.plot((-d, +d), (1 - d, 1 + d), **kw)
        bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
    fig.text(0.02, 0.5, "Test accuracy (%)", rotation=90, va="center", ha="center")
    handles = [Line2D([], [], marker="o", ls="", color=c, mec="black", mew=0.4,
                      ms=4, label=f"$L={L}$") for L, c in zip(Ls, OKABE_ITO)]
    handles.append(Line2D([], [], color="grey", ls="--", lw=1.1,
                          label="classical control (mean)"))
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    plt.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.17)
    plt.savefig(os.path.join(out, "ablation.pdf"))
    plt.close()


# ------------------------------------------------------------ gradient norms
def fig_grad_norms(recs, out):
    def runs(emb, q, layers=(1, 2, 3)):
        return [r for L in layers
                for r in need(recs, exp="ablation30", emb=emb, qubits=q, layers=L).values()]

    a2 = runs("angle", 2)
    groups = [("angle q2, collapsed", [r for r in a2 if acc(r) < COLLAPSE]),
              ("angle q2, non-collapsed", [r for r in a2 if acc(r) >= COLLAPSE]),
              ("angle q3", runs("angle", 3)),
              ("amplitude q3", runs("amplitude", 3))]
    for label, rs in groups:
        if not rs:
            raise Missing(f"exp=ablation30 group '{label}'")
    panels = [("grad_norm_quantum", "Quantum-layer gradient norm", True),
              ("grad_norm_encoder", "Encoder gradient norm", True),
              ("train_loss", "Training loss", False)]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
    for ax, (k, title, logy) in zip(axes, panels):
        for (label, rs), colr in zip(groups, OKABE_ITO):
            arr = np.array([r["history"][k] for r in rs])
            ep = np.arange(1, arr.shape[1] + 1)
            ax.plot(ep, arr.mean(0), color=colr, lw=1.3, label=label)
            ax.fill_between(ep, np.percentile(arr, 25, 0), np.percentile(arr, 75, 0),
                            color=colr, alpha=0.18, lw=0)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch")
    axes[2].axhline(PRIOR_ENTROPY, color="gray", ls=":", lw=1)
    axes[2].text(axes[2].get_xlim()[1], PRIOR_ENTROPY + 0.005, "class-prior entropy",
                 fontsize=7, color="gray", ha="right", va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(os.path.join(out, "grad_norms.pdf"))
    plt.close()


# ------------------------------------------------------------ input noise
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
    groups = {m: need(recs, exp="main", model=m) for m in draw}
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)
    for ax, (ntype, xlab) in zip(axes, panels):
        for mdl in draw:
            d = groups[mdl]
            levels = sorted(next(iter(d.values()))["input_noise"][ntype], key=float)
            per = np.array([accs(d, noise=ntype, level=lv) for lv in levels])
            lv = [float(x) for x in levels]
            mu, sd = per.mean(1), per.std(1, ddof=1)
            ax.plot(lv, mu, ms=3.5, color=C[mdl], label=NAME[mdl],
                    markerfacecolor="none" if STYLE[mdl]["ls"] == "--" else C[mdl],
                    **STYLE[mdl])
            ax.fill_between(lv, mu - sd, mu + sd, color=C[mdl], alpha=0.10, lw=0,
                            zorder=STYLE[mdl]["zorder"] - 1)
        ax.set_xlabel(xlab)
        ax.set_title(ntype.capitalize(), fontsize=9)
    axes[0].set_ylabel("Test accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    hl = dict(zip(l, h))
    fig.legend([hl[NAME[k]] for k in C], [NAME[k] for k in C], frameon=False,
               fontsize=7.5, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               columnspacing=1.0, handlelength=1.8)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(os.path.join(out, "input_noise.pdf"))
    plt.close()


# ------------------------------------------------------------ noise matrix
def fig_noise_matrix(recs, out):
    conds = ["none", "depolarizing", "amplitude_damping"]
    offs = dict(zip(conds, [-0.25, 0.0, 0.25]))
    cols = dict(zip(conds, OKABE_ITO[:3]))
    cell = {}
    for tr in conds:
        for ev in conds:
            if tr != "none" and ev == "none":
                continue
            cell[(tr, ev)] = need(recs, exp="noisematrix", qnoise=tr, enoise=ev)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    ref = accs(cell[("none", "none")]).mean()
    ax.axhline(ref, color="gray", ls=":", lw=1, zorder=1)
    for i, tr in enumerate(conds):
        for ev in conds:
            if (tr, ev) not in cell:
                continue
            v = accs(cell[(tr, ev)])
            x = i + offs[ev]
            ax.scatter(x + rng.uniform(-0.05, 0.05, len(v)), v, s=12, color=cols[ev],
                       alpha=0.5, lw=0, zorder=2)
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt="D", ms=4.5,
                        color=cols[ev], mec="black", mew=0.5, capsize=3, lw=1.2,
                        zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels([CH[c] for c in conds])
    ax.set_xlabel("Training channel noise")
    ax.set_ylabel("Test accuracy (%)")
    handles = [Line2D([], [], marker="D", ls="", color=cols[c], mec="black", mew=0.5,
                      ms=4.5, label=f"eval: {CH[c]}") for c in conds]
    handles.append(Line2D([], [], color="gray", ls=":", lw=1,
                          label="none $\\to$ none mean"))
    fig.legend(handles=handles, frameon=False, fontsize=7.5, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.04), columnspacing=1.2)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(os.path.join(out, "noise_matrix.pdf"))
    plt.close()


# ------------------------------------------------------------ quantum noise
def fig_quantum_noise(recs, out):
    base = accs(need(recs, exp="main", model="hybrid"))
    qn = {(t, p_): need(recs, exp="qnoise", qnoise=t, qnoise_p=p_)
          for t in ["depolarizing", "amplitude_damping"] for p_ in [0.01, 0.05]}
    labels, mus, sds = ["Noiseless"], [base.mean()], [base.std(ddof=1)]
    for t in ["depolarizing", "amplitude_damping"]:
        for p_ in [0.01, 0.05]:
            v = accs(qn[(t, p_)])
            labels.append(f"{'Depol.' if t == 'depolarizing' else 'Amp. damp.'}\np={p_}")
            mus.append(v.mean())
            sds.append(v.std(ddof=1))
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    xs = np.arange(len(labels))
    ax.bar(xs, mus, yerr=sds, capsize=3, width=0.6,
           color=["#444444"] + [OKABE_ITO[0]] * 4, alpha=0.9)
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
    figs = [("convergence.pdf", fig_convergence), ("ablation.pdf", fig_ablation),
            ("grad_norms.pdf", fig_grad_norms), ("input_noise.pdf", fig_input_noise),
            ("noise_matrix.pdf", fig_noise_matrix),
            ("quantum_noise.pdf", fig_quantum_noise)]
    n = 0
    for name, fn in figs:
        try:
            fn(recs, args.out)
            n += 1
        except Missing as e:
            plt.close("all")
            print(f"skipped {name}: no runs in {args.results} for {e}")
    print(f"Wrote {n} figures to {args.out}/")


if __name__ == "__main__":
    main()
