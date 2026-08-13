"""Across-seed statistics: paired tests and effect sizes.

Designed for the 30-seed protocol. All pairwise comparisons are paired by
seed (same seed => same train/test split), so Wilcoxon signed-rank is the
appropriate test. Effect sizes reported: rank-biserial correlation (natural
companion to Wilcoxon) and paired Cohen's d for interpretability.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "median": float(np.median(a)),
    }


def paired_comparison(a: list[float], b: list[float]) -> dict:
    """Compare model A vs model B on a per-seed paired metric (e.g. accuracy).

    Returns Wilcoxon signed-rank p-value, rank-biserial effect size, and
    paired Cohen's d. Positive effect sizes favor A.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired comparison requires equal-length arrays "
                         "(same seeds for both models).")
    diff = a - b

    if np.allclose(diff, 0):
        return {"wilcoxon_p": 1.0, "rank_biserial": 0.0, "cohens_d": 0.0,
                "mean_diff": 0.0, "n_pairs": int(a.size)}

    w_stat, p = stats.wilcoxon(a, b)
    # Rank-biserial correlation from the Wilcoxon statistic.
    nz = diff[diff != 0]
    n = nz.size
    total = n * (n + 1) / 2
    rank_biserial = float((2 * w_stat / total) - 1) * (-1 if np.mean(nz) < 0 else 1)
    # Sign convention: make positive favor A explicitly.
    rank_biserial = abs(rank_biserial) * (1 if diff.mean() > 0 else -1)

    d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0

    return {
        "wilcoxon_p": float(p),
        "rank_biserial": rank_biserial,
        "cohens_d": d,
        "mean_diff": float(diff.mean()),
        "n_pairs": int(a.size),
    }
