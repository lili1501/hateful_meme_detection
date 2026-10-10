"""Binary-classification metrics. AUROC is the rank-statistic (Mann-Whitney U) form with tie handling."""
from __future__ import annotations

import numpy as np


def _rank(x: np.ndarray) -> np.ndarray:
    """1-based ranks, ties get the average rank."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    n_pos = int((y == 1).sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both classes present")
    r = _rank(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def accuracy(y: np.ndarray, s: np.ndarray, threshold: float = 0.5) -> float:
    return float(((np.asarray(s) >= threshold).astype(int) == np.asarray(y)).mean())


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Accuracy-maximising threshold on (y, s); among ties the one closest to 0.5. Fit on VALIDATION data only."""
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    cands = np.unique(np.concatenate([s, [0.5]]))
    mids = np.concatenate([[cands[0] - 1e-9], (cands[:-1] + cands[1:]) / 2, [cands[-1] + 1e-9]])
    accs = np.array([accuracy(y, s, t) for t in mids])
    best = np.flatnonzero(accs == accs.max())
    return float(mids[best[np.argmin(np.abs(mids[best] - 0.5))]])


def confusion(y: np.ndarray, s: np.ndarray, threshold: float = 0.5) -> dict:
    p = (np.asarray(s) >= threshold).astype(int); y = np.asarray(y)
    return {"tp": int(((p == 1) & (y == 1)).sum()), "fp": int(((p == 1) & (y == 0)).sum()),
            "tn": int(((p == 0) & (y == 0)).sum()), "fn": int(((p == 0) & (y == 1)).sum())}


def bootstrap_ci(y, s, fn=auroc, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], s[idx]))
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def summarize(y, s, threshold: float = 0.5) -> dict:
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    lo, hi = bootstrap_ci(y, s)
    return {"n": int(len(y)), "pos_rate": float(y.mean()), "auroc": auroc(y, s), "auroc_ci95": [lo, hi],
            "threshold": float(threshold), "accuracy": accuracy(y, s, threshold),
            "confusion": confusion(y, s, threshold)}
