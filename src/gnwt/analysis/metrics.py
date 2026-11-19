"""Metrics of stage1.md section 9: pooled AUC, within-chooser AUC, Brier, ECE, log-loss, match AUC,
chooser-cluster bootstrap of differences against a0, and TOST equivalence on +-SESOI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_auc(y: np.ndarray, s: np.ndarray, group: np.ndarray) -> float:
    vals = []
    for g in np.unique(group):
        m = group == g
        a = auc(y[m], s[m])
        if np.isfinite(a):
            vals.append(a)
    return float(np.mean(vals)) if vals else float("nan")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def accuracy(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> float:
    return float(np.mean((np.asarray(p) >= thr) == (np.asarray(y) == 1)))


@dataclass(frozen=True)
class Scored:
    """Directed predictions of one branch, rows aligned with the evaluation frame."""

    name: str
    p: np.ndarray                 # P(y_ij = 1)
    match_score: np.ndarray       # per directed row: score of the dyad for the match AUC


def summarize(y: np.ndarray, chooser: np.ndarray, first: np.ndarray, match: np.ndarray, sc: Scored,
              bins: int = 10) -> dict[str, float]:
    return {"auc": auc(y, sc.p), "auc_within": within_auc(y, sc.p, chooser), "brier": brier(y, sc.p),
            "ece": ece(y, sc.p, bins), "log_loss": log_loss(y, sc.p), "accuracy": accuracy(y, sc.p),
            "match_auc": auc(match[first], sc.match_score[first])}


def cluster_bootstrap(y: np.ndarray, chooser: np.ndarray, first: np.ndarray, match: np.ndarray,
                      branches: list[Scored], ref: Scored, B: int, seed: int) -> dict[str, np.ndarray]:
    """Bootstrap over choosers of AUC(branch) - AUC(ref) and match-AUC differences.

    Match AUC is computed on the dyads whose first row's chooser is resampled."""
    rng = np.random.default_rng(seed)
    groups = np.unique(chooser)
    rows_of = {g: np.where(chooser == g)[0] for g in groups}
    first_mask = np.zeros(len(y), bool)
    first_mask[first] = True
    out = {b.name: np.full((B, 2), np.nan) for b in branches}
    for k in range(B):
        pick = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([rows_of[g] for g in pick])
        fidx = idx[first_mask[idx]]
        a_ref = auc(y[idx], ref.p[idx])
        m_ref = auc(match[fidx], ref.match_score[fidx])
        for b in branches:
            out[b.name][k, 0] = auc(y[idx], b.p[idx]) - a_ref
            out[b.name][k, 1] = auc(match[fidx], b.match_score[fidx]) - m_ref
    return out


def tost(diff: np.ndarray, sesoi: float, alpha: float = 0.05) -> dict[str, float | bool]:
    """Equivalence on +-sesoi: (1 - 2 alpha) percentile CI inside the margin."""
    d = diff[np.isfinite(diff)]
    lo, hi = np.quantile(d, [alpha, 1 - alpha])
    lo95, hi95 = np.quantile(d, [0.025, 0.975])
    return {"mean": float(d.mean()), "ci95_lo": float(lo95), "ci95_hi": float(hi95), "ci90_lo": float(lo),
            "ci90_hi": float(hi), "equivalent": bool(lo > -sesoi and hi < sesoi),
            "better": bool(lo95 > 0), "worse": bool(hi95 < 0)}
