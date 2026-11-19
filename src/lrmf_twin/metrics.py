"""Metrics and tests of design section 5.3.

* AUC pooled over cells and macro-AUC over questions (mean of per-question AUCs), log-loss, Brier, accuracy,
  Brier decomposition into reliability / resolution / uncertainty (Murphy 1973, 10 bins).
* Paired cluster bootstrap by person for ΔAUC (pooled and macro) and Δlog-loss: each replicate draws persons
  with replacement and re-weights their cells; the same weights are used for both models (paired).
  Weighted AUC uses tie groups (ties count 1/2), so constant predictions get AUC 0.5.
* Likelihood-ratio test of beta in logit p = eta (offset) + beta * D~ (the primary test of the design;
  DeLong is not used for nested models).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats

EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)


def logloss_cells(y, p):
    p = _clip(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


class _AUCGroups:
    """Precomputed tie groups of one score vector, sorted by (item, score)."""

    def __init__(self, scores: np.ndarray, items: np.ndarray | None):
        items = np.zeros(scores.size, dtype=np.int64) if items is None else np.asarray(items, dtype=np.int64)
        order = np.lexsort((scores, items))
        s, it = scores[order], items[order]
        new = np.ones(s.size, dtype=bool)
        new[1:] = (s[1:] != s[:-1]) | (it[1:] != it[:-1])
        gid_sorted = np.cumsum(new) - 1
        self.gid = np.empty(s.size, dtype=np.int64)
        self.gid[order] = gid_sorted
        self.n_groups = int(gid_sorted[-1]) + 1 if s.size else 0
        self.g_item = it[new]
        self.n_items = int(items.max()) + 1 if items.size else 0

    def auc(self, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Per-item weighted AUC (NaN where an item lacks a class)."""
        pos = np.bincount(self.gid, weights=w * y, minlength=self.n_groups)
        neg = np.bincount(self.gid, weights=w * (1 - y), minlength=self.n_groups)
        cum = np.cumsum(neg)
        item_tot_neg = np.bincount(self.g_item, weights=neg, minlength=self.n_items)
        item_start = np.concatenate([[0.0], np.cumsum(item_tot_neg)[:-1]])
        below = cum - neg - item_start[self.g_item]
        num = np.bincount(self.g_item, weights=pos * (below + 0.5 * neg), minlength=self.n_items)
        wpos = np.bincount(self.g_item, weights=pos, minlength=self.n_items)
        den = wpos * item_tot_neg
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)


def auc(y, p, w=None) -> float:
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if w is None else w
    return float(_AUCGroups(np.asarray(p, dtype=np.float64), None).auc(y, w)[0])


def macro_auc(y, p, items, w=None) -> float:
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if w is None else w
    _, inv = np.unique(items, return_inverse=True)
    per = _AUCGroups(np.asarray(p, dtype=np.float64), inv).auc(y, w)
    return float(np.nanmean(per)) if np.isfinite(per).any() else float("nan")


def brier_decomposition(y, p, bins: int = 10) -> dict:
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    n = np.bincount(b, minlength=bins)
    py = np.bincount(b, weights=y, minlength=bins)
    pp = np.bincount(b, weights=p, minlength=bins)
    ok = n > 0
    ybar = y.mean()
    rel = float(np.sum((pp[ok] / n[ok] - py[ok] / n[ok]) ** 2 * n[ok]) / y.size)
    res = float(np.sum((py[ok] / n[ok] - ybar) ** 2 * n[ok]) / y.size)
    return {"reliability": rel, "resolution": res, "uncertainty": float(ybar * (1 - ybar))}


def summary(y, p, items=None) -> dict:
    y, p = np.asarray(y, dtype=np.float64), _clip(p)
    out = {"n": int(y.size), "auc": auc(y, p), "logloss": float(logloss_cells(y, p).mean()),
           "brier": float(np.mean((p - y) ** 2)), "acc": float(np.mean((p > 0.5) == (y > 0.5)))}
    out["auc_item"] = macro_auc(y, p, items) if items is not None else float("nan")
    out.update(brier_decomposition(y, p))
    return out


def paired_bootstrap(y, p_model, p_ref, persons, items=None, B: int = 300, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """CI of model - ref for pooled AUC, macro AUC (if items) and log-loss, resampling persons."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float64)
    _, pid = np.unique(persons, return_inverse=True)
    n_p = int(pid.max()) + 1
    gm, gr = _AUCGroups(np.asarray(p_model, dtype=np.float64), None), _AUCGroups(np.asarray(p_ref, dtype=np.float64), None)
    if items is not None:
        _, iid = np.unique(items, return_inverse=True)
        im, ir = _AUCGroups(np.asarray(p_model, dtype=np.float64), iid), _AUCGroups(np.asarray(p_ref, dtype=np.float64), iid)
    dll = logloss_cells(y, p_model) - logloss_cells(y, p_ref)
    rows = []
    for _ in range(B):
        w = rng.multinomial(n_p, np.full(n_p, 1.0 / n_p)).astype(np.float64)[pid]
        r = [gm.auc(y, w)[0] - gr.auc(y, w)[0], float(np.sum(w * dll) / np.sum(w))]
        if items is not None:
            r.append(float(np.nanmean(im.auc(y, w)) - np.nanmean(ir.auc(y, w))))
        rows.append(r)
    arr = np.asarray(rows)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    ones = np.ones_like(y)
    out = {"d_auc": gm.auc(y, ones)[0] - gr.auc(y, ones)[0],
           "d_auc_ci": [float(np.nanpercentile(arr[:, 0], lo)), float(np.nanpercentile(arr[:, 0], hi))],
           "d_logloss": float(dll.mean()),
           "d_logloss_ci": [float(np.percentile(arr[:, 1], lo)), float(np.percentile(arr[:, 1], hi))]}
    if items is not None:
        out["d_auc_item"] = float(np.nanmean(im.auc(y, ones)) - np.nanmean(ir.auc(y, ones)))
        out["d_auc_item_ci"] = [float(np.nanpercentile(arr[:, 2], lo)), float(np.nanpercentile(arr[:, 2], hi))]
    return out


def _loglik_offset(beta: float, y, eta, x) -> float:
    z = eta + beta * x
    return float(np.sum(y * z - np.logaddexp(0.0, z)))


def lr_test(y, eta, x) -> dict:
    """LR test of beta in logit p = eta + beta * x (eta is an offset)."""
    y, eta, x = (np.asarray(a, dtype=np.float64) for a in (y, eta, x))
    if y.size < 3 or np.allclose(x, 0):
        return {"beta": 0.0, "chi2": 0.0, "p_value": 1.0, "n": int(y.size)}
    res = optimize.minimize_scalar(lambda b: -_loglik_offset(b, y, eta, x), bounds=(-20, 20), method="bounded")
    ll1, ll0 = -res.fun, _loglik_offset(0.0, y, eta, x)
    chi2 = max(2.0 * (ll1 - ll0), 0.0)
    return {"beta": float(res.x), "chi2": float(chi2), "p_value": float(stats.chi2.sf(chi2, 1)), "n": int(y.size)}


def corr_ci(a, b, alpha: float = 0.05) -> dict:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = a.size
    if n < 4 or a.std() < 1e-12 or b.std() < 1e-12:
        return {"r": float("nan"), "ci": [float("nan"), float("nan")], "n": int(n)}
    r = float(np.corrcoef(a, b)[0, 1])
    z, se = math.atanh(max(min(r, 0.999999), -0.999999)), 1.0 / math.sqrt(n - 3)
    q = stats.norm.ppf(1 - alpha / 2)
    return {"r": r, "ci": [math.tanh(z - q * se), math.tanh(z + q * se)], "n": int(n)}
