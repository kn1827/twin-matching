"""Evaluation of a0 / m0 (tech spec 9): pooled AUC, within-chooser AUC, Brier, ECE, log loss, match AUC,
and a paired cluster bootstrap (clusters = choosers, B = 2000) with a TOST equivalence test on the AUC
difference against the smallest effect size of interest (SESOI, 0.03).

The bootstrap draws multinomial cluster weights, so every metric of every method is evaluated on the
same resamples (paired). AUC under weights is computed exactly from tie groups of the scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score


def auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece_edges(p, bins: int = 10) -> np.ndarray:
    q = np.quantile(np.asarray(p, float), np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    return q


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-mass bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.clip(np.searchsorted(ece_edges(p, bins), p, side="right") - 1, 0, bins - 1)
    tot = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(tot / len(p))


def within_auc_values(y, p, group) -> tuple[np.ndarray, np.ndarray]:
    """Per-group AUC (nan where a group has one class only) and the group labels."""
    df = pd.DataFrame({"y": y, "p": p, "g": group})
    gs, vals = [], []
    for g, d in df.groupby("g", sort=True):
        gs.append(g)
        vals.append(auc(d["y"].values, d["p"].values) if d["y"].nunique() == 2 else np.nan)
    return np.array(vals), np.array(gs)


def dyads(pred: pd.DataFrame, col: str, female: np.ndarray | None = None) -> pd.DataFrame:
    """Unordered pairs with both directions present: p_ij, p_ji, match label and a cluster id.

    `pred` has columns chooser, chosen, match and the directed probability `col`. The dyad's cluster is
    its female participant when genders are known (each dyad has exactly one), else the smaller id.
    """
    key = {(u, v): k for k, (u, v) in enumerate(zip(pred["chooser"], pred["chosen"]))}
    rows = []
    for k, (u, v) in enumerate(zip(pred["chooser"], pred["chosen"])):
        r = key.get((v, u))
        if r is None or u > v:
            continue
        c = u if female is None else (u if female[u] else v)
        rows.append((u, v, pred[col].iat[k], pred[col].iat[r], int(pred["match"].iat[k]), c))
    return pd.DataFrame(rows, columns=["i", "j", "p_ij", "p_ji", "match", "cluster"])


# -------------------------------------------------------------------------------- weighted metrics
class _WAUC:
    """Exact weighted AUC for many weight vectors at once: W (R, n) -> (R,)."""

    def __init__(self, scores: np.ndarray, y: np.ndarray):
        u, inv = np.unique(np.asarray(scores, float), return_inverse=True)
        n, G = len(scores), len(u)
        self.Ip = sp.csr_matrix((np.asarray(y, float), (np.arange(n), inv)), shape=(n, G))
        self.In = sp.csr_matrix((1.0 - np.asarray(y, float), (np.arange(n), inv)), shape=(n, G))

    def __call__(self, W: np.ndarray) -> np.ndarray:
        wp = np.asarray((self.Ip.T @ W.T).T)              # (R, G) positive weight per tie group
        wn = np.asarray((self.In.T @ W.T).T)
        below = np.cumsum(wn, axis=1) - wn
        num = (wp * (below + 0.5 * wn)).sum(1)
        den = wp.sum(1) * wn.sum(1)
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _wmean(W: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (W @ v) / W.sum(1)


def _wece(W: np.ndarray, y: np.ndarray, p: np.ndarray, bins: int) -> np.ndarray:
    b = np.clip(np.searchsorted(ece_edges(p, bins), p, side="right") - 1, 0, bins - 1)
    B = sp.csr_matrix((np.ones(len(p)), (np.arange(len(p)), b)), shape=(len(p), bins))
    dy = np.asarray((B.T @ (W * (y - p)).T).T)
    return np.abs(dy).sum(1) / W.sum(1)


@dataclass
class EvalSet:
    """Everything needed to score directed and mutual predictions of several methods on the same rows."""
    y: np.ndarray
    chooser: np.ndarray
    probs: dict                        # method -> directed probability (n,)
    match_scores: dict                 # method -> (dyad score (d,)) aligned with dyad_frame
    dyad_frame: pd.DataFrame           # i, j, match, cluster
    ece_bins: int = 10


def point_metrics(es: EvalSet, method: str) -> dict:
    y, p = es.y, es.probs[method]
    wa, _ = within_auc_values(y, p, es.chooser)
    out = {"auc": auc(y, p), "within_auc": float(np.nanmean(wa)), "brier": brier(y, p), "log_loss": log_loss(y, p),
           "ece": ece(y, p, es.ece_bins)}
    if method in es.match_scores:
        out["match_auc"] = auc(es.dyad_frame["match"].values, es.match_scores[method])
    return out


def bootstrap(es: EvalSet, reference: str, B: int = 2000, seed: int = 0, chunk: int = 250,
              sesoi: float = 0.03) -> pd.DataFrame:
    """Paired cluster bootstrap over choosers. For each method: point estimate and 95% CI of every metric,
    and Delta vs `reference` (method - reference) with 95% CI and a TOST verdict on Delta AUC (90% CI
    inside +-sesoi)."""
    clusters, cidx = np.unique(es.chooser, return_inverse=True)
    C = len(clusters)
    dyc = np.searchsorted(clusters, es.dyad_frame["cluster"].values)
    valid_d = (dyc < C) & (clusters[np.minimum(dyc, C - 1)] == es.dyad_frame["cluster"].values)
    rng = np.random.default_rng(seed)
    methods = list(es.probs)
    wauc = {m: _WAUC(es.probs[m], es.y) for m in methods}
    mauc = {m: _WAUC(es.match_scores[m][valid_d], es.dyad_frame["match"].values[valid_d])
            for m in methods if m in es.match_scores}
    wa_vals = {}
    for m in methods:
        v, g = within_auc_values(es.y, es.probs[m], es.chooser)
        wa_vals[m] = (np.searchsorted(clusters, g), v)
    ll = {m: -(es.y * np.log(np.clip(es.probs[m], 1e-12, 1)) + (1 - es.y) * np.log(np.clip(1 - es.probs[m], 1e-12, 1)))
          for m in methods}
    br = {m: (es.probs[m] - es.y) ** 2 for m in methods}
    stats = {m: {k: [] for k in ("auc", "within_auc", "brier", "log_loss", "ece", "match_auc")} for m in methods}
    done = 0
    while done < B:
        R = min(chunk, B - done)
        counts = rng.multinomial(C, np.full(C, 1.0 / C), size=R).astype(float)   # (R, C)
        W = counts[:, cidx]
        Wd = counts[:, dyc[valid_d]]
        for m in methods:
            s = stats[m]
            s["auc"].append(wauc[m](W))
            s["brier"].append(_wmean(W, br[m]))
            s["log_loss"].append(_wmean(W, ll[m]))
            s["ece"].append(_wece(W, es.y, es.probs[m], es.ece_bins))
            gi, v = wa_vals[m]
            ok = ~np.isnan(v)
            cw = counts[:, gi[ok]]
            s["within_auc"].append(cw @ v[ok] / cw.sum(1))
            if m in mauc:
                s["match_auc"].append(mauc[m](Wd))
        done += R
    rows = []
    ref = {k: np.concatenate(v) for k, v in stats[reference].items() if v}
    ref_pm = point_metrics(es, reference)
    for m in methods:
        pm = point_metrics(es, m)
        row = {"method": m}
        for k, v in stats[m].items():
            if not v:
                continue
            arr = np.concatenate(v)
            row[k] = pm.get(k, np.nan)
            row[f"{k}_lo"], row[f"{k}_hi"] = np.nanpercentile(arr, [2.5, 97.5])
            if m != reference and k in ref:
                d = arr - ref[k]
                row[f"d_{k}"] = pm.get(k, np.nan) - ref_pm.get(k, np.nan)
                row[f"d_{k}_lo"], row[f"d_{k}_hi"] = np.nanpercentile(d, [2.5, 97.5])
                if k == "auc":
                    lo90, hi90 = np.nanpercentile(d, [5, 95])
                    row["tost_equivalent"] = bool(lo90 > -sesoi and hi90 < sesoi)
                    row["p_two_sided"] = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))
        rows.append(row)
    return pd.DataFrame(rows)
