"""Offline estimation of the context reader (tech spec 7, steps O2 and O5). Never runs at inference.

* `estimate_catalog`: catalog profiles D^def (mean, SE, ICC(2,k)) from >= 3 raters per context.
* `estimate_reader`: from the labelled context sentences (table `demand_labels`), alternating
    1. k   - Buhlmann non-parametric prior strength: k = sigma^2 / tau^2, sigma^2 = mean n_s (ybar_s - mu_s)^2
             (variance of the keyword evidence around the true level of the sentence, per hit) and
             tau^2 = mean (mu_s - D^def_s)^2 - rater noise (variance of true levels around the default);
    2. theta - least squares of (11b) against the rater means with theta_neg <= theta_weak <= theta_base
             <= theta_strong (theta_rev free), solved exactly by bounded least squares on non-negative
             increments (the PAV solution when every sentence carries one cue class);
  until theta stops changing. Shared (one theta, one k) is the default; per-style versions are kept
  only if grouped CV MAE drops by more than 1 SE. Styles with ICC(2,k) < 0.70 are dropped from S_d.
* `evaluate_reader`: section 9 measures on held-out labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import pearsonr, spearmanr

from stage1.context.catalog import Catalog
from stage1.context.demand import CLS_INDEX, ReaderArtifacts
from stage1.context.lexicon import Lexicon
from stage1.context.styles import CUE_CLASSES, STYLES
from stage1.context.text import split_clauses

LABEL_COLUMNS = ("text_id", "text", "lang", "context_id", "kind", "rater_id", "style", "rating")


def icc2k(M: np.ndarray) -> float:
    """ICC(2,k) of Shrout & Fleiss (two-way random, mean of k raters) for a complete targets x raters matrix."""
    M = np.asarray(M, float)
    M = M[~np.isnan(M).any(axis=1)]
    n, k = M.shape
    if n < 2 or k < 2:
        return np.nan
    gm = M.mean()
    msr = k * np.sum((M.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((M.mean(0) - gm) ** 2) / (k - 1)
    sse = np.sum((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else np.nan


def estimate_catalog(ratings: pd.DataFrame) -> dict:
    """ratings: context_id, rater_id, style, rating (-3..3). Returns {context: {"profile": ..., "n_raters"}}."""
    out = {}
    icc = {}
    for s in STYLES:
        piv = ratings[ratings["style"] == s].pivot_table(index="context_id", columns="rater_id", values="rating")
        icc[s] = icc2k(piv.values) if len(piv) else np.nan
    for cid, g in ratings.groupby("context_id"):
        prof = {}
        for s in STYLES:
            v = g.loc[g["style"] == s, "rating"].astype(float).values
            if len(v) < 3:
                raise ValueError(f"context {cid}, style {s}: fewer than 3 raters")
            prof[s] = {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))),
                       "icc2k": None if np.isnan(icc[s]) else float(icc[s])}
        out[str(cid)] = {"profile": prof, "n_raters": int(g["rater_id"].nunique())}
    return out


# ---------------------------------------------------------------------------------------- reader
class _Prepared:
    """Per-text hits, priors and rater means."""

    def __init__(self, labels: pd.DataFrame, lex: Lexicon, cat: Catalog):
        texts = labels.drop_duplicates("text_id").set_index("text_id")
        self.ids = list(texts.index)
        self.group = texts["context_id"].fillna("other").astype(str).values
        self.kind = texts["kind"].astype(str).values if "kind" in texts else np.array(["cue"] * len(texts))
        self.text = texts["text"].astype(str).values
        piv = labels.pivot_table(index="text_id", columns="style", values="rating", aggfunc="mean").reindex(self.ids)
        self.mu = piv.reindex(columns=list(STYLES)).values                       # (T, 12) rater means
        nr = labels.groupby(["text_id", "style"])["rating"].agg(["var", "count"]).reset_index()
        noise = (nr["var"].fillna(0) / nr["count"]).groupby(nr["style"]).mean()
        self.rater_noise = noise.reindex(list(STYLES)).fillna(0).values          # var of a rater mean
        self.counts = np.zeros((len(self.ids), len(STYLES), len(CUE_CLASSES)))
        self.prior = np.zeros((len(self.ids), len(STYLES)))
        self.c_star = []
        self.hits = []
        for t, txt in enumerate(self.text):
            cl = split_clauses(txt, lex.punct, lex.split_words, lex.keep_bigrams)
            m = cat.match(cl)
            self.c_star.append(m.c_star)
            self.prior[t] = cat.prior(m)
            hits = lex.read(cl)
            self.hits.append(hits)
            for h in hits:
                self.counts[t, STYLES.index(h.style), CLS_INDEX[h.cls]] += 1
        self.n = self.counts.sum(-1)
        icc = {}
        for s in STYLES:
            p2 = labels[labels["style"] == s].pivot_table(index="text_id", columns="rater_id", values="rating")
            icc[s] = icc2k(p2.values)
        self.icc = icc


def _predict(P: _Prepared, theta: np.ndarray, k: np.ndarray, rows=slice(None)) -> np.ndarray:
    n = P.n[rows]
    ybar = np.divide(np.einsum("tsc,sc->ts", P.counts[rows], theta), n, out=np.zeros_like(n), where=n > 0)
    Z = n / (n + k[None, :])
    return np.where(n > 0, Z * ybar + (1 - Z) * P.prior[rows], P.prior[rows])


def _fit_k(P: _Prepared, theta: np.ndarray, rows, per_style: bool) -> np.ndarray:
    n = P.n[rows]
    ybar = np.divide(np.einsum("tsc,sc->ts", P.counts[rows], theta), n, out=np.zeros_like(n), where=n > 0)
    mu = P.mu[rows]
    has = (n > 0) & ~np.isnan(mu)
    obs = ~np.isnan(mu)
    sig = n * (ybar - np.nan_to_num(mu)) ** 2
    tau = (np.nan_to_num(mu) - P.prior[rows]) ** 2

    def ratio(mask_h, mask_o, noise):
        s2 = sig[mask_h].mean() if mask_h.any() else 1.0
        t2 = max(tau[mask_o].mean() - noise if mask_o.any() else 1.0, 1e-3)
        return float(np.clip(s2 / t2, 1e-3, 1e3))

    if not per_style:
        return np.full(len(STYLES), ratio(has, obs, float(np.mean(P.rater_noise))))
    return np.array([ratio(has[:, s], obs[:, s], P.rater_noise[s]) for s in range(len(STYLES))])


_ORDER = [CLS_INDEX[c] for c in ("neg", "weak", "base", "strong")]
_REV = CLS_INDEX["rev"]


def _fit_theta(P: _Prepared, k: np.ndarray, rows, per_style: bool) -> np.ndarray:
    """Monotone least squares of (11b) on the rater means; parameters (t0, d1, d2, d3, t_rev)."""
    n = P.n[rows]
    mu = P.mu[rows]
    Z = n / (n + k[None, :])
    use = (n > 0) & ~np.isnan(mu)
    # theta_class = M @ params ; M maps (t0, d1, d2, d3, trev) -> (neg, rev, weak, base, strong)
    M = np.zeros((5, 5))
    M[_ORDER[0], 0] = 1
    M[_ORDER[1], [0, 1]] = 1
    M[_ORDER[2], [0, 1, 2]] = 1
    M[_ORDER[3], [0, 1, 2, 3]] = 1
    M[_REV, 4] = 1
    lb, ub = np.array([-3, 0, 0, 0, -3.0]), np.array([3, 6, 6, 6, 3.0])

    def solve(mask_ts):
        tt, ss = np.nonzero(mask_ts)
        if len(tt) == 0:
            return np.array([-1.0, -1.0, 1.0, 2.0, 2.5])          # no data: neutral start values
        coef = (Z[tt, ss] / n[tt, ss])[:, None] * P.counts[rows][tt, ss]          # (m, 5 classes)
        A = coef @ M
        b = mu[tt, ss] - (1 - Z[tt, ss]) * P.prior[rows][tt, ss]
        res = lsq_linear(A, b, bounds=(lb, ub))
        return np.clip(M @ res.x, -3, 3)

    if not per_style:
        return np.tile(solve(use), (len(STYLES), 1))
    out = np.zeros((len(STYLES), 5))
    shared = solve(use)
    for s in range(len(STYLES)):
        m = np.zeros_like(use)
        m[:, s] = use[:, s]
        out[s] = solve(m) if m.sum() >= 5 else shared
    return out


def _fit(P: _Prepared, rows, per_style_theta: bool, per_style_k: bool, max_iter: int = 50):
    theta = np.tile(np.array([-1.0, -1.0, 1.0, 2.0, 2.5]), (len(STYLES), 1))
    k = np.ones(len(STYLES))
    for _ in range(max_iter):
        k = _fit_k(P, theta, rows, per_style_k)
        new = _fit_theta(P, k, rows, per_style_theta)
        if np.max(np.abs(new - theta)) < 1e-6:
            theta = new
            break
        theta = new
    return theta, k


def _group_cv_mae(P: _Prepared, per_theta: bool, per_k: bool, folds: int = 5) -> tuple[float, float]:
    groups = np.unique(P.group)
    parts = np.array_split(groups, min(folds, len(groups)))
    errs = []
    for g in parts:
        te = np.isin(P.group, g)
        if not te.any() or te.all():
            continue
        th, k = _fit(P, np.flatnonzero(~te), per_theta, per_k)
        pred = _predict(P, th, k, np.flatnonzero(te))
        mu = P.mu[te]
        errs.append(np.nanmean(np.abs(pred - mu)))
    errs = np.array(errs)
    return float(errs.mean()), float(errs.std(ddof=1) / np.sqrt(len(errs))) if len(errs) > 1 else 0.0


def estimate_reader(labels: pd.DataFrame, lex: Lexicon, cat: Catalog, icc_min: float = 0.70,
                    cv_folds: int = 5) -> ReaderArtifacts:
    """theta (cue levels), k (prior strength) and the ICC filter from labelled sentences."""
    missing = [c for c in ("text_id", "text", "context_id", "rater_id", "style", "rating") if c not in labels]
    if missing:
        raise ValueError(f"labels lack columns {missing}")
    P = _Prepared(labels, lex, cat)
    allrows = np.arange(len(P.ids))
    cv = {}
    for per_theta, per_k in ((False, False), (False, True), (True, False), (True, True)):
        cv[(per_theta, per_k)] = _group_cv_mae(P, per_theta, per_k, cv_folds)
    base_mae, base_se = cv[(False, False)]
    choice = (False, False)
    for key in ((False, True), (True, False), (True, True)):
        if cv[key][0] < base_mae - base_se and cv[key][0] < cv[choice][0]:
            choice = key
    theta, k = _fit(P, allrows, *choice)
    keep = np.array([not (np.isnan(P.icc[s]) or P.icc[s] < icc_min) for s in STYLES])
    meta = {"n_texts": len(P.ids), "n_labels": int(len(labels)), "per_style_theta": choice[0],
            "per_style_k": choice[1], "cv_mae": {f"theta_{'style' if a else 'shared'}_k_{'style' if b else 'shared'}":
                                                 {"mae": m, "se": se} for (a, b), (m, se) in cv.items()},
            "icc2k": {s: (None if np.isnan(v) else float(v)) for s, v in P.icc.items()},
            "dropped_styles": [s for s, kk in zip(STYLES, keep) if not kk]}
    return ReaderArtifacts(lex, cat, theta, k, keep, "estimated", meta)


def evaluate_reader(art: ReaderArtifacts, labels: pd.DataFrame) -> dict:
    """Section 9 measures of the full reader on (held-out) labels."""
    P = _Prepared(labels, art.lexicon, art.catalog)
    pred = _predict(P, art.theta, art.k)
    out = {"mae": float(np.nanmean(np.abs(pred - P.mu))), "per_style": {}}
    for s, name in enumerate(STYLES):
        ok = ~np.isnan(P.mu[:, s])
        if ok.sum() > 2 and np.std(pred[ok, s]) > 0 and np.std(P.mu[ok, s]) > 0:
            out["per_style"][name] = {"pearson": float(pearsonr(pred[ok, s], P.mu[ok, s])[0]),
                                      "spearman": float(spearmanr(pred[ok, s], P.mu[ok, s])[0])}
    cue = P.kind == "cue"
    if cue.any():
        hit_any = np.array([bool(h) for h in P.hits])
        out["coverage"] = float(hit_any[cue].mean())
    texts = labels.drop_duplicates("text_id").set_index("text_id").reindex(P.ids)
    ctx = texts["context_id"].astype(str).values
    known = np.isin(ctx, list(art.catalog.contexts))
    if known.any():
        out["basic_match_accuracy"] = float(np.mean([c == m for c, m, k in zip(ctx, P.c_star, known) if k]))
    return out
