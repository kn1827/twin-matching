"""Logistic-regression baselines (scikit-learn) and the item-mean floor, trained on one fold's cells.

* `per_item_lr`  one L2 logistic regression per question, features = demographics (+ the person's other
                  answers, `features.answer_features`, own column dropped). With answers this is the
                  information-balanced LR of the design (6.4: the LR sees what the twin sees); without them
                  it is the "demographic look-up" LR.
* `text_lr`      for new questions (R2), where a per-question LR cannot exist: one pooled logistic
                  regression on the bilinear features [1, e_j] (x) [1, x_p, s_p], with e_j = PCA of the
                  question-text embedding and s_p = PCA of the person's answers (own-answer part optional).
                  It is the "LR with demographics and a text-predicted item mean" the design asks for in R2.
* `item_mean`    B0 floor: the item's training positive rate (global rate for an item without answers).
* `torch_lr`     the same per-question L2 logistic regression as `per_item_lr` (same features, lambda = 1/C,
                  unpenalised intercept up to 1e-4), fitted for all questions at once by L-BFGS on the GPU when
                  available: the default engine, since thousands of sklearn fits per fold do not scale to GSS or
                  MovieLens. `tests/unit/test_lrmf_twin.py` checks it against sklearn.
"""

from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def item_mean(Y: np.ndarray, train: np.ndarray, test: np.ndarray, prior_n: float = 2.0) -> np.ndarray:
    """Item positive rate, smoothed with `prior_n` pseudo-answers at the global rate (a cold item with 5
    identical answers must not predict exactly 0 or 1)."""
    y = np.where(train, Y, 0.0)
    glob = float(Y[train].mean()) if train.any() else 0.5
    mu = (y.sum(0) + prior_n * glob) / (train.sum(0) + prior_n)
    out = np.full(Y.shape, np.nan, dtype=np.float32)
    out[test] = np.broadcast_to(mu, Y.shape)[test]
    return out


def _fit_one(j, X, y, rows_tr, rows_te, C):
    if rows_te.size == 0:
        return j, rows_te, np.array([], dtype=np.float32)
    yt = y[rows_tr]
    if rows_tr.size < 10 or yt.min() == yt.max():
        rate = float(yt.mean()) if yt.size else 0.5
        return j, rows_te, np.full(rows_te.size, rate, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LogisticRegression(C=C, max_iter=500).fit(X[rows_tr], yt)
    return j, rows_te, model.predict_proba(X[rows_te])[:, 1].astype(np.float32)


def per_item_lr(Y: np.ndarray, train: np.ndarray, test: np.ndarray, Xd: np.ndarray,
                A: np.ndarray | None = None, C: float = 0.05, n_jobs: int = 4) -> np.ndarray:
    P, J = Y.shape
    out = np.full((P, J), np.nan, dtype=np.float32)
    jobs = []
    for j in range(J):
        rows_te = np.flatnonzero(test[:, j])
        if rows_te.size == 0:
            continue
        X = Xd if A is None else np.hstack([Xd, np.delete(A, j, axis=1)])
        jobs.append(delayed(_fit_one)(j, X, Y[:, j], np.flatnonzero(train[:, j]), rows_te, C))
    for j, rows_te, p in Parallel(n_jobs=n_jobs, prefer="threads")(jobs):
        out[rows_te, j] = p
    return out


def torch_lr(Y: np.ndarray, train: np.ndarray, test: np.ndarray, Xd: np.ndarray, A: np.ndarray | None = None,
             C: float = 0.005, max_iter: int = 300, device: str = "auto", max_feat_items: int = 800) -> np.ndarray:
    from lrmf_twin.t1_base import BaseConfig, LRMFBase
    cfg = BaseConfig(rank=0, use_theta=A is not None, semantic=False, person_bias=False, lam_b=1e-4,
                     lam_W=1.0 / C, lam_theta=1.0 / C, max_iter=max_iter, device=device, max_feat_items=max_feat_items)
    model = LRMFBase(cfg).fit(Y, train, Xd, A if A is not None else np.zeros((Y.shape[0], 1), dtype=np.float32))
    z = model.logits()
    model.release()
    out = np.full(Y.shape, np.nan, dtype=np.float32)
    out[test] = (1.0 / (1.0 + np.exp(-z)))[test]
    return out


def text_lr(Y: np.ndarray, train: np.ndarray, test: np.ndarray, Xd: np.ndarray, E: np.ndarray,
            A: np.ndarray | None = None, C: float = 0.05, k_item: int = 8, k_person: int = 8,
            max_rows: int = 40000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    P, J = Y.shape
    seen = train.any(0)
    pca_e = PCA(n_components=min(k_item, int(seen.sum()) - 1), random_state=seed).fit(E[seen])
    e = pca_e.transform(E)
    e = (e - e[seen].mean(0)) / (e[seen].std(0) + 1e-6)
    G = np.hstack([np.ones((J, 1)), e])                                   # (J, 1 + k_item)
    parts = [np.ones((P, 1)), Xd]
    if A is not None:
        s = PCA(n_components=k_person, random_state=seed).fit_transform(A[:, seen])
        parts.append((s - s.mean(0)) / (s.std(0) + 1e-6))
    H = np.hstack(parts)                                                  # (P, 1 + D [+ k_person])
    cells = np.argwhere(train)
    if len(cells) > max_rows:
        cells = cells[rng.choice(len(cells), size=max_rows, replace=False)]
    Phi = np.einsum("na,nb->nab", G[cells[:, 1]], H[cells[:, 0]]).reshape(len(cells), -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LogisticRegression(C=C, max_iter=1000).fit(Phi, Y[cells[:, 0], cells[:, 1]])
    Wm = model.coef_.reshape(G.shape[1], H.shape[1])
    logits = H @ (G @ Wm).T + model.intercept_[0]                        # (P, J)
    out = np.full((P, J), np.nan, dtype=np.float32)
    out[test] = (1.0 / (1.0 + np.exp(-logits)))[test]
    return out
