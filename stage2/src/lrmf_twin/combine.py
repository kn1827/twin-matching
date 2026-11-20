"""T6 orthogonaliser, T7 monotone gate, T8 non-negative super learner, T9 calibration.

All of them are fitted ONLY on out-of-fold base predictions (design 4.4 step 5) and are themselves
cross-fitted with folds grouped by person, so every level-2 prediction is out-of-fold too.

* T6  D~ = D - m(X), m = OLS of D on [1, eta, eta^2] (X = the base logit, the "compact" X of 4.5b).
      Cells whose twin call failed get D~ = 0 (no twin information).
* T7  w(z) = sigmoid(a0 - softplus(a1) * log1p(n_p) - softplus(a2) * log1p(n_j) + softplus(a3) * new_item):
      non-increasing in the amount of data of the person and of the question, by construction.
      Fitted jointly with beta in   logit p = eta + w(z) * beta * D~.
* T8  p = sum_k alpha_k p_k, alpha on the simplex, minimising log-loss; the base alone is a candidate, so
      the final model cannot lose to it by more than the CV noise.
* T9  Platt scaling a + b * logit(p) of the super learner output.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.special import expit, logit, softmax
from sklearn.model_selection import GroupKFold

EPS = 1e-6


def _lg(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return logit(p)


def _nll(z, y):
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def group_folds(groups: np.ndarray, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    n_groups = np.unique(groups).size
    k = max(2, min(n_splits, n_groups))
    return list(GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=groups))


# -- T6 ---------------------------------------------------------------------------------------------------
def orthogonalise(D: np.ndarray, eta: np.ndarray, folds) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    ok = np.isfinite(D)
    X = np.column_stack([np.ones_like(eta), eta, eta ** 2])
    out = np.zeros_like(D)
    for tr, te in folds:
        tr_ok = tr[ok[tr]]
        if tr_ok.size < 4:
            continue
        coef, *_ = np.linalg.lstsq(X[tr_ok], D[tr_ok], rcond=None)
        out[te] = np.where(ok[te], D[te] - X[te] @ coef, 0.0)
    return out


# -- T7 / offset models --------------------------------------------------------------------------------------
def _softplus(x):
    return np.logaddexp(0.0, x)


def gate(theta: np.ndarray, Z: np.ndarray) -> np.ndarray:
    a0, a1, a2, a3 = theta
    return expit(a0 - _softplus(a1) * Z[:, 0] - _softplus(a2) * Z[:, 1] + _softplus(a3) * Z[:, 2])


def gate_features(n_p, n_j, new_item) -> np.ndarray:
    return np.column_stack([np.log1p(n_p) / 5.0, np.log1p(n_j) / 7.0, np.asarray(new_item, dtype=np.float64)])


def fit_gated(y, eta, Dt, Z, l2: float = 1e-3) -> np.ndarray:
    def obj(params):
        beta, th = params[0], params[1:]
        z = eta + gate(th, Z) * beta * Dt
        return _nll(z, y) + l2 * np.sum(params ** 2)
    x0 = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
    return optimize.minimize(obj, x0, method="L-BFGS-B").x


def predict_gated(params, eta, Dt, Z):
    return expit(eta + gate(params[1:], Z) * params[0] * Dt)


def fit_offset_beta(y, eta, x) -> float:
    res = optimize.minimize_scalar(lambda b: _nll(eta + b * x, y), bounds=(-20, 20), method="bounded")
    return float(res.x)


def fit_logistic(X, y, l2: float = 1e-3) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(y)), X])

    def obj(w):
        return _nll(X1 @ w, y) + l2 * np.sum(w[1:] ** 2)
    return optimize.minimize(obj, np.zeros(X1.shape[1]), method="L-BFGS-B").x


def predict_logistic(w, X):
    return expit(np.column_stack([np.ones(X.shape[0]), X]) @ w)


# -- T8 / T9 ------------------------------------------------------------------------------------------------
def fit_super_learner(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    P = np.clip(P, EPS, 1 - EPS)

    def obj(v):
        p = np.clip(P @ softmax(v), EPS, 1 - EPS)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    res = optimize.minimize(obj, np.zeros(P.shape[1]), method="L-BFGS-B")
    return softmax(res.x)


def fit_platt(p, y, prior_n: float = 5.0) -> np.ndarray:
    """logit p' = a + b * logit p, shrunk toward the identity (a = 0, b = 1) with the weight of `prior_n`
    pseudo-observations: a sample of ~20 cells barely moves it, a sample of thousands calibrates freely."""
    z = _lg(p)
    l2 = prior_n / max(len(y), 1)
    res = optimize.minimize(lambda q: _nll(q[0] + (1.0 + q[1]) * z, y) + l2 * (q[0] ** 2 + q[1] ** 2),
                            np.zeros(2), method="L-BFGS-B")
    return np.array([res.x[0], 1.0 + res.x[1]])


def predict_platt(w, p):
    return expit(w[0] + w[1] * _lg(p))


# -- level 2 ------------------------------------------------------------------------------------------------
OFFSET_CANDIDATES = ("LR", "B2", "B3", "B4")


def level2(cells: dict[str, np.ndarray], variants: list[str], n_splits: int = 5,
           sl_candidates: tuple[str, ...] = ("LR", "B2", "B3", "B4", "B7", "B8")) -> dict:
    """Cross-fitted twin combinations on the LLM sample. `cells` holds 1-D arrays:
    y, person, eta_B2, eta_B3, eta_B4, p_LR, p_B0, n_p, n_j, new_item and D_a / D_b for the T5 variants run.

    v0.3 (A4, A5): inside each level-2 fold the offset of the twin is the base with the lowest training
    log-loss among {LR, B2, B3, B4} (a broken T2/T3 base is no longer inherited), T6 is fitted on that offset,
    and LR is a candidate of the super learner (the oracle guarantee only covers listed candidates)."""
    y = cells["y"].astype(np.float64)
    folds = group_folds(cells["person"], n_splits)
    Z = gate_features(cells["n_p"], cells["n_j"], cells["new_item"])
    base = {"LR": _lg(cells["p_LR"]), "B2": cells["eta_B2"].astype(np.float64),
            "B3": cells["eta_B3"].astype(np.float64), "B4": cells["eta_B4"].astype(np.float64)}
    out: dict[str, np.ndarray] = {"LR": expit(base["LR"]), "B2": expit(base["B2"]), "B3": expit(base["B3"]),
                                  "B4": expit(base["B4"])}
    offset_by_fold = []
    eta = np.zeros_like(y)
    for tr, te in folds:
        best = min(OFFSET_CANDIDATES, key=lambda k: _nll(base[k][tr], y[tr]))
        offset_by_fold.append(best)
        eta[te] = base[best][te]
    out["_offset (best base)"] = expit(eta)
    params: dict[str, list] = {}
    Dt: dict[str, np.ndarray] = {}
    lg_b0 = _lg(cells["p_B0"])
    for v in variants:
        D = cells[f"D_{v}"].astype(np.float64)
        ok = np.isfinite(D)
        Dt[v] = np.zeros_like(y)
        p_raw, p_off, p_gate, p_alone, p_b0tw = (np.zeros_like(y) for _ in range(5))
        Dz = np.where(ok, D, 0.0)
        miss = (~ok).astype(np.float64)
        for (tr, te), key in zip(folds, offset_by_fold):
            e = base[key]
            # T6 on the chosen offset, fitted on the training part of the fold
            X = np.column_stack([np.ones_like(e), e, e ** 2])
            tr_ok = tr[ok[tr]]
            if tr_ok.size >= 4:
                coef, *_ = np.linalg.lstsq(X[tr_ok], D[tr_ok], rcond=None)
                Dt[v][te] = np.where(ok[te], D[te] - X[te] @ coef, 0.0)
                Dt[v][tr] = np.where(ok[tr], D[tr] - X[tr] @ coef, 0.0)
            dtr, dte = Dt[v][tr].copy(), Dt[v][te].copy()
            D_fill = np.where(ok, D, e)
            w = fit_logistic(np.column_stack([e[tr], D_fill[tr]]), y[tr])
            p_raw[te] = predict_logistic(w, np.column_stack([e[te], D_fill[te]]))
            b = fit_offset_beta(y[tr], e[tr], dtr)
            p_off[te] = expit(e[te] + b * dte)
            g = fit_gated(y[tr], e[tr], dtr, Z[tr])
            p_gate[te] = predict_gated(g, e[te], dte, Z[te])
            params.setdefault(f"gate_{v}", []).append([float(x) for x in g])
            wa = fit_logistic(np.column_stack([Dz[tr], miss[tr]]), y[tr])
            p_alone[te] = predict_logistic(wa, np.column_stack([Dz[te], miss[te]]))
            q = optimize.minimize(lambda q: _nll(lg_b0[tr] + q[0] + q[1] * Dz[tr] + q[2] * miss[tr], y[tr]),
                                  np.zeros(3), method="L-BFGS-B").x
            p_b0tw[te] = expit(lg_b0[te] + q[0] + q[1] * Dz[te] + q[2] * miss[te])
        # D~ reported for tests: the held-out part of every fold
        dt_all = np.zeros_like(y)
        for (tr, te), key in zip(folds, offset_by_fold):
            e = base[key]
            X = np.column_stack([np.ones_like(e), e, e ** 2])
            tr_ok = tr[ok[tr]]
            if tr_ok.size >= 4:
                coef, *_ = np.linalg.lstsq(X[tr_ok], D[tr_ok], rcond=None)
                dt_all[te] = np.where(ok[te], D[te] - X[te] @ coef, 0.0)
        Dt[v] = dt_all
        if v == "a":
            out["B5"], out["B6"], out["B7"] = p_raw, p_off, p_gate
            out["T-b (twin alone)"] = p_alone
            out["_B0+twin"] = p_b0tw
        else:
            out["B8"] = p_gate
            out["_B6 with T5b"] = p_off
            out["_B0+twin_b"] = p_b0tw
    cands = [c for c in sl_candidates if c in out]
    P = np.column_stack([out[c] for c in cands])
    p_sl = np.zeros_like(y)
    p_cal = np.zeros_like(y)
    alphas = []
    for tr, te in folds:
        a = fit_super_learner(P[tr], y[tr])
        alphas.append(a.tolist())
        p_sl[te] = P[te] @ a
        w = fit_platt(P[tr] @ a, y[tr])
        p_cal[te] = predict_platt(w, P[te] @ a)
    out["B9 super learner (+T9)"] = p_cal
    out["_B9 before T9"] = p_sl
    return {"pred": out, "Dt": Dt, "folds": folds, "sl_candidates": cands, "offset_by_fold": offset_by_fold,
            "eta_offset": eta, "sl_alpha_mean": np.mean(alphas, axis=0).tolist(), "params": params}
