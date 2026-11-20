"""Hyper-parameters learned on a metric, never fixed by hand (report "phuong an sua", A1 and the general rule).

For each regime, on the TRAINING cells of outer fold 0 only, an inner split is built with the same regime
generator (`make_folds(train0, regime)`): an inner new-question split for R2, inner sparse persons for R4, etc.
Every candidate is fitted on the inner training cells and scored by log-loss on the inner held-out cells (the
outer test cells are never touched: nested validation, Varma & Simon 2006). The winners are then used for all
outer folds of the regime.

Searched (coordinate-wise to keep the cost bounded):
* LR: C (= 1 / lambda) of the per-question logistic regression, also used as the LR part of B2 (the
  baseline gets the same tuning as the twin's base: a fair comparison);
* text LR (R2 baseline): C;
* B2 = LR + MF: lambda of the MF part (person bias, U, V) then the rank, B2 warm-started from the fitted LR;
  acceptance rule: if no MF candidate beats LR on the inner validation log-loss, the MF part is dropped and
  B2 = LR (rank 0, no person bias), so B2 cannot be worse than LR on validation.
Post-hoc weights that need no refit (T2 kappa, T3 omega, calibration, level-2 offset and stacking) are
selected by cross-fitting across the outer folds instead (`pipeline.py`, `combine.py`).
"""

from __future__ import annotations

import time

import numpy as np

from lrmf_twin import baselines
from lrmf_twin.features import answer_features
from lrmf_twin.regimes import make_folds
from lrmf_twin.t1_base import BaseConfig, LRMFBase

LR_C_GRID = (0.002, 0.005, 0.02, 0.05)
TEXT_C_GRID = (0.001, 0.005, 0.05)
MF_LAMBDA_GRID = (3.0, 10.0, 30.0, 100.0)
MF_RANK_GRID = (8, 16, 32)


def _ll(y, z):
    z = np.clip(z, -30, 30)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def lr_config(base: BaseConfig, C: float) -> BaseConfig:
    return BaseConfig(**{**base.as_dict(), "rank": 0, "person_bias": False, "semantic": False, "use_theta": True,
                         "lam_b": 1e-4, "lam_W": 1.0 / C, "lam_theta": 1.0 / C})


def mf_config(base: BaseConfig, C: float, lam: float, rank: int) -> BaseConfig:
    return BaseConfig(**{**base.as_dict(), "rank": rank, "person_bias": True, "semantic": False, "use_theta": True,
                         "lam_b": 1e-4, "lam_W": 1.0 / C, "lam_theta": 1.0 / C, "lam_c": lam, "lam_U": lam,
                         "lam_V": lam})


def fit_lr(Y, train, Xd, A, C, base: BaseConfig) -> LRMFBase:
    return LRMFBase(lr_config(base, C)).fit(Y, train, Xd, A)


def fit_b2(Y, train, Xd, A, C, mf: dict | None, base: BaseConfig, lr_model: LRMFBase) -> LRMFBase:
    """B2 warm-started from the fitted LR; with mf=None (rejected on validation) B2 is the LR itself."""
    if mf is None:
        return lr_model
    init = lr_model.components()
    return LRMFBase(mf_config(base, C, mf["lam"], mf["rank"])).fit(Y, train, Xd, A, init=init)


def tune_regime(data, E, regime: str, train0: np.ndarray, Xd: np.ndarray, base: BaseConfig, seed: int,
                n_sparse: int, n_known: int, k_cold: int, log=print) -> dict:
    t0 = time.time()
    inner = make_folds(train0, regime, n_folds=5, seed=seed + 101, n_sparse=max(20, n_sparse // 2),
                       n_known=n_known, k_cold=k_cold, wave=data.wave)[0]
    if not inner.test.any():
        return {"lr_C": 0.005, "text_lr_C": 0.005, "mf": {"lam": 10.0, "rank": 16}, "table": [], "note": "no inner split"}
    A = answer_features(data, inner.train)
    yv = data.Y[inner.test].astype(np.float64)
    table = []

    best_lr, lr_models = None, {}
    for C in LR_C_GRID:
        m = fit_lr(data.Y, inner.train, Xd, A, C, base)
        ll = _ll(yv, m.logits()[inner.test])
        table.append({"model": "LR", "C": C, "val_logloss": ll, "converged": m.converged})
        lr_models[C] = (m, ll)
        if best_lr is None or ll < best_lr[1]:
            best_lr = (C, ll)
    C = best_lr[0]
    for c2, (m, _) in lr_models.items():
        if c2 != C:
            m.release()
    lr_model = lr_models[C][0]

    text_C = 0.005
    if regime == "R2":
        best = None
        for tc in TEXT_C_GRID:
            p = baselines.text_lr(data.Y, inner.train, inner.test, Xd, E, A, C=tc, seed=seed)[inner.test]
            p = np.clip(p, 1e-6, 1 - 1e-6)
            ll = float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p)))
            table.append({"model": "text LR", "C": tc, "val_logloss": ll})
            if best is None or ll < best[1]:
                best = (tc, ll)
        text_C = best[0]

    best_mf = None
    for lam in MF_LAMBDA_GRID:
        m = fit_b2(data.Y, inner.train, Xd, A, C, {"lam": lam, "rank": 16}, base, lr_model)
        ll = _ll(yv, m.logits()[inner.test])
        table.append({"model": "B2", "lambda": lam, "rank": 16, "val_logloss": ll, "converged": m.converged})
        m.release()
        if best_mf is None or ll < best_mf[2]:
            best_mf = (lam, 16, ll)
    for r in MF_RANK_GRID:
        if r == 16:
            continue
        m = fit_b2(data.Y, inner.train, Xd, A, C, {"lam": best_mf[0], "rank": r}, base, lr_model)
        ll = _ll(yv, m.logits()[inner.test])
        table.append({"model": "B2", "lambda": best_mf[0], "rank": r, "val_logloss": ll, "converged": m.converged})
        m.release()
        if ll < best_mf[2]:
            best_mf = (best_mf[0], r, ll)
    lr_model.release()
    mf = {"lam": best_mf[0], "rank": best_mf[1]} if best_mf[2] < best_lr[1] else None
    log(f"    tuning {regime} (inner split of fold 0, {int(inner.test.sum())} validation cells, "
        f"{time.time() - t0:.0f}s): LR C={C} (ll {best_lr[1]:.4f}); "
        + (f"MF lambda={mf['lam']}, rank={mf['rank']} (ll {best_mf[2]:.4f})" if mf else
           f"MF rejected (best ll {best_mf[2]:.4f} >= LR): B2 = LR")
        + (f"; text LR C={text_C}" if regime == "R2" else ""))
    return {"lr_C": C, "text_lr_C": text_C, "mf": mf, "table": table,
            "lr_val_logloss": best_lr[1], "mf_val_logloss": best_mf[2]}
