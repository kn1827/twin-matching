"""Comparison branches of stage1.md section 9, cross-fitted on the same wave folds as a0.

| group      | branch                                                                     |
|------------|----------------------------------------------------------------------------|
| floor      | random; base rate; by gender; actor only; partner only; actor + partner;   |
|            | pooled demographics                                                        |
| heuristic  | cosine profile similarity (LFKL "Similarity"), Platt-calibrated            |
| statistical| B1 (logit, same features and missing coding as a0, unpenalised);           |
|            | LR L2 with pairwise interactions (C by inner group CV);                    |
|            | GLMM with crossed random effects (probit SRRM with rho = 0, diagonal Sigma)|
|            | SRRM without the dyad block                                                |
| learner    | CatBoost; EBM (TabPFN only when the package and a pinned checkpoint exist) |
| context    | S8: delta^SD = 0                                                           |

B1 is a control branch: it is never used to set, replace or calibrate a0.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from gnwt.persona import INTERESTS, SELF_KEYS
from gnwt.stage1.features import DesignSpec, build_design, delta_basis_for, dyad_raw
from gnwt.stage1.offline.srrm_fit import DyadData, FitConfig, crossfit, fold_standardizer

ACTOR_PREFIX = ("intercept", "gender_i", "age_i", "miss_age_i", "self_i", "miss_self_i", "bf_i", "miss_bf_i")
PARTNER_PREFIX = ("intercept", "age_j", "miss_age_j", "self_j", "miss_self_j", "bf_j", "miss_bf_j", "fit")
DEMO_PREFIX = ("intercept", "gender_i", "age_i", "miss_age_i", "age_j", "miss_age_j", "abs_age_gap",
               "miss_abs_age_gap", "same_field", "miss_same_field")


def _cols(names: list[str], prefixes: tuple[str, ...]) -> list[int]:
    return [k for k, n in enumerate(names) if n.startswith(prefixes)]


def _lr(X: np.ndarray, y: np.ndarray, Xte: np.ndarray, C: float | None = None) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    keep = X.std(0) > 1e-10
    keep[0] = False                                # sklearn adds its own intercept
    m = LogisticRegression(penalty=None if C is None else "l2", C=1.0 if C is None else C, max_iter=5000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X[:, keep], y)
    return m.predict_proba(Xte[:, keep])[:, 1]


def raw_frame(d: DyadData) -> pd.DataFrame:
    """Profile fields of both persons + dyad features, for tree learners (no label, no date ratings)."""
    T = d.T
    out: dict[str, Any] = {}
    for side, idx in (("i", d.i), ("j", d.j)):
        out[f"gender_{side}"] = T.gender[idx]
        out[f"age_{side}"] = T.age[idx]
        out[f"field_{side}"] = pd.Series(T.field[idx]).fillna("NA").astype(str).to_numpy()
        out[f"goal_{side}"] = pd.Series(T.goal[idx]).fillna("NA").astype(str).to_numpy()
        for k, it in enumerate(INTERESTS):
            out[f"{it}_{side}"] = T.interests[idx, k]
        for k, sk in enumerate(SELF_KEYS):
            out[f"self_{sk}_{side}"] = T.selfr[idx, k]
        for k, t in enumerate(("O", "C", "E", "A", "N")):
            out[f"bf_{t}_{side}"] = T.bf[idx, k]
    raw = dyad_raw(T, d.i, d.j, DesignSpec())
    out.update(raw)
    return pd.DataFrame(out)


CAT_COLS = ("field_i", "goal_i", "field_j", "goal_j")


def _catboost(F: pd.DataFrame, y: np.ndarray, tr: np.ndarray, te: np.ndarray, seed: int) -> np.ndarray:
    from catboost import CatBoostClassifier

    m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, loss_function="Logloss",
                           random_seed=seed, verbose=False, thread_count=2, allow_writing_files=False)
    m.fit(F.iloc[tr], y[tr], cat_features=[c for c in CAT_COLS if c in F])
    return m.predict_proba(F.iloc[te])[:, 1]


def _ebm(F: pd.DataFrame, y: np.ndarray, tr: np.ndarray, te: np.ndarray, seed: int) -> np.ndarray:
    from interpret.glassbox import ExplainableBoostingClassifier

    m = ExplainableBoostingClassifier(random_state=seed, n_jobs=2)
    m.fit(F.iloc[tr], y[tr])
    return m.predict_proba(F.iloc[te])[:, 1]


def _lr_interactions(X: np.ndarray, y: np.ndarray, Xte: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    keep = X.std(0) > 1e-10
    keep[0] = False
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    Z = poly.fit_transform(X[:, keep])
    Zte = poly.transform(Xte[:, keep])
    nz = Z.std(0) > 1e-10
    sc = StandardScaler().fit(Z[:, nz])
    cv = list(GroupKFold(n_splits=5).split(Z, y, groups))
    m = LogisticRegressionCV(Cs=np.logspace(-4, 0, 5), cv=cv, scoring="neg_log_loss", max_iter=3000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(sc.transform(Z[:, nz]), y)
    return m.predict_proba(sc.transform(Zte[:, nz]))[:, 1]


def _cosine(d: DyadData, std: Any, tr: np.ndarray, te: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    T = d.T
    persons = np.unique(np.concatenate([d.i[tr], d.j[tr]]))
    V = np.column_stack([T.interests, T.selfr, T.bf])
    mu = np.nanmean(V[persons], axis=0)
    sd = np.nanstd(V[persons], axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    Z = np.nan_to_num((V - mu) / sd)
    a, b = Z[d.i], Z[d.j]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    cos = np.nan_to_num(cos)
    m = LogisticRegression().fit(cos[tr, None], y[tr])
    return m.predict_proba(cos[te, None])[:, 1]


FAST_BRANCHES = ("random", "base_rate", "by_gender", "actor_only", "partner_only", "actor_partner",
                 "demographics", "cosine_profile", "B1", "LR_L2_interactions")
SLOW_BRANCHES = ("CatBoost", "EBM")


def _fold_job(args: tuple[DyadData, np.ndarray, np.ndarray, tuple[str, ...], int]) -> dict[str, np.ndarray]:
    d, tr_all, te, names, seed = args
    from threadpoolctl import threadpool_limits

    with threadpool_limits(2):
        tr = tr_all[d.observed[tr_all]]
        y = d.y
        spec = DesignSpec()
        dtr = d.subset(np.sort(tr_all))
        std = fold_standardizer(dtr, spec)
        basis = delta_basis_for(d.T, np.unique(d.i[tr]))
        des = build_design(d.T, d.i, d.j, std, spec, basis)
        X = des.X
        out: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(seed)
        base = float(y[tr].mean())
        F = raw_frame(d) if any(n in SLOW_BRANCHES for n in names) else None
        for name in names:
            if name == "random":
                out[name] = np.full(len(te), base)
                out[name + "__score"] = rng.random(len(te))
            elif name == "base_rate":
                out[name] = np.full(len(te), base)
            elif name == "by_gender":
                g = d.T.gender[d.i]
                rate = {v: float(y[tr][g[tr] == v].mean()) for v in np.unique(g[tr][np.isfinite(g[tr])])}
                out[name] = np.array([rate.get(v, base) for v in g[te]])
            elif name == "actor_only":
                c = _cols(des.names, ACTOR_PREFIX)
                out[name] = _lr(X[tr][:, c], y[tr], X[te][:, c])
            elif name == "partner_only":
                c = _cols(des.names, PARTNER_PREFIX)
                out[name] = _lr(X[tr][:, c], y[tr], X[te][:, c])
            elif name == "actor_partner":
                c = sorted(set(_cols(des.names, ACTOR_PREFIX)) | set(_cols(des.names, PARTNER_PREFIX)))
                out[name] = _lr(X[tr][:, c], y[tr], X[te][:, c])
            elif name == "demographics":
                c = _cols(des.names, DEMO_PREFIX)
                out[name] = _lr(X[tr][:, c], y[tr], X[te][:, c])
            elif name == "cosine_profile":
                out[name] = _cosine(d, std, tr, te, y)
            elif name == "B1":
                out[name] = _lr(X[tr], y[tr], X[te])
            elif name == "LR_L2_interactions":
                out[name] = _lr_interactions(X[tr], y[tr], X[te], d.wave[tr], seed)
            elif name == "CatBoost":
                out[name] = _catboost(F, y, tr, te, seed)
            elif name == "EBM":
                out[name] = _ebm(F, y, tr, te, seed)
            else:
                raise KeyError(name)
        return out


def crossfit_branches(d: DyadData, names: tuple[str, ...], seed: int, workers: int = 1,
                      fold_of: np.ndarray | None = None) -> pd.DataFrame:
    """Predictions of the non-SRRM branches for every row of d (folds: waves unless fold_of given)."""
    from concurrent.futures import ProcessPoolExecutor

    folds = d.wave if fold_of is None else fold_of
    units = np.unique(folds)
    jobs = [(d, np.where(folds != u)[0], np.where(folds == u)[0], names, seed + k) for k, u in enumerate(units)]
    res = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for (dd, tr, te, *_), out in zip(jobs, ex.map(_fold_job, jobs)):
                res[tuple(te)] = (te, out)
    else:
        for job in jobs:
            res[tuple(job[2])] = (job[2], _fold_job(job))
    frame = {}
    for te, out in res.values():
        for k, v in out.items():
            frame.setdefault(k, np.full(len(d.i), np.nan))[te] = v
    return pd.DataFrame(frame)


def srrm_variants(d: DyadData, base_spec: DesignSpec, cfg: FitConfig, workers: int,
                  names: tuple[str, ...], fold_of: dict[int, int] | None = None) -> dict[str, pd.DataFrame]:
    """S8 (delta = 0), SRRM without the dyad block, GLMM (rho = 0, diagonal Sigma)."""
    out: dict[str, pd.DataFrame] = {}
    variants: dict[str, tuple[DesignSpec, FitConfig]] = {
        "S8_delta0": (replace(base_spec, delta_mode="zero"), cfg),
        "SRRM_no_dyad": (replace(base_spec, dyad=()), cfg),
        "GLMM_crossed": (base_spec, replace(cfg, fix_rho=0.0, diag_sigma=True)),
    }
    for name in names:
        spec, c = variants[name]
        pred, _ = crossfit(d, spec, c, workers=workers, folds=fold_of)
        out[name] = pred
    return out
