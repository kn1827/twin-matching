"""Offline estimation of the context reader (stage1.md, section 7 "Bộ đọc ngữ cảnh", steps O2, O4, O5).

* `rate_catalog`   O2: rater scores of bare context names -> default profile mean, SE, ICC(2,k).
* `icc2k`          ICC(2,k) of Shrout & Fleiss (1979), two-way random effects, average of k raters.
* `estimate_reader` O5: prior strength k (Bühlmann non-parametric) and cue levels theta (least squares
  under the order theta_neg <= theta_weak <= theta_base <= theta_strong; this is PAV when every row
  carries one cue class), iterated to a fixed point; group CV by basic context for MAE.

Labels (`demand_labels`): one row per (text_id, rater_id, style) with rating in -3..3, plus text,
context_id (basic context the text was written for, or "ooc") and kind (bare / cue / minimal_pair / ooc).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from gnwt.stage1.context.catalog import BasicMatch, Catalog
from gnwt.stage1.context.lexicon import Lexicon
from gnwt.stage1.context.styles import CUE_CLASSES, STYLE_INDEX, STYLES

ORDER = ("neg", "weak", "base", "strong")


def icc2k(M: np.ndarray) -> float:
    """ICC(2,k) for an (n targets x k raters) matrix without missing cells."""
    M = np.asarray(M, float)
    n, k = M.shape
    if n < 2 or k < 2:
        return float("nan")
    gm = M.mean()
    msr = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((M.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else float("nan")


def rate_catalog(ratings: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """ratings: columns context_id, rater_id, style, rating. Returns {context: {style: {mean, se, icc2k}}}."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    iccs: dict[str, float] = {}
    for s in STYLES:
        sub = ratings[ratings["style"] == s].pivot_table(index="context_id", columns="rater_id", values="rating")
        iccs[s] = icc2k(sub.dropna(axis=1).to_numpy()) if sub.shape[0] >= 2 else float("nan")
    for (c, s), g in ratings.groupby(["context_id", "style"]):
        x = g["rating"].to_numpy(float)
        out.setdefault(str(c), {})[str(s)] = {
            "mean": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan"),
            "icc2k": iccs[str(s)]}
    return out


@dataclass
class _Rows:
    """One row per (text, style) with at least one hit: counts by class, Z inputs, target, prior."""

    text_id: np.ndarray
    group: np.ndarray
    style: np.ndarray
    counts: np.ndarray        # (m, 5) hits per cue class, order CUE_CLASSES
    n: np.ndarray
    target: np.ndarray        # mean rater rating
    prior: np.ndarray         # D^def_s(c*)


def _build_rows(labels: pd.DataFrame, lex: Lexicon, cat: Catalog) -> tuple[_Rows, pd.DataFrame]:
    tgt = labels.groupby(["text_id", "style"])["rating"].mean()
    texts = labels.drop_duplicates("text_id").set_index("text_id")
    rec, all_rows = [], []
    for tid, row in texts.iterrows():
        clauses, hits = lex.read(str(row["text"]))
        m: BasicMatch = cat.match(clauses)
        prior = cat.prior(m)
        group = m.c_star or (str(row.get("context_id")) if pd.notna(row.get("context_id")) else "other")
        cnt = np.zeros((len(STYLES), len(CUE_CLASSES)))
        for h in hits:
            cnt[STYLE_INDEX[h.style], CUE_CLASSES.index(h.cls)] += 1
        for s in STYLES:
            if (tid, s) not in tgt.index:
                continue
            i = STYLE_INDEX[s]
            all_rows.append({"text_id": tid, "group": group, "style": s, "n": cnt[i].sum(),
                             "target": float(tgt[(tid, s)]), "prior": float(prior[i]),
                             **{f"c_{c}": cnt[i, k] for k, c in enumerate(CUE_CLASSES)}})
    df = pd.DataFrame(all_rows)
    hit = df[df["n"] > 0]
    rows = _Rows(hit["text_id"].to_numpy(), hit["group"].to_numpy(), hit["style"].to_numpy(),
                 hit[[f"c_{c}" for c in CUE_CLASSES]].to_numpy(float), hit["n"].to_numpy(float),
                 hit["target"].to_numpy(float), hit["prior"].to_numpy(float))
    return rows, df


def _fit_theta(R: _Rows, k: np.ndarray) -> np.ndarray:
    """Least squares of (11b) on the targets under the class order; theta in [-3, 3]."""
    Z = R.n / (R.n + k)
    A = (Z / R.n)[:, None] * R.counts                         # columns CUE_CLASSES
    b = R.target - (1 - Z) * R.prior
    # theta = L delta: neg = d1, weak = d1 + d2, base = d1 + d2 + d3, strong = d1 + .. + d4, rev = d5
    idx = {c: CUE_CLASSES.index(c) for c in CUE_CLASSES}
    L = np.zeros((5, 5))
    for r, c in enumerate(ORDER):
        L[idx[c], : r + 1] = 1.0
    L[idx["rev"], 4] = 1.0
    used = R.counts.sum(0) > 0
    lb = np.array([-3.0, 0.0, 0.0, 0.0, -3.0])
    ub = np.array([3.0, 6.0, 6.0, 6.0, 3.0])
    sol = lsq_linear(A @ L, b, bounds=(lb, ub))
    theta = L @ sol.x
    theta[~used] = np.nan
    return np.clip(theta, -3, 3)


def buhlmann_k(ybar: np.ndarray, n: np.ndarray, target: np.ndarray, prior: np.ndarray) -> float:
    """Bühlmann non-parametric prior strength k = sigma^2 / tau^2 (stage1.md, section 7, step 1).

    sigma^2: variance of one keyword's evidence around the true level of its text, estimated by
    mean(n * (ybar - target)^2); tau^2: variance of the true levels around the default profile of their
    basic context, mean((target - prior)^2)."""
    s2 = float(np.mean(n * (ybar - target) ** 2))
    tau2 = float(np.mean((target - prior) ** 2))
    return max(s2, 1e-6) / max(tau2, 1e-6)


def _buhlmann_k(R: _Rows, theta: np.ndarray) -> float:
    ybar = (R.counts @ np.nan_to_num(theta)) / R.n
    return buhlmann_k(ybar, R.n, R.target, R.prior)


def _predict(R: _Rows, theta: np.ndarray, k: float | np.ndarray) -> np.ndarray:
    Z = R.n / (R.n + k)
    ybar = (R.counts @ np.nan_to_num(theta)) / R.n
    return Z * ybar + (1 - Z) * R.prior


def estimate_reader(labels: pd.DataFrame, lex: Lexicon, cat: Catalog, P: Any = None, *,
                    max_iter: int = 50, icc_min: float = 0.70, n_folds: int = 5,
                    k_fixed: float | None = None) -> dict[str, Any]:
    """O5. Returns the JSON-able reader artifact (theta, k, icc, CV report).

    Steps 1-2 alternate until neither theta nor k moves; `k_fixed` freezes k (sensitivity analysis)."""
    R, full = _build_rows(labels, lex, cat)
    if len(R.n) == 0:
        raise ValueError("no label text fires any keyword: theta cannot be estimated")
    k = 1.0 if k_fixed is None else float(k_fixed)
    theta = np.zeros(len(CUE_CLASSES))
    for _ in range(max_iter):
        theta_new = _fit_theta(R, np.full(len(R.n), k))
        k_new = _buhlmann_k(R, theta_new) if k_fixed is None else k
        done = np.allclose(np.nan_to_num(theta_new), np.nan_to_num(theta), atol=1e-6) and abs(k_new - k) < 1e-6
        theta, k = theta_new, k_new
        if done:
            break

    # ICC(2,k) per style over texts
    icc = []
    for s in STYLES:
        sub = labels[labels["style"] == s].pivot_table(index="text_id", columns="rater_id", values="rating")
        icc.append(icc2k(sub.dropna(axis=1).to_numpy()) if sub.shape[0] >= 2 else float("nan"))

    # group CV by basic context: MAE of the full reader (X2) vs catalog only (X0) vs keywords only (X1)
    groups = np.unique(R.group)
    folds = {g: f % n_folds for f, g in enumerate(sorted(groups))}
    fold = np.array([folds[g] for g in R.group])
    err = {"X0_catalog": [], "X1_keywords": [], "X2_full": []}
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        Rt = _Rows(*(getattr(R, a)[tr] for a in R.__dataclass_fields__))
        Re = _Rows(*(getattr(R, a)[te] for a in R.__dataclass_fields__))
        kk = 1.0 if k_fixed is None else float(k_fixed)
        th = np.zeros(5)
        for _ in range(max_iter):
            th = _fit_theta(Rt, np.full(len(Rt.n), kk))
            kk = _buhlmann_k(Rt, th) if k_fixed is None else kk
        err["X0_catalog"].append(np.abs(Re.prior - Re.target))
        err["X1_keywords"].append(np.abs((Re.counts @ np.nan_to_num(th)) / Re.n - Re.target))
        err["X2_full"].append(np.abs(_predict(Re, th, kk) - Re.target))
    mae = {k_: float(np.mean(np.concatenate(v))) for k_, v in err.items()}
    sens = {str(g): float(np.mean(np.abs(_predict(R, _fit_theta(R, np.full(len(R.n), g)), g) - R.target)))
            for g in (P.stage1.context_reader.prior_strength.sensitivity_grid if P is not None else (0.25, 0.5, 1, 2, 4))}
    return {
        "provenance": "estimated", "lexicon_sha256": lex.sha256, "catalog_sha256": cat.sha256,
        "theta": {c: (None if np.isnan(theta[i]) else float(theta[i])) for i, c in enumerate(CUE_CLASSES)},
        "k": float(k), "icc": [None if np.isnan(x) else float(x) for x in icc], "icc_min": icc_min,
        "meta": {"n_texts": int(labels["text_id"].nunique()), "n_rows_with_hits": int(len(R.n)),
                 "cv_mae": mae, "k_sensitivity_mae": sens},
    }


def save_reader(artifact: dict[str, Any], path: str | Path) -> str:
    import hashlib

    data = json.dumps(artifact, indent=1, sort_keys=True).encode("utf-8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()
