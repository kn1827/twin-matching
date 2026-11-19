"""Why is a0 below the published Speed Dating baselines? Factorial diagnosis with fast models.

    python experiments/diagnose_gap.py [--markets columbia speed_dating] [--workers 4]

The published number to explain is biDeepFM / biLorentzFM Part-A AUC 0.68 / 0.70 (biLorentzFM, Applied
Sciences 2025, Table 13). Their set-up differs from Stage 1 in three ways, and a fourth factor is the
model family. Each is switched on separately on the same rows:

* protocol  `lowo` (Stage 1: the test wave's people are never seen) vs `person5` (5 folds grouped by
            chooser, as in biLorentzFM: the partner j of a test row is still a *chosen* person in the
            training rows of other choosers);
* features  `stage1` (what a0 reads) -> `+attr` (self-rated attractiveness, attr3_1) -> `literature`
            (+ race, same-race, importance of race/religion, 6 preference weights);
* model     L2 logistic regression vs LightGBM (best GBDT family on SpeedDating, McElfresh et al. 2023);
* leakage   `+partner_pop`: under person5 the mean decision j received from the training choosers is
            known; under lowo it is never available (unseen wave).

Two ceilings come from the locked SRRM: the latent R^2 of the profile part eta, and the in-sample AUC
reachable if the person effects a_i, b_j were known (two-way fixed-effects logit), i.e. what a
profile-only model can never recover for new people.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnwt.analysis.metrics import auc, within_auc  # noqa: E402

REPORTS = ROOT / "reports"


def _cmp():
    spec = importlib.util.spec_from_file_location("cmp_baselines", ROOT / "experiments" / "compare_literature_baselines.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cmp_baselines"] = mod
    spec.loader.exec_module(mod)
    return mod


ATTR_COLS = ("attr_self_i", "attr_self_j")


def design(frame: pd.DataFrame, cat: list[str], num: list[str], tr: np.ndarray) -> np.ndarray:
    """One-hot categoricals (training levels), z-scored numerics with NaN -> 0 plus indicators."""
    cols = []
    for c in cat:
        v = np.array([str(x) for x in frame[c]], dtype=object)
        for lv in np.unique(v[tr]):
            cols.append((v == lv).astype(float))
    for c in num:
        x = frame[c].to_numpy(float)
        m = np.isfinite(x)
        mu, sd = np.nanmean(x[tr]), np.nanstd(x[tr]) or 1.0
        cols.append(np.where(m, (x - mu) / sd, 0.0))
        if (~m[tr]).any():
            cols.append((~m).astype(float))
    return np.column_stack(cols)


def _job(args: tuple) -> dict:
    frame, cat, num, y, chosen, groups, tr, te, model, seed = args
    from threadpoolctl import threadpool_limits

    with threadpool_limits(2), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model == "logit":
            from sklearn.linear_model import LogisticRegression

            X = design(frame, cat, num, tr)
            p = LogisticRegression(C=0.1, max_iter=3000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        else:
            import lightgbm as lgb

            rng = np.random.default_rng(seed)
            g = np.unique(groups[tr])
            gv = rng.choice(g, max(1, len(g) // 10), replace=False)
            va, fit = tr[np.isin(groups[tr], gv)], tr[~np.isin(groups[tr], gv)]
            F = frame.copy()
            for c in cat:
                F[c] = pd.Categorical([str(x) for x in F[c]])
            m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                                   n_jobs=2, verbose=-1)
            m.fit(F.iloc[fit], y[fit], eval_set=[(F.iloc[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
            p = m.predict_proba(F.iloc[te])[:, 1]
    return {"te": te, "p": p}


def partner_pop(y: np.ndarray, chosen: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Mean decision received by the chosen person from the TRAINING rows of each fold (nan if unseen)."""
    out = np.full(len(y), np.nan)
    for f in np.unique(fold):
        tr = fold != f
        s = pd.Series(y[tr]).groupby(chosen[tr]).mean()
        out[fold == f] = pd.Series(chosen[fold == f]).map(s).to_numpy(float)
    return out


def run(market: str, workers: int) -> dict:
    cmp = _cmp()
    from gnwt.config import load_params, resolve_path
    from gnwt.data.bfi10 import build_proxy
    from gnwt.data.speed_dating import load_market
    from gnwt.stage1.features import PersonTable
    from gnwt.stage1.offline.srrm_fit import DyadData

    P = load_params()
    m = load_market(market)
    bf = build_proxy(m.persons, P.bf_proxy.min_items_per_trait)
    T = PersonTable.from_frame(m.persons, bf, None, default_context="speed_dating")
    d = DyadData.from_rows(T, m.rows)
    d = d.subset(np.where(d.observed & d.observed[d.rev])[0])    # dyads with both decisions in the file
    path = resolve_path(P.data.raw_dir) / P.data.datasets[market].file
    extras = cmp._extras_columbia(m, path) if market == "columbia" else cmp._extras_openml(m, path)
    lit, lcat, lnum = cmp.feature_table(d, extras)
    s1, scat, snum = cmp.feature_table(d, None)
    y = d.y.astype(float)
    chosen = d.j
    sets = {
        "stage1": (s1, scat, snum),
        "+attr": (lit, scat, snum + [c for c in ATTR_COLS if c in lit]),
        "literature": (lit, lcat, lnum),
    }
    from sklearn.model_selection import GroupKFold

    folds = {"lowo": d.wave.copy()}
    f5 = np.zeros(len(y), int)
    for k, (_, te) in enumerate(GroupKFold(n_splits=5).split(lit, y, d.i)):
        f5[te] = k
    folds["person5"] = f5

    cells = []
    for proto, fold in folds.items():
        pop = partner_pop(y, chosen, fold)
        for fs, (frame, cat, num) in sets.items():
            variants = [(fs, frame, num)]
            if fs == "literature":
                variants.append((fs + "+partner_pop", frame.assign(partner_pop=pop), num + ["partner_pop"]))
            for name, fr, nm in variants:
                for model in ("logit", "lightgbm"):
                    cells.append((proto, name, model, fr, cat, nm, fold))
    jobs, owner = [], []
    for k, (proto, name, model, fr, cat, nm, fold) in enumerate(cells):
        groups = d.wave if proto == "lowo" else d.i
        for f in np.unique(fold):
            jobs.append((fr[cat + nm], cat, nm, y, chosen, groups, np.where(fold != f)[0], np.where(fold == f)[0],
                         model, P.global_seed + int(f)))
            owner.append(k)
    preds = [np.full(len(y), np.nan) for _ in cells]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, out in zip(owner, ex.map(_job, jobs)):
            preds[k][out["te"]] = out["p"]
    table = []
    for (proto, name, model, *_), p in zip(cells, preds):
        table.append({"protocol": proto, "features": name, "model": model, "auc": auc(y, p),
                      "auc_within": within_auc(y, p, d.i), "match_auc": auc(np.nan_to_num(d.match)[d.i < d.j],
                                                                          (p * p[d.rev])[d.i < d.j])})

    # how identifiable are people from their profile? (share of persons with a unique profile vector)
    jcols = [c for c in s1 if c.endswith("_j")]
    prof = ["|".join(str(x) for x in row) for row in s1[jcols].to_numpy()]
    uniq = float(pd.Series(prof).groupby(d.j).first().duplicated(keep=False).eq(False).mean())

    # ceilings from the locked SRRM and the two-way fixed-effects logit
    ceil = {}
    pr = pd.read_csv(REPORTS / f"predictions_{market}.csv")
    rep = json.loads((REPORTS / f"stage1_{market}.json").read_text(encoding="utf-8"))
    s = rep["locked"]["s"]
    var_eta = float(np.var(pr["eta"]))
    ceil["latent_R2_profile"] = var_eta / (var_eta + s * s)
    ceil["sigma_a2"], ceil["sigma_b2"] = rep["locked"]["sigma_a2"], rep["locked"]["sigma_b2"]
    ok = pr["y"].notna().to_numpy()
    ceil["a0_auc"] = auc(pr["y"].to_numpy()[ok], pr["a0"].to_numpy()[ok])
    var_eta = float(np.var(pr["eta"].to_numpy()[ok]))
    ceil["latent_R2_profile"] = var_eta / (var_eta + s * s)
    from scipy.sparse import csr_matrix, hstack
    from sklearn.linear_model import LogisticRegression

    n = len(y)
    A = csr_matrix((np.ones(n), (np.arange(n), d.i)), shape=(n, len(T)))
    Bm = csr_matrix((np.ones(n), (np.arange(n), d.j)), shape=(n, len(T)))
    fe = LogisticRegression(C=1.0, max_iter=5000).fit(hstack([A, Bm]).tocsr(), y)
    ceil["fixed_effects_in_sample_auc"] = auc(y, fe.predict_proba(hstack([A, Bm]).tocsr())[:, 1])
    pop_only = partner_pop(y, chosen, folds["person5"])
    ceil["partner_pop_only_auc_person5"] = auc(y[np.isfinite(pop_only)], pop_only[np.isfinite(pop_only)])
    return {"market": market, "cells": table, "share_unique_profiles": uniq, "ceilings": ceil}


def render(results: list[dict]) -> str:
    L = ["# Diagnosis: why a0 is below the published Speed Dating baselines", ""]
    for r in results:
        c = r["ceilings"]
        L += [f"## {r['market']}", "",
              f"a0 (Stage 1 SRRM, LOWO) AUC = {c['a0_auc']:.4f}. Latent R² of the profile part η = "
              f"{c['latent_R2_profile']:.3f} (σa² = {c['sigma_a2']:.2f}, σb² = {c['sigma_b2']:.2f}). "
              f"Two-way fixed effects (in-sample, knows every a_i and b_j): AUC = {c['fixed_effects_in_sample_auc']:.3f}. "
              f"Partner popularity alone under person5: AUC = {c['partner_pop_only_auc_person5']:.3f}. "
              f"Persons with a unique profile vector: {100 * r['share_unique_profiles']:.1f}%.", "",
              "| protocol | features | model | Part A AUC | AUC within | mutual AUC |", "|---|---|---|---|---|---|"]
        for t in r["cells"]:
            L.append(f"| {t['protocol']} | {t['features']} | {t['model']} | {t['auc']:.4f} | {t['auc_within']:.4f} | {t['match_auc']:.4f} |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", nargs="+", default=["columbia", "speed_dating"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    res = [run(mk, args.workers) for mk in args.markets]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "diagnosis_gap.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    md = render(res)
    (REPORTS / "diagnosis_gap.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
