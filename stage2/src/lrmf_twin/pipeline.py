"""Run the twin and the LR baseline on one regime, everything out-of-fold (design 4.4 training order).

Version 0.3 (report "phuong an sua"):

* tuning (A1): LR C, text-LR C, MF lambda and rank learned on a regime-mimicking inner split of fold 0
  (`tuning.py`); B2 is warm-started from the fitted LR and falls back to LR when MF does not validate.
* T2 (A2): item parameters (b_j, v_j) = rho_j * learned + (1 - rho_j) * generated-from-text, with
  rho_j = n_j / (n_j + kappa); the generator is a kernel ridge chosen by CV over items; kappa is chosen by
  cross-fitting across the outer folds (each fold uses the kappa that is best on the other folds).
* T3 (A3): LLM soft labels enter the person refit with weight omega * n0 / (n0 + n_p); omega in
  {0, 0.1, 0.3, 1} is cross-fitted the same way, so omega = 0 (no LLM prior) is chosen when the prior hurts
  (PPI++ power tuning: lambda = 0 recovers the classical estimator).
* calibration (A2, T9): Platt per data group (new item / cold item / sparse person / warm), cross-fitted.
* T5 default = T5a only (A6: the twin does not see the base prediction, no anchoring).

Level 1 (statistical models) covers every cell of the regime's scope; level 2 (anything that needs D) only
the `llm_cells` sampled cells. `ft_context` keeps what the fine-tuned central twin (twin_ft.py) needs.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.special import expit

from lrmf_twin import baselines, combine
from lrmf_twin.data.twin2k import SurveyData
from lrmf_twin.features import answer_features, centred_demographics
from lrmf_twin.llm import LLMClient, parse_list, parse_probs
from lrmf_twin.regimes import make_folds, regime_flags
from lrmf_twin.t1_base import BaseConfig, logits_from, refit_rows
from lrmf_twin.t2_encoder import ItemGenerator, shrink_item_params
from lrmf_twin.t3_prior import build_prompt as t3_prompt
from lrmf_twin.t3_prior import select_anchors
from lrmf_twin.t4_retrieval import normalise, retrieve
from lrmf_twin.t5_twin import build_prompt as t5_prompt
from lrmf_twin.t5_twin import to_logit
from lrmf_twin.tuning import fit_b2, fit_lr, tune_regime


@dataclass
class RunConfig:
    regimes: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5", "R6")
    n_folds: int = 5
    seed: int = 20261118
    n_sparse: int = 80              # persons made sparse (R4) / new (R5)
    n_known: int = 10               # answers kept per sparse person (R4)
    k_cold: int = 5                 # answers kept per cold item (R6)
    t3_max_known: int = 25
    t3_anchors: int = 8
    t3_n0: float = 10.0             # shape of the prior weight n0 / (n0 + n_p); its scale omega is learned
    llm_cells: int = 30
    t5_variants: str = "a"
    top_k: int = 5
    lr_engine: str = "torch"        # "torch" (tuned, all datasets) or "sklearn" (reference, fixed C)
    lr_C: float = 0.005             # used only when tune=False
    text_lr_C: float = 0.005
    tune: bool = True
    kappa_grid: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0, 100.0)
    omega_grid: tuple[float, ...] = (0.0, 0.1, 0.3, 1.0)
    calibrate: bool = True
    n_jobs: int = 4
    base: BaseConfig = field(default_factory=BaseConfig)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["regimes"] = list(self.regimes)
        return d


LEVEL1_MODELS = ("LR", "B0 item mean", "B2 LR+MF (T1)", "B3 +T2", "B4 +T3")


def _sample_cells(scope: np.ndarray, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Up to n cells of the scope, at most one per person while persons last."""
    cells = np.argwhere(scope)
    if n <= 0 or cells.size == 0:
        return []
    rng.shuffle(cells)
    _, first = np.unique(cells[:, 0], return_index=True)       # one random cell per person
    first = rng.permutation(first)
    order = np.concatenate([first, np.setdiff1d(np.arange(len(cells)), first)])
    return [(int(p), int(j)) for p, j in cells[order[:n]]]


def _ll_vec(y, z):
    z = np.clip(z, -30, 30)
    return np.logaddexp(0.0, z) - y * z


def crossfit_select(cands: dict, y: np.ndarray, fold: np.ndarray, rng: np.random.Generator,
                    mask: np.ndarray | None = None, max_cells: int = 300_000) -> tuple[np.ndarray, dict]:
    """For each outer fold k, pick the candidate with the lowest log-loss on the OTHER folds' cells."""
    keys = list(cands)
    out = np.array(cands[keys[0]], copy=True)
    chosen = {}
    use = np.ones_like(y, dtype=bool) if mask is None else mask
    for k in np.unique(fold):
        other = np.flatnonzero(use & (fold != k))
        if other.size > max_cells:
            other = rng.choice(other, max_cells, replace=False)
        if other.size == 0:
            best = keys[0]
        else:
            best = min(keys, key=lambda key: float(np.mean(_ll_vec(y[other], cands[key][other]))))
        chosen[int(k)] = best
        here = fold == k
        out[here] = cands[best][here]
    return out, chosen


def crossfit_calibrate(z: np.ndarray, y: np.ndarray, fold: np.ndarray, group: np.ndarray,
                       rng: np.random.Generator, min_cells: int = 300, max_cells: int = 200_000) -> tuple[np.ndarray, dict]:
    """Platt scaling per data group, each fold calibrated with parameters fitted on the other folds."""
    out = np.array(z, copy=True)
    params = {}
    for k in np.unique(fold):
        for g in np.unique(group[fold == k]):
            tr = np.flatnonzero((fold != k) & (group == g))
            if tr.size < min_cells:
                tr = np.flatnonzero(fold != k)
            if tr.size == 0:
                continue
            if tr.size > max_cells:
                tr = rng.choice(tr, max_cells, replace=False)
            w = combine.fit_platt(expit(z[tr]), y[tr])
            here = (fold == k) & (group == g)
            out[here] = w[0] + w[1] * z[here]
            params[f"fold{int(k)}/group{int(g)}"] = [float(w[0]), float(w[1])]
    return out, params


def run_regime(data: SurveyData, E: np.ndarray, regime: str, cfg: RunConfig, llm: LLMClient,
               log=print) -> dict:
    P, J = data.shape
    rng = np.random.default_rng(cfg.seed + 3)
    observed = ~np.isnan(data.Y)
    folds = make_folds(observed, regime, cfg.n_folds, seed=cfg.seed, n_sparse=cfg.n_sparse, n_known=cfg.n_known,
                       k_cold=cfg.k_cold, wave=data.wave)
    scope = np.zeros((P, J), dtype=bool)
    for f in folds:
        scope |= f.test
    flat = np.flatnonzero(scope)
    pos = np.full(P * J, -1, dtype=np.int64)
    pos[flat] = np.arange(flat.size)
    n = flat.size
    y_vec = data.Y.ravel()[flat].astype(np.float64)
    fold_vec = np.full(n, -1, dtype=np.int16)
    group_vec = np.zeros(n, dtype=np.int8)

    use_llm = llm.enabled and cfg.llm_cells > 0
    sample = _sample_cells(scope, cfg.llm_cells if use_llm else 0, np.random.default_rng(cfg.seed + 7))
    sample_set = {c: i for i, c in enumerate(sample)}
    variants = [v for v in cfg.t5_variants if v in "ab"] if use_llm else []
    Xd = centred_demographics(data.X_demo)
    En = normalise(E)
    t_start = time.time()

    tuned = (tune_regime(data, E, regime, folds[0].train, Xd, cfg.base, cfg.seed, cfg.n_sparse, cfg.n_known,
                         cfg.k_cold, log)
             if cfg.tune else {"lr_C": cfg.lr_C, "text_lr_C": cfg.text_lr_C,
                               "mf": {"lam": cfg.base.lam_U, "rank": cfg.base.rank}, "table": []})
    C, mf = tuned["lr_C"], tuned["mf"]
    lam_person = mf["lam"] if mf else 10.0

    vec = {m: np.full(n, np.nan) for m in ("LR", "B0 item mean", "B2")}
    cand3 = {k: np.full(n, np.nan) for k in cfg.kappa_grid}
    do_t3 = use_llm and regime in ("R4", "R5")
    cand4 = {(k, w): np.full(n, np.nan) for k in cfg.kappa_grid for w in cfg.omega_grid} if do_t3 else {}
    l2_rows: dict[int, dict] = {}
    prompts: list[dict] = []
    t3_info = {"persons": 0, "with_prior": 0}
    ft_context = None
    fit_info = []

    def progress(tag):
        def _p(done, total):
            if done == total or done % 20 == 0:
                st = llm.stats
                log(f"    {tag}: {done}/{total} new calls ({st.calls} total, ~{st.seconds / max(st.calls, 1):.1f}s/call, "
                    f"{st.cache_hits} cached, {st.failures} failed)")
        return _p

    for f in folds:
        t0 = time.time()
        idx = pos[np.flatnonzero(f.test.ravel())]
        test_flat = np.flatnonzero(f.test.ravel())
        fold_vec[idx] = f.k
        A = answer_features(data, f.train)
        n_p = f.train.sum(1).astype(np.float64)
        n_j = f.train.sum(0).astype(np.float64)
        pp, jj = np.divmod(test_flat, J)
        g = np.zeros(idx.size, dtype=np.int8)
        g[n_p[pp] <= cfg.t3_max_known] = 3
        g[(n_j[jj] > 0) & (n_j[jj] < 20)] = 2
        g[n_j[jj] == 0] = 1
        group_vec[idx] = g

        vec["B0 item mean"][idx] = baselines.item_mean(data.Y, f.train, f.test)[f.test]
        lr_model = fit_lr(data.Y, f.train, Xd, A, C, cfg.base)
        if regime == "R2":          # a per-question LR cannot exist for a question without answers
            p_lr = baselines.text_lr(data.Y, f.train, f.test, Xd, E, A, C=tuned["text_lr_C"], seed=cfg.seed)[f.test]
        elif cfg.lr_engine == "sklearn":
            p_lr = baselines.per_item_lr(data.Y, f.train, f.test, Xd, A, C=C, n_jobs=cfg.n_jobs)[f.test]
        else:
            p_lr = expit(lr_model.logits()[f.test])
        vec["LR"][idx] = p_lr
        b2 = fit_b2(data.Y, f.train, Xd, A, C, mf, cfg.base, lr_model)
        comp = b2.components()
        fit_info.append({"fold": f.k, "lr_iter": lr_model.n_iter, "lr_converged": lr_model.converged,
                         "b2_iter": b2.n_iter, "b2_converged": b2.converged})
        b2.release()
        if b2 is not lr_model:
            lr_model.release()
        eta2 = logits_from(comp)
        vec["B2"][idx] = eta2[f.test]

        # T2: generated item parameters, shrunk toward the learned ones by n_j / (n_j + kappa)
        free = np.column_stack([comp["b"], comp["V"]])
        gen = ItemGenerator(seed=cfg.seed).fit(E, free, n_j)
        G = gen.predict(E)
        item_k = {}
        for kappa in cfg.kappa_grid:
            par = shrink_item_params(free, G, n_j, kappa)
            item_k[kappa] = (par[:, 0], par[:, 1:])
            cand3[kappa][idx] = logits_from(comp, b=par[:, 0], V=par[:, 1:])[f.test]

        # T3: LLM soft labels for sparse / new persons, weight omega * n0 / (n0 + n_p), omega cross-fitted
        sparse = np.array([int(p) for p in f.sparse_persons if n_p[p] <= cfg.t3_max_known], dtype=int)
        if do_t3 and sparse.size:
            anchors = select_anchors(comp["V"] if comp["V"].shape[1] else free[:, :1], n_j,
                                     n_anchor=2 * cfg.t3_anchors, min_answers=min(500, int(np.quantile(n_j, 0.5))))
            jobs = []
            for p in sparse:
                known = [int(j) for j in np.flatnonzero(f.train[p])]
                use = [a for a in anchors if a not in set(known)][:cfg.t3_anchors]
                if use:
                    jobs.append((p, use, t3_prompt(data, p, known, use)))
            texts = llm.complete_many([j[2] for j in jobs], progress(f"T3 {regime} fold {f.k}"))
            q = np.zeros((sparse.size, J), dtype=np.float32)
            shape = np.zeros((sparse.size, J), dtype=np.float32)
            row_of = {p: i for i, p in enumerate(sparse)}
            for (p, use, _), text in zip(jobs, texts):
                probs = parse_list(text, len(use))
                if probs is not None:
                    q[row_of[p], use] = np.clip(np.asarray(probs, dtype=np.float32), 0.02, 0.98)
                    shape[row_of[p], use] = cfg.t3_n0 / (cfg.t3_n0 + n_p[p])
                    t3_info["with_prior"] += 1
            t3_info["persons"] += int(sparse.size)
            ys = np.nan_to_num(data.Y[sparse], nan=0.0)
            ms = f.train[sparse].astype(np.float32)
            test_rows = f.test[sparse]
            cell_idx = pos[(sparse[:, None] * J + np.arange(J)[None, :])[test_rows]]
            for kappa in cfg.kappa_grid:
                b_k, V_k = item_k[kappa]
                fixed = comp["rest"][sparse] + b_k[None, :]
                base_rows = logits_from(comp, b=b_k, V=V_k)[sparse]
                for omega in cfg.omega_grid:
                    if omega == 0.0:
                        rows = base_rows
                    else:
                        c_new, U_new = refit_rows(fixed, V_k, ys, ms, q, omega * shape, lam_person, lam_person,
                                                  comp["c"][sparse], comp["U"][sparse], cfg.base.device)
                        rows = fixed + c_new[:, None] + U_new @ V_k.T
                    cand4[(kappa, omega)][cell_idx] = rows[test_rows]

        # T4 + T5 for the sampled cells of this fold (T5a by default: no base prediction in the prompt)
        z = regime_flags(f.train)
        mine = [c for c in sample if f.test[c]]
        jobs = []
        for p, j in mine:
            ev = retrieve(p, j, f.train, En, k=cfg.top_k)
            row = {"regime": regime, "fold": f.k, "person": p, "pid": int(data.pids[p]), "item": j,
                   "item_id": data.items[j].item_id, "y": float(data.Y[p, j]), "cell": int(pos[p * J + j]),
                   "n_p": float(n_p[p]), "n_j": float(z["n_j"][j]), "new_item": float(z["new_item"][j]),
                   "n_evidence": len(ev)}
            l2_rows[sample_set[(p, j)]] = row
            for v in variants:
                base_prob = float(expit(eta2[p, j])) if v == "b" else None
                jobs.append((row, v, t5_prompt(data, p, j, ev, base_prob)))
        texts = llm.complete_many([jb[2] for jb in jobs], progress(f"T5 {regime} fold {f.k}")) if jobs else []
        for (row, v, prompt), text in zip(jobs, texts):
            item = data.items[row["item"]]
            D = to_logit(parse_probs(text, len(item.options)), item.positive)
            row[f"D_{v}"] = D
            prompts.append({"regime": regime, "pid": row["pid"], "item_id": row["item_id"], "variant": f"T5{v}",
                            "D": None if np.isnan(D) else D, "prompt": prompt, "response": text})
        if f.k == 0:
            ft_context = {"train": f.train.copy(), "test": f.test.copy(), "U": comp["U"].copy(),
                          "V": comp["V"].copy(), "eta2": eta2[f.test].copy()}
        del comp, eta2
        log(f"  {regime} fold {f.k + 1}/{len(folds)} done in {time.time() - t0:.0f}s "
            f"(test cells {int(f.test.sum())})")

    # -- cross-fitted selection of kappa (T2) and omega (T3), then calibration ------------------------------
    eta3, kappa_by_fold = crossfit_select(cand3, y_vec, fold_vec, rng)
    omega_by_fold = {}
    if cand4:
        eta4 = eta3.copy()
        for k in np.unique(fold_vec):
            kap = kappa_by_fold[int(k)]
            sub = {w: cand4[(kap, w)] for w in cfg.omega_grid}
            ok = np.isfinite(sub[cfg.omega_grid[0]])
            best, chosen = crossfit_select(sub, y_vec, fold_vec, rng, mask=ok)
            here = (fold_vec == k) & ok
            eta4[here] = best[here]
            omega_by_fold[int(k)] = chosen[int(k)]
    else:
        eta4 = eta3.copy()
    logits = {"B2": vec["B2"], "B3": eta3, "B4": eta4}
    calib = {}
    if cfg.calibrate:
        for m in ("B2", "B3", "B4"):
            logits[m], calib[m] = crossfit_calibrate(logits[m], y_vec, fold_vec, group_vec, rng)
    log(f"    selection {regime}: kappa by fold {kappa_by_fold}" + (f", omega by fold {omega_by_fold}" if cand4 else ""))

    preds = {}
    for m, v in (("LR", vec["LR"]), ("B0 item mean", vec["B0 item mean"]), ("B2 LR+MF (T1)", expit(logits["B2"])),
                 ("B3 +T2", expit(logits["B3"])), ("B4 +T3", expit(logits["B4"]))):
        arr = np.full(P * J, np.nan, dtype=np.float32)
        arr[flat] = v
        preds[m] = arr.reshape(P, J)
    fold_of = np.full(P * J, -1, dtype=np.int16)
    fold_of[flat] = fold_vec
    for row in l2_rows.values():
        c = row.pop("cell")
        row["eta_B2"], row["eta_B3"], row["eta_B4"] = (float(logits[m][c]) for m in ("B2", "B3", "B4"))
        row["p_LR"] = float(vec["LR"][c])
        row["p_B0 item mean"] = float(vec["B0 item mean"][c])
    if ft_context is not None:
        ft_context.update({"regime": regime, "flat_test0": np.flatnonzero(ft_context["test"].ravel()),
                           "pos": pos, "preds": {m: preds[m] for m in ("LR", "B0 item mean", "B2 LR+MF (T1)", "B3 +T2", "B4 +T3")}})
    return {"regime": regime, "scope": scope, "fold_of": fold_of.reshape(P, J), "preds": preds,
            "level2_rows": [l2_rows[i] for i in sorted(l2_rows)], "prompts": prompts, "variants": variants,
            "t3": t3_info, "seconds": time.time() - t_start, "ft_context": ft_context,
            "selection": {"tuning": tuned, "kappa_by_fold": {k: float(v) for k, v in kappa_by_fold.items()},
                          "omega_by_fold": {k: float(v) for k, v in omega_by_fold.items()},
                          "calibration": calib, "fits": fit_info},
            "folds": [{"k": f.k, "train_cells": int(f.train.sum()), "test_cells": int(f.test.sum()),
                       "sparse_persons": int(f.sparse_persons.size), "new_items": int(f.new_items.size)}
                      for f in folds]}


def run_level2(rows: list[dict], variants: list[str], n_splits: int = 5) -> dict | None:
    """T6-T9 on the pooled LLM sample of all regimes (the gate learns from z across regimes)."""
    if not rows or not variants:
        return None
    cols = ["y", "person", "eta_B2", "eta_B3", "eta_B4", "n_p", "n_j", "new_item"] + [f"D_{v}" for v in variants]
    cells = {c: np.array([r.get(c, np.nan) for r in rows], dtype=np.float64) for c in cols}
    cells["person"] = np.array([r["pid"] for r in rows])
    cells["p_B0"] = np.array([r["p_B0 item mean"] for r in rows], dtype=np.float64)
    cells["p_LR"] = np.array([r["p_LR"] for r in rows], dtype=np.float64)
    res = combine.level2(cells, variants, n_splits=n_splits)
    res["cells"] = cells
    return res
