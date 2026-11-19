"""Run the twin and the LR baselines on one regime, everything out-of-fold (design 4.4 training order).

Per fold:  A (own answers, training cells only) -> B0 item mean, LR-demo, LR
           -> B2 = T1 (LR + MF) -> B3 = T1 + T2 (semantic item heads) -> B4 = B3 + T3 (sparse / new persons)
           -> for the sampled cells of this fold: T4 retrieval -> T5a / T5b twin signal D (batched LLM calls).
After the folds: T6-T9 on the LLM sample (combine.level2), cross-fitted by person.

Level 1 (statistical models) covers every cell of the regime's scope; level 2 (anything that needs D) only
the `llm_cells` sampled cells, because each needs one LLM call.
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
from lrmf_twin.t1_base import BaseConfig, LRMFBase
from lrmf_twin.t3_prior import build_prompt as t3_prompt
from lrmf_twin.t3_prior import select_anchors
from lrmf_twin.t4_retrieval import normalise, retrieve
from lrmf_twin.t5_twin import build_prompt as t5_prompt
from lrmf_twin.t5_twin import to_logit


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
    t3_n0: float = 10.0
    llm_cells: int = 30
    t5_variants: str = "ab"
    top_k: int = 5
    lr_engine: str = "torch"        # "torch" (all datasets) or "sklearn" (reference, small data only)
    lr_C: float = 0.005
    lr_demo_C: float = 0.03
    text_lr_C: float = 0.005        # tuned on an inner new-item split of R2 fold-0 training items (Twin-2K)
    n_jobs: int = 4
    base: BaseConfig = field(default_factory=BaseConfig)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["regimes"] = list(self.regimes)
        return d


LEVEL1_MODELS = ("LR-demo", "LR", "B0 item mean", "B2 LR+MF (T1)", "B3 +T2", "B4 +T3")


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


def _lr(cfg: RunConfig, Y, train, test, Xd, A, C):
    if cfg.lr_engine == "sklearn":
        return baselines.per_item_lr(Y, train, test, Xd, A, C=C, n_jobs=cfg.n_jobs)
    return baselines.torch_lr(Y, train, test, Xd, A, C=C, max_iter=cfg.base.max_iter, device=cfg.base.device,
                              max_feat_items=cfg.base.max_feat_items)


def run_regime(data: SurveyData, E: np.ndarray, regime: str, cfg: RunConfig, llm: LLMClient,
               log=print) -> dict:
    P, J = data.shape
    observed = ~np.isnan(data.Y)
    folds = make_folds(observed, regime, cfg.n_folds, seed=cfg.seed, n_sparse=cfg.n_sparse, n_known=cfg.n_known,
                       k_cold=cfg.k_cold, wave=data.wave)
    scope = np.zeros((P, J), dtype=bool)
    for f in folds:
        scope |= f.test
    use_llm = llm.enabled and cfg.llm_cells > 0
    sample = _sample_cells(scope, cfg.llm_cells if use_llm else 0, np.random.default_rng(cfg.seed + 7))
    sample_set = {c: i for i, c in enumerate(sample)}
    variants = [v for v in cfg.t5_variants if v in "ab"] if use_llm else []

    preds = {m: np.full((P, J), np.nan, dtype=np.float32) for m in LEVEL1_MODELS}
    fold_of = np.full((P, J), -1, dtype=np.int16)
    Xd = centred_demographics(data.X_demo)
    En = normalise(E)
    l2_rows: dict[int, dict] = {}
    prompts: list[dict] = []
    t3_info = {"persons": 0, "with_prior": 0}
    t_start = time.time()

    def progress(tag):
        def _p(done, total):
            if done == total or done % 20 == 0:
                st = llm.stats
                log(f"    {tag}: {done}/{total} new calls ({st.calls} total, ~{st.seconds / max(st.calls, 1):.1f}s/call, "
                    f"{st.cache_hits} cached, {st.failures} failed)")
        return _p

    for f in folds:
        t0 = time.time()
        A = answer_features(data, f.train)
        preds["B0 item mean"][f.test] = baselines.item_mean(data.Y, f.train, f.test)[f.test]
        if regime == "R2":          # a per-question LR cannot exist for a question without answers
            preds["LR"][f.test] = baselines.text_lr(data.Y, f.train, f.test, Xd, E, A, C=cfg.text_lr_C,
                                                    seed=cfg.seed)[f.test]
            preds["LR-demo"][f.test] = baselines.text_lr(data.Y, f.train, f.test, Xd, E, None, C=cfg.text_lr_C,
                                                         seed=cfg.seed)[f.test]
        else:
            preds["LR"][f.test] = _lr(cfg, data.Y, f.train, f.test, Xd, A, cfg.lr_C)[f.test]
            preds["LR-demo"][f.test] = _lr(cfg, data.Y, f.train, f.test, Xd, None, cfg.lr_demo_C)[f.test]
        t_lr = time.time() - t0
        fold_of[f.test] = f.k

        b2 = LRMFBase(BaseConfig(**{**cfg.base.as_dict(), "semantic": False})).fit(data.Y, f.train, Xd, A)
        eta2 = b2.logits()
        b2.release()
        b3 = LRMFBase(BaseConfig(**{**cfg.base.as_dict(), "semantic": True})).fit(data.Y, f.train, Xd, A, E)
        eta3 = b3.logits()
        eta4 = eta3.copy()

        # T3: LLM prior for sparse / new persons (gate: n_p <= t3_max_known)
        n_p = f.train.sum(1)
        n_j = f.train.sum(0)
        sparse = [int(p) for p in f.sparse_persons if n_p[p] <= cfg.t3_max_known]
        if use_llm and sparse:
            anchors = select_anchors(b3.item_factors(), n_j, n_anchor=2 * cfg.t3_anchors,
                                     min_answers=min(500, int(np.quantile(n_j, 0.5))))
            jobs = []
            for p in sparse:
                known = [int(j) for j in np.flatnonzero(f.train[p])]
                use = [a for a in anchors if a not in set(known)][:cfg.t3_anchors]
                if use:
                    jobs.append((p, use, t3_prompt(data, p, known, use)))
            texts = llm.complete_many([j[2] for j in jobs], progress(f"T3 {regime} fold {f.k}"))
            pseudo = {}
            for (p, use, _), text in zip(jobs, texts):
                probs = parse_list(text, len(use))
                if probs is not None:
                    pseudo[p] = (np.asarray(use), np.clip(np.asarray(probs, dtype=np.float32), 0.02, 0.98))
            eta4[sparse] = b3.refit_persons(np.array(sparse), pseudo, n0=cfg.t3_n0)
            t3_info["persons"] += len(sparse)
            t3_info["with_prior"] += len(pseudo)
        b3.release()

        for name, eta in (("B2 LR+MF (T1)", eta2), ("B3 +T2", eta3), ("B4 +T3", eta4)):
            preds[name][f.test] = expit(eta[f.test])
        z = regime_flags(f.train)

        # T4 + T5 for the sampled cells of this fold, all prompts of the fold in one batched call
        mine = [c for c in sample if f.test[c]]
        jobs = []
        for p, j in mine:
            ev = retrieve(p, j, f.train, En, k=cfg.top_k)
            row = {"regime": regime, "fold": f.k, "person": p, "pid": int(data.pids[p]), "item": j,
                   "item_id": data.items[j].item_id, "y": float(data.Y[p, j]),
                   "eta_B2": float(eta2[p, j]), "eta_B3": float(eta3[p, j]), "eta_B4": float(eta4[p, j]),
                   "n_p": float(n_p[p]), "n_j": float(z["n_j"][j]), "new_item": float(z["new_item"][j]),
                   "n_evidence": len(ev)}
            l2_rows[sample_set[(p, j)]] = row
            for v in variants:
                base_prob = float(expit(eta4[p, j])) if v == "b" else None
                jobs.append((row, v, t5_prompt(data, p, j, ev, base_prob)))
        texts = llm.complete_many([jb[2] for jb in jobs], progress(f"T5 {regime} fold {f.k}")) if jobs else []
        for (row, v, prompt), text in zip(jobs, texts):
            item = data.items[row["item"]]
            D = to_logit(parse_probs(text, len(item.options)), item.positive)
            row[f"D_{v}"] = D
            prompts.append({"regime": regime, "pid": row["pid"], "item_id": row["item_id"], "variant": f"T5{v}",
                            "D": None if np.isnan(D) else D, "prompt": prompt, "response": text})
        log(f"  {regime} fold {f.k + 1}/{len(folds)} done in {time.time() - t0:.0f}s "
            f"(LR baselines {t_lr:.0f}s, test cells {int(f.test.sum())})")

    for row in l2_rows.values():
        p, j = row["person"], row["item"]
        for m in ("LR", "LR-demo", "B0 item mean"):
            row[f"p_{m}"] = float(preds[m][p, j])
    return {"regime": regime, "scope": scope, "fold_of": fold_of, "preds": preds,
            "level2_rows": [l2_rows[i] for i in sorted(l2_rows)], "prompts": prompts, "variants": variants,
            "t3": t3_info, "seconds": time.time() - t_start,
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
    res = combine.level2(cells, variants, n_splits=n_splits)
    res["cells"] = cells
    return res
