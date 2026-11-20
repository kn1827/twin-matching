"""Unit tests of the LR + MF residual twin (src/lrmf_twin) on small synthetic data; no network, no LLM."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from lrmf_twin import baselines, combine
from lrmf_twin.data.twin2k import Item, SurveyData
from lrmf_twin.features import answer_features, centred_demographics
from lrmf_twin.llm import LLMClient, parse_list, parse_probs
from lrmf_twin.metrics import auc, lr_test, macro_auc, paired_bootstrap
from lrmf_twin.pipeline import RunConfig, run_level2, run_regime
from lrmf_twin.regimes import make_folds
from lrmf_twin.t1_base import BaseConfig, LRMFBase
from lrmf_twin.t4_retrieval import normalise, retrieve
from lrmf_twin.t5_twin import build_prompt, to_logit


def synthetic(P=160, J=30, r=2, seed=0, missing=0.1) -> tuple[SurveyData, np.ndarray]:
    """Low-rank logistic data; item embeddings carry the item factors (so text is informative)."""
    rng = np.random.default_rng(seed)
    U, V, b = rng.normal(size=(P, r)), rng.normal(size=(J, r)), rng.normal(scale=0.5, size=J)
    Y = (rng.random((P, J)) < 1 / (1 + np.exp(-(U @ V.T * 1.5 + b)))).astype(np.float32)
    Y[rng.random((P, J)) < missing] = np.nan
    items = [Item(f"Q{j}", f"Q{j}", "Block", "ordinal", f"Statement {j}", None, ["Yes", "No"], [1], 1.5)
             for j in range(J)]
    X_demo = np.eye(3, dtype=np.float32)[rng.integers(0, 3, P)]
    wave = np.where(np.isnan(Y), -1, rng.integers(0, 4, size=(P, 1))).astype(np.int16)   # person's survey year
    data = SurveyData(np.arange(1, P + 1), items, Y, np.where(np.isnan(Y), np.nan, 2 - Y).astype(np.float32),
                      X_demo, ["g=0", "g=1", "g=2"], [f"group {k}" for k in X_demo.argmax(1)], {}, wave, "synthetic")
    E = np.hstack([V, b[:, None], rng.normal(scale=0.1, size=(J, 5))]).astype(np.float32)
    return data, E


# -- metrics ------------------------------------------------------------------------------------------------
def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500).astype(float)
    p = np.round(rng.random(500), 1)                 # many ties
    assert auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert auc(y, np.full(500, 0.3)) == pytest.approx(0.5)


def test_macro_auc_is_mean_of_item_aucs():
    rng = np.random.default_rng(2)
    y, p, it = rng.integers(0, 2, 600).astype(float), rng.random(600), rng.integers(0, 4, 600)
    expected = np.mean([roc_auc_score(y[it == k], p[it == k]) for k in range(4)])
    assert macro_auc(y, p, it) == pytest.approx(expected, abs=1e-12)


def test_paired_bootstrap_same_model_is_zero():
    rng = np.random.default_rng(3)
    y, p = rng.integers(0, 2, 300).astype(float), rng.random(300)
    out = paired_bootstrap(y, p, p, rng.integers(0, 50, 300), rng.integers(0, 5, 300), B=20)
    assert out["d_auc"] == 0 and out["d_auc_ci"] == [0, 0] and out["d_logloss_ci"] == [0, 0]


def test_lr_test_detects_signal_not_noise():
    rng = np.random.default_rng(4)
    x = rng.normal(size=3000)
    y = (rng.random(3000) < 1 / (1 + np.exp(-0.5 * x))).astype(float)
    assert lr_test(y, np.zeros(3000), x)["p_value"] < 1e-6
    assert lr_test(y, 0.5 * x, rng.normal(size=3000))["p_value"] > 0.001


# -- T0 -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("regime", ["R1", "R2", "R3", "R4", "R5", "R6"])
def test_folds_are_out_of_fold(regime):
    data, _ = synthetic()
    obs = ~np.isnan(data.Y)
    folds = make_folds(obs, regime, 4, seed=0, n_sparse=40, n_known=5, k_cold=3, wave=data.wave)
    covered = np.zeros_like(obs)
    for f in folds:
        assert not (f.train & f.test).any()
        assert not (covered & f.test).any()          # every scope cell predicted once
        covered |= f.test
        if regime == "R2":
            assert not f.train[:, f.new_items].any()
        if regime == "R4":
            assert (f.train[f.sparse_persons].sum(1) <= 5).all()
        if regime == "R5":
            assert not f.train[f.sparse_persons].any()
        if regime == "R6":
            assert (f.train[:, f.new_items].sum(0) <= 3).all()
        if regime == "R3":                            # a held-out (item, wave) block is never split
            for j in range(obs.shape[1]):
                for w in np.unique(data.wave[f.test[:, j], j]):
                    assert not (f.train[:, j] & (data.wave[:, j] == w)).any()
    if regime in ("R1", "R3"):
        assert (covered == obs).all()


def test_r3_needs_waves():
    data, _ = synthetic()
    with pytest.raises(ValueError):
        make_folds(~np.isnan(data.Y), "R3", 4, wave=None)


def test_torch_lr_matches_sklearn():
    data, _ = synthetic(P=200, J=12, seed=7)
    f = make_folds(~np.isnan(data.Y), "R1", 4, seed=0)[0]
    Xd, A = centred_demographics(data.X_demo), answer_features(data, f.train)
    ref = baselines.per_item_lr(data.Y, f.train, f.test, Xd, A, C=0.05, n_jobs=1)[f.test]
    got = baselines.torch_lr(data.Y, f.train, f.test, Xd, A, C=0.05, max_iter=500, device="cpu")[f.test]
    assert np.corrcoef(ref, got)[0, 1] > 0.999 and np.max(np.abs(ref - got)) < 0.02


def test_answer_features_never_use_test_cells():
    data, _ = synthetic()
    f = make_folds(~np.isnan(data.Y), "R1", 4, seed=0)[0]
    A = answer_features(data, f.train)
    assert np.all(A[~f.train] == 0)


# -- T1 / T2 / T3 -------------------------------------------------------------------------------------------
def test_mf_beats_rank0_on_low_rank_data():
    data, _ = synthetic(P=200, J=30)
    f = make_folds(~np.isnan(data.Y), "R1", 4, seed=0)[0]
    Xd, A = centred_demographics(data.X_demo), answer_features(data, f.train)
    y = data.Y[f.test]
    ll = {}
    for r in (0, 2):
        m = LRMFBase(BaseConfig(rank=r, use_theta=False, lam_W=10, lam_U=1, lam_V=1, max_iter=150))
        p = 1 / (1 + np.exp(-m.fit(data.Y, f.train, Xd, A).logits()[f.test]))
        ll[r] = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert ll[2] < ll[0] - 0.02


def test_semantic_heads_predict_new_items():
    data, E = synthetic(P=200, J=40, seed=5)
    f = make_folds(~np.isnan(data.Y), "R2", 4, seed=0)[0]
    Xd, A = centred_demographics(data.X_demo), answer_features(data, f.train)
    y = data.Y[f.test]
    cfg = dict(rank=2, use_theta=False, lam_U=1, lam_V=1, lam_mlp=1.0, max_iter=200)
    plain = LRMFBase(BaseConfig(**cfg)).fit(data.Y, f.train, Xd, A).logits()[f.test]
    sem = LRMFBase(BaseConfig(semantic=True, sem_hidden=16, **cfg)).fit(data.Y, f.train, Xd, A, E).logits()[f.test]
    assert roc_auc_score(y, sem) > roc_auc_score(y, plain) + 0.05


def test_refit_persons_moves_toward_prior():
    data, _ = synthetic()
    f = make_folds(~np.isnan(data.Y), "R4", 4, seed=0, n_sparse=40, n_known=3)[0]
    Xd, A = centred_demographics(data.X_demo), answer_features(data, f.train)
    m = LRMFBase(BaseConfig(rank=2, use_theta=False, max_iter=100)).fit(data.Y, f.train, Xd, A)
    p = int(f.sparse_persons[0])
    before = m.logits()[p]
    items = np.arange(5)
    after = m.refit_persons(np.array([p]), {p: (items, np.full(5, 0.98, dtype=np.float32))})[0]
    assert np.mean(after[items]) > np.mean(before[items])


# -- T4 / T5 / LLM ------------------------------------------------------------------------------------------
def test_retrieval_excludes_target_and_test_cells():
    data, E = synthetic()
    f = make_folds(~np.isnan(data.Y), "R1", 4, seed=0)[0]
    p, j = map(int, np.argwhere(f.test)[0])
    ev = retrieve(p, j, f.train, normalise(E), k=8)
    assert j not in ev and all(f.train[p, k] for k in ev)


def test_parse_and_logit():
    assert parse_probs('{"p": [1, 1, 2]}', 3) == [0.25, 0.25, 0.5]
    assert parse_probs('noise {"probabilities": {"1": 0.2, "2": 0.8}} tail', 2) == [0.2, 0.8]
    assert parse_probs('{"p": [0.5]}', 2) is None and parse_probs("not json", 2) is None
    assert parse_list('{"p": [20, 90, 50]}', 3) == [0.2, 0.9, 0.5]
    assert parse_list('{"p": [0.3, 1.4]}', 2) == [0.3, 1.0]
    assert np.isnan(to_logit(None, [1]))
    assert to_logit([0.9, 0.1], [1]) == pytest.approx(np.log(0.9 / 0.1))


def test_prompt_hides_target_answer():
    data, E = synthetic()
    p, j = 0, int(np.flatnonzero(~np.isnan(data.Y[0]))[0])
    prompt = build_prompt(data, p, j, [], base_prob=0.62)
    assert "62%" in prompt and "exactly 2 numbers" in prompt and "->" not in prompt


# -- T6-T9 + pipeline ---------------------------------------------------------------------------------------
def test_super_learner_on_simplex():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 400).astype(float)
    P = np.column_stack([np.clip(y * 0.6 + 0.2 + rng.normal(0, 0.1, 400), 0.01, 0.99), rng.random(400)])
    a = combine.fit_super_learner(P, y)
    assert a.min() >= 0 and a.sum() == pytest.approx(1.0) and a[0] > 0.8


def test_gate_is_monotone_in_data():
    th = np.array([0.3, -1.0, 0.5, 0.2])
    small = combine.gate(th, combine.gate_features([5], [100], [0]))
    big = combine.gate(th, combine.gate_features([500], [100], [0]))
    assert small[0] > big[0]


def test_end_to_end_with_mock_llm(tmp_path):
    data, E = synthetic(P=120, J=20)
    llm = LLMClient("mock", cache_path=tmp_path / "cache.sqlite")
    cfg = RunConfig(n_folds=3, n_sparse=12, n_known=4, k_cold=3, llm_cells=12, n_jobs=1, lr_engine="torch",
                    base=BaseConfig(rank=2, max_iter=40, sem_hidden=8))
    rows = []
    for rg in ("R1", "R2", "R3", "R4", "R5", "R6"):
        res = run_regime(data, E, rg, cfg, llm, log=lambda *_: None)
        scope = res["scope"]
        for m in ("LR", "B2 LR+MF (T1)", "B3 +T2"):
            assert np.isfinite(res["preds"][m][scope]).all()
        rows += res["level2_rows"]
    out = run_level2(rows, ["a", "b"], n_splits=3)
    for name in ("B5", "B6", "B7", "B8", "B9 super learner (+T9)"):
        p = out["pred"][name]
        assert p.shape == (len(rows),) and np.all((p > 0) & (p < 1))


# -- v0.3: learned hyper-parameters, cross-fitted weights, central twin ------------------------------------
def test_b2_warm_start_begins_at_lr_and_does_not_lose():
    from lrmf_twin.tuning import fit_b2, fit_lr
    data, _ = synthetic(P=150, J=20, seed=3)
    f = make_folds(~np.isnan(data.Y), "R1", 4, seed=0)[0]
    Xd, A = centred_demographics(data.X_demo), answer_features(data, f.train)
    base = BaseConfig(max_iter=300, device="cpu")
    lr = fit_lr(data.Y, f.train, Xd, A, 0.05, base)
    b2 = fit_b2(data.Y, f.train, Xd, A, 0.05, {"lam": 1e6, "rank": 2}, base, lr)   # MF frozen by a huge penalty
    assert np.max(np.abs(b2.logits() - lr.logits())) < 0.05
    assert fit_b2(data.Y, f.train, Xd, A, 0.05, None, base, lr) is lr                # MF rejected -> B2 = LR


def test_tune_regime_returns_learned_values():
    from lrmf_twin.tuning import tune_regime
    data, E = synthetic(P=150, J=20, seed=4)
    f = make_folds(~np.isnan(data.Y), "R1", 4, seed=0)[0]
    out = tune_regime(data, E, "R1", f.train, centred_demographics(data.X_demo), BaseConfig(max_iter=100, device="cpu"),
                      seed=0, n_sparse=20, n_known=4, k_cold=3, log=lambda *_: None)
    assert out["lr_C"] in (0.002, 0.005, 0.02, 0.05) and len(out["table"]) >= 4
    assert out["mf"] is None or out["mf_val_logloss"] < out["lr_val_logloss"]


def test_crossfit_select_prefers_informative_candidate():
    from lrmf_twin.pipeline import crossfit_select
    rng = np.random.default_rng(0)
    z = rng.normal(size=4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-z))).astype(float)
    fold = np.repeat(np.arange(4), 1000)
    out, chosen = crossfit_select({"good": z, "noise": rng.normal(size=4000), "zero": np.zeros(4000)}, y, fold, rng)
    assert set(chosen.values()) == {"good"} and np.allclose(out, z)


def test_item_generator_shrinkage():
    from lrmf_twin.t2_encoder import ItemGenerator, shrink_item_params
    rng = np.random.default_rng(1)
    E = rng.normal(size=(60, 5))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    target = np.column_stack([E[:, 0] * 2, E[:, 1]])
    n_j = np.full(60, 100.0)
    gen = ItemGenerator(seed=0).fit(E, target, n_j)
    assert np.corrcoef(gen.predict(E)[:, 0], target[:, 0])[0, 1] > 0.8
    free = np.zeros_like(target)
    warm = shrink_item_params(free, target, np.full(60, 1e6), kappa=10.0)
    cold = shrink_item_params(free, target, np.zeros(60), kappa=10.0)
    assert np.allclose(warm, free, atol=1e-3) and np.allclose(cold, target)


def test_level2_offset_picks_best_base_and_lists_lr():
    rng = np.random.default_rng(2)
    n = 400
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    cells = {"y": y, "person": np.arange(n), "p_LR": 1 / (1 + np.exp(-z)), "p_B0": np.full(n, y.mean()),
             "eta_B2": rng.normal(size=n) * 3, "eta_B3": rng.normal(size=n) * 3, "eta_B4": rng.normal(size=n) * 3,
             "n_p": np.full(n, 50.0), "n_j": np.full(n, 100.0), "new_item": np.zeros(n), "D_a": rng.normal(size=n)}
    out = combine.level2(cells, ["a"], n_splits=4)
    assert set(out["offset_by_fold"]) == {"LR"} and "LR" in out["sl_candidates"]


def test_central_twin_tiny_model_runs():
    pytest.importorskip("peft")
    from lrmf_twin.twin_ft import FTConfig, run_twin_ft
    data, E = synthetic(P=60, J=12, seed=5)
    obs = ~np.isnan(data.Y)
    f = make_folds(obs, "R1", 4, seed=0)[0]
    U, V = np.random.default_rng(0).normal(size=(60, 2)), np.random.default_rng(1).normal(size=(12, 2))
    preds = {m: np.full(data.shape, 0.5, dtype=np.float32) for m in ("LR", "B0 item mean", "B2 LR+MF (T1)")}
    ctx = {"train": f.train, "test": f.test, "U": U, "V": V, "preds": preds}
    tuned = {"lr_C": 0.05, "mf": {"lam": 10.0, "rank": 2}}
    cfg = FTConfig(model="tiny-random", max_steps=6, eval_every=3, batch_size=4, n_train=40, n_val=12, n_eval=20,
                   max_len=128)
    res = run_twin_ft(data, E, "R1", ctx, tuned, BaseConfig(max_iter=50, device="cpu"), cfg,
                      n_sparse=10, n_known=3, k_cold=3, seed=0, log=lambda *_: None)
    z = np.array([r["z_twin"] for r in res["rows"]])
    assert len(z) == 20 and np.isfinite(z).all() and res["fit"]["best_step"] > 0
