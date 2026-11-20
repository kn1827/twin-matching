"""SRRM core: a0 / m0 properties (U1, U2), ranking invariance (U3), fit (U4), single-context
identification (U9), every missing-block mask (U10), and the no-LLM static scan (U11)."""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import multivariate_normal

from stage1.features import (ACTOR_ONLY, BLOCKS, BLOCK_COLUMNS, COL, COLUMNS, Persona, PersonTable, Standardizer,
                             design_from_raw, pair_raw)
from stage1.fit import fit_score
from stage1.offline.probit import probit_map
from stage1.srrm import Identification, LockedSRRM, a0, bvn_cdf, eta_blocks, m0

SRC = Path(__file__).resolve().parents[1] / "src" / "stage1"


@settings(max_examples=300, deadline=None)
@given(st.floats(-8, 8), st.floats(-8, 8), st.floats(-0.95, 0.95), st.floats(1.0, 3.0))
def test_u1_a0_m0_bounds(e1, e2, r, s):
    p1, p2 = float(a0(e1, s)), float(a0(e2, s))
    assert 0.0 < p1 < 1.0 and 0.0 < p2 < 1.0
    m = float(m0(e1, e2, r, s))
    assert 0.0 <= m <= min(p1, p2) + 1e-12
    assert float(m0(e1, e2, 0.0, s)) == pytest.approx(p1 * p2, rel=1e-12, abs=1e-300)


def test_bvn_matches_scipy():
    rng = np.random.default_rng(1)
    for _ in range(40):
        h, k = rng.normal(0, 1.5, 2)
        r = rng.uniform(-0.9, 0.9)
        ref = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([h, k])
        assert float(bvn_cdf(h, k, r)) == pytest.approx(ref, abs=2e-6)
    assert float(bvn_cdf(0.0, 0.0, 0.5)) == pytest.approx(0.25 + np.arcsin(0.5) / (2 * np.pi), abs=1e-12)


def test_u2_monte_carlo():
    """(12)-(13) against 2e6 draws of the generative model (10) for two new people."""
    rng = np.random.default_rng(7)
    sa2, sb2, sab, rho = 0.77, 0.40, -0.22, 0.10
    eta_ij, eta_ji = 0.22, 0.004
    n = 2_000_000
    Sig = np.array([[sa2, sab], [sab, sb2]])
    ab_i = rng.multivariate_normal([0, 0], Sig, n)
    ab_j = rng.multivariate_normal([0, 0], Sig, n)
    eps = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    z_ij = eta_ij + ab_i[:, 0] + ab_j[:, 1] + eps[:, 0]
    z_ji = eta_ji + ab_j[:, 0] + ab_i[:, 1] + eps[:, 1]
    s = np.sqrt(1 + sa2 + sb2)
    r = (2 * sab + rho) / (1 + sa2 + sb2)
    for emp, th in (((z_ij > 0).mean(), float(a0(eta_ij, s))), (((z_ij > 0) & (z_ji > 0)).mean(),
                                                                  float(m0(eta_ij, eta_ji, r, s)))):
        se = np.sqrt(th * (1 - th) / n)
        assert abs(emp - th) < 3 * se
    # the worked example of spec section 8 (illustrative numbers)
    assert s == pytest.approx(1.473, abs=1e-3) and r == pytest.approx(-0.157, abs=1e-3)
    assert float(a0(0.220, s)) == pytest.approx(0.559, abs=1e-3)
    assert float(m0(0.220, 0.004, r, s)) == pytest.approx(0.256, abs=1e-3)


def _people(n, rng, bf=True):
    out = []
    for k in range(n):
        out.append(Persona(pid=str(k), wave=1, gender=k % 2, age=float(rng.integers(20, 35)),
                           field=str(rng.integers(0, 4)), goal=str(rng.integers(0, 3)),
                           interests=rng.integers(1, 11, 17).astype(float),
                           self_ratings=rng.integers(3, 11, 4).astype(float),
                           bf=rng.uniform(0, 1, 5) if bf else np.full(5, np.nan)))
    return out


def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    t = PersonTable.from_personas(_people(n, rng))
    men, women = np.arange(0, n, 2), np.arange(1, n, 2)
    a = np.concatenate([np.repeat(men, len(women)), np.repeat(women, len(men))])
    p = np.concatenate([np.tile(women, len(men)), np.tile(men, len(women))])
    raw = pair_raw(t, a, p)
    std = Standardizer.fit(t, a, p, raw=raw)
    return t, a, p, raw, std, rng


def _model(beta, D=None, mode="orthogonal"):
    std = Standardizer({})
    return LockedSRRM(beta=beta, sigma_a2=0.5, sigma_b2=0.4, sigma_ab=-0.1, rho=0.2, standardizer=std,
                      ident=Identification(mode, np.zeros(5) if D is None else D))


def test_u3_actor_terms_do_not_change_ranking():
    t, a, p, raw, std, rng = _toy()
    X = design_from_raw(raw, np.array([0.1, 0.4, 0.2, 0.3, 0.1]), std).X
    beta = rng.normal(0, 0.3, len(COLUMNS))
    beta2 = beta.copy()
    for c in ACTOR_ONLY:
        beta2[COL[c]] += rng.normal(0, 1)
    e1, e2 = eta_blocks(X, _model(beta)).eta, eta_blocks(X, _model(beta2)).eta
    for i in np.unique(a):
        m = a == i
        assert np.array_equal(np.argsort(e1[m], kind="stable"), np.argsort(e2[m], kind="stable"))


def test_u4_fit_linear_masked_and_parts():
    rng = np.random.default_rng(3)
    D = rng.uniform(-1, 1, 5)
    P1, P2 = rng.normal(size=5), rng.normal(size=5)
    ok = np.ones(5, bool)
    f1, parts = fit_score(P1, ok, D)
    f2, _ = fit_score(P2, ok, D)
    f12, _ = fit_score(2 * P1 - 3 * P2, ok, D)
    assert abs(f12 - (2 * f1 - 3 * f2)) < 1e-12
    assert abs(parts.sum() - f1) < 1e-12
    assert fit_score(P1, np.zeros(5, bool), D)[0] == 0.0


def test_u9_single_context_identification():
    """delta is orthogonal to D~ and predictions equal the free partner-Big-Five model."""
    t, a, p, raw, std, rng = _toy(60, seed=4)
    D = np.array([0.2, 0.59, 0.17, 0.5, 0.33])
    X = design_from_raw(raw, D, std).X
    true = np.zeros(len(COLUMNS))
    true[COL["intercept"]] = -0.3
    for d in ("O", "C", "E", "A", "ES"):
        true[COL[f"partner_bf_{d}"]] = rng.normal(0, 0.4)
    y = (X @ true + rng.normal(size=len(X)) > 0).astype(int)
    ident = Identification("orthogonal", D)
    Xm = ident.to_model(X)
    keep = Xm.std(0) > 1e-12
    keep[ident.model_columns().index("intercept")] = True
    b, _ = probit_map(Xm[:, keep], y, ident.prior_sd(0.5, 5.0)[keep])
    bm = np.zeros(Xm.shape[1])
    bm[keep] = b
    raw_b = ident.to_raw_coef(bm)
    delta = raw_b[[COL[f"partner_bf_{d}"] for d in ("O", "C", "E", "A", "ES")]]
    assert abs(delta @ D) < 1e-10
    # free model: partner Big Five columns with the same isotropic prior, no fit column
    cols = [c for c in COLUMNS if c != "fit"]
    Xf = X[:, [COL[c] for c in cols]]
    kf = Xf.std(0) > 1e-12
    kf[cols.index("intercept")] = True
    sd = np.where(np.array(cols) == "intercept", 5.0, 0.5)[kf]
    bf, _ = probit_map(Xf[:, kf], y, sd)
    assert np.max(np.abs(Xm[:, keep] @ b - Xf[:, kf] @ bf)) < 1e-8


def test_u10_every_mask_is_finite_and_equals_omega():
    t, a, p, raw, std, rng = _toy(20, seed=5)
    beta = rng.normal(0, 0.5, len(COLUMNS))
    model = _model(beta)
    D = np.array([0.1, 0.5, 0.2, 0.4, 0.3])
    for bits in itertools.product([False, True], repeat=len(BLOCKS)):
        hide = {b: np.full(len(a), h) for b, h in zip(BLOCKS, bits)}
        X = design_from_raw(raw, D, std, hide).X
        eta = eta_blocks(X, model).eta
        assert np.all(np.isfinite(a0(eta, model.s)))
        for b, h in zip(BLOCKS, bits):
            contrib = X[:, [COL[c] for c in BLOCK_COLUMNS[b]]] @ beta[[COL[c] for c in BLOCK_COLUMNS[b]]]
            if h:
                assert np.all(contrib == 0) and np.all(X[:, COL[f"miss_{b}"]] == 1)


def test_u11_static_scan():
    banned = re.compile(r"^\s*(from|import)\s+(llm|gnwt\.llm|openai|anthropic|httpx|requests|transformers|"
                        r"sentence_transformers|ollama)\b", re.M)
    for f in SRC.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        if f.name == "fallback_embed.py":
            # allowed only lazily, inside a function (never at module import)
            assert not re.search(r"^(from|import)\s+sentence_transformers", text, re.M)
            continue
        assert not banned.search(text), f
        assert "data/sealed" not in text and "sealed" not in text.lower(), f
    demand = (SRC / "context" / "demand.py").read_text(encoding="utf-8")
    assert "fallback_embed" in demand and "if m.how == \"generic\" and art.fallback_embedding" in demand
