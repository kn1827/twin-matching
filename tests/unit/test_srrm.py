"""U1-U4, U9, U10 and the worked example of stage1.md section 8."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr

from gnwt.stage1.features import (PERSON_BLOCKS, DesignSpec, PersonTable, Standardizer, build_design,
                                  delta_basis_for, orthonormal_complement)
from gnwt.stage1.fit import fit_score
from gnwt.stage1.offline.srrm_fit import map_probit
from gnwt.stage1.srrm import LockedSRRM, a0, bvn_cdf, eta_blocks, m0

finite = st.floats(-6, 6, allow_nan=False)


@settings(max_examples=300, deadline=None)
@given(finite, finite, st.floats(0.0, 3.0), st.floats(0.0, 3.0), st.floats(-0.9, 0.9), st.floats(-0.9, 0.9))
def test_u1_bounds(eta_ij, eta_ji, sa2, sb2, cab, rho):
    sab = cab * np.sqrt(sa2 * sb2)
    s = np.sqrt(1 + sa2 + sb2)
    r = (2 * sab + rho) / (1 + sa2 + sb2)
    p1, p2 = a0(eta_ij, s), a0(eta_ji, s)
    assert 0.0 < p1 < 1.0 and 0.0 < p2 < 1.0
    mm = m0(eta_ij, eta_ji, r, s)
    assert mm <= min(p1, p2) + 1e-12
    assert m0(eta_ij, eta_ji, 0.0, s) == pytest.approx(p1 * p2, abs=1e-12)


def test_u2_monte_carlo():
    rng = np.random.default_rng(7)
    sa2, sb2, sab, rho = 0.77, 0.40, -0.22, 0.10
    eta_ij, eta_ji = 0.22, 0.004
    n = 2_000_000
    S = np.array([[sa2, sab], [sab, sb2]])
    ab_i = rng.multivariate_normal([0, 0], S, n)
    ab_j = rng.multivariate_normal([0, 0], S, n)
    eps = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    z_ij = eta_ij + ab_i[:, 0] + ab_j[:, 1] + eps[:, 0]
    z_ji = eta_ji + ab_j[:, 0] + ab_i[:, 1] + eps[:, 1]
    s = np.sqrt(1 + sa2 + sb2)
    r = (2 * sab + rho) / (1 + sa2 + sb2)
    for emp, th in (((z_ij > 0).mean(), a0(eta_ij, s)), (((z_ij > 0) & (z_ji > 0)).mean(), m0(eta_ij, eta_ji, r, s))):
        se = np.sqrt(th * (1 - th) / n)
        assert abs(emp - th) < 3 * se


def test_spec_example_section_8():
    s = np.sqrt(1 + 0.77 + 0.40)
    r = (2 * -0.22 + 0.10) / (1 + 0.77 + 0.40)
    assert s == pytest.approx(1.473, abs=5e-4) and r == pytest.approx(-0.157, abs=5e-4)
    D = np.array([0.2037, 0.5889, 0.1667, 0.5, 0.3333])
    fit, parts = fit_score(np.array([0.5, 1.0, -0.5, 0.2, -0.3]), np.ones(5, bool), D)
    assert fit == pytest.approx(0.61, abs=0.005)
    eta_ij = -0.10 + 0.28 * fit + 0.15
    assert a0(eta_ij, s) == pytest.approx(0.559, abs=0.001)
    fit_r, _ = fit_score(np.array([-0.2, 0.3, 0.8, -0.4, 0.5]), np.ones(5, bool), np.array([0.2778, 0.5, 0.1667, 0.5, 0.3333]))
    eta_ji = -0.20 + 0.28 * fit_r + 0.15
    assert a0(eta_ji, s) == pytest.approx(0.501, abs=0.001)
    assert m0(eta_ij, eta_ji, r, s) == pytest.approx(0.256, abs=0.001)


@settings(max_examples=200, deadline=None)
@given(st.lists(finite, min_size=5, max_size=5), st.lists(finite, min_size=5, max_size=5),
       st.lists(st.floats(-1, 1), min_size=5, max_size=5), st.floats(-3, 3))
def test_u4_fit_linear(p1, p2, D, c):
    p1, p2, D = np.array(p1), np.array(p2), np.array(D)
    m = np.ones(5, bool)
    f1, parts = fit_score(p1, m, D)
    f2, _ = fit_score(p2, m, D)
    f12, _ = fit_score(c * p1 + p2, m, D)
    assert abs(f12 - (c * f1 + f2)) < 1e-9 * (1 + abs(f12))
    assert abs(parts.sum() - f1) < 1e-12 * (1 + abs(f1))
    assert fit_score(p1, np.zeros(5, bool), D)[0] == 0.0


# ------------------------------------------------------------------------------- synthetic market

def _synthetic(n_per_side: int = 30, waves: int = 4, seed: int = 0, D=None):  # noqa: ANN001
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows, persons = [], []
    for w in range(waves):
        men = [f"m{w}_{k}" for k in range(n_per_side)]
        women = [f"f{w}_{k}" for k in range(n_per_side)]
        for pid, g in [(p, "male") for p in men] + [(p, "female") for p in women]:
            persons.append({"person_id": pid, "wave": w, "gender": g, "age": rng.integers(20, 35),
                            "field": rng.choice(["a", "b", "c"]), "goal": rng.choice(["x", "y"]),
                            **{f"i{k}": rng.integers(1, 11) for k in range(17)},
                            **{s: rng.integers(3, 11) for s in ("self_sinc", "self_intel", "self_fun", "self_amb")}})
        for a in men:
            for b in women:
                rows += [{"chooser": a, "chosen": b, "wave": w}, {"chooser": b, "chosen": a, "wave": w}]
    from gnwt.persona import INTERESTS

    P = pd.DataFrame(persons).rename(columns={f"i{k}": INTERESTS[k] for k in range(17)}).set_index("person_id")
    bf = pd.DataFrame(rng.random((len(P), 5)), index=P.index, columns=list("OCEAN"))
    T = PersonTable.from_frame(P, bf, None, default_context="speed_dating")
    if D is not None:
        T.D_tilde[:] = D
    R = pd.DataFrame(rows)
    return T, R


def test_u9_orthogonal_delta_equals_free_partner_bf():
    D = np.array([0.2, 0.6, 0.15, 0.5, 0.3])
    T, R = _synthetic(D=D)
    i = np.array([T.index[c] for c in R["chooser"]])
    j = np.array([T.index[c] for c in R["chosen"]])
    std = Standardizer.fit(T, np.arange(len(T)), {})
    rng = np.random.default_rng(3)
    basis = delta_basis_for(T, np.unique(i))
    Q = np.array(basis["speed_dating"])
    assert np.allclose(Q.T @ D, 0) and np.allclose(Q.T @ Q, np.eye(4))
    spec_o = DesignSpec(dyad=(), actor=("bf",), partner=("bf",))
    spec_free = DesignSpec(dyad=(), actor=("bf",), partner=("bf",), delta_mode="free", use_fit=False)
    Xo = build_design(T, i, j, std, spec_o, basis)
    Xf = build_design(T, i, j, std, spec_free, basis)
    keep_o = Xo.X.std(0) > 1e-12
    keep_o[0] = True
    keep_f = Xf.X.std(0) > 1e-12
    keep_f[0] = True
    beta_true = rng.normal(0, 0.3, keep_f.sum())
    y = (Xf.X[:, keep_f] @ beta_true + rng.standard_normal(len(i)) > 0).astype(float)
    big = np.full(keep_o.sum(), 1e6)
    bo = map_probit(Xo.X[:, keep_o], y, big)
    bf = map_probit(Xf.X[:, keep_f], y, np.full(keep_f.sum(), 1e6))
    po, pf = ndtr(Xo.X[:, keep_o] @ bo), ndtr(Xf.X[:, keep_f] @ bf)
    assert np.max(np.abs(po - pf)) < 1e-6
    # <delta, D> = 0 by construction: the residual columns are P~ Q
    np.testing.assert_allclose(orthonormal_complement(D).T @ D, 0, atol=1e-12)


def _locked_from(design, beta, sa2=0.5, sb2=0.4, sab=-0.1, rho=0.1, spec=None, std=None):  # noqa: ANN001
    return LockedSRRM(design.names, np.asarray(beta, float), design.block_of, sa2, sb2, sab, rho, std, spec, {}, {})


def test_u10_all_masks_and_u3_actor_invariance():
    T, R = _synthetic(n_per_side=6, waves=1)
    i = np.array([T.index[c] for c in R["chooser"]])
    j = np.array([T.index[c] for c in R["chosen"]])
    std = Standardizer.fit(T, np.arange(len(T)), {})
    spec = DesignSpec()
    full = build_design(T, i, j, std, spec, {})
    rng = np.random.default_rng(1)
    beta = rng.normal(0, 0.3, len(full.names))
    model = _locked_from(full, beta, spec=spec, std=std)
    nb = len(PERSON_BLOCKS)
    for bits in itertools.product([False, True], repeat=nb):
        pm = np.tile(np.array(bits), (len(T), 1))
        des = build_design(T, i, j, std, spec, {}, pmask=pm)
        bl = eta_blocks(des, model)
        p = a0(bl.eta, model.s)
        assert np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1))
        if not bits[PERSON_BLOCKS.index("bf")]:
            # masked Big Five: the actor block reduces to omega_bf_i, the residual block to omega_bf_j
            assert np.allclose(bl.etaP_actor, model.coef("miss_bf_i"))
            assert np.allclose(bl.etaP_resid, model.coef("miss_bf_j"))
            assert np.allclose(bl.etaP_fit, 0.0)
    # U3: changing beta_A and the actor part of eta0 does not change the ranking of i's candidates
    actor_cols = [k for k, n in enumerate(full.names) if n.startswith(("bf_i", "miss_bf_i", "self_i", "age_i",
                                                                         "miss_age_i", "miss_self_i", "gender_i"))]
    beta2 = beta.copy()
    beta2[actor_cols] += rng.normal(0, 1, len(actor_cols))
    e1 = eta_blocks(full, model).eta
    e2 = eta_blocks(full, _locked_from(full, beta2, spec=spec, std=std)).eta
    for c in np.unique(i):
        sel = i == c
        assert np.array_equal(np.argsort(e1[sel], kind="stable"), np.argsort(e2[sel], kind="stable"))


def test_bvn_matches_scipy():
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(0)
    for r in (-0.95, -0.5, -0.1, 0.0, 0.3, 0.8, 0.97):
        h, k = rng.normal(size=5) * 1.5, rng.normal(size=5) * 1.5
        ref = [multivariate_normal([0, 0], [[1, r], [r, 1]]).cdf([a, b]) for a, b in zip(h, k)]
        assert np.max(np.abs(bvn_cdf(h, k, r) - ref)) < 1e-7
