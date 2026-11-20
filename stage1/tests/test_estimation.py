"""Offline estimators: exact Gaussian block of the Gibbs sampler, recovery on simulated SRRM data, the
Laplace batch update, the reader estimator (Buhlmann k + monotone theta + ICC), catalog profiles, the
weighted bootstrap AUC, the data loaders (when the raw files are present) and the static_scores writer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from stage1.context.demand import ReaderArtifacts, extract_demands
from stage1.context.styles import STYLES
from stage1.metrics import _WAUC
from stage1.offline.gibbs import GibbsConfig, SRRMGibbs, split_rhat
from stage1.offline.probit import probit_map
from stage1.offline.reader import estimate_catalog, estimate_reader, evaluate_reader, icc2k


def _sim_srrm(n_side=30, waves=3, p=3, seed=0, beta=(-0.2, 0.6, -0.4), sa2=0.5, sb2=0.3, sab=-0.1, rho=0.3):
    rng = np.random.default_rng(seed)
    rows, X, y = [], [], []
    pid = 0
    Sig = np.array([[sa2, sab], [sab, sb2]])
    for w in range(waves):
        men = np.arange(pid, pid + n_side)
        women = np.arange(pid + n_side, pid + 2 * n_side)
        pid += 2 * n_side
        ab = {i: rng.multivariate_normal([0, 0], Sig) for i in np.concatenate([men, women])}
        feats = {i: rng.normal(size=p - 1) for i in np.concatenate([men, women])}
        for i in men:
            for j in women:
                e = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]])
                for (u, v, eps) in ((i, j, e[0]), (j, i, e[1])):
                    x = np.concatenate([[1.0], feats[v]])            # partner features
                    z = x @ np.asarray(beta) + ab[u][0] + ab[v][1] + eps
                    rows.append((u, v, w))
                    X.append(x)
                    y.append(int(z > 0))
    r = np.array(rows)
    return np.array(X), np.array(y), r[:, 0], r[:, 1], r[:, 2]


def test_gaussian_block_is_exact():
    """The perturbation + Schur draw has mean Q^-1 r and covariance Q^-1."""
    X, y, a, b, _ = _sim_srrm(n_side=4, waves=1, seed=1)
    g = SRRMGibbs(X, y, a, b, np.array([2.0, 0.5, 0.5]), GibbsConfig())
    rng = np.random.default_rng(0)
    z = rng.normal(size=len(y))
    rho, Sigma, s2 = 0.3, np.array([[0.5, -0.1], [-0.1, 0.3]]), 1.8
    draws = np.array([g._draw_theta(z, rho, Sigma, s2, rng) for _ in range(6000)])
    # exact posterior from dense algebra
    n = len(y)
    D = g.D.toarray()
    Om = np.zeros((n, n))
    for i, j in zip(g.first, g.second):
        Om[np.ix_([i, j], [i, j])] = np.linalg.inv([[1, rho], [rho, 1]])
    for i in g.single:
        Om[i, i] = 1
    P = g.p + 2 * g.N
    L = np.zeros((P, P))
    L[np.arange(g.p), np.arange(g.p)] = 1 / (g.c2 * s2)
    Si = np.linalg.inv(Sigma)
    for k in range(g.N):
        ia, ib = g.p + k, g.p + g.N + k
        L[np.ix_([ia, ib], [ia, ib])] = Si
    Q = D.T @ Om @ D + L
    mean = np.linalg.solve(Q, D.T @ Om @ z)
    cov = np.linalg.inv(Q)
    se = np.sqrt(np.diag(cov) / len(draws))
    assert np.all(np.abs(draws.mean(0) - mean) < 5 * se)
    assert np.allclose(np.cov(draws, rowvar=False), cov, atol=0.06 * np.max(np.diag(cov)))


@pytest.mark.slow
def test_gibbs_recovers_simulated_srrm():
    X, y, a, b, _ = _sim_srrm(n_side=30, waves=3, seed=2)
    g = SRRMGibbs(X, y, a, b, np.array([5.0, 1.0, 1.0]), GibbsConfig(iterations=700, burn_in=250, chains=2, seed=3))
    res = g.run()
    s = np.sqrt(1 + 0.5 + 0.3)
    bm = res.mean("beta_marginal")
    assert np.allclose(bm, np.array([-0.2, 0.6, -0.4]) / s, atol=0.12)
    assert res.mean("rho") == pytest.approx(0.3, abs=0.2)
    assert 0.15 < res.mean("sigma_a2") < 1.2 and 0.05 < res.mean("sigma_b2") < 0.9
    assert np.nanmax(split_rhat(res.draws["beta"])) < 1.1


def test_probit_map_matches_unpenalised_limit():
    rng = np.random.default_rng(4)
    X = np.column_stack([np.ones(3000), rng.normal(size=(3000, 2))])
    y = (X @ [0.3, 0.8, -0.5] + rng.normal(size=3000) > 0).astype(int)
    b, H = probit_map(X, y, np.full(3, 100.0))
    assert np.allclose(b, [0.3, 0.8, -0.5], atol=0.1)
    assert np.all(np.linalg.eigvalsh(H) > 0)


def test_laplace_update_moves_toward_new_data():
    from stage1.features import COL, COLUMNS, Standardizer
    from stage1.offline.estimate import laplace_update
    from stage1.srrm import Identification, LockedSRRM
    ident = Identification("free", np.zeros(5))
    cols = ident.model_columns()
    active = ["intercept", "actor_male"]
    model = LockedSRRM(beta=np.zeros(len(COLUMNS)), sigma_a2=0.5, sigma_b2=0.5, sigma_ab=0.0, rho=0.0,
                       standardizer=Standardizer({}), ident=ident,
                       meta={"active_model_columns": active, "marginal_mean_model": [0.0, 0.0],
                             "marginal_cov_model": [[0.01, 0.0], [0.0, 0.01]]})
    rng = np.random.default_rng(0)
    Xr = np.zeros((2000, len(COLUMNS)))
    Xr[:, COL["intercept"]] = 1
    Xr[:, COL["actor_male"]] = rng.choice([-1.0, 1.0], 2000)
    y = (0.8 * Xr[:, COL["actor_male"]] + rng.normal(size=2000) > 0).astype(int)
    new = laplace_update(model, Xr, y)
    assert new.beta[COL["actor_male"]] > 0.1 * model.s
    assert new.meta["laplace_updates"] == 1
    assert len(cols) == len(COLUMNS)


# ---------------------------------------------------------------------------------------- reader
TEMPLATES = [("bare", "{a}", None), ("cue", "{a}, {k}", "base"), ("cue", "{a}, really {k}", "strong"),
             ("minimal_pair", "{a}, somewhat {k}", "weak"), ("minimal_pair", "{a}, no need for {k}", "neg")]
KW = {"DEP": "punctual", "INN": "creative", "COO": "teamwork", "SOC": "sociable", "STR": "pressure",
      "CUR": "curious"}


def _synthetic_labels(art, true_theta, true_k, n_raters=4, noise=0.4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    tid = 0
    for cid, e in art.catalog.contexts.items():
        prior = art.catalog.profile_of(cid)
        for kind, tmpl, cls in TEMPLATES:
            for s, kw in KW.items():
                text = tmpl.format(a=e.aliases[0], k=kw)
                true = prior.copy()
                if cls is not None:
                    i = STYLES.index(s)
                    Z = 1 / (1 + true_k)
                    true[i] = Z * true_theta[cls] + (1 - Z) * prior[i] + rng.normal(0, 0.15)
                for r in range(n_raters):
                    for i, st in enumerate(STYLES):
                        val = int(np.clip(np.round(true[i] + rng.normal(0, noise)), -3, 3))
                        rows.append((f"t{tid}", text, "en", cid, kind, f"r{r}", st, val))
                tid += 1
                if cls is None:
                    break
    return pd.DataFrame(rows, columns=["text_id", "text", "lang", "context_id", "kind", "rater_id", "style", "rating"])


def test_reader_estimation_recovers_order(art):
    theta = {"neg": -1.0, "weak": 1.0, "base": 2.0, "strong": 2.8}
    labels = _synthetic_labels(art, theta, true_k=0.5)
    est = estimate_reader(labels, art.lexicon, art.catalog)
    th = est.theta[0]
    i = {c: k for k, c in enumerate(("neg", "rev", "weak", "base", "strong"))}
    assert th[i["neg"]] <= th[i["weak"]] <= th[i["base"]] <= th[i["strong"]]
    assert th[i["strong"]] > th[i["neg"]] + 1.5
    assert est.status == "estimated" and np.all(est.k > 0)
    ev = evaluate_reader(est, labels)
    assert ev["mae"] < 0.6 and ev["basic_match_accuracy"] > 0.95
    # round trip through the hashed artifact
    from tests.conftest import LEX  # noqa: F401
    cd = extract_demands("study group, really punctual", None, est)
    assert cd.reader_status == "estimated"


def test_icc_and_catalog():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=30)
    M = truth[:, None] + rng.normal(0, 0.3, (30, 4))
    assert icc2k(M) > 0.9
    assert icc2k(rng.normal(size=(30, 4))) < 0.5
    rows = [(c, f"r{r}", s, int(np.clip(round(v + rng.normal(0, 0.5)), -3, 3)))
            for c, base in (("study_group", 1.0), ("running", -0.5), ("gym", 0.5)) for r in range(3)
            for s, v in zip(STYLES, base + rng.normal(0, 1, 12))]
    out = estimate_catalog(pd.DataFrame(rows, columns=["context_id", "rater_id", "style", "rating"]))
    assert set(out) == {"study_group", "running", "gym"} and out["gym"]["n_raters"] == 3
    assert all(-3 <= out[c]["profile"][s]["mean"] <= 3 for c in out for s in STYLES)


def test_reader_artifact_roundtrip(tmp_path, art):
    p = tmp_path / "reader.json"
    art.save(p)
    back = ReaderArtifacts.load(p, art.lexicon, art.catalog)
    assert back.hash == art.hash and np.array_equal(back.theta, art.theta)


# ---------------------------------------------------------------------------------------- metrics
def test_weighted_auc_equals_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(size=500) + y, 1)                          # ties on purpose
    W = np.vstack([np.ones(500), rng.integers(0, 3, 500)]).astype(float)
    got = _WAUC(s, y)(W)
    assert got[0] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    rep = np.repeat(np.arange(500), W[1].astype(int))
    assert got[1] == pytest.approx(roc_auc_score(y[rep], s[rep]), abs=1e-12)


# ---------------------------------------------------------------------------------------- data + db
def _have(name):
    from stage1.data.speed_dating import DEFAULT_DIR, FILES
    return (DEFAULT_DIR / FILES[name]).exists()


@pytest.mark.skipif(not _have("columbia"), reason="Columbia CSV not downloaded")
def test_columbia_loader():
    from stage1.data import load
    d = load("columbia", download=False)
    assert len(d.personas) == 551 and d.waves.tolist() == list(range(1, 22))
    assert 0.40 < d.pairs["y"].mean() < 0.44
    assert all(p.bf_source == "none" for p in d.personas)


@pytest.mark.skipif(not _have("openml"), reason="OpenML ARFF not downloaded")
def test_openml_loader_rebuilds_persons():
    from stage1.data import load
    d = load("openml", download=False)
    assert d.info["persons_rebuilt"] == 551
    assert d.info["partner_linked"] / d.info["rows"] > 0.99
    assert d.info["decision_o_consistency"] == 1.0


@pytest.mark.skipif(not _have("columbia"), reason="Columbia CSV not downloaded")
def test_static_scores_writer(tmp_path, art):
    from stage1.data import load
    from stage1.offline.estimate import Problem, SRRMSettings, _active
    from stage1.srrm import LockedSRRM
    from stage1.static import Stage1DB, run_stage1
    d = load("columbia", download=False)
    prob = Problem(d, art, SRRMSettings())
    rows = np.flatnonzero(np.isin(prob.wave, [1, 2, 3]))
    std = prob.standardizer(rows)
    Xm = prob.ident.to_model(prob.raw_design(rows, std))
    act = _active(Xm, prob.ident.model_columns())
    b, _ = probit_map(Xm[:, act], prob.y[rows], np.full(act.sum(), 0.5))
    bm = np.zeros(Xm.shape[1])
    bm[act] = b * 1.4
    model = LockedSRRM(beta=prob.ident.to_raw_coef(bm), sigma_a2=0.5, sigma_b2=0.46, sigma_ab=-0.1, rho=0.2,
                       standardizer=std, ident=prob.ident)
    db = Stage1DB(tmp_path / "s1.db")
    info = run_stage1(db, d, art, model)
    n = db.con.execute("SELECT COUNT(*), MIN(a0), MAX(a0) FROM static_scores").fetchone()
    db.close()
    assert info["static_rows"] == n[0] > 0 and 0 < n[1] <= n[2] < 1
    con = __import__("sqlite3").connect(tmp_path / "s1.db")
    bad = con.execute("SELECT COUNT(*) FROM static_scores s JOIN static_scores t ON s.pair_id = t.pair_id "
                      "AND s.chooser <> t.chooser WHERE s.m0 > MIN(s.a0, t.a0) + 1e-12").fetchone()[0]
    con.close()
    assert bad == 0                                                     # m0 <= min(a0_ij, a0_ji)
