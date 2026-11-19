"""Offline estimators on synthetic data with known truth: the Gibbs sampler of (10), the reader estimator
(theta by ordered least squares, k by Bühlmann), ICC(2,k); and the Speed Dating loaders when the raw
files are present."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnwt.config import PROJECT_ROOT
from gnwt.stage1.context.demand import ReaderArtifacts, extract_demands
from gnwt.stage1.context.styles import STYLES
from gnwt.stage1.offline.reader import buhlmann_k, estimate_reader, icc2k
from gnwt.stage1.offline.srrm_fit import gibbs_srrm, split_rhat


def test_gibbs_recovers_variance_components():
    rng = np.random.default_rng(11)
    W, K = 8, 18                                # 8 waves, 18 men x 18 women
    sa2, sb2, sab, rho = 0.6, 0.4, -0.15, 0.25
    beta = np.array([-0.3, 0.4, -0.2])
    S = np.array([[sa2, sab], [sab, sb2]])
    send, recv, X = [], [], []
    pid = 0
    ab = []
    for _ in range(W):
        men = list(range(pid, pid + K))
        women = list(range(pid + K, pid + 2 * K))
        pid += 2 * K
        for a in men:
            for b in women:
                for s_, r_ in ((a, b), (b, a)):
                    send.append(s_)
                    recv.append(r_)
                    X.append([1.0, rng.normal(), rng.normal()])
    N = pid
    ab = rng.multivariate_normal([0, 0], S, N)
    send, recv, X = np.array(send), np.array(recv), np.array(X)
    rev = np.arange(len(send)) ^ 1               # rows come in (i->j, j->i) pairs
    eps = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], len(send) // 2).reshape(-1)
    z = X @ beta + ab[send, 0] + ab[recv, 1] + eps
    y = (z > 0).astype(float)
    res = gibbs_srrm(X, y, send, recv, rev, N, np.array([1.0, 1.0, 1.0]), iters=1500, burn=500, thin=2,
                     seed=3, chains=2)
    b, a2, b2, ab_, r = res.posterior_mean()
    assert np.allclose(b, beta, atol=0.12), b
    assert abs(a2 - sa2) < 0.2 and abs(b2 - sb2) < 0.15 and abs(ab_ - sab) < 0.12, (a2, b2, ab_)
    assert abs(r - rho) < 0.12, r
    assert res.rhat["beta_max"] < 1.1


def test_split_rhat_detects_disagreement():
    rng = np.random.default_rng(0)
    same = rng.normal(size=(4, 500))
    assert split_rhat(same) < 1.01
    shifted = same + np.array([0, 0, 0, 3.0])[:, None]
    assert split_rhat(shifted) > 1.2


def test_icc2k_perfect_and_noise():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=40)
    assert icc2k(np.column_stack([truth] * 3)) == pytest.approx(1.0)
    noisy = np.column_stack([truth + rng.normal(0, 3, 40) for _ in range(3)])
    assert icc2k(noisy) < 0.7


def test_estimate_reader_recovers_theta(lexicon, catalog_illustrative):
    """Labels generated from (11b) with known theta and k; the estimator must recover them."""
    theta = {"neg": -0.5, "rev": -1.5, "weak": 1.0, "base": 2.0, "strong": 2.7}
    k_true = 0.5
    art = ReaderArtifacts(lexicon, catalog_illustrative, theta, np.full(len(STYLES), k_true), None, 0.7, "illustrative")
    rng = np.random.default_rng(5)
    cues = ["kỷ luật", "sáng tạo", "hợp tác", "cẩn thận", "bình tĩnh", "lắng nghe", "giao lưu", "dẫn dắt",
            "nghiên cứu", "linh hoạt", "áp lực", "phấn đấu"]
    mods = ["rất cần {}", "cần {}", "hơi cần {}", "không cần {}", "{}"]
    revs = ["tuỳ hứng", "cạnh tranh", "nóng nảy", "qua loa"]
    rows = []
    tid = 0
    for ctx in ("học nhóm", "hẹn hò", "ôn thi", ""):
        for cue in cues:
            for mtpl in mods:
                txt = (f"{ctx}, " if ctx else "") + mtpl.format(cue)
                cd = extract_demands(txt, None, art)
                for r in range(3):
                    for s_i, s in enumerate(STYLES):
                        rows.append({"text_id": f"t{tid}", "text": txt, "context_id": "x", "rater_id": f"r{r}",
                                     "style": s, "rating": cd.D_style[s_i] + rng.normal(0, 0.05)})
                tid += 1
        for rv in revs:
            txt = (f"{ctx}, " if ctx else "") + rv
            cd = extract_demands(txt, None, art)
            for r in range(3):
                for s_i, s in enumerate(STYLES):
                    rows.append({"text_id": f"t{tid}", "text": txt, "context_id": "x", "rater_id": f"r{r}",
                                 "style": s, "rating": cd.D_style[s_i] + rng.normal(0, 0.05)})
            tid += 1
    labels = pd.DataFrame(rows)
    out = estimate_reader(labels, lexicon, catalog_illustrative, k_fixed=k_true)
    th = out["theta"]
    assert th["neg"] <= th["weak"] <= th["base"] <= th["strong"]
    for c, v in theta.items():
        assert th[c] == pytest.approx(v, abs=0.35), (c, th[c])
    assert out["meta"]["cv_mae"]["X2_full"] <= out["meta"]["cv_mae"]["X0_catalog"]


def test_buhlmann_k_recovers_variance_ratio():
    """Bühlmann world: true level mu ~ N(prior, tau^2); n hits of evidence each N(mu, sigma^2)."""
    rng = np.random.default_rng(2)
    m, tau, sigma = 20000, 0.8, 1.2
    n = rng.integers(1, 4, m).astype(float)
    prior = rng.normal(1.0, 0.5, m)
    mu = prior + tau * rng.standard_normal(m)
    ybar = mu + sigma / np.sqrt(n) * rng.standard_normal(m)
    k = buhlmann_k(ybar, n, mu, prior)
    assert k == pytest.approx(sigma ** 2 / tau ** 2, rel=0.05)


RAW = PROJECT_ROOT / "data" / "raw"


@pytest.mark.skipif(not (RAW / "SpeedDating.arff").exists() or not (RAW / "Speed Dating Data.csv").exists(),
                    reason="raw Speed Dating files not downloaded")
def test_openml_reconstruction_matches_columbia():
    from gnwt.data.speed_dating import load_market

    c = load_market("columbia")
    o = load_market("speed_dating")
    assert o.report["persons"] == c.report["persons"]
    assert o.report["rows"] >= 0.99 * c.report["rows"]
    # every reconstructed dyad has its reverse row with the swapped decisions
    key = {(a, b): y for a, b, y in zip(o.rows["chooser"], o.rows["chosen"], o.rows["y"])}
    both = [(a, b) for (a, b) in key if (b, a) in key]
    assert len(both) >= 0.99 * len(key)
    # same decision rate and the same wave sizes as the original file
    assert abs(o.rows["y"].mean() - c.rows["y"].mean()) < 0.002
    assert (o.persons.groupby("wave").size() == c.persons.groupby("wave").size()).all()
