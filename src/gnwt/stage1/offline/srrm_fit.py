"""Offline estimation of the probit SRRM (stage1.md, section 7, steps O7-O8). Never run at inference.

Procedure (locked; the numbers are what it computes):

1. Features are standardised within the training fold (persons of the training waves).
2. Prior scale (grid `priors.scale_grid`) and block-masking rate (grid `block_masking_rate_grid`) are
   chosen by nested leave-one-wave-out CV on log-loss, 1-SE rule preferring the stronger shrinkage
   (then the lower masking rate). The inner fits use the Laplace/MAP approximation of the marginal
   probit P(y=1) = Phi(x'beta / s) with s taken from a short pilot Gibbs run on the same training
   waves; this keeps 21 x 20 x 20 inner fits tractable (see CHANGELOG).
3. With the chosen configuration the full model (10) is sampled with Hoff's Gibbs sampler
   (Hoff 2021, section 6; the `amen` decorrelation trick for the dyadic correlation rho):
   z | . truncated normal per dyad; beta, (a, b) | . conjugate normal after the transform
   Z~ = td Z + to Z'; Sigma_ab | . LKJ(2) + half-Cauchy MH (or inverse Wishart); rho | . MH.
   Several chains, split-R^ reported for beta, Sigma_ab and rho.
4. Cross-fit: every development wave receives a0 from the model fitted on the other waves (I1).

Coefficient priors are N(0, scale^2) centred at 0, the intercept N(0, intercept_sd^2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtr, ndtri

from gnwt.stage1.features import (Design, DesignSpec, PersonTable, Standardizer, build_design,
                                  delta_basis_for, dyad_columns_for_standardizer, person_mask)
from gnwt.stage1.srrm import LockedSRRM, align, bvn_cdf


# ------------------------------------------------------------------------------------ data


@dataclass
class DyadData:
    """Directed rows of one market, every dyad present in both directions (missing y = nan)."""

    T: PersonTable
    i: np.ndarray
    j: np.ndarray
    y: np.ndarray                 # float, nan = direction absent from the file
    wave: np.ndarray
    match: np.ndarray
    rev: np.ndarray               # index of the reverse row
    observed: np.ndarray          # bool

    @classmethod
    def from_rows(cls, T: PersonTable, rows: pd.DataFrame) -> DyadData:
        i = np.array([T.index[c] for c in rows["chooser"]], int)
        j = np.array([T.index[c] for c in rows["chosen"]], int)
        y = rows["y"].to_numpy(float)
        wave = rows["wave"].to_numpy(int)
        match = rows["match"].to_numpy(float)
        pos = {(a, b): k for k, (a, b) in enumerate(zip(i, j))}
        extra_i, extra_j, extra_w = [], [], []
        for k, (a, b) in enumerate(zip(i, j)):
            if (b, a) not in pos:
                pos[(b, a)] = len(i) + len(extra_i)
                extra_i.append(b)
                extra_j.append(a)
                extra_w.append(wave[k])
        if extra_i:
            i = np.concatenate([i, extra_i])
            j = np.concatenate([j, extra_j])
            y = np.concatenate([y, np.full(len(extra_i), np.nan)])
            wave = np.concatenate([wave, extra_w])
            match = np.concatenate([match, np.full(len(extra_i), np.nan)])
        rev = np.array([pos[(b, a)] for a, b in zip(i, j)], int)
        return cls(T, i, j, y, wave, match, rev, np.isfinite(y))

    def subset(self, rows: np.ndarray) -> DyadData:
        """Rows closed under reversal (callers pass whole waves)."""
        rows = np.asarray(rows, int)
        new = -np.ones(len(self.i), int)
        new[rows] = np.arange(len(rows))
        rev = new[self.rev[rows]]
        if np.any(rev < 0):
            raise ValueError("subset is not closed under reversal")
        return DyadData(self.T, self.i[rows], self.j[rows], self.y[rows], self.wave[rows], self.match[rows],
                        rev, self.observed[rows])


def fold_standardizer(d: DyadData, spec: DesignSpec) -> Standardizer:
    persons = np.unique(np.concatenate([d.i, d.j]))
    return Standardizer.fit(d.T, persons, dyad_columns_for_standardizer(d.T, d.i, d.j, spec))


def design_for(d: DyadData, std: Standardizer, spec: DesignSpec, basis: dict[str, Any],
               pmask: np.ndarray | None = None) -> Design:
    return build_design(d.T, d.i, d.j, std, spec, basis, pmask)


def kept_columns(design: Design) -> list[int]:
    sd = design.X.std(0)
    return [k for k, n in enumerate(design.names) if n == "intercept" or sd[k] > 1e-10]


# ------------------------------------------------------------------------------ MAP probit


def map_probit(X: np.ndarray, y: np.ndarray, prior_sd: np.ndarray, beta0: np.ndarray | None = None,
               tol: float = 1e-10, max_iter: int = 100) -> np.ndarray:
    """argmax sum log Phi(q x'b) - 0.5 sum (b/prior_sd)^2,  q = 2y - 1 (Newton with step halving;
    the objective is concave, so this converges in a handful of steps)."""
    q = 2.0 * y - 1.0
    iv = 1.0 / prior_sd ** 2

    def obj(b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        t = q * (X @ b)
        lp = log_ndtr(t)
        lam = np.exp(-0.5 * t * t - 0.5 * math.log(2 * math.pi) - lp)
        return float(lp.sum() - 0.5 * np.sum(iv * b * b)), t, lam

    b = np.zeros(X.shape[1]) if beta0 is None else np.array(beta0, float)
    f, t, lam = obj(b)
    for _ in range(max_iter):
        g = X.T @ (q * lam) - iv * b
        w = lam * (lam + t)
        H = (X * w[:, None]).T @ X + np.diag(iv)
        step = np.linalg.solve(H, g)
        a = 1.0
        while True:
            bn = b + a * step
            fn, tn, lamn = obj(bn)
            if fn >= f - 1e-12 or a < 1e-6:
                break
            a *= 0.5
        b, done = bn, abs(fn - f) < tol * (1 + abs(f))
        f, t, lam = fn, tn, lamn
        if done:
            break
    return b


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# -------------------------------------------------------------------------------- Gibbs


@dataclass
class GibbsResult:
    beta: np.ndarray              # (chains, draws, p)
    sigma: np.ndarray             # (chains, draws, 3): sigma_a2, sigma_b2, sigma_ab
    rho: np.ndarray               # (chains, draws)
    rhat: dict[str, float] = field(default_factory=dict)
    accept: dict[str, float] = field(default_factory=dict)

    def posterior_mean(self) -> tuple[np.ndarray, float, float, float, float]:
        b = self.beta.reshape(-1, self.beta.shape[-1]).mean(0)
        s = self.sigma.reshape(-1, 3).mean(0)
        return b, float(s[0]), float(s[1]), float(s[2]), float(self.rho.mean())


def _trunc_norm(m: np.ndarray, sd: float, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(m.shape)
    u = np.clip(u, 1e-300, 1.0)
    out = m + sd * rng.standard_normal(m.shape)
    pos = y == 1
    neg = y == 0
    if np.any(pos):
        x = -ndtri(u[pos] * ndtr(m[pos] / sd))
        x = np.where(np.isfinite(x), x, -m[pos] / sd + 1e-9)
        out[pos] = np.maximum(m[pos] + sd * x, 1e-12)
    if np.any(neg):
        x = ndtri(u[neg] * ndtr(-m[neg] / sd))
        x = np.where(np.isfinite(x), x, -m[neg] / sd - 1e-9)
        out[neg] = np.minimum(m[neg] + sd * x, 0.0)
    return out


def _half(rho: float) -> tuple[float, float, float, float]:
    """Se^{-1/2} = [[td, to], [to, td]], Se^{1/2} = [[sd, so], [so, sd]], Se = [[1, rho], [rho, 1]]."""
    ip, im = 1.0 / math.sqrt(1.0 + rho), 1.0 / math.sqrt(1.0 - rho)
    p, m = math.sqrt(1.0 + rho), math.sqrt(1.0 - rho)
    return (ip + im) / 2.0, (ip - im) / 2.0, (p + m) / 2.0, (p - m) / 2.0


def _log_post_sigma(theta: np.ndarray, S: np.ndarray, n: int, prior: str, A: float) -> float:
    la, lb, zr = theta
    sa, sb, r = math.exp(la), math.exp(lb), math.tanh(zr)
    c = sa * sb * r
    det = sa * sa * sb * sb * (1 - r * r)
    if det <= 0:
        return -np.inf
    inv = np.array([[sb * sb, -c], [-c, sa * sa]]) / det
    ll = -0.5 * n * math.log(det) - 0.5 * float(np.sum(inv * S))
    lp = -math.log1p((sa / A) ** 2) - math.log1p((sb / A) ** 2)
    if prior == "lkj_eta_2":
        lp += math.log(1 - r * r)                      # LKJ(eta = 2), 2 x 2
    jac = la + lb + math.log(1 - r * r)
    return ll + lp + jac


def _components(send: np.ndarray, recv: np.ndarray, n_persons: int) -> list[np.ndarray]:
    """Connected components of the person graph (the waves in Speed Dating)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    A = coo_matrix((np.ones(len(send)), (send, recv)), shape=(n_persons, n_persons))
    _, lab = connected_components(A, directed=False)
    used = np.zeros(n_persons, bool)
    used[send] = True
    used[recv] = True
    return [np.where((lab == c) & used)[0] for c in np.unique(lab[used])]


def gibbs_srrm(X: np.ndarray, y: np.ndarray, send: np.ndarray, recv: np.ndarray, rev: np.ndarray,
               n_persons: int, prior_sd: np.ndarray, *, iters: int, burn: int, thin: int, seed: int,
               covariance: str = "lkj_eta_2", var_scale: float = 1.0, fix_rho: float | None = None,
               diag_sigma: bool = False, chains: int = 1) -> GibbsResult:
    """Hoff-style Gibbs sampler for (10) with multiplicative rank 0.

    After the decorrelating transform Z~ = td Z + to Z' the errors are iid N(0, 1) and
    z~_ij = x~_ij' beta + a~_i + b~_j + e~_ij with (a~_k, b~_k) ~ N(0, T Sigma T). (beta, a~, b~) are
    drawn jointly from their Gaussian full conditional: beta from its marginal with the person effects
    integrated out, then the person effects given beta, one connected component (wave) at a time,
    batched with padding. This keeps the intercept from trading off slowly against the person effects.
    """
    n, p = X.shape
    first = np.where(np.arange(n) < rev)[0]
    second = rev[first]
    Xr = X[rev]
    XtX, XrXr, XXr = X.T @ X, Xr.T @ Xr, X.T @ Xr
    iv = 1.0 / prior_sd ** 2
    n_send = np.bincount(send, minlength=n_persons).astype(float)
    n_recv = np.bincount(recv, minlength=n_persons).astype(float)
    obs = np.isfinite(y)
    yy = np.where(obs, y, -1.0)

    def psum(idx: np.ndarray, A: np.ndarray) -> np.ndarray:
        out = np.zeros((n_persons, A.shape[1]))
        np.add.at(out, idx, A)
        return out

    SaX0, SaXr0, SbX0, SbXr0 = psum(send, X), psum(send, Xr), psum(recv, X), psum(recv, Xr)
    comps = _components(send, recv, n_persons)
    used = np.concatenate(comps)
    nP = len(used)
    K = max(len(c) for c in comps)
    nc = len(comps)
    P = -np.ones((nc, K), int)
    for q, c in enumerate(comps):
        P[q, :len(c)] = c
    real = P >= 0
    Pz = np.where(real, P, n_persons)                       # padding slot -> extra zero row
    M = np.zeros((n_persons + 1, n_persons + 1))
    np.add.at(M, (send, recv), 1.0)
    # precision of (a~, b~) per component: [[Da, Mo], [Mo', Db]] with Da, Db diagonal; padding slots
    # get Da = Db = 1 and no coupling, so they never touch the real slots
    Mc = np.stack([M[np.ix_(Pz[q], Pz[q])] * np.outer(real[q], real[q]) for q in range(nc)])
    ns_pad = np.append(n_send, 0.0)[Pz]
    nr_pad = np.append(n_recv, 0.0)[Pz]
    eyeK = np.eye(K)[None] * real[:, :, None]

    def gather(V: np.ndarray) -> np.ndarray:            # (N, ...) -> (nc, K, ...), zero padded
        Vz = np.concatenate([V, np.zeros((1,) + V.shape[1:])], axis=0)
        return Vz[Pz]

    ybar = float(np.nanmean(y))
    keep_draws = (iters - burn) // thin
    Bd = np.zeros((chains, keep_draws, p))
    SG = np.zeros((chains, keep_draws, 3))
    RH = np.zeros((chains, keep_draws))
    acc_s = acc_r = 0
    for ch in range(chains):
        rng = np.random.default_rng([seed, ch])
        beta = np.zeros(p)
        beta[0] = float(ndtri(min(max(ybar, 1e-3), 1 - 1e-3)))
        a = np.zeros(n_persons)
        b = np.zeros(n_persons)
        Sig = np.diag([0.3, 0.3])
        rho = 0.0 if fix_rho is None else fix_rho
        z = np.where(y == 1, 0.5, -0.5) + rng.normal(0, 0.1, n)
        step_s, step_r = 0.15, 0.05
        k_out = 0
        for t in range(iters):
            # z | .
            mu = X @ beta + a[send] + b[recv]
            sd = math.sqrt(1.0 - rho * rho)
            for A_, B_ in ((first, second), (second, first)):
                z[A_] = _trunc_norm(mu[A_] + rho * (z[B_] - mu[B_]), sd, yy[A_], rng)
            # decorrelating transform
            td, to, sdh, soh = _half(rho)
            zt = td * z + to * z[rev]
            Tm = np.array([[td, to], [to, td]])
            Si = np.linalg.inv(Tm @ Sig @ Tm)
            SaX = td * SaX0 + to * SaXr0
            SbX = td * SbX0 + to * SbXr0
            Saz = np.bincount(send, weights=zt, minlength=n_persons)
            Sbz = np.bincount(recv, weights=zt, minlength=n_persons)
            Aprec = td * td * XtX + to * to * XrXr + td * to * (XXr + XXr.T) + np.diag(iv)
            g = td * (X.T @ zt) + to * (Xr.T @ zt)
            # joint (beta, a~, b~)
            Da = np.where(real, ns_pad + Si[0, 0], 1.0)                      # (nc, K)
            Db = np.where(real, nr_pad + Si[1, 1], 1.0)
            Mo = Mc + Si[0, 1] * eyeK                                         # (nc, K, K)
            MoT = np.swapaxes(Mo, 1, 2)
            Sch = Db[..., None] * np.eye(K)[None] - MoT @ (Mo / Da[..., None])  # Schur complement
            Ls = np.linalg.cholesky(Sch)
            Ra = np.concatenate([gather(SaX), gather(Saz)[..., None]], axis=2)   # (nc, K, p + 1)
            Rb = np.concatenate([gather(SbX), gather(Sbz)[..., None]], axis=2)
            Xb = np.linalg.solve(Sch, Rb - MoT @ (Ra / Da[..., None]))
            Xa = (Ra - Mo @ Xb) / Da[..., None]
            Aprec -= (np.einsum("ckp,ckq->pq", Ra[..., :p], Xa[..., :p])
                      + np.einsum("ckp,ckq->pq", Rb[..., :p], Xb[..., :p]))
            g -= (np.einsum("ckp,ck->p", Ra[..., :p], Xa[..., p])
                  + np.einsum("ckp,ck->p", Rb[..., :p], Xb[..., p]))
            La = np.linalg.cholesky(Aprec)
            mean = np.linalg.solve(La.T, np.linalg.solve(La, g))
            beta = mean + np.linalg.solve(La.T, rng.standard_normal(p))
            ub = Xb[..., p] - Xb[..., :p] @ beta
            ua = Xa[..., p] - Xa[..., :p] @ beta
            nb = np.linalg.solve(np.swapaxes(Ls, 1, 2), rng.standard_normal((nc, K, 1)))[..., 0]
            na = -(Mo @ nb[..., None])[..., 0] / Da + rng.standard_normal((nc, K)) / np.sqrt(Da)
            u = np.concatenate([ua + na, ub + nb], axis=1)
            at = np.zeros(n_persons)
            bt = np.zeros(n_persons)
            at[P[real]] = u[:, :K][real]
            bt[P[real]] = u[:, K:][real]
            a = sdh * at + soh * bt
            b = soh * at + sdh * bt
            # Sigma | a, b
            au, bu = a[used], b[used]
            S = np.array([[au @ au, au @ bu], [au @ bu, bu @ bu]])
            if covariance == "inverse_wishart" and not diag_sigma:
                W = np.linalg.inv(np.eye(2) + S)
                Zw = rng.standard_normal((int(4 + nP), 2)) @ np.linalg.cholesky(W).T
                Sig = np.linalg.inv(Zw.T @ Zw)
            else:
                cr = Sig[0, 1] / math.sqrt(Sig[0, 0] * Sig[1, 1])
                th = np.array([0.5 * math.log(Sig[0, 0]), 0.5 * math.log(Sig[1, 1]),
                               0.0 if diag_sigma else math.atanh(float(np.clip(cr, -0.99, 0.99)))])
                prop = th + step_s * rng.standard_normal(3)
                if diag_sigma:
                    prop[2] = 0.0
                if math.log(rng.random()) < (_log_post_sigma(prop, S, nP, covariance, var_scale)
                                             - _log_post_sigma(th, S, nP, covariance, var_scale)):
                    th = prop
                    acc_s += t >= burn
                    step_s *= 1.02 if t < burn else 1.0
                elif t < burn:
                    step_s *= 0.99
                sa, sb, rr = math.exp(th[0]), math.exp(th[1]), math.tanh(th[2])
                Sig = np.array([[sa * sa, rr * sa * sb], [rr * sa * sb, sb * sb]])
            # rho | .
            if fix_rho is None:
                eps = z - (X @ beta + a[send] + b[recv])
                e1, e2 = eps[first], eps[second]
                s11, s22, s12 = float(e1 @ e1), float(e2 @ e2), float(e1 @ e2)
                D = len(first)

                def ll(rv: float) -> float:
                    return -0.5 * D * math.log(1 - rv * rv) - (s11 - 2 * rv * s12 + s22) / (2 * (1 - rv * rv))

                rn = math.tanh(math.atanh(rho) + step_r * rng.standard_normal())
                if math.log(rng.random()) < ll(rn) - ll(rho) + math.log(1 - rn * rn) - math.log(1 - rho * rho):
                    rho = rn
                    acc_r += t >= burn
                    step_r *= 1.02 if t < burn else 1.0
                elif t < burn:
                    step_r *= 0.99
            if t >= burn and (t - burn) % thin == 0 and k_out < keep_draws:
                Bd[ch, k_out] = beta
                SG[ch, k_out] = (Sig[0, 0], Sig[1, 1], Sig[0, 1])
                RH[ch, k_out] = rho
                k_out += 1
    total = chains * (iters - burn)
    res = GibbsResult(Bd, SG, RH, accept={"sigma": acc_s / max(total, 1), "rho": acc_r / max(total, 1)})
    res.rhat = {"beta_max": max_rhat(Bd), "sigma_max": max_rhat(SG), "rho": max_rhat(RH[..., None])}
    return res


def split_rhat(x: np.ndarray) -> float:
    """Split-R^ (Gelman et al. 2013) of draws x with shape (chains, draws)."""
    c, n = x.shape
    if n < 4:
        return float("nan")
    h = n // 2
    parts = np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)
    m = parts.shape[1]
    W = parts.var(axis=1, ddof=1).mean()
    Bv = m * parts.mean(axis=1).var(ddof=1)
    if W <= 0:
        return 1.0
    var = (m - 1) / m * W + Bv / m
    return float(np.sqrt(var / W))


def max_rhat(arr: np.ndarray) -> float:
    vals = [split_rhat(arr[..., k]) for k in range(arr.shape[-1])]
    vals = [v for v in vals if np.isfinite(v)]
    return float(max(vals)) if vals else float("nan")


# --------------------------------------------------------------------------------- fitting


@dataclass(frozen=True)
class FitConfig:
    scale_grid: tuple[float, ...]
    mask_grid: tuple[float, ...]
    intercept_sd: float
    iters: int
    burn: int
    thin: int
    chains: int
    covariance: str
    var_scale: float
    seed: int
    rhat_max: float = 1.01
    fix_rho: float | None = None
    diag_sigma: bool = False
    nested: bool = True
    pilot_iters: int = 400

    @classmethod
    def from_params(cls, P: Any, chains: int | None = None, **over: Any) -> FitConfig:
        s = P.stage1
        cfg = cls(tuple(s.priors.scale_grid), tuple(s.block_masking_rate_grid), s.priors.intercept_sd,
                  s.core.sampler.iter, s.core.sampler.burn, s.core.sampler.thin,
                  chains or s.core.sampler.chains, s.priors.covariance, s.priors.variance_scale.scale,
                  P.global_seed, s.core.sampler.rhat_max)
        return cls(**{**cfg.__dict__, **over})


def _prior_sd(names: list[str], scale: float, intercept_sd: float) -> np.ndarray:
    return np.array([intercept_sd if n == "intercept" else scale for n in names])


def _gibbs_on(d: DyadData, des: Design, cols: list[int], scale: float, cfg: FitConfig, seed: int,
              chains: int, iters: int, burn: int) -> GibbsResult:
    X = des.X[:, cols]
    names = [des.names[k] for k in cols]
    return gibbs_srrm(X, d.y, d.i, d.j, d.rev, len(d.T), _prior_sd(names, scale, cfg.intercept_sd),
                      iters=iters, burn=burn, thin=cfg.thin, seed=seed, covariance=cfg.covariance,
                      var_scale=cfg.var_scale, fix_rho=cfg.fix_rho, diag_sigma=cfg.diag_sigma, chains=chains)


def select_config(d: DyadData, spec: DesignSpec, cfg: FitConfig, s_hat: float,
                  std: Standardizer, basis: dict[str, Any], seed: int) -> dict[str, Any]:
    """Nested LOWO over the training waves of `d`, MAP marginal probit, 1-SE rule."""
    waves = np.unique(d.wave)
    rng = np.random.default_rng([seed, 7])
    results = []
    obs = d.observed
    for rate in cfg.mask_grid:
        pm = person_mask(len(d.T), rate, np.random.default_rng([seed, int(rate * 1000)]))
        des_tr = design_for(d, std, spec, basis, pm)          # masked design for fitting
        des_ev = design_for(d, std, spec, basis, None)        # deployment design for scoring
        cols = kept_columns(des_tr)
        Xtr, Xev = des_tr.X[:, cols], des_ev.X[:, cols]
        names = [des_tr.names[k] for k in cols]
        for scale in cfg.scale_grid:
            sd = _prior_sd(names, scale, cfg.intercept_sd) / s_hat
            losses, sizes = [], []
            b0 = None
            for w in waves:
                tr = (d.wave != w) & obs
                te = (d.wave == w) & obs
                if te.sum() == 0:
                    continue
                b = map_probit(Xtr[tr], d.y[tr], sd, b0)
                b0 = b
                p = ndtr(Xev[te] @ b)
                losses.append(log_loss(d.y[te], p))
                sizes.append(int(te.sum()))
            L = np.array(losses)
            wts = np.array(sizes, float)
            results.append({"scale": scale, "mask": rate, "loss": float(np.average(L, weights=wts)),
                            "se": float(L.std(ddof=1) / math.sqrt(len(L))) if len(L) > 1 else 0.0})
    best = min(results, key=lambda r: r["loss"])
    thr = best["loss"] + best["se"]
    ok = [r for r in results if r["loss"] <= thr]
    chosen = min(ok, key=lambda r: (r["scale"], r["mask"]))
    del rng
    return {"chosen": chosen, "best": best, "grid": results}


def fit_srrm(d: DyadData, spec: DesignSpec, cfg: FitConfig, seed: int, chains: int | None = None,
             fixed: dict[str, float] | None = None) -> LockedSRRM:
    """Steps 1-3 on the waves of `d`; returns the locked model."""
    std = fold_standardizer(d, spec)
    basis = delta_basis_for(d.T, np.unique(d.i))
    des0 = design_for(d, std, spec, basis)
    # pilot run for s (marginal scale) at the middle of the grid
    mid = cfg.scale_grid[len(cfg.scale_grid) // 2]
    if fixed is None and cfg.nested:
        pil = _gibbs_on(d, des0, kept_columns(des0), mid, cfg, seed + 11, 1, cfg.pilot_iters, cfg.pilot_iters // 2)
        _, sa2, sb2, _, _ = pil.posterior_mean()
        s_hat = math.sqrt(1 + sa2 + sb2)
        sel = select_config(d, spec, cfg, s_hat, std, basis, seed)
        scale, rate = sel["chosen"]["scale"], sel["chosen"]["mask"]
    else:
        fx = fixed or {"scale": mid, "mask": 0.0}
        scale, rate, sel, s_hat = fx["scale"], fx["mask"], {"chosen": fx}, float("nan")
    pm = person_mask(len(d.T), rate, np.random.default_rng([seed, int(rate * 1000)]))
    des = design_for(d, std, spec, basis, pm)
    cols = kept_columns(des)
    n_ch = chains or cfg.chains
    res = _gibbs_on(d, des, cols, scale, cfg, seed, n_ch, cfg.iters, cfg.burn)
    beta, sa2, sb2, sab, rho = res.posterior_mean()
    names = [des.names[k] for k in cols]
    meta = {"scale": scale, "mask_rate": rate, "s_hat_pilot": s_hat, "selection": sel.get("grid"),
            "chosen": sel.get("chosen"), "rhat": res.rhat, "accept": res.accept, "chains": n_ch,
            "iters": cfg.iters, "burn": cfg.burn, "n_rows": int(d.observed.sum()),
            "waves": sorted(int(w) for w in np.unique(d.wave)),
            "beta_sd": [float(x) for x in res.beta.reshape(-1, len(beta)).std(0)],
            "fix_rho": cfg.fix_rho, "diag_sigma": cfg.diag_sigma}
    return LockedSRRM(names, beta, [des.block_of[k] for k in cols], sa2, sb2, sab, rho, std, spec, basis, meta)


def predict(m: LockedSRRM, d: DyadData) -> pd.DataFrame:
    """a0 for both directions, m0 per dyad and the block decomposition of every row of d."""
    from gnwt.stage1.srrm import eta_blocks

    des = build_design(d.T, d.i, d.j, m.std, m.spec, m.delta_basis)
    bl = eta_blocks(des, m)
    eta = bl.eta
    s, r = m.s, m.r
    a0v = ndtr(eta / s)
    m0v = bvn_cdf(eta / s, eta[d.rev] / s, r)
    return pd.DataFrame({"i": d.i, "j": d.j, "wave": d.wave, "y": d.y, "match": d.match,
                         "eta0": bl.eta0, "etaP_fit": bl.etaP_fit, "etaP_resid": bl.etaP_resid,
                         "etaP_actor": bl.etaP_actor, "etaG": bl.etaG, "eta": eta, "a0": a0v,
                         "m0": m0v, "fit": des.fit})


def _crossfit_one(args: tuple[DyadData, DesignSpec, FitConfig, int, int]) -> tuple[int, LockedSRRM, pd.DataFrame]:
    from threadpoolctl import threadpool_limits

    d, spec, cfg, w, seed = args
    tr = np.where(d.wave != w)[0]
    te = np.where(d.wave == w)[0]
    with threadpool_limits(2):
        m = fit_srrm(d.subset(tr), spec, cfg, seed, chains=cfg.chains)
        out = predict(m, d.subset(te))
    out["row"] = te
    return w, m, out


def crossfit(d: DyadData, spec: DesignSpec, cfg: FitConfig, waves: Sequence[int] | None = None,
             workers: int = 1, folds: dict[int, int] | None = None) -> tuple[pd.DataFrame, dict[int, LockedSRRM]]:
    """Leave-one-wave-out cross-fit (or any fold map wave -> fold). Rows of d keep their order."""
    if folds is not None:
        fold_of = np.array([folds[int(w)] for w in d.wave])
        d2 = DyadData(d.T, d.i, d.j, d.y, fold_of, d.match, d.rev, d.observed)
        units = sorted(set(folds.values()))
    else:
        d2 = d
        units = sorted(int(w) for w in (waves if waves is not None else np.unique(d.wave)))
    jobs = [(d2, spec, cfg, w, cfg.seed + 101 * k) for k, w in enumerate(units)]
    outs: list[pd.DataFrame] = []
    models: dict[int, LockedSRRM] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for w, m, out in ex.map(_crossfit_one, jobs):
                models[w] = m
                outs.append(out)
    else:
        for job in jobs:
            w, m, out = _crossfit_one(job)
            models[w] = m
            outs.append(out)
    pred = pd.concat(outs).sort_values("row").reset_index(drop=True)
    pred["wave"] = d.wave[pred["row"].to_numpy()]
    return pred, models


def estimate_srrm(d: DyadData, spec: DesignSpec, cfg: FitConfig) -> LockedSRRM:
    """Locked model on every development wave (spec 10.1 `estimate_srrm`)."""
    return fit_srrm(d, spec, cfg, cfg.seed, chains=cfg.chains)
