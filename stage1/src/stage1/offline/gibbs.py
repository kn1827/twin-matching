"""Gibbs sampler for the SRRM probit (10), following Hoff's data-augmentation scheme (amen 1.4.5):

1. (beta, a, b) | z, rho, Sigma  -- one Gaussian block. With R = [[1, rho], [rho, 1]] for the two
   directions of a dyad, the precision is D' Omega D + prior, where D' Omega D =
   (S_same - rho S_cross) / (1 - rho^2) + S_single is assembled from three cross-product matrices
   precomputed once (they do not depend on rho).
2. z | theta, rho, y  -- truncated normals; inside a dyad z_ij | z_ji has mean mu_ij + rho (z_ji - mu_ji)
   and variance 1 - rho^2.
3. rho | residual pairs  -- random-walk Metropolis on atanh(rho), uniform prior on (-1, 1).
4. Sigma_ab | a, b, beta  -- random-walk Metropolis on (log sigma_a, log sigma_b, atanh corr) with an
   LKJ(eta = 2) prior on the correlation and half-Cauchy priors on the scales (Gelman 2006), or an
   inverse-Wishart(4, I) prior (Hoff) as the sensitivity option.

Coefficient prior on the MARGINAL scale: beta_k / s ~ N(0, c_k^2), i.e. beta_k ~ N(0, c_k^2 s^2) with
s^2 = 1 + sigma_a^2 + sigma_b^2. The width c is the one chosen by nested LOWO on the marginal probit, so
selection and sampling shrink the same quantity (the one a0 = Phi(eta / s) depends on).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.special import log_ndtr, ndtri_exp

from stage1.offline.probit import probit_map


@dataclass
class GibbsConfig:
    iterations: int = 1500
    burn_in: int = 500
    thin: int = 1
    chains: int = 4
    covariance: str = "lkj_eta_2"          # or "inverse_wishart"
    variance_scale: float = 1.0            # half-Cauchy scale of sigma_a, sigma_b
    seed: int = 0


@dataclass
class GibbsResult:
    draws: dict                             # name -> (chains, kept, ...) arrays
    accept: dict = field(default_factory=dict)

    def mean(self, name: str) -> np.ndarray:
        d = self.draws[name]
        return d.reshape(-1, *d.shape[2:]).mean(axis=0)

    def rhat(self) -> dict:
        out = {}
        for name in ("beta", "sigma_a2", "sigma_b2", "sigma_ab", "rho"):
            d = self.draws[name]
            d = d.reshape(d.shape[0], d.shape[1], -1)
            out[name] = float(np.nanmax(split_rhat(d)))
        return out


def split_rhat(d: np.ndarray) -> np.ndarray:
    """Split-R-hat (Gelman et al. BDA3) for draws (chains, n, k) -> (k,)."""
    c, n, k = d.shape
    h = n // 2
    if h < 2:
        return np.full(k, np.nan)
    s = np.concatenate([d[:, :h], d[:, h:2 * h]], axis=0)              # (2c, h, k)
    m = s.mean(axis=1)
    W = s.var(axis=1, ddof=1).mean(axis=0)
    B = h * m.var(axis=0, ddof=1)
    var = (h - 1) / h * W + B / h
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.sqrt(var / W)
    return np.where(W > 0, r, 1.0)


def _rtrunc_sign(mean: np.ndarray, sd: np.ndarray | float, y: np.ndarray, rng) -> np.ndarray:
    """z ~ N(mean, sd^2) truncated to z > 0 when y = 1 and z < 0 when y = 0 (log-space inverse CDF)."""
    a = -mean / sd
    logu = np.log(rng.random(len(mean)))
    x_pos = -ndtri_exp(logu + log_ndtr(-a))          # upper tail beyond a
    x_neg = ndtri_exp(logu + log_ndtr(a))            # lower tail below a
    x = np.where(y == 1, x_pos, x_neg)
    x = np.where(y == 1, np.maximum(x, a), np.minimum(x, a))
    return mean + sd * x


class SRRMGibbs:
    def __init__(self, X: np.ndarray, y: np.ndarray, actor: np.ndarray, partner: np.ndarray,
                 prior_sd_marginal: np.ndarray, cfg: GibbsConfig = GibbsConfig()):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, int)
        self.cfg = cfg
        n, p = self.X.shape
        persons, inv = np.unique(np.concatenate([np.asarray(actor), np.asarray(partner)]), return_inverse=True)
        self.persons = persons
        self.ai, self.pi = inv[:n], inv[n:]
        N = len(persons)
        self.n, self.p, self.N = n, p, N
        self.c2 = np.asarray(prior_sd_marginal, float) ** 2

        # dyad structure
        where = {}
        for r, (u, v) in enumerate(zip(self.ai, self.pi)):
            where.setdefault((u, v), r)
        first, second = [], []
        paired = np.zeros(n, bool)
        for r, (u, v) in enumerate(zip(self.ai, self.pi)):
            q = where.get((v, u))
            if q is not None and q != r and not paired[r] and not paired[q] and where[(u, v)] == r:
                first.append(r)
                second.append(q)
                paired[r] = paired[q] = True
        self.first, self.second = np.array(first, int), np.array(second, int)
        self.single = np.flatnonzero(~paired)

        A = sp.csr_matrix((np.ones(n), (np.arange(n), self.ai)), shape=(n, N))
        B = sp.csr_matrix((np.ones(n), (np.arange(n), self.pi)), shape=(n, N))
        self.D = sp.hstack([sp.csr_matrix(self.X), A, B]).tocsr()
        D1, D2, Ds = self.D[self.first], self.D[self.second], self.D[self.single]
        self.D1T, self.D2T, self.DsT = D1.T.tocsr(), D2.T.tocsr(), Ds.T.tocsr()
        self.idx_a = p + np.arange(N)
        self.idx_b = p + N + np.arange(N)

        def blocks(M):
            M = M.tocsc()
            return (M[:p, :p].toarray(), M[:p, p:].toarray(), M[p:, p:].tocsc())

        # cross products split into beta / person blocks; none of them depends on rho
        self.S_same = blocks(D1.T @ D1 + D2.T @ D2)
        self.S_cross = blocks(D1.T @ D2 + D2.T @ D1)
        self.S_single = blocks(Ds.T @ Ds)
        ia, ib = np.arange(N), N + np.arange(N)
        self._uu_pattern = (np.concatenate([ia, ib, ia, ib]), np.concatenate([ia, ib, ib, ia]))

    # ------------------------------------------------------------------------------------ pieces
    def _draw_theta(self, z, rho, Sigma, s2, rng):
        """Exact Gaussian draw of theta = (beta, a, b) | z, rho, Sigma.

        Perturbation + solve: theta = Q^-1 (D' Omega z~ + Lambda theta~) with z~ = z + eps,
        eps ~ N(0, Omega^-1) and Lambda theta~ ~ N(0, Lambda), which is distributed N(Q^-1 D' Omega z, Q^-1).
        Q is solved through the Schur complement of its sparse person block (people only meet within a
        wave, so that block is block-diagonal by wave and factorises cheaply)."""
        k = 1.0 / (1.0 - rho * rho)
        p, N = self.p, self.N
        Si = np.linalg.inv(Sigma)
        Qbb = (self.S_same[0] - rho * self.S_cross[0]) * k + self.S_single[0] + np.diag(1.0 / (self.c2 * s2))
        Qbu = (self.S_same[1] - rho * self.S_cross[1]) * k + self.S_single[1]
        rows, cols = self._uu_pattern
        prior_uu = sp.csc_matrix((np.concatenate([np.full(N, Si[0, 0]), np.full(N, Si[1, 1]),
                                                  np.full(2 * N, Si[0, 1])]), (rows, cols)), shape=(2 * N, 2 * N))
        Quu = ((self.S_same[2] - rho * self.S_cross[2]) * k + self.S_single[2] + prior_uu).tocsc()
        m = len(self.first)
        x1, x2 = rng.standard_normal(m), rng.standard_normal(m)
        z1 = z[self.first] + x1
        z2 = z[self.second] + rho * x1 + np.sqrt(1 - rho * rho) * x2
        zs = z[self.single] + rng.standard_normal(len(self.single))
        rhs = (self.D1T @ (z1 - rho * z2) + self.D2T @ (z2 - rho * z1)) * k + self.DsT @ zs
        rhs[:p] += rng.standard_normal(p) / np.sqrt(self.c2 * s2)
        w = rng.standard_normal((N, 2)) @ np.linalg.cholesky(Si).T
        rhs[self.idx_a] += w[:, 0]
        rhs[self.idx_b] += w[:, 1]
        lu = splu(Quu, permc_spec="MMD_AT_PLUS_A")
        Y = lu.solve(np.ascontiguousarray(Qbu.T))                      # Quu^-1 Qub, (2N, p)
        yu = lu.solve(rhs[p:])
        S = Qbb - Qbu @ Y
        beta = np.linalg.solve(S, rhs[:p] - Qbu @ yu)
        u = yu - Y @ beta
        return np.concatenate([beta, u])

    def _draw_z(self, z, mu, rho, rng):
        y = self.y
        sd = np.sqrt(1.0 - rho * rho)
        f, s = self.first, self.second
        z[f] = _rtrunc_sign(mu[f] + rho * (z[s] - mu[s]), sd, y[f], rng)
        z[s] = _rtrunc_sign(mu[s] + rho * (z[f] - mu[f]), sd, y[s], rng)
        z[self.single] = _rtrunc_sign(mu[self.single], 1.0, y[self.single], rng)
        return z

    def _rho_logpost(self, t, S11, S12, m):
        rho = np.tanh(t)
        q = 1.0 - rho * rho
        return -0.5 * m * np.log(q) - (S11 - 2 * rho * S12) / (2 * q) + np.log(q)

    def _sigma_logpost(self, par, Saa, Sbb, Sab, beta):
        la, lb, tc = par
        sa, sb, c = np.exp(la), np.exp(lb), np.tanh(tc)
        q = 1.0 - c * c
        N = self.N
        ll = -N * (la + lb) - 0.5 * N * np.log(q) - (Saa / sa ** 2 - 2 * c * Sab / (sa * sb) + Sbb / sb ** 2) / (2 * q)
        if self.cfg.covariance == "inverse_wishart":
            nu = 4.0
            det = (sa * sb) ** 2 * q
            tr_inv = (1 / sa ** 2 + 1 / sb ** 2) / q                    # tr(I Sigma^-1)
            lp = -(nu + 3) / 2 * np.log(det) - 0.5 * tr_inv + np.log(4 * sa ** 2 * sb ** 2)
        else:
            A = self.cfg.variance_scale
            lp = np.log(q) - np.log1p((sa / A) ** 2) - np.log1p((sb / A) ** 2)   # LKJ(2) + half-Cauchy
        jac = la + lb + np.log(q)
        s2 = 1.0 + sa ** 2 + sb ** 2
        lb_prior = -0.5 * len(beta) * np.log(s2) - 0.5 * np.sum(beta ** 2 / self.c2) / s2
        return ll + lp + jac + lb_prior

    # ------------------------------------------------------------------------------------ run
    def run(self, b_init: np.ndarray | None = None, verbose: bool = False) -> GibbsResult:
        cfg = self.cfg
        if b_init is None:
            b_init, _ = probit_map(self.X, self.y, np.sqrt(self.c2))
        keep = list(range(cfg.burn_in, cfg.iterations, cfg.thin))
        C, K = cfg.chains, len(keep)
        draws = {"beta": np.zeros((C, K, self.p)), "sigma_a2": np.zeros((C, K)), "sigma_b2": np.zeros((C, K)),
                 "sigma_ab": np.zeros((C, K)), "rho": np.zeros((C, K)), "beta_marginal": np.zeros((C, K, self.p))}
        acc = {"rho": [], "sigma": []}
        for ch in range(C):
            rng = np.random.default_rng([cfg.seed, ch])
            par = np.array([np.log(0.5) + 0.1 * rng.standard_normal(), np.log(0.5) + 0.1 * rng.standard_normal(),
                            0.1 * rng.standard_normal()])
            t_rho = 0.1 * rng.standard_normal()
            sa, sb, c = np.exp(par[0]), np.exp(par[1]), np.tanh(par[2])
            Sigma = np.array([[sa ** 2, c * sa * sb], [c * sa * sb, sb ** 2]])
            s2 = 1 + sa ** 2 + sb ** 2
            theta = np.zeros(self.p + 2 * self.N)
            theta[:self.p] = b_init * np.sqrt(s2)
            mu = self.D @ theta
            z = _rtrunc_sign(mu, 1.0, self.y, rng)
            step_rho, step_sig = 0.05, np.array([0.1, 0.1, 0.15])
            na_r = na_s = 0
            k_out = 0
            for it in range(cfg.iterations):
                rho = np.tanh(t_rho)
                theta = self._draw_theta(z, rho, Sigma, s2, rng)
                mu = self.D @ theta
                z = self._draw_z(z, mu, rho, rng)
                e = z - mu
                e1, e2 = e[self.first], e[self.second]
                S11, S12, m = float(e1 @ e1 + e2 @ e2), float(e1 @ e2), len(e1)
                if m > 0:
                    prop = t_rho + step_rho * rng.standard_normal()
                    if np.log(rng.random()) < self._rho_logpost(prop, S11, S12, m) - self._rho_logpost(t_rho, S11, S12, m):
                        t_rho, na_r = prop, na_r + 1
                a, b = theta[self.idx_a], theta[self.idx_b]
                beta = theta[:self.p]
                Saa, Sbb, Sab = float(a @ a), float(b @ b), float(a @ b)
                prop = par + step_sig * rng.standard_normal(3)
                if np.log(rng.random()) < (self._sigma_logpost(prop, Saa, Sbb, Sab, beta)
                                           - self._sigma_logpost(par, Saa, Sbb, Sab, beta)):
                    par, na_s = prop, na_s + 1
                sa, sb, c = np.exp(par[0]), np.exp(par[1]), np.tanh(par[2])
                Sigma = np.array([[sa ** 2, c * sa * sb], [c * sa * sb, sb ** 2]])
                s2 = 1 + sa ** 2 + sb ** 2
                if it < cfg.burn_in and (it + 1) % 50 == 0:          # adapt during burn-in only
                    ar, as_ = na_r / 50, na_s / 50
                    step_rho *= 1.25 if ar > 0.45 else (0.8 if ar < 0.2 else 1.0)
                    step_sig *= 1.25 if as_ > 0.45 else (0.8 if as_ < 0.2 else 1.0)
                    na_r = na_s = 0
                if it >= cfg.burn_in and (it - cfg.burn_in) % cfg.thin == 0:
                    draws["beta"][ch, k_out] = beta
                    draws["beta_marginal"][ch, k_out] = beta / np.sqrt(s2)
                    draws["sigma_a2"][ch, k_out] = sa ** 2
                    draws["sigma_b2"][ch, k_out] = sb ** 2
                    draws["sigma_ab"][ch, k_out] = c * sa * sb
                    draws["rho"][ch, k_out] = np.tanh(t_rho)
                    k_out += 1
                if verbose and (it + 1) % 250 == 0:
                    print(f"    chain {ch} it {it + 1}: sa2={sa ** 2:.3f} sb2={sb ** 2:.3f} rho={np.tanh(t_rho):.3f}",
                          flush=True)
            n_after = max(cfg.iterations - cfg.burn_in, 1)
            acc["rho"].append(na_r / n_after)
            acc["sigma"].append(na_s / n_after)
        return GibbsResult(draws, acc)
