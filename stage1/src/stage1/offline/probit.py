"""Penalised (MAP) probit by Newton's method, used for (a) the nested-LOWO choice of the prior width and
masking rate and (b) the Laplace batch update of a locked model.

Under the SRRM (10) the marginal model of one decision for new people is a probit with coefficient
beta / s, so the MAP probit estimates exactly the quantity a0 needs. Prior: b ~ N(mean, prior_cov).
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr


def _mills(u: np.ndarray) -> np.ndarray:
    """phi(u) / Phi(u), stable for very negative u."""
    return np.exp(-0.5 * u * u - 0.5 * np.log(2 * np.pi) - log_ndtr(u))


def probit_map(X: np.ndarray, y: np.ndarray, prior_sd: np.ndarray | None = None, *,
               prior_mean: np.ndarray | None = None, prior_prec: np.ndarray | None = None,
               weights: np.ndarray | None = None, tol: float = 1e-10, max_iter: int = 100,
               b0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (b_hat, H) with H the negative Hessian of the log posterior at b_hat (posterior precision
    of the Laplace approximation)."""
    n, p = X.shape
    q = 2.0 * np.asarray(y, float) - 1.0
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    if prior_prec is None:
        prior_prec = np.diag(1.0 / np.asarray(prior_sd, float) ** 2)
    m0 = np.zeros(p) if prior_mean is None else np.asarray(prior_mean, float)
    b = m0.copy() if b0 is None else np.asarray(b0, float).copy()

    def objective(bb):
        u = q * (X @ bb)
        d = bb - m0
        return float(np.sum(w * log_ndtr(u)) - 0.5 * d @ prior_prec @ d)

    f = objective(b)
    H = prior_prec
    for _ in range(max_iter):
        eta = X @ b
        u = q * eta
        m = _mills(u)
        g = X.T @ (w * q * m) - prior_prec @ (b - m0)
        wt = w * m * (u + m)
        H = (X * wt[:, None]).T @ X + prior_prec
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:
            nb = b + t * step
            nf = objective(nb)
            if nf >= f - 1e-12 or t < 1e-8:
                break
            t *= 0.5
        b, f_old, f = nb, f, nf
        if np.max(np.abs(t * step)) < tol or abs(f - f_old) < tol * (1 + abs(f)):
            break
    u = q * (X @ b)
    m = _mills(u)
    H = (X * (w * m * (u + m))[:, None]).T @ X + prior_prec
    return b, H
