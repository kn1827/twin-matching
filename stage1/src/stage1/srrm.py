"""SRRM probit core (Hoff 2021, eq. 2.3 / 4.1) for directed binary decisions (tech spec 2).

    z_{i->j} = eta_ij + a_i + b_j + eps_ij,  y = 1[z > 0],
    (a_i, b_i) ~ N(0, Sigma_ab),  (eps_ij, eps_ji) ~ N(0, [[1, rho], [rho, 1]])                (10)
    a0(i->j) = Phi(eta_ij / s),  s = sqrt(1 + sigma_a^2 + sigma_b^2)                               (12)
    m0_ij    = Phi2(eta_ij / s, eta_ji / s; r),  r = (2 sigma_ab + rho) / (1 + sigma_a^2 + sigma_b^2) (13)

For two new people a_i, b_j are independent of everything observed, so z - eta has variance s^2 and
the two directions have covariance 2 sigma_ab + rho. The multiplicative term u_i'v_j is dropped (its
expectation is 0 for new people). The split eta = eta0 + etaP_fit + etaP_resid + etaP_actor + etaG (14)
is exact on the latent scale because eta is linear in the coefficients.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.special import ndtr, owens_t

from stage1.features import COL, COLUMNS, ETA_PART, Features, FeatureConfig, Standardizer

ETA_PARTS = ("eta0", "etaP_fit", "etaP_resid", "etaP_actor", "etaG")


def a0(eta, s: float):
    """(12) directed probability of "yes"."""
    return ndtr(np.asarray(eta, float) / s)


def bvn_cdf(h, k, r):
    """Phi2(h, k; r): standard bivariate normal CDF via Owen's T (vectorised, exact to ~1e-15)."""
    h, k = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float))
    r = float(r)
    if abs(r) < 1e-15:
        return ndtr(h) * ndtr(k)
    if r >= 1 - 1e-15:
        return ndtr(np.minimum(h, k))
    if r <= -1 + 1e-15:
        return np.maximum(ndtr(h) + ndtr(k) - 1, 0.0)
    h = np.where(h == 0, 1e-300, h)
    k = np.where(k == 0, 1e-300, k)
    q = np.sqrt((1 - r) * (1 + r))
    ah = (k - r * h) / (h * q)
    ak = (h - r * k) / (k * q)
    beta = np.where((h * k < 0) | ((h * k == 0) & (h + k < 0)), 0.5, 0.0)
    out = 0.5 * (ndtr(h) + ndtr(k)) - owens_t(h, ah) - owens_t(k, ak) - beta
    return np.clip(out, 0.0, np.minimum(ndtr(h), ndtr(k)))


def m0(eta_ij, eta_ji, r: float, s: float = 1.0):
    """(13) probability that both say "yes"; equals a0_ij * a0_ji when r = 0."""
    return bvn_cdf(np.asarray(eta_ij, float) / s, np.asarray(eta_ji, float) / s, r)


@dataclass
class Identification:
    """Single-context identification (tech spec 4): the partner Big Five weight is gamma_P * D~_ref + delta
    with delta orthogonal to D~_ref ("orthogonal"), or unconstrained ("free", multi-context data)."""
    mode: str = "orthogonal"
    D_ref: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.D_ref))

    def basis(self) -> np.ndarray:
        if self.mode == "orthogonal" and self.norm > 0:
            return null_space(self.D_ref[None, :])                        # (5, 4) orthonormal
        return np.eye(5)

    def model_columns(self) -> list[str]:
        Q = self.basis()
        cols = [c for c in COLUMNS if not c.startswith("partner_bf_")]
        k = cols.index("fit") + 1
        return cols[:k] + [f"partner_bf_q{m + 1}" for m in range(Q.shape[1])] + cols[k:]

    def to_model(self, X: np.ndarray) -> np.ndarray:
        """Raw design (COLUMNS) -> model design (model_columns)."""
        Q = self.basis()
        P = X[:, [COL[f"partner_bf_{d}"] for d in ("O", "C", "E", "A", "ES")]]
        keep = [COL[c] for c in COLUMNS if not c.startswith("partner_bf_")]
        Xk = X[:, keep]
        k = [c for c in COLUMNS if not c.startswith("partner_bf_")].index("fit") + 1
        return np.hstack([Xk[:, :k], P @ Q, Xk[:, k:]])

    def to_raw_coef(self, b_model: np.ndarray) -> np.ndarray:
        """Model coefficients -> coefficients on the raw COLUMNS (delta = Q delta')."""
        Q = self.basis()
        cols = self.model_columns()
        out = np.zeros(len(COLUMNS))
        qidx = [i for i, c in enumerate(cols) if c.startswith("partner_bf_q")]
        delta = Q @ b_model[qidx]
        for i, c in enumerate(cols):
            if not c.startswith("partner_bf_q"):
                out[COL[c]] = b_model[i]
        for d, dom in enumerate(("O", "C", "E", "A", "ES")):
            out[COL[f"partner_bf_{dom}"]] = delta[d]
        return out

    def prior_sd(self, scale: float, intercept_sd: float) -> np.ndarray:
        """Prior sd per model column. In orthogonal mode gamma_P gets scale/||D_ref|| so that the
        reparametrisation is the same isotropic prior on the partner weight (exact equivalence, U9)."""
        sd = np.full(len(self.model_columns()), float(scale))
        cols = self.model_columns()
        sd[cols.index("intercept")] = intercept_sd
        if self.mode == "orthogonal" and self.norm > 0:
            sd[cols.index("fit")] = scale / self.norm
        return sd


@dataclass
class Blocks:
    eta0: np.ndarray
    etaP_fit: np.ndarray
    etaP_resid: np.ndarray
    etaP_actor: np.ndarray
    etaG: np.ndarray

    @property
    def eta(self) -> np.ndarray:
        return self.eta0 + self.etaP_fit + self.etaP_resid + self.etaP_actor + self.etaG


@dataclass
class LockedSRRM:
    beta: np.ndarray                  # latent-scale coefficients on the raw COLUMNS (posterior means)
    sigma_a2: float
    sigma_b2: float
    sigma_ab: float
    rho: float
    standardizer: Standardizer
    ident: Identification
    feature_cfg: FeatureConfig = field(default_factory=FeatureConfig)
    meta: dict = field(default_factory=dict)

    @property
    def s(self) -> float:
        return float(np.sqrt(1 + self.sigma_a2 + self.sigma_b2))

    @property
    def r(self) -> float:
        return float((2 * self.sigma_ab + self.rho) / (1 + self.sigma_a2 + self.sigma_b2))

    @property
    def gamma_P(self) -> float:
        return float(self.beta[COL["fit"]])

    @property
    def delta(self) -> np.ndarray:
        return self.beta[[COL[f"partner_bf_{d}"] for d in ("O", "C", "E", "A", "ES")]]

    def to_json(self) -> dict:
        return {"columns": list(COLUMNS), "beta": [float(v) for v in self.beta],
                "sigma_a2": self.sigma_a2, "sigma_b2": self.sigma_b2, "sigma_ab": self.sigma_ab, "rho": self.rho,
                "s": self.s, "r": self.r, "standardizer": self.standardizer.to_json(),
                "ident": {"mode": self.ident.mode, "D_ref": [float(v) for v in self.ident.D_ref]},
                "feature_cfg": self.feature_cfg.__dict__, "meta": self.meta}

    @property
    def hash(self) -> str:
        d = self.to_json()
        d.pop("meta")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"srrm_{self.hash[:16]}.json"
        path.write_text(json.dumps({**self.to_json(), "hash": self.hash}, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "LockedSRRM":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d["columns"] != list(COLUMNS):
            raise ValueError("locked SRRM was fitted with another feature layout")
        m = cls(np.array(d["beta"]), d["sigma_a2"], d["sigma_b2"], d["sigma_ab"], d["rho"],
                Standardizer.from_json(d["standardizer"]),
                Identification(d["ident"]["mode"], np.array(d["ident"]["D_ref"])),
                FeatureConfig(**d["feature_cfg"]), d.get("meta", {}))
        if m.hash != d["hash"]:
            raise ValueError(f"locked SRRM {path} does not match its recorded hash")
        return m


def eta_blocks(f: Features | np.ndarray, th: LockedSRRM) -> Blocks:
    """Exact split of eta (14) for one Features object or a raw design matrix (n, len(COLUMNS))."""
    X = np.atleast_2d(f.x if isinstance(f, Features) else f)
    parts = {p: np.zeros(len(X)) for p in ETA_PARTS}
    for c, j in COL.items():
        if th.beta[j] != 0.0:
            parts[ETA_PART[c]] += X[:, j] * th.beta[j]
    return Blocks(**parts)
