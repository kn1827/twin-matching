"""Run-time core: Hoff's probit SRRM read out for two new people (stage1.md, section 2).

    z_{i->j} = eta_ij + a_i + b_j + eps_ij,  (a_i, b_i) ~ N(0, Sigma_ab),  corr(eps_ij, eps_ji) = rho  (10)
    a0(i->j) = Phi(eta_ij / s),  s = sqrt(1 + sigma_a^2 + sigma_b^2)                                 (12)
    m0_ij    = Phi2(eta_ij / s, eta_ji / s; r),  r = (2 sigma_ab + rho) / (1 + sigma_a^2 + sigma_b^2)  (13)

`LockedSRRM` is the read-only artifact written by `gnwt.stage1.offline.srrm_fit.estimate_srrm`: posterior
means of the coefficients and variance components, the standardiser of the development fold, the design
spec and the sha256 of its own content. Everything here is table lookup and linear algebra.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr

from gnwt.stage1.features import BLOCKS, Design, DesignSpec, Standardizer


@dataclass(frozen=True)
class LockedSRRM:
    names: list[str]
    beta: np.ndarray
    block_of: list[str]
    sigma_a2: float
    sigma_b2: float
    sigma_ab: float
    rho: float
    std: Standardizer
    spec: DesignSpec
    delta_basis: dict[str, list[list[float]]]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def s(self) -> float:
        return float(np.sqrt(1.0 + self.sigma_a2 + self.sigma_b2))

    @property
    def r(self) -> float:
        return float((2.0 * self.sigma_ab + self.rho) / (1.0 + self.sigma_a2 + self.sigma_b2))

    def coef(self, name: str) -> float:
        return float(self.beta[self.names.index(name)]) if name in self.names else 0.0

    # --------------------------------------------------------------------------------- serialise

    def to_dict(self) -> dict[str, Any]:
        return {"names": self.names, "beta": [float(b) for b in self.beta], "block_of": self.block_of,
                "sigma_a2": self.sigma_a2, "sigma_b2": self.sigma_b2, "sigma_ab": self.sigma_ab,
                "rho": self.rho, "std": self.std.to_json(), "spec": self.spec.to_json(),
                "delta_basis": self.delta_basis, "meta": self.meta}

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def save(self, path: str | Path) -> str:
        d = self.to_dict()
        digest = self.sha256()
        d["sha256"] = digest
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=1, sort_keys=True), encoding="utf-8")
        return digest

    @classmethod
    def load(cls, path: str | Path, sha256: str | None = None) -> LockedSRRM:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        stored = d.pop("sha256", None)
        m = cls(list(d["names"]), np.array(d["beta"], float), list(d["block_of"]), float(d["sigma_a2"]),
                float(d["sigma_b2"]), float(d["sigma_ab"]), float(d["rho"]), Standardizer.from_json(d["std"]),
                DesignSpec.from_json(d["spec"]), dict(d["delta_basis"]), dict(d.get("meta", {})))
        digest = m.sha256()
        if stored is not None and stored != digest:
            raise ValueError(f"{path}: content does not match its stored sha256")
        if sha256 is not None and sha256 != digest:
            raise ValueError(f"{path}: sha256 mismatch with the manifest")
        return m


@dataclass(frozen=True)
class Blocks:
    eta0: np.ndarray
    etaP_fit: np.ndarray
    etaP_resid: np.ndarray
    etaP_actor: np.ndarray
    etaG: np.ndarray

    @property
    def eta(self) -> np.ndarray:
        return self.eta0 + self.etaP_fit + self.etaP_resid + self.etaP_actor + self.etaG


def align(design: Design, names: list[str]) -> np.ndarray:
    """Columns of `design` in the order of the locked model (absent columns = 0)."""
    pos = {n: k for k, n in enumerate(design.names)}
    X = np.zeros((design.X.shape[0], len(names)))
    for k, n in enumerate(names):
        if n in pos:
            X[:, k] = design.X[:, pos[n]]
    return X


def eta_blocks(f: Design, th: LockedSRRM) -> Blocks:
    """Exact decomposition (14) on the latent scale."""
    X = align(f, th.names)
    contrib = X * th.beta
    out = {b: np.zeros(X.shape[0]) for b in BLOCKS}
    for k, b in enumerate(th.block_of):
        out[b] = out[b] + contrib[:, k]
    return Blocks(**out)


def a0(eta: np.ndarray | float, s: float) -> np.ndarray | float:
    return ndtr(np.asarray(eta, float) / s) if np.ndim(eta) else float(ndtr(float(eta) / s))


def m0(eta_ij: np.ndarray | float, eta_ji: np.ndarray | float, r: float, s: float = 1.0) -> np.ndarray | float:
    """Phi2(eta_ij/s, eta_ji/s; r). With the default s = 1 the inputs are already on the a0 scale."""
    h = np.atleast_1d(np.asarray(eta_ij, float)) / s
    k = np.atleast_1d(np.asarray(eta_ji, float)) / s
    out = bvn_cdf(h, k, r)
    return out if np.ndim(eta_ij) else float(out[0])


# ------------------------------------------------------------------------------- bivariate normal

_GL = {
    6: ([0.1713244923791705, 0.3607615730481384, 0.4679139345726904],
        [0.9324695142031522, 0.6612093864662647, 0.2386191860831970]),
    12: ([0.04717533638651177, 0.1069393259953183, 0.1600783285433464, 0.2031674267230659,
          0.2334925365383547, 0.2491470458134029],
         [0.9815606342467191, 0.9041172563704750, 0.7699026741943050, 0.5873179542866171,
          0.3678314989981802, 0.1252334085114692]),
    20: ([0.01761400713915212, 0.04060142980038694, 0.06267204833410906, 0.08327674157670475,
          0.1019301198172404, 0.1181945319615184, 0.1316886384491766, 0.1420961093183821,
          0.1491729864726037, 0.1527533871307259],
         [0.9931285991850949, 0.9639719272779138, 0.9122344282513259, 0.8391169718222188,
          0.7463319064601508, 0.6360536807265150, 0.5108670019508271, 0.3737060887154196,
          0.2277858511416451, 0.07652652113349733]),
}


def _bvnu(dh: np.ndarray, dk: np.ndarray, r: float) -> np.ndarray:
    """P(X > dh, Y > dk), corr r: Genz (2004), 'Numerical computation of rectangular bivariate and
    trivariate normal and t probabilities', Statistics and Computing 14, 251-260."""
    if r == 0.0:
        return ndtr(-dh) * ndtr(-dk)
    tp = 2.0 * np.pi
    h, k = dh.astype(float), dk.astype(float)
    hk = h * k
    ng = 6 if abs(r) < 0.3 else (12 if abs(r) < 0.75 else 20)
    w0, x0 = (np.array(v) for v in _GL[ng])
    w = np.concatenate([w0, w0])
    x = np.concatenate([1 - x0, 1 + x0])
    if abs(r) < 0.925:
        hs = (h * h + k * k) / 2.0
        asr = np.arcsin(r) / 2.0
        sn = np.sin(asr * x)                                   # (m,)
        bvn = np.exp((sn[None, :] * hk[:, None] - hs[:, None]) / (1.0 - sn[None, :] ** 2)) @ w
        bvn = bvn * asr / tp + ndtr(-h) * ndtr(-k)
    else:
        if r < 0:
            k = -k
            hk = -hk
        bvn = np.zeros_like(h)
        if abs(r) < 1:
            as_ = 1.0 - r * r
            a = np.sqrt(as_)
            bs = (h - k) ** 2
            asr = -(bs / as_ + hk) / 2.0
            c = (4.0 - hk) / 8.0
            d = (12.0 - hk) / 80.0
            bvn = np.where(asr > -100, a * np.exp(asr) * (1 - c * (bs - as_) * (1 - d * bs) / 3 + c * d * as_ ** 2), 0.0)
            b = np.sqrt(bs)
            sp = np.sqrt(tp) * ndtr(-b / a)
            bvn = np.where(hk > -100, bvn - np.exp(-hk / 2) * sp * b * (1 - c * bs * (1 - d * bs) / 3), bvn)
            a2 = a / 2.0
            xs = (a2 * x[None, :]) ** 2                        # (1, m)
            asr2 = -(bs[:, None] / xs + hk[:, None]) / 2.0     # (n, m)
            sp2 = 1 + c[:, None] * xs * (1 + 5 * d[:, None] * xs)
            rs = np.sqrt(1 - xs)
            ep = np.exp(-(hk[:, None] / 2) * xs / (1 + rs) ** 2) / rs
            terms = np.where(asr2 > -100, np.exp(asr2) * (sp2 - ep), 0.0) @ w
            bvn = (a2 * terms - bvn) / tp
        if r > 0:
            bvn = bvn + ndtr(-np.maximum(h, k))
        else:
            L = np.where(h < 0, ndtr(k) - ndtr(h), ndtr(-h) - ndtr(-k))
            bvn = np.where(h >= k, -bvn, L - bvn)
    return np.clip(bvn, 0.0, 1.0)


def bvn_cdf(h: np.ndarray, k: np.ndarray, r: float) -> np.ndarray:
    """Phi2(h, k; r) = P(X < h, Y < k)."""
    h = np.asarray(h, float)
    k = np.asarray(k, float)
    return _bvnu(-h, -k, float(r))
