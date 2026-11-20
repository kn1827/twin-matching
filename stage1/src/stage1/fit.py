"""Trait-context fit (11a): fit(j, c) = sum_d D~_d(c) * P~_{j,d}.

Person and context share one signed scale: P~ is the z-scored Big Five with ES = -N (0 = population
mean), D~ is the demand on -1..1 (0 = the trait neither helps nor hinders). They are combined by a
product (demands-abilities fit), never by a distance |P~ - D~|.
"""

from __future__ import annotations

import numpy as np


def fit_score(P_tilde_j: np.ndarray, mask_j: np.ndarray, D_tilde: np.ndarray) -> tuple[float, np.ndarray]:
    """(fit, five parts D~_d * P~_{j,d}). `mask_j` is True where the trait is PRESENT; absent traits
    contribute 0, and a fully masked block gives fit = 0 (tech spec 6)."""
    P = np.where(np.asarray(mask_j, bool), np.nan_to_num(np.asarray(P_tilde_j, float)), 0.0)
    parts = np.asarray(D_tilde, float) * P
    return float(parts.sum()), parts


def fit_scores(P_tilde: np.ndarray, mask: np.ndarray, D_tilde: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised fit_score over rows: P_tilde (n, 5), mask (n, 5), D_tilde (n, 5) or (5,)."""
    P = np.where(np.asarray(mask, bool), np.nan_to_num(np.asarray(P_tilde, float)), 0.0)
    parts = np.broadcast_to(np.asarray(D_tilde, float), P.shape) * P
    return parts.sum(axis=1), parts
