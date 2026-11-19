"""Trait-context fit (stage1.md, section 4, formula (11a)).

    fit(j, c) = sum_d D~_d(c) * P~_{j,d}

P~_j is the z-standardised Big Five of the chosen person on the signed scale O, C, E, A, ES (ES = -N) and
D~(c) the demand vector of the chooser's context on [-1, 1]. Missing traits contribute 0; when the
whole Big Five block is missing (>= `trait_block_missing_if_ge` traits missing) fit = 0. fit has no free
coefficient; its slope gamma_P is learned in the SRRM.
"""

from __future__ import annotations

import numpy as np


def fit_score(P_tilde_j: np.ndarray, mask_j: np.ndarray, D_tilde: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (fit, parts) with parts_d = D~_d * P~_{j,d} * mask_{j,d}; sum(parts) == fit exactly."""
    P = np.where(mask_j.astype(bool), np.nan_to_num(np.asarray(P_tilde_j, float)), 0.0)
    parts = np.asarray(D_tilde, float) * P
    return float(parts.sum()), parts


def fit_matrix(P_tilde: np.ndarray, mask: np.ndarray, D_tilde: np.ndarray) -> np.ndarray:
    """Vectorised (11a): P_tilde, mask (n, 5) of the chosen persons, D_tilde (n, 5) of the choosers."""
    P = np.where(mask.astype(bool), np.nan_to_num(P_tilde), 0.0)
    return np.einsum("nd,nd->n", D_tilde, P)
