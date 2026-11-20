"""Published constants of the demand space (tech spec 3.1).

Twelve of the 21 O*NET 2025 Work Styles, two or three per Big Five domain, with the domain assignment
of the O*NET development report (Higher_Order_Styles.pdf). The demand scale is the O*NET
work-importance scale -3 ("very detrimental") .. +3 ("very beneficial"); dividing by WI_MAX only changes
units. Neuroticism is reversed into emotional stability (ES = -N) so that "high" is always the
demanded direction.
"""

from __future__ import annotations

import numpy as np

STYLES: tuple[str, ...] = ("DEP", "DET", "ACH", "INN", "CUR", "AMB", "SOC", "LEA", "COO", "EMP", "STR", "SCT")
STYLE_NAMES: dict[str, str] = {
    "DEP": "Dependability", "DET": "Attention to Detail", "ACH": "Achievement Orientation",
    "INN": "Innovation", "CUR": "Intellectual Curiosity", "AMB": "Tolerance for Ambiguity",
    "SOC": "Social Orientation", "LEA": "Leadership Orientation", "COO": "Cooperation",
    "EMP": "Empathy", "STR": "Stress Tolerance", "SCT": "Self-Control",
}
DOMAINS: tuple[str, ...] = ("O", "C", "E", "A", "ES")
DOMAIN_OF: dict[str, str] = {
    "DEP": "C", "DET": "C", "ACH": "C",
    "INN": "O", "CUR": "O", "AMB": "O",
    "SOC": "E", "LEA": "E",
    "COO": "A", "EMP": "A",
    "STR": "ES", "SCT": "ES",
}
WI_MAX = 3.0
CUE_CLASSES: tuple[str, ...] = ("neg", "rev", "weak", "base", "strong")
CUE_ORDER: tuple[str, ...] = ("neg", "weak", "base", "strong")      # theta must be non-decreasing here
BIG5_ORDER: tuple[str, ...] = ("O", "C", "E", "A", "N")             # order of persona Big Five inputs

STYLE_INDEX = {s: i for i, s in enumerate(STYLES)}
DOMAIN_INDEX = {d: i for i, d in enumerate(DOMAINS)}


def domain_matrix(keep: np.ndarray | None = None) -> np.ndarray:
    """(5, 12) matrix A with D~ = A @ D_hat: A[d, s] = 1 / (WI_MAX * |S_d|) for s in S_d (11c).

    `keep` (bool, 12) drops styles that failed the ICC filter; a domain with no style left gets a zero
    row, i.e. a neutral demand.
    """
    keep = np.ones(len(STYLES), bool) if keep is None else np.asarray(keep, bool)
    A = np.zeros((len(DOMAINS), len(STYLES)))
    for d, dom in enumerate(DOMAINS):
        members = [STYLE_INDEX[s] for s in STYLES if DOMAIN_OF[s] == dom and keep[STYLE_INDEX[s]]]
        for s in members:
            A[d, s] = 1.0 / (WI_MAX * len(members))
    return A
