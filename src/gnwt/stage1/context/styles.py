"""Published constants of the context reader (stage1.md, section 3.1).

Twelve of the 21 O*NET 2024-2025 Work Styles, each assigned to one Big Five domain by the O*NET
higher-order style report (Higher_Order_Styles.pdf). The demand scale is the O*NET Work Style impact
scale, -3 ("very detrimental") .. +3 ("very beneficial").
"""

from __future__ import annotations

STYLES = ("DEP", "DET", "ACH", "INN", "CUR", "AMB", "SOC", "LEA", "COO", "EMP", "STR", "SCT")
ONET_NAME = {
    "DEP": "Dependability", "DET": "Attention to Detail", "ACH": "Achievement Orientation",
    "INN": "Innovation", "CUR": "Intellectual Curiosity", "AMB": "Tolerance for Ambiguity",
    "SOC": "Social Orientation", "LEA": "Leadership Orientation", "COO": "Cooperation",
    "EMP": "Empathy", "STR": "Stress Tolerance", "SCT": "Self-Control",
}
DOMAIN_OF: dict[str, str] = {
    "DEP": "C", "DET": "C", "ACH": "C",
    "INN": "O", "CUR": "O", "AMB": "O",
    "SOC": "E", "LEA": "E",
    "COO": "A", "EMP": "A",
    "STR": "ES", "SCT": "ES",
}
DOMAINS = ("O", "C", "E", "A", "ES")
WI_MAX = 3.0
STYLE_INDEX = {s: i for i, s in enumerate(STYLES)}
CUE_CLASSES = ("neg", "rev", "weak", "base", "strong")
ORDERED_CLASSES = ("neg", "weak", "base", "strong")   # theta_neg <= theta_weak <= theta_base <= theta_strong


def styles_of(domain: str) -> tuple[str, ...]:
    return tuple(s for s in STYLES if DOMAIN_OF[s] == domain)


assert all(len(styles_of(d)) >= 2 for d in DOMAINS)
