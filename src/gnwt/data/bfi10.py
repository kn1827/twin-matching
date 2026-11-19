"""Offline Big Five proxy for Speed Dating personas (stage1.md, input table: "proxy BFI-10 ... ngoài chặng 1").

Speed Dating has no Big Five. Rev 110 built a BFI-10 proxy with an LLM; Stage 1 must not depend on an
LLM, so this module builds a deterministic **unit-weighted composite** instead (see CHANGELOG):

    P_d = Phi( mean_k  sign_k * z(x_k) ),   x_k in ITEMS[d]

z is taken over the persons of the market, every item has weight +-1 (no fitted or hand-tuned weight)
and only the *sign* of each item comes from published trait-activity correlates:

* O: aesthetic / intellectual activities (art, museums, theater, reading, concerts, music);
  Openness is the trait most tied to cultural participation (McCrae & Costa 1997; Kraaykamp & van
  Eijck 2005, Personality and Individual Differences 38).
* C: self-rated ambition (achievement striving) and exercise (C r = 0.20 with physical activity,
  Rhodes & Smith 2006).
* E: clubbing, self-rated fun and going-out frequency (E r = 0.23 with activity, Rhodes & Smith 2006;
  sociability items of the BFI). `go_out` is coded 1 = several times a week, so its sign is -1.
* A: self-rated sincerity (straightforwardness facet of A).
* N: no item in either file; N stays missing, so ES = -N is masked by the trait-level rule.

Persons with fewer than `min_items_per_trait` valid items for a trait get that trait missing. The proxy
has unknown validity: Stage 1 treats it as a proxy (`bf_source = 'proxy'`) and gamma_P learned on it is
a conservative estimate (Mệnh đề 0h of the proposal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

ITEMS: dict[str, tuple[tuple[str, int], ...]] = {
    "O": (("art", 1), ("museums", 1), ("theater", 1), ("reading", 1), ("concerts", 1), ("music", 1)),
    "C": (("self_amb", 1), ("exercise", 1)),
    "E": (("clubbing", 1), ("self_fun", 1), ("go_out", -1)),
    "A": (("self_sinc", 1),),
    "N": (),
}
PROXY_VERSION = "ucw-1.0.0"


def build_proxy(persons: pd.DataFrame, min_items_per_trait: int = 1) -> pd.DataFrame:
    """Return a frame indexed like `persons` with columns O, C, E, A, N in [0, 1] (NaN = missing)."""
    out = pd.DataFrame(index=persons.index, columns=list(ITEMS), dtype=float)
    for trait, items in ITEMS.items():
        zs = []
        for col, sign in items:
            if col not in persons or persons[col].notna().sum() < 2:
                continue
            x = persons[col].astype(float)
            sd = x.std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                continue
            zs.append(sign * (x - x.mean()) / sd)
        if not zs:
            continue
        z = pd.concat(zs, axis=1)
        n_ok = z.notna().sum(axis=1)
        comp = z.mean(axis=1, skipna=True)
        comp[n_ok < min_items_per_trait] = np.nan
        out[trait] = norm.cdf(comp.to_numpy())
        out.loc[comp.isna(), trait] = np.nan
    return out
