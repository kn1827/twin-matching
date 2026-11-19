"""Catalog of basic contexts (stage1.md, sections 3.4 and 10.4).

Each row has aliases (vi/en), goal codes and a default demand profile D^def on the 12 styles (mean, se,
ICC(2,k)) written by the offline rating step O2 (`gnwt.stage1.offline.reader.rate_catalog`), never by
hand. A profile that has not been rated yet is `null`; such a context counts as **unrated** and its
profile is neutral (0 on every style), which is exactly the "ICC unknown or too low -> D~_d = 0" rule of
section 6.

Matching [quy ước]: aliases go through the same syllable matcher as the lexicon (toned, then toneless).
The longest alias wins; a tie between contexts gives the mean profile and the flag `multi`. No match
gives the generic prior (mean of the rated profiles) and the text is logged for catalog review.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from gnwt.stage1.context.styles import STYLES
from gnwt.stage1.context.text import PhraseMatcher

How = Literal["exact", "toneless", "multi", "embed", "generic", "event", "market"]


@dataclass(frozen=True)
class BasicMatch:
    c_star: str | None
    how: How
    ids: tuple[str, ...]
    alias: str | None = None


@dataclass(frozen=True)
class ContextRow:
    id: str
    aliases: dict[str, list[str]]
    goal_codes: tuple[str, ...]
    mean: np.ndarray | None              # (12,), None = unrated
    se: np.ndarray | None
    icc2k: np.ndarray | None
    n_raters: int | None
    sign_check: dict[str, str] = field(default_factory=dict)

    @property
    def rated(self) -> bool:
        return self.mean is not None


class Catalog:
    def __init__(self, raw: dict[str, Any], sha256: str, ngram_max: int = 4, toneless: bool = True) -> None:
        self.version = str(raw["version"])
        self.sha256 = sha256
        self.illustrative = bool(raw.get("illustrative", False))
        self.rows: dict[str, ContextRow] = {}
        aliases: dict[str, str] = {}
        for c in raw["contexts"]:
            cid = str(c["id"])
            prof = c.get("profile")
            mean = se = icc = None
            if prof:
                vals = [prof.get(s) or {} for s in STYLES]
                if all(v.get("mean") is not None for v in vals):
                    mean = np.array([float(v["mean"]) for v in vals])
                    se = np.array([np.nan if v.get("se") is None else float(v["se"]) for v in vals])
                    icc = np.array([np.nan if v.get("icc2k") is None else float(v["icc2k"]) for v in vals])
                    if np.any(np.abs(mean) > 3.0):
                        raise ValueError(f"catalog {cid}: profile outside -3..3")
                elif any(v.get("mean") is not None for v in vals):
                    raise ValueError(f"catalog {cid}: profile must be complete (12 styles) or null")
            al = {lg: [str(a) for a in (c.get("aliases", {}).get(lg) or [])] for lg in ("vi", "en")}
            for lg in al:
                for a in al[lg]:
                    if a in aliases and aliases[a] != cid:
                        raise ValueError(f"alias {a!r} used by {aliases[a]} and {cid}")
                    aliases[a] = cid
            self.rows[cid] = ContextRow(cid, al, tuple(c.get("goal_codes") or ()), mean, se, icc,
                                        c.get("n_raters"), dict(c.get("sign_check") or {}))
        self.matcher = PhraseMatcher(aliases, ngram_max=ngram_max, toneless=toneless)
        self._alias_len = {a: len(a.split()) for a in aliases}
        rated = [r.mean for r in self.rows.values() if r.mean is not None]
        self.generic = np.mean(rated, axis=0) if rated else np.zeros(len(STYLES))

    @classmethod
    def load(cls, path: str | Path, sha256: str | None = None, **kw: Any) -> Catalog:
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if sha256 is not None and sha256 != digest:
            raise ValueError(f"catalog {path}: sha256 mismatch (expected {sha256[:12]}, got {digest[:12]})")
        return cls(yaml.safe_load(data.decode("utf-8")), digest, **kw)

    def match(self, clauses: list[list[str]]) -> BasicMatch:
        found: list[tuple[int, str, bool, str]] = []
        for cl in clauses:
            for m in self.matcher.find_all(cl):
                found.append((m.end - m.start, str(m.payload), m.toneless, " ".join(cl[m.start:m.end])))
        if not found:
            return BasicMatch(None, "generic", ())
        best = max(f[0] for f in found)
        top = [f for f in found if f[0] == best]
        ids = tuple(sorted({f[1] for f in top}))
        if len(ids) > 1:
            return BasicMatch(None, "multi", ids, top[0][3])
        tl = all(f[2] for f in top if f[1] == ids[0])
        return BasicMatch(ids[0], "toneless" if tl else "exact", ids, top[0][3])

    def prior(self, m: BasicMatch) -> np.ndarray:
        """D^def, shape (12,), scale -3..3. Unrated contexts are neutral (zeros)."""
        if m.how == "generic" or not m.ids:
            return self.generic.copy()
        profs = [self._profile(cid) for cid in m.ids]
        return np.mean(profs, axis=0)

    def _profile(self, cid: str) -> np.ndarray:
        r = self.rows[cid]
        return np.zeros(len(STYLES)) if r.mean is None else r.mean.copy()

    def icc_of(self, m: BasicMatch) -> np.ndarray | None:
        if not m.ids:
            return None
        iccs = [self.rows[c].icc2k for c in m.ids if self.rows[c].icc2k is not None]
        if not iccs:
            return None
        return np.fmin.reduce(np.vstack(iccs), axis=0)       # NaN (unknown) only where all are unknown

    def is_rated(self, m: BasicMatch) -> bool:
        if m.how == "generic" or not m.ids:
            return any(r.rated for r in self.rows.values())
        return all(self.rows[c].rated for c in m.ids)
