"""Demand vector of one context sentence (stage1.md, sections 3.5 and 8; formulas (11b), (11c)).

    D^_s(t) = Z_s * ybar_s(t) + (1 - Z_s) * D^def_s(c*),   Z_s = n_s / (n_s + k_s),
    ybar_s  = mean of theta_kappa(h) over the hits h of style s                    (11b)
    D~_d(t) = 1 / (3 |S_d|) * sum_{s in S_d} D^_s(t)                                (11c)

theta (one level per cue class), k (prior strength) and the per-style ICC come from `ReaderArtifacts`,
written by the offline estimator. When no estimated artifact exists yet (`theta is None`), keyword hits
are still detected and logged, but Z_s = 0, so D^ = D^def: the formula with k_s = infinity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gnwt.stage1.context.catalog import BasicMatch, Catalog
from gnwt.stage1.context.lexicon import Hit, Lexicon
from gnwt.stage1.context.styles import CUE_CLASSES, DOMAINS, STYLE_INDEX, STYLES, WI_MAX, styles_of
from gnwt.stage1.context.text import normalize


@dataclass(frozen=True)
class ReaderArtifacts:
    lexicon: Lexicon
    catalog: Catalog
    theta: dict[str, float] | np.ndarray | None      # {cls: level} shared, or (12, 5) per style
    k: np.ndarray | None                              # (12,) prior strength; None with theta None
    icc: np.ndarray | None = None                     # (12,) ICC(2,k) of the label set, None = unknown
    icc_min: float = 0.70
    provenance: str = "unestimated"                   # estimated | illustrative | unestimated
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def artifact_hash(self) -> str:
        theta = (None if self.theta is None else
                 (self.theta.tolist() if isinstance(self.theta, np.ndarray) else
                  {c: float(self.theta[c]) for c in sorted(self.theta)}))
        payload = {"lexicon": self.lexicon.sha256, "catalog": self.catalog.sha256, "theta": theta,
                   "k": None if self.k is None else [float(x) for x in self.k],
                   "icc": None if self.icc is None else [None if np.isnan(x) else float(x) for x in self.icc],
                   "icc_min": self.icc_min, "provenance": self.provenance}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def theta_of(self, style: str, cls: str) -> float:
        assert self.theta is not None
        if isinstance(self.theta, np.ndarray):
            return float(self.theta[STYLE_INDEX[style], CUE_CLASSES.index(cls)])
        return float(self.theta[cls])

    def kept_styles(self, catalog_icc: np.ndarray | None) -> np.ndarray:
        """Styles kept in S_d: ICC known and below icc_min drops the style [quy ước]."""
        keep = np.ones(len(STYLES), dtype=bool)
        for icc in (self.icc, catalog_icc):
            if icc is not None:
                keep &= ~(np.isfinite(icc) & (icc < self.icc_min))
        return keep

    @classmethod
    def from_json(cls, path: str | Path, lexicon: Lexicon, catalog: Catalog) -> ReaderArtifacts:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("lexicon_sha256") not in (None, lexicon.sha256):
            raise ValueError("reader artifact was estimated with another lexicon version")
        if raw.get("catalog_sha256") not in (None, catalog.sha256):
            raise ValueError("reader artifact was estimated with another catalog version")
        th = raw.get("theta")
        theta = (None if th is None else np.array(th, dtype=float) if isinstance(th, list)
                 else {c: (float("nan") if v is None else float(v)) for c, v in th.items()})
        k = raw.get("k")
        k_arr = None if k is None else (np.full(len(STYLES), float(k)) if np.isscalar(k) else np.array(k, float))
        icc = raw.get("icc")
        icc_arr = None if icc is None else np.array([np.nan if v is None else v for v in icc], float)
        return cls(lexicon, catalog, theta, k_arr, icc_arr, float(raw.get("icc_min", 0.70)),
                   str(raw.get("provenance", "estimated")), dict(raw.get("meta", {})))

    @classmethod
    def unestimated(cls, lexicon: Lexicon, catalog: Catalog, icc_min: float = 0.70) -> ReaderArtifacts:
        return cls(lexicon, catalog, None, None, None, icc_min, "unestimated")


@dataclass(frozen=True)
class ContextDemand:
    D_style: np.ndarray          # (12,) on -3..3
    D_tilde: np.ndarray          # (5,) on -1..1, order O C E A ES
    Z: np.ndarray                # (12,)
    match: BasicMatch
    hits: list[Hit]
    text_hash: str
    artifact_hash: str
    source: str = "user"         # user | event | market | none
    flags: tuple[str, ...] = ()
    n: np.ndarray = field(default_factory=lambda: np.zeros(len(STYLES)))

    def as_row(self, persona_id: str) -> dict[str, Any]:
        """Row of `context_demands` (section 10.2)."""
        return {
            "persona_id": persona_id, "text_hash": self.text_hash, "c_star": self.match.c_star,
            "match_how": self.match.how, "D_style": json.dumps([round(float(x), 6) for x in self.D_style]),
            **{f"D_{d}": float(v) for d, v in zip(DOMAINS, self.D_tilde)},
            "Z": json.dumps([round(float(x), 6) for x in self.Z]),
            "hits": json.dumps([{"style": h.style, "cls": h.cls, "span": [h.clause, *h.span],
                                 "toneless": h.toneless, "phrase": h.phrase} for h in self.hits],
                               ensure_ascii=False),
            "lexicon_ver": None, "catalog_ver": None, "artifact_hash": self.artifact_hash,
        }


def _domain_vector(D_hat: np.ndarray, keep: np.ndarray) -> np.ndarray:
    out = np.zeros(len(DOMAINS))
    for k, d in enumerate(DOMAINS):
        idx = [STYLE_INDEX[s] for s in styles_of(d) if keep[STYLE_INDEX[s]]]
        out[k] = float(np.sum(D_hat[idx])) / (WI_MAX * len(idx)) if idx else 0.0
    return out


def extract_demands(text: str | None, event_ctx: str | None, art: ReaderArtifacts,
                    market_ctx: str | None = None) -> ContextDemand:
    """(11b), (11c) for one sentence. Empty text -> event description -> market default context."""
    flags: list[str] = []
    source = "user"
    if not (text and text.strip()):
        flags.append("no_text")
        if event_ctx and event_ctx.strip():
            text, source = event_ctx, "event"
        else:
            text, source = "", ("market" if market_ctx else "none")
    clauses, hits = art.lexicon.read(text) if text else ([], [])
    if source == "market" and market_ctx:
        match = BasicMatch(market_ctx, "market", (market_ctx,))
    else:
        match = art.catalog.match(clauses)
        if source == "event" and match.c_star is not None:
            match = BasicMatch(match.c_star, "event", match.ids, match.alias)
    if match.how in ("multi", "generic", "toneless"):
        flags.append(match.how)
    if any(h.toneless for h in hits):
        flags.append("toneless_hit")
    if not art.catalog.is_rated(match):
        flags.append("unrated_context")

    D_def = art.catalog.prior(match)
    n = np.zeros(len(STYLES))
    ysum = np.zeros(len(STYLES))
    for h in hits:
        i = STYLE_INDEX[h.style]
        if art.theta is not None:
            lv = art.theta_of(h.style, h.cls)
            if not np.isfinite(lv):            # class never seen in the label set: not evidence
                continue
            ysum[i] += lv
        n[i] += 1
    if art.theta is None or art.k is None:
        if hits:
            flags.append("reader_unestimated")
        Z = np.zeros(len(STYLES))
        ybar = np.zeros(len(STYLES))
    else:
        Z = n / (n + art.k)
        ybar = np.divide(ysum, n, out=np.zeros(len(STYLES)), where=n > 0)
    D_hat = Z * ybar + (1.0 - Z) * D_def
    keep = art.kept_styles(art.catalog.icc_of(match))
    D_tilde = _domain_vector(D_hat, keep)
    text_hash = hashlib.sha256(normalize(text or "").encode("utf-8")).hexdigest()
    return ContextDemand(D_hat, D_tilde, Z, match, hits, text_hash, art.artifact_hash, source,
                         tuple(dict.fromkeys(flags)), n)
