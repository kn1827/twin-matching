"""Context demands: catalog prior + keyword evidence merged by Buhlmann credibility (11b), then mapped
to the five Big Five domains (11c).

    D_hat_s(t) = Z_s * ybar_s(t) + (1 - Z_s) * D^def_s(c*),   Z_s = n_s / (n_s + k_s),
    ybar_s(t)  = mean of theta_{class(h)} over the hits h of style s in t                    (11b)
    D~_d(t)    = sum_{s in S_d} D_hat_s(t) / (3 |S_d|)                                         (11c)

theta (cue-class levels) and k (prior strength) come from ReaderArtifacts, written by the offline
estimator (stage1/offline/reader.py). Nothing here calls a language model or the network.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from stage1.context.catalog import BasicMatch, Catalog
from stage1.context.lexicon import Hit, Lexicon
from stage1.context.styles import CUE_CLASSES, STYLE_INDEX, STYLES, domain_matrix
from stage1.context.text import normalize, split_clauses

CLS_INDEX = {c: i for i, c in enumerate(CUE_CLASSES)}


@dataclass
class ReaderArtifacts:
    """Locked numbers of the reader plus the lexicon and catalog they were estimated with."""
    lexicon: Lexicon
    catalog: Catalog
    theta: np.ndarray                 # (12, 5) level of each cue class per style, -3..3, order CUE_CLASSES
    k: np.ndarray                     # (12,) prior strength
    keep: np.ndarray                  # (12,) bool, styles that passed the ICC filter
    status: str = "unestimated"       # estimated | illustrative | unestimated
    meta: dict = field(default_factory=dict)
    fallback_embedding: dict | None = None

    @property
    def hash(self) -> str:
        payload = {"theta": np.round(self.theta, 12).tolist(), "k": np.round(self.k, 12).tolist(),
                   "keep": self.keep.astype(int).tolist(), "status": self.status,
                   "lexicon": self.lexicon.sha256, "catalog": self.catalog.sha256}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def to_json(self) -> dict:
        return {"theta": {s: dict(zip(CUE_CLASSES, map(float, self.theta[i]))) for i, s in enumerate(STYLES)},
                "k": dict(zip(STYLES, map(float, self.k))), "keep": dict(zip(STYLES, map(bool, self.keep))),
                "status": self.status, "lexicon_sha256": self.lexicon.sha256,
                "lexicon_version": self.lexicon.version, "catalog_sha256": self.catalog.sha256,
                "catalog_version": self.catalog.version, "meta": self.meta, "hash": self.hash}

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return self.hash

    @classmethod
    def load(cls, path: Path, lexicon: Lexicon, catalog: Catalog) -> "ReaderArtifacts":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw["lexicon_sha256"] != lexicon.sha256 or raw["catalog_sha256"] != catalog.sha256:
            raise ValueError("reader artifacts were estimated with another lexicon/catalog version: re-estimate")
        theta = np.array([[raw["theta"][s][c] for c in CUE_CLASSES] for s in STYLES])
        art = cls(lexicon, catalog, theta, np.array([raw["k"][s] for s in STYLES]),
                  np.array([raw["keep"][s] for s in STYLES]), raw["status"], raw.get("meta", {}))
        if art.hash != raw["hash"]:
            raise ValueError(f"reader artifact {path} does not match its recorded hash")
        return art

    @classmethod
    def illustrative(cls, lexicon: Lexicon, catalog: Catalog) -> "ReaderArtifacts":
        """The ILLUSTRATIVE numbers of spec section 3.5 (theta_neg -0.5, base 2.0, strong 2.7, k 0.5).

        weak and rev are not given there; 1.0 and -1.0 are placeholders. For demos and tests only:
        every output built from it carries status "illustrative".
        """
        row = np.array([-0.5, -1.0, 1.0, 2.0, 2.7])          # neg, rev, weak, base, strong
        return cls(lexicon, catalog, np.tile(row, (len(STYLES), 1)), np.full(len(STYLES), 0.5),
                   np.ones(len(STYLES), bool), "illustrative", {"source": "tech spec 3.5 example"})


@dataclass(frozen=True)
class ContextDemand:
    D_style: np.ndarray               # (12,) D_hat on -3..3
    D_tilde: np.ndarray               # (5,) on -1..1, order DOMAINS
    Z: np.ndarray                     # (12,) weight given to the keywords
    match: BasicMatch
    hits: list[Hit]
    text_hash: str
    artifact_hash: str
    source: str = "user"              # user | event | market | none
    rated_prior: bool = True          # False when the catalog profile is not yet rated (neutral 0)
    reader_status: str = "estimated"

    def provenance(self) -> dict:
        return {"c_star": self.match.c_star, "how": self.match.how, "ids": list(self.match.ids),
                "source": self.source, "rated_prior": self.rated_prior, "reader_status": self.reader_status,
                "hits": [{"style": h.style, "pole": h.pole, "cls": h.cls, "clause": h.clause,
                          "span": list(h.span), "toneless": h.toneless} for h in self.hits],
                "Z": dict(zip(STYLES, np.round(self.Z, 6).tolist()))}


def text_hash(text: str | None) -> str:
    return hashlib.sha256(normalize(text or "").encode("utf-8")).hexdigest()


def _clauses(text: str, lex: Lexicon) -> list[list[str]]:
    return split_clauses(text, lex.punct, lex.split_words, lex.keep_bigrams)


def evidence(hits: list[Hit], theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """n_s and ybar_s for the 12 styles (ybar = 0 where n_s = 0)."""
    n = np.zeros(len(STYLES))
    tot = np.zeros(len(STYLES))
    for h in hits:
        s = STYLE_INDEX[h.style]
        n[s] += 1
        tot[s] += theta[s, CLS_INDEX[h.cls]]
    ybar = np.divide(tot, n, out=np.zeros_like(tot), where=n > 0)
    return n, ybar


def credibility(prior: np.ndarray, n: np.ndarray, ybar: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(11b): D_hat = Z ybar + (1 - Z) D^def with Z = n / (n + k). Exact prior when n = 0."""
    Z = n / (n + k)
    D = np.where(n > 0, Z * ybar + (1 - Z) * prior, prior)
    return D, Z


def extract_demands(text: str | None, event_ctx: str | None, art: ReaderArtifacts,
                    market: str | None = None) -> ContextDemand:
    """Demand vector of a context sentence (tech spec 3.5).

    Fallback when the sentence is empty [no_text_fallback: event_then_market]: the event description,
    then the market's basic context `market`, then the generic prior.
    """
    lex, cat = art.lexicon, art.catalog
    if text and text.strip():
        source, body = "user", text
    elif event_ctx and event_ctx.strip():
        source, body = "event", event_ctx
    elif market:
        source, body = "market", None
    else:
        source, body = "none", None

    hits: list[Hit] = []
    if body is not None:
        clauses = _clauses(body, lex)
        m = cat.match(clauses)
        if m.how == "generic" and art.fallback_embedding and art.fallback_embedding.get("enabled"):
            from stage1.context.fallback_embed import nearest_basic   # optional, off by default
            m = nearest_basic(body, cat, art.fallback_embedding)
        if source == "event" and m.c_star is not None:
            m = BasicMatch(m.c_star, "event", m.ids)
        hits = lex.read(clauses)
    elif market:
        m = cat.by_id(market, "market")
    else:
        m = BasicMatch(None, "generic", ())

    prior = cat.prior(m)
    n, ybar = evidence(hits, art.theta)
    D_style, Z = credibility(prior, n, ybar, art.k)
    D_tilde = domain_matrix(art.keep) @ D_style
    return ContextDemand(D_style=D_style, D_tilde=D_tilde, Z=Z, match=m, hits=hits,
                         text_hash=text_hash(body), artifact_hash=art.hash, source=source,
                         rated_prior=cat.is_rated(m), reader_status=art.status)
