"""OPTIONAL, OFF BY DEFAULT: nearest basic context for an unknown context name (tech spec 3.4).

A small local embedding model (default intfloat/multilingual-e5-small) is used ONLY to find the
closest basic context of a name the alias table does not know; it never scores a demand level. The
similarity threshold `q_hat` must be calibrated (conformal) on held-out paraphrases; below it the match
stays generic. Imported lazily by demand.extract_demands only when `fallback_embedding.enabled`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from stage1.context.catalog import BasicMatch, Catalog


@lru_cache(maxsize=2)
def _model(model_id: str):
    from sentence_transformers import SentenceTransformer   # optional dependency
    return SentenceTransformer(model_id)


def nearest_basic(text: str, catalog: Catalog, cfg: dict) -> BasicMatch:
    model_id = cfg.get("model", "intfloat/multilingual-e5-small")
    q_hat = cfg.get("q_hat")
    if q_hat is None:
        raise ValueError("fallback_embedding.q_hat must be calibrated before the fallback is enabled")
    ids, names = [], []
    for e in catalog.contexts.values():
        for a in e.aliases:
            ids.append(e.id)
            names.append(f"query: {a}")
    m = _model(model_id)
    E = m.encode(names, normalize_embeddings=True)
    q = m.encode([f"query: {text}"], normalize_embeddings=True)[0]
    sims = E @ q
    j = int(np.argmax(sims))
    if sims[j] < q_hat:
        return BasicMatch(None, "generic", ())
    return BasicMatch(ids[j], "embed", (ids[j],))


def nearest_basic_by_model(text: str, model_id: str, q_hat: float, catalog: Catalog) -> BasicMatch:
    """Signature of the spec (text, model_id, q_hat) plus the catalog to search."""
    return nearest_basic(text, catalog, {"model": model_id, "q_hat": q_hat})
