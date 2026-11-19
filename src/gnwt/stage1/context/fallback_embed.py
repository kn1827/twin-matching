"""OPTIONAL fallback, off by default (stage1.md, section 3.4): nearest basic context of an unknown name.

A small local sentence encoder (default intfloat/multilingual-e5-small) compares the text with every
catalog alias and returns the closest basic context when the cosine similarity clears a threshold
`q_hat` calibrated offline on held-out phrasings (conformal, target error 0.10). It never scores demand
levels. The encoder is imported lazily, only when `fallback_embedding.enabled = true`, so the default
Stage 1 run imports no embedding library (test U11).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from gnwt.stage1.context.catalog import BasicMatch, Catalog


@lru_cache(maxsize=2)
def _encoder(model_id: str) -> Any:  # pragma: no cover - needs a local model
    from sentence_transformers import SentenceTransformer  # lazy: never imported when disabled

    return SentenceTransformer(model_id)


def nearest_basic(text: str, model_id: str, q_hat: float, catalog: Catalog) -> BasicMatch:  # pragma: no cover
    enc = _encoder(model_id)
    names, cids = [], []
    for cid, row in catalog.rows.items():
        for lang in row.aliases.values():
            for a in lang:
                names.append(f"query: {a}")
                cids.append(cid)
    E = enc.encode([f"query: {text}", *names], normalize_embeddings=True)
    sims = E[1:] @ E[0]
    best = int(np.argmax(sims))
    if float(sims[best]) < q_hat:
        return BasicMatch(None, "generic", ())
    return BasicMatch(cids[best], "embed", (cids[best],), names[best][7:])
