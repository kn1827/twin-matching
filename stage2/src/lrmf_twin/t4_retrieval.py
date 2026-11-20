"""T4 evidence retriever: top-k of the person's own training answers closest to the target question.

Similarity = cosine of the T2 item-text embeddings. The target item is always excluded, and only cells of
the fold's training mask are candidates, so a held-out answer can never reach the prompt (design 4.4 step 3).
"""

from __future__ import annotations

import numpy as np

from lrmf_twin.data.twin2k import SurveyData


def normalise(E: np.ndarray) -> np.ndarray:
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)


def retrieve(p: int, j: int, train: np.ndarray, En: np.ndarray, k: int = 5) -> list[int]:
    cand = np.flatnonzero(train[p])
    cand = cand[cand != j]
    if cand.size == 0:
        return []
    sims = En[cand] @ En[j]
    order = np.argsort(-sims, kind="stable")[:k]
    return [int(c) for c in cand[order]]


def short_question(data: SurveyData, j: int, stem_chars: int = 110) -> str:
    item = data.items[j]
    stem = item.stem if len(item.stem) <= stem_chars else item.stem[:stem_chars].rsplit(" ", 1)[0] + "..."
    return f"{stem} {item.row}" if item.row else stem


def evidence_lines(data: SurveyData, p: int, items: list[int], stem_chars: int = 110) -> list[str]:
    return [f"- {short_question(data, k, stem_chars)} -> {data.items[k].answer_label(float(data.V[p, k]))}"
            for k in items]
