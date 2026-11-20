"""T3 sparse-person prior: one LLM call per sparse person, once.

The LLM reads the person's demographics and the few answers already known, and estimates P(target answer)
on a fixed list of anchor questions. These soft labels become pseudo-observations in
`LRMFBase.refit_persons`, so the person's MF factor is pulled toward what the LLM expects instead of
toward 0. The gate: only persons with n_p <= `max_known` answers get a prior, and its weight
n0 / (n0 + n_p) decreases with the person's own data (design T3 / V3, S-LLMR "history <= k").

Anchors: items with the largest MF factor norm among well-answered items, greedily kept only if their factor
is not almost collinear (|cos| < 0.7) with an anchor already chosen, so the pseudo-answers pin down
different directions of u_p.
"""

from __future__ import annotations

import numpy as np

from lrmf_twin.data.twin2k import SurveyData
from lrmf_twin.llm import LLMClient, parse_list
from lrmf_twin.t4_retrieval import evidence_lines, short_question


def select_anchors(V: np.ndarray, n_j: np.ndarray, n_anchor: int = 16, min_answers: int = 500,
                   max_cos: float = 0.7) -> list[int]:
    norms = np.linalg.norm(V, axis=1)
    order = [int(j) for j in np.argsort(-norms) if n_j[j] >= min_answers]
    chosen: list[int] = []
    for j in order:
        vj = V[j] / max(norms[j], 1e-9)
        if all(abs(float(vj @ (V[c] / max(norms[c], 1e-9)))) < max_cos for c in chosen):
            chosen.append(j)
        if len(chosen) == n_anchor:
            break
    return chosen


def build_prompt(data: SurveyData, p: int, known: list[int], anchors: list[int]) -> str:
    lines = [f"Respondent: {data.demo_text[p]}"]
    if known:
        lines.append("Answers we already know:")
        lines += evidence_lines(data, p, known, stem_chars=70)
    lines.append("For each question below, estimate the probability that this respondent's answer is the "
                 "TARGET answer.")
    for i, j in enumerate(anchors, start=1):
        target = " or ".join(data.items[j].positive_labels())
        target = target if len(target) <= 100 else target[:100] + "..."
        lines.append(f"{i}. {short_question(data, j, 90)} TARGET: {target}")
    lines.append(f'Return JSON {{"p": [...]}} with exactly {len(anchors)} numbers between 0 and 1.')
    return "\n".join(lines)


def person_prior(llm: LLMClient, data: SurveyData, p: int, known: list[int], anchors: list[int],
                 n_use: int = 8) -> tuple[np.ndarray, np.ndarray] | None:
    use = [a for a in anchors if a not in set(known)][:n_use]
    if not use:
        return None
    probs = parse_list(llm.complete(build_prompt(data, p, known, use)), len(use))
    if probs is None:
        return None
    return np.asarray(use, dtype=int), np.clip(np.asarray(probs, dtype=np.float32), 0.02, 0.98)
