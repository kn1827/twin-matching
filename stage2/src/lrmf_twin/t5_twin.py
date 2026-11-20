"""T5 corrective twin: one LLM call per cell, T = 0, no chain of thought -> verbalised distribution D.

* T5a (independent): the twin reads the person's profile, the top-k related answers (T4) and the target
  question; it does not see the base prediction.
* T5b (corrective, design V4 / CTRL pattern): same prompt plus "a statistical model estimates x% for the
  target answer"; the twin returns its corrected distribution.

The distribution over the question's options is mapped to P(y = 1) by summing the options of the positive
set, then to D = logit(P) with P clipped to [0.02, 0.98]. A call that fails or cannot be parsed gives
D = NaN; downstream (T6) treats it as "no twin information" for that cell.
"""

from __future__ import annotations

import math

from lrmf_twin.data.twin2k import SurveyData
from lrmf_twin.llm import LLMClient, parse_probs
from lrmf_twin.t4_retrieval import evidence_lines, short_question

OPTION_CHARS = 90
P_CLIP = (0.02, 0.98)


def _options_block(options: list[str]) -> str:
    out = []
    for k, o in enumerate(options, start=1):
        o = o if len(o) <= OPTION_CHARS else o[:OPTION_CHARS].rsplit(" ", 1)[0] + "..."
        out.append(f"{k}) {o}")
    return "\n".join(out)


def build_prompt(data: SurveyData, p: int, j: int, evidence: list[int], base_prob: float | None = None) -> str:
    item = data.items[j]
    lines = [f"Respondent: {data.demo_text[p]}"]
    if evidence:
        lines.append("Their answers to related questions:")
        lines += evidence_lines(data, p, evidence)
    if base_prob is not None:
        target = " or ".join(item.positive_labels())
        target = target if len(target) <= 120 else target[:120] + "..."
        lines.append(f"A statistical model trained on other respondents estimates a {100 * base_prob:.0f}% chance "
                     f"that this respondent answers: {target}. Correct it if their answers suggest otherwise.")
    lines.append(f"Question: {short_question(data, j, stem_chars=300)}")
    lines.append("Options:\n" + _options_block(item.options))
    n = len(item.options)
    lines.append(f'Give the probability that this respondent chose each option. Return JSON {{"p": [...]}} '
                 f"with exactly {n} numbers summing to 1, in option order.")
    return "\n".join(lines)


def to_logit(dist: list[float] | None, positive: list[int]) -> float:
    if dist is None:
        return float("nan")
    prob = sum(dist[k - 1] for k in positive)
    prob = min(max(prob, P_CLIP[0]), P_CLIP[1])
    return math.log(prob / (1.0 - prob))


def twin_signal(llm: LLMClient, data: SurveyData, p: int, j: int, evidence: list[int],
                base_prob: float | None = None) -> tuple[float, str]:
    prompt = build_prompt(data, p, j, evidence, base_prob)
    dist = parse_probs(llm.complete(prompt), len(data.items[j].options))
    return to_logit(dist, data.items[j].positive), prompt
