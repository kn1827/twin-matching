"""Text normalisation, clause splitting and longest-match phrase lookup (stage1.md, section 3.3).

Vietnamese is matched on syllables: the text is NFC-normalised, lower-cased and split on whitespace, and
every dictionary entry is stored as a syllable sequence, so a longest match over syllable n-grams
(n <= ngram_max) needs no word segmenter. Tone placement is canonicalised ("hoà" == "hòa",
"tuỳ" == "tùy") because both spellings are in common use.

Toneless matching [quy ước]: a span may match an entry through its toneless form only when (a) no toned
entry matches at that position, (b) the toneless form is unique in the dictionary and (c) every syllable
of the span was typed without diacritics. Rule (c) keeps a toned word such as "hỏi" from matching the
toneless form "hoi" of the modifier "hơi".
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass

CLAUSE_PUNCT = ",;.!?"
CLAUSE_WORDS = ("nhưng", "but")
NO_SPLIT_AFTER = {"but": ("also",)}         # "not only X but also Y" stays one clause

_TONE_FIX = {}
for _base, _tones in (("o", "àáảãạ"), ("o", "èéẻẽẹ"), ("u", "ỳýỷỹỵ")):
    for _t in _tones:
        # old style puts the tone on the second vowel of an open syllable "oa", "oe", "uy"
        _plain2 = unicodedata.normalize("NFD", _t)[0]
        _toned1 = unicodedata.normalize("NFC", _base + unicodedata.normalize("NFD", _t)[1:])
        _TONE_FIX[_base + _t] = _toned1 + _plain2
_TONE_RE = re.compile(r"(?<![q])(" + "|".join(map(re.escape, _TONE_FIX)) + r")$")


def _canon_syllable(s: str) -> str:
    return _TONE_RE.sub(lambda m: _TONE_FIX[m.group(1)], s)


def normalize(text: str) -> str:
    """NFC, lower case, canonical tone placement, punctuation kept as separate tokens, single spaces."""
    t = unicodedata.normalize("NFC", text).lower()
    t = t.replace("’", "'").replace("‘", "'")
    t = re.sub(r"([,;.!?])", r" \1 ", t)
    t = re.sub(r"[\"“”()\[\]{}:/\\|*_+=<>~`^#@&%-]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(_canon_syllable(s) for s in t.split(" ")) if t else ""


def strip_tones(text: str) -> str:
    """Remove Vietnamese diacritics, đ -> d."""
    t = unicodedata.normalize("NFD", text)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", t).replace("đ", "d").replace("Đ", "D")


def split_clauses(text: str, *, punct: str = CLAUSE_PUNCT, words: Iterable[str] = CLAUSE_WORDS) -> list[list[str]]:
    """Clauses as syllable lists. Split at , ; . ! ? and at "nhưng" / "but", never at "và" / "and"."""
    toks = normalize(text).split(" ") if text and text.strip() else []
    words = set(words)
    out: list[list[str]] = [[]]
    for i, tok in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if tok in punct and len(tok) == 1:
            out.append([])
        elif tok in words and nxt not in NO_SPLIT_AFTER.get(tok, ()):
            out.append([])
        else:
            out[-1].append(tok)
    return [c for c in out if c]


def phrase_key(phrase: str) -> tuple[str, ...]:
    return tuple(normalize(phrase).split(" "))


@dataclass(frozen=True)
class Match:
    start: int
    end: int                    # exclusive, syllable index within the clause
    payload: Hashable
    toneless: bool


class PhraseMatcher:
    """Left-to-right longest match of phrases over syllables, toned first, then toneless (unique only)."""

    def __init__(self, phrases: Mapping[str, Hashable], ngram_max: int = 4, toneless: bool = True) -> None:
        self.ngram_max = ngram_max
        self.toned: dict[tuple[str, ...], Hashable] = {}
        for ph, payload in phrases.items():
            key = phrase_key(ph)
            if len(key) > ngram_max:
                raise ValueError(f"phrase longer than ngram_max={ngram_max}: {ph!r}")
            old = self.toned.get(key)
            if old is not None and old != payload:
                raise ValueError(f"phrase {ph!r} maps to two different targets: {old} / {payload}")
            self.toned[key] = payload
        groups: dict[tuple[str, ...], set[Hashable]] = {}
        for key, payload in self.toned.items():
            groups.setdefault(tuple(strip_tones(s) for s in key), set()).add(payload)
        self.collisions = {k: v for k, v in groups.items() if len(v) > 1}
        self.toneless = ({k: next(iter(v)) for k, v in groups.items() if len(v) == 1} if toneless else {})

    def find_all(self, toks: list[str]) -> list[Match]:
        plain = [strip_tones(t) for t in toks]
        out: list[Match] = []
        i, n = 0, len(toks)
        while i < n:
            hit: Match | None = None
            for L in range(min(self.ngram_max, n - i), 0, -1):
                key = tuple(toks[i:i + L])
                if key in self.toned:
                    hit = Match(i, i + L, self.toned[key], False)
                    break
            if hit is None and self.toneless:
                for L in range(min(self.ngram_max, n - i), 0, -1):
                    span = toks[i:i + L]
                    if any(s != p for s, p in zip(span, plain[i:i + L])):
                        continue                      # rule (c): typed with diacritics
                    key = tuple(plain[i:i + L])
                    if key in self.toneless:
                        hit = Match(i, i + L, self.toneless[key], True)
                        break
            if hit is None:
                i += 1
            else:
                out.append(hit)
                i = hit.end
        return out
