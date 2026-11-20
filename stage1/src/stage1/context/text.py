"""Text normalisation and clause splitting for the keyword reader (tech spec 3.3).

English only. Text is Unicode-NFC normalised, lower-cased, typographic quotes are mapped to ASCII,
hyphens are read as spaces, and every word becomes one token. Clauses are split at punctuation and at
contrast words ("but", "however", ...), never at "and", so that "very disciplined and punctual" keeps
"very" for both keywords. "but also" (as in "not only X but also Y") does not split.
"""

from __future__ import annotations

import re
import unicodedata

_QUOTES = {"’": "'", "‘": "'", "ʼ": "'", "“": '"', "”": '"', "`": "'"}
_DASHES = re.compile(r"[‐-―\-_/]+")
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[,;.!?:]")

DEFAULT_PUNCT = (",", ";", ".", "!", "?", ":")
DEFAULT_SPLIT_WORDS = ("but", "however", "although", "though", "whereas")
DEFAULT_KEEP_BIGRAMS = (("but", "also"),)


def normalize(text: str) -> str:
    """NFC, lower case, ASCII quotes, hyphens/slashes as spaces, collapsed whitespace."""
    t = unicodedata.normalize("NFC", text or "").lower()
    for a, b in _QUOTES.items():
        t = t.replace(a, b)
    t = _DASHES.sub(" ", t)
    return _WS.sub(" ", t).strip()


def strip_tones(text: str) -> str:
    """Remove diacritics (the spec's tone stripping; for English: café -> cafe, naïve -> naive)."""
    t = unicodedata.normalize("NFD", text).replace("đ", "d").replace("Đ", "D")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", t)


def tokenize(text: str) -> list[str]:
    """Normalised word and punctuation tokens. Letters outside a-z are folded first (accents)."""
    return _TOKEN.findall(strip_tones(normalize(text)))


def tokenize_keep_accents(text: str) -> list[str]:
    """Tokens before accent folding, used for the exact (accented) pass of the matcher."""
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?|[,;.!?:]", normalize(text))


def split_clauses(text: str, punct: tuple[str, ...] = DEFAULT_PUNCT,
                  words: tuple[str, ...] = DEFAULT_SPLIT_WORDS,
                  keep_bigrams: tuple[tuple[str, str], ...] = DEFAULT_KEEP_BIGRAMS,
                  fold: bool = False) -> list[list[str]]:
    """Clauses as lists of tokens. Splits at `punct` and at `words` unless the pair is in `keep_bigrams`."""
    toks = tokenize(text) if fold else tokenize_keep_accents(text)
    keep = set(keep_bigrams)
    clauses: list[list[str]] = [[]]
    for k, tok in enumerate(toks):
        nxt = toks[k + 1] if k + 1 < len(toks) else None
        if tok in punct or (tok in words and (tok, nxt) not in keep):
            if clauses[-1]:
                clauses.append([])
            continue
        clauses[-1].append(tok)
    return [c for c in clauses if c]
