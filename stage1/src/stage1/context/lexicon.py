"""Keyword lexicon + modifier rules (tech spec 3.3, 10.3).

The lexicon holds only words and their direction (pole). `find` locates keyword hits with a longest
match over token n-grams (accented pass first, then an accent-folded pass restricted to entries whose
folded form is unique, flagged `toneless`). `classify` assigns each hit a cue class from the rule table
of section 3.3, adapted to English:

| pattern                                         | example                          | class  |
|-------------------------------------------------|----------------------------------|--------|
| keyword, no modifier (or "need", "want")        | "discipline", "need creativity"  | base   |
| strong / obligation modifier                    | "really disciplined", "must be"  | strong |
| softener                                        | "somewhat careful", "not too"    | weak   |
| "no need for" / "not important" (post)          | "no need for creativity"         | neg    |
| reverse-pole keyword                            | "competitive" (COO), "late"      | rev    |
| prohibition + reverse keyword                   | "don't be late"                  | strong |
| plain negator + reverse keyword                 | "no competition", "not late"     | base   |
| fixed expressions (matched first)               | "X is a must" / "not only X but also Y" | strong / base |

Scope [convention]: a pre modifier acts on every keyword after it until the next pre modifier or the end
of the clause; a post modifier ("... is not important") acts on every keyword before it in its clause.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from stage1.context.styles import STYLES
from stage1.context.text import (DEFAULT_KEEP_BIGRAMS, DEFAULT_PUNCT, DEFAULT_SPLIT_WORDS, strip_tones,
                                 tokenize_keep_accents)

PRE_KINDS = ("strong", "weak", "neg_pre", "prohibit", "negator", "neutral")
POST_KINDS = ("neg_post", "strong_post")


@dataclass(frozen=True)
class Hit:
    style: str
    pole: int
    span: tuple[int, int]      # token span [start, end) inside the clause
    clause: int
    cls: str                   # "" until classify() runs
    toneless: bool             # matched only after accent folding


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _phrase(s: str) -> tuple[str, ...]:
    return tuple(tokenize_keep_accents(str(s)))


class Lexicon:
    """Keyword and modifier tables loaded from demand_lexicon_v*.yaml."""

    def __init__(self, raw: dict, sha256: str, path: Path | None = None, min_entries_per_style: int = 5):
        self.version = str(raw.get("version"))
        self.sha256 = sha256
        self.path = path
        self.keywords: dict[tuple[str, ...], tuple[str, int]] = {}
        self.modifiers: dict[tuple[str, ...], str] = {}
        per_style: Counter = Counter()
        for e in raw["entries"]:
            style, pole = str(e["style"]), int(e["pole"])
            if style not in STYLES or pole not in (-1, 1):
                raise ValueError(f"bad lexicon entry {e}")
            for w in e.get("en", []):
                if not isinstance(w, str):          # YAML turns bare no/yes/on/off into booleans
                    raise ValueError(f"lexicon entry {w!r} of {style} is not a string: quote it in the YAML")
                ph = _phrase(w)
                if ph in self.keywords and self.keywords[ph] != (style, pole):
                    raise ValueError(f"lexicon entry {w!r} belongs to two styles/poles: "
                                     f"{self.keywords[ph]} and {(style, pole)}")
                self.keywords[ph] = (style, pole)
                per_style[style] += 1
        short = {s: per_style[s] for s in STYLES if per_style[s] < min_entries_per_style}
        if short:
            raise ValueError(f"styles with fewer than {min_entries_per_style} entries: {short}")
        for kind, spec in raw.get("modifiers", {}).items():
            pos = spec.get("pos", "pre")
            k = kind if kind in PRE_KINDS + POST_KINDS else None
            if k is None:
                raise ValueError(f"unknown modifier kind {kind!r}")
            if (pos == "post") != (k in POST_KINDS):
                raise ValueError(f"modifier kind {kind!r} has wrong position {pos!r}")
            for w in spec.get("en", []):
                if not isinstance(w, str):
                    raise ValueError(f"modifier {w!r} ({kind}) is not a string: quote it in the YAML")
                self._add_modifier(_phrase(w), k)
        for fx in raw.get("fixed", []):
            self._add_fixed(str(fx["pattern"]), str(fx["cls"]))
        both = set(self.keywords) & set(self.modifiers)
        if both:
            raise ValueError(f"phrases that are both keyword and modifier: {sorted(both)}")
        split = raw.get("clause_split", {})
        self.punct = tuple(split.get("punct", DEFAULT_PUNCT))
        self.split_words = tuple(split.get("words", DEFAULT_SPLIT_WORDS))
        self.keep_bigrams = tuple(tuple(_phrase(b)) for b in split.get("keep_bigrams", [])) or DEFAULT_KEEP_BIGRAMS
        self.max_len = max(len(p) for p in list(self.keywords) + list(self.modifiers))
        # accent-folded tables, unique folded forms only (collision table built at load)
        self.folded_keywords, self.collisions = self._fold_table(self.keywords)
        self.folded_modifiers, mcoll = self._fold_table(self.modifiers)
        self.collisions.update(mcoll)

    def _add_modifier(self, ph: tuple[str, ...], kind: str) -> None:
        if ph in self.modifiers and self.modifiers[ph] != kind:
            raise ValueError(f"modifier {' '.join(ph)!r} has two kinds: {self.modifiers[ph]}, {kind}")
        self.modifiers[ph] = kind

    def _add_fixed(self, pattern: str, cls: str) -> None:
        """Literal pieces of a fixed pattern become modifiers: the piece before a slot is a pre modifier,
        the piece after the last slot of a pattern that starts with a slot is a post modifier."""
        kind_pre = {"strong": "strong", "base": "neutral", "weak": "weak", "neg": "neg_pre"}[cls]
        kind_post = {"strong": "strong_post", "neg": "neg_post"}.get(cls)
        parts = [p.strip() for p in pattern.replace("{Y}", "{X}").split("{X}")]
        starts_with_slot = pattern.strip().startswith("{")
        for k, piece in enumerate(parts):
            if not piece:
                continue
            is_last = k == len(parts) - 1
            if is_last and starts_with_slot and len(parts) == 2:
                if kind_post is None:
                    raise ValueError(f"fixed pattern {pattern!r}: class {cls!r} cannot follow the keyword")
                self._add_modifier(_phrase(piece), kind_post)
            else:
                self._add_modifier(_phrase(piece), kind_pre)

    @staticmethod
    def _fold_table(table: dict) -> tuple[dict, dict]:
        groups: dict[tuple[str, ...], list] = defaultdict(list)
        for ph, val in table.items():
            groups[tuple(strip_tones(t) for t in ph)].append(val)
        unique = {f: v[0] for f, v in groups.items() if len({repr(x) for x in v}) == 1}
        coll = {f: v for f, v in groups.items() if len({repr(x) for x in v}) > 1}
        return unique, coll

    @classmethod
    def load(cls, path: Path | str, sha256: str | None = None, min_entries_per_style: int = 5) -> "Lexicon":
        path = Path(path)
        digest = file_sha256(path)
        if sha256 and sha256 != digest:
            raise ValueError(f"lexicon {path} sha256 {digest} does not match the locked {sha256}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(raw, digest, path, min_entries_per_style)

    # -------------------------------------------------------------------------------------- scanning
    def _scan(self, clause: list[str]) -> list[tuple[str, tuple[int, int], object, bool]]:
        """Longest match left to right over keywords and modifiers.

        Returns items (kind, span, payload, toneless) with kind "kw" (payload (style, pole)) or "mod"
        (payload modifier kind).
        """
        out = []
        folded = [strip_tones(t) for t in clause]
        i, n = 0, len(clause)
        while i < n:
            found = None
            for L in range(min(self.max_len, n - i), 0, -1):
                ph = tuple(clause[i:i + L])
                if ph in self.keywords:
                    found = ("kw", (i, i + L), self.keywords[ph], False)
                elif ph in self.modifiers:
                    found = ("mod", (i, i + L), self.modifiers[ph], False)
                else:
                    fph = tuple(folded[i:i + L])
                    if fph in self.folded_keywords:
                        found = ("kw", (i, i + L), self.folded_keywords[fph], True)
                    elif fph in self.folded_modifiers:
                        found = ("mod", (i, i + L), self.folded_modifiers[fph], True)
                if found:
                    break
            if found:
                out.append(found)
                i = found[1][1]
            else:
                i += 1
        return out

    def find(self, clause: list[str], clause_index: int = 0) -> list[Hit]:
        """Keyword hits of one clause (class not yet assigned)."""
        return [Hit(style=p[0], pole=p[1], span=sp, clause=clause_index, cls="", toneless=tl)
                for kind, sp, p, tl in self._scan(clause) if kind == "kw"]

    def classify(self, clause: list[str], hits: list[Hit]) -> list[Hit]:
        """Assign the cue class of every hit of `clause` (rule table in the module docstring)."""
        items = self._scan(clause)
        out = []
        for h in hits:
            pre = None
            post_neg = post_strong = False
            for kind, sp, payload, _ in items:
                if kind != "mod":
                    continue
                if sp[1] <= h.span[0] and payload in PRE_KINDS:
                    pre = payload                      # latest pre modifier before the keyword wins
                elif sp[0] >= h.span[1] and payload == "neg_post":
                    post_neg = True
                elif sp[0] >= h.span[1] and payload == "strong_post":
                    post_strong = True
            out.append(replace(h, cls=resolve_class(h.pole, pre, post_neg, post_strong)))
        return out

    def read(self, clauses: list[list[str]]) -> list[Hit]:
        hits: list[Hit] = []
        for c, clause in enumerate(clauses):
            hits.extend(self.classify(clause, self.find(clause, c)))
        return hits


_POS_CLASS = {None: "base", "neutral": "base", "strong": "strong", "weak": "weak",
              "neg_pre": "neg", "prohibit": "neg", "negator": "neg"}
_REV_CLASS = {None: "rev", "neutral": "rev", "strong": "rev", "weak": "rev",
              "neg_pre": "base", "prohibit": "strong", "negator": "base"}


def resolve_class(pole: int, pre: str | None, post_neg: bool, post_strong: bool) -> str:
    """Rule table of section 3.3 as a pure function."""
    if post_neg:
        return "neg"
    if pole > 0:
        cls = _POS_CLASS[pre]
        return "strong" if (post_strong and cls == "base") else cls
    return _REV_CLASS[pre]
