"""Bilingual demand lexicon with VADER-style rules (stage1.md, sections 3.3 and 10.3).

`Lexicon.find` returns the keyword hits of one clause (longest syllable n-gram match, toned first, then
toneless); `Lexicon.classify` assigns each hit one of the five cue classes of table 3.3:

| pattern                                    | class  |
|--------------------------------------------|--------|
| keyword, no modifier (or "cần", "có")       | base   |
| strong modifier + keyword                   | strong |
| weak modifier + keyword                     | weak   |
| negation + keyword                          | neg    |
| reverse keyword                             | rev    |
| prohibition + reverse keyword               | strong |
| negation + reverse keyword                  | base   |
| fixed expression                            | its own class |

Scope [quy ước]: a modifier placed before keywords acts on every keyword after it up to the next modifier
or the end of the clause. A post-negation ("không quan trọng", "thì không cần") acts on every keyword
before it in the clause. A pre-modifier with no keyword after it in its clause ("kỷ luật là bắt buộc",
"kỷ luật không cần") acts backwards: a negation on every earlier keyword, a strong or weak modifier on the
earlier keywords that carry no modifier of their own. Cases the table leaves open are fixed here:
prohibition + direct keyword -> rev; strong or weak modifier + reverse keyword -> rev.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from gnwt.stage1.context.styles import STYLES
from gnwt.stage1.context.text import PhraseMatcher, phrase_key, split_clauses, strip_tones

MOD_CLASSES = ("strong", "weak", "neg_pre", "neg_post", "prohibit", "ignore")
_PRE = {"strong": "strong", "weak": "weak", "neg_pre": "neg", "prohibit": "prohibit"}


@dataclass(frozen=True)
class Hit:
    style: str
    pole: int
    span: tuple[int, int]
    clause: int
    cls: str
    toneless: bool
    phrase: str = ""


@dataclass(frozen=True)
class _Mod:
    kind: str          # "pre" | "post" | "ignore"
    cls: str           # strong | weak | neg | prohibit | base


class Lexicon:
    def __init__(self, raw: dict[str, Any], sha256: str, ngram_max: int = 4, toneless: bool = True,
                 min_entries_per_style_lang: int = 5) -> None:
        self.version = str(raw["version"])
        self.sha256 = sha256
        self.ngram_max = ngram_max
        split = raw.get("clause_split", {})
        self.split_punct = "".join(split.get("punct", [",", ";", ".", "!", "?"]))
        self.split_words = tuple(split.get("words", ["nhưng", "but"]))

        phrases: dict[str, Any] = {}
        counts: dict[tuple[str, str], int] = {}
        seen: dict[tuple[tuple[str, ...], int], str] = {}
        for e in raw["entries"]:
            style, pole = str(e["style"]), int(e["pole"])
            if style not in STYLES or pole not in (-1, 1):
                raise ValueError(f"bad lexicon entry: {e}")
            for lang in ("vi", "en"):
                for ph in e.get(lang) or []:
                    key = phrase_key(str(ph))
                    other = seen.get((key, pole))
                    if other is not None and other != style:
                        raise ValueError(f"{ph!r} belongs to {other} and {style} with the same pole")
                    seen[(key, pole)] = style
                    phrases[str(ph)] = ("kw", style, pole)
                    counts[(style, lang)] = counts.get((style, lang), 0) + 1
        short = [f"{s}/{lg}={counts.get((s, lg), 0)}" for s in STYLES for lg in ("vi", "en")
                 if counts.get((s, lg), 0) < min_entries_per_style_lang]
        if short:
            raise ValueError(f"fewer than {min_entries_per_style_lang} entries for: {', '.join(short)}")

        mods = raw.get("modifiers", {})
        for mcls in MOD_CLASSES:
            spec = mods.get(mcls) or {}
            for lang in ("vi", "en"):
                for ph in spec.get(lang) or []:
                    if str(ph) in phrases:
                        raise ValueError(f"{ph!r} is both a keyword and a modifier")
                    if mcls == "ignore":
                        phrases[str(ph)] = ("mod", _Mod("ignore", "base"))
                    elif mcls == "neg_post":
                        phrases[str(ph)] = ("mod", _Mod("post", "neg"))
                    else:
                        phrases[str(ph)] = ("mod", _Mod("pre", _PRE[mcls]))
        for fx in raw.get("fixed") or []:
            self._add_fixed(phrases, str(fx["pattern"]), str(fx["cls"]))

        self.matcher = PhraseMatcher(phrases, ngram_max=ngram_max, toneless=toneless)
        self.collisions = {" ".join(k): sorted(map(str, v)) for k, v in self.matcher.collisions.items()}
        self.n_entries = sum(1 for v in phrases.values() if v[0] == "kw")

    @staticmethod
    def _add_fixed(phrases: dict[str, Any], pattern: str, cls: str) -> None:
        """'lit1 {X} lit2 {Y} lit3' -> lit1, lit2 act as pre-modifiers, lit3 as a post-modifier."""
        import re

        parts = re.split(r"\{[A-Z]\}", pattern)
        for k, lit in enumerate(parts):
            lit = lit.strip()
            if not lit:
                continue
            kind = "post" if k == len(parts) - 1 and len(parts) > 1 else "pre"
            if lit in phrases and phrases[lit][0] == "kw":
                raise ValueError(f"fixed literal {lit!r} collides with a keyword")
            phrases[lit] = ("mod", _Mod(kind, cls))

    @classmethod
    def load(cls, path: str | Path, sha256: str | None = None, **kw: Any) -> Lexicon:
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if sha256 is not None and sha256 != digest:
            raise ValueError(f"lexicon {path}: sha256 mismatch (expected {sha256[:12]}, got {digest[:12]})")
        return cls(yaml.safe_load(data.decode("utf-8")), digest, **kw)

    # ------------------------------------------------------------------------------------ reading

    def clauses(self, text: str) -> list[list[str]]:
        return split_clauses(text, punct=self.split_punct, words=self.split_words)

    def _items(self, clause: list[str]) -> list[tuple[int, int, Any, bool]]:
        return [(m.start, m.end, m.payload, m.toneless) for m in self.matcher.find_all(clause)]

    def find(self, clause: list[str], clause_idx: int = 0) -> list[Hit]:
        out = []
        for s, e, payload, tl in self._items(clause):
            if payload[0] == "kw":
                out.append(Hit(payload[1], payload[2], (s, e), clause_idx, "", tl, " ".join(clause[s:e])))
        return out

    def classify(self, clause: list[str], hits: list[Hit]) -> list[Hit]:
        items = self._items(clause)
        pre_of: dict[tuple[int, int], str | None] = {}
        post_of: dict[tuple[int, int], str] = {}
        kw_spans = [(s, e) for s, e, p, _ in items if p[0] == "kw"]
        current: str | None = None
        for s, e, payload, _ in items:
            if payload[0] == "kw":
                pre_of[(s, e)] = current
                continue
            mod: _Mod = payload[1]
            if mod.kind == "ignore":
                continue
            before = [sp for sp in kw_spans if sp[1] <= s]
            after = [sp for sp in kw_spans if sp[0] >= e]
            if mod.kind == "post" or (mod.kind == "pre" and not after):
                for sp in before:
                    if mod.cls == "neg" or mod.kind == "post":
                        post_of[sp] = mod.cls
                    elif pre_of.get(sp) is None and sp not in post_of:
                        post_of[sp] = mod.cls
                current = None
            else:
                current = mod.cls
        out = []
        for h in hits:
            mod = post_of.get(h.span, pre_of.get(h.span))
            out.append(replace(h, cls=_class(h.pole, mod)))
        return out

    def read(self, text: str) -> tuple[list[list[str]], list[Hit]]:
        """Clauses and classified hits of a whole text."""
        clauses = self.clauses(text)
        hits: list[Hit] = []
        for ci, cl in enumerate(clauses):
            hits.extend(self.classify(cl, self.find(cl, ci)))
        return clauses, hits


def _class(pole: int, mod: str | None) -> str:
    if pole > 0:
        return {None: "base", "base": "base", "strong": "strong", "weak": "weak", "neg": "neg",
                "prohibit": "rev"}[mod]
    return {None: "rev", "base": "rev", "strong": "rev", "weak": "rev", "neg": "base",
            "prohibit": "strong"}[mod]


__all__ = ["Hit", "Lexicon", "strip_tones"]
