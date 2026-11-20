"""Basic-context catalog: aliases, goal codes and default demand profiles D^def (tech spec 3.4, 10.4).

Alias matching uses the same longest-n-gram matcher as the lexicon (accented pass, then accent-folded
pass on unique folded aliases). Several contexts matched: keep the one with the longest alias; still
tied: the prior is the mean of the tied profiles and `how = "multi"` [convention]. Nothing matched: the
generic prior (mean of the rated catalog profiles) with `how = "generic"`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from stage1.context.lexicon import _phrase, file_sha256
from stage1.context.styles import STYLES
from stage1.context.text import strip_tones

MatchHow = Literal["exact", "toneless", "multi", "embed", "generic", "event", "market"]


@dataclass(frozen=True)
class BasicMatch:
    c_star: str | None
    how: MatchHow
    ids: tuple[str, ...]


@dataclass(frozen=True)
class ContextEntry:
    id: str
    aliases: tuple[str, ...]
    goal_codes: tuple[str, ...]
    profile: np.ndarray | None        # (12,) mean on -3..3, None when not yet rated
    se: np.ndarray | None
    icc2k: np.ndarray | None
    n_raters: int | None
    sign_check: dict


class Catalog:
    def __init__(self, raw: dict, sha256: str, path: Path | None = None):
        self.version = str(raw.get("version"))
        self.sha256 = sha256
        self.path = path
        self.contexts: dict[str, ContextEntry] = {}
        self.alias_table: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for c in raw["contexts"]:
            prof = c.get("profile")
            mean = se = icc = None
            if prof:
                missing = [s for s in STYLES if s not in prof]
                if missing:
                    raise ValueError(f"context {c['id']}: profile lacks styles {missing}")
                mean = np.array([float(prof[s]["mean"]) for s in STYLES])
                se = np.array([float(prof[s].get("se") or np.nan) for s in STYLES])
                icc = np.array([float(prof[s].get("icc2k") or np.nan) for s in STYLES])
                if np.any(np.abs(mean) > 3):
                    raise ValueError(f"context {c['id']}: profile outside the -3..3 scale")
            aliases = tuple(c.get("aliases", {}).get("en", []))
            entry = ContextEntry(c["id"], aliases, tuple(c.get("goal_codes", [])), mean, se, icc,
                                 c.get("n_raters"), dict(c.get("sign_check") or {}))
            if entry.id in self.contexts:
                raise ValueError(f"duplicate context id {entry.id}")
            self.contexts[entry.id] = entry
            for a in aliases + (entry.id.replace("_", " "),):
                self.alias_table[_phrase(a)].add(entry.id)
        self.max_len = max(len(p) for p in self.alias_table)
        folded: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for ph, ids in self.alias_table.items():
            folded[tuple(strip_tones(t) for t in ph)] |= ids
        self.folded_alias = {k: v for k, v in folded.items() if len(v) == 1}
        rated = [e.profile for e in self.contexts.values() if e.profile is not None]
        self.generic = np.mean(rated, axis=0) if rated else np.zeros(len(STYLES))

    @classmethod
    def load(cls, path: Path | str, sha256: str | None = None) -> "Catalog":
        path = Path(path)
        digest = file_sha256(path)
        if sha256 and sha256 != digest:
            raise ValueError(f"catalog {path} sha256 {digest} does not match the locked {sha256}")
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")), digest, path)

    @property
    def rated(self) -> bool:
        return all(e.profile is not None for e in self.contexts.values())

    def match(self, clauses: list[list[str]]) -> BasicMatch:
        """Basic context named in the text (longest alias wins)."""
        best_len, best_ids, toneless = 0, set(), False
        for clause in clauses:
            folded = [strip_tones(t) for t in clause]
            for i in range(len(clause)):
                for L in range(min(self.max_len, len(clause) - i), 0, -1):
                    ph = tuple(clause[i:i + L])
                    ids, tl = self.alias_table.get(ph), False
                    if ids is None:
                        ids, tl = self.folded_alias.get(tuple(folded[i:i + L])), True
                    if ids:
                        if L > best_len:
                            best_len, best_ids, toneless = L, set(ids), tl
                        elif L == best_len:
                            best_ids |= ids
                            toneless = toneless and tl
                        break
        if not best_ids:
            return BasicMatch(None, "generic", ())
        ids = tuple(sorted(best_ids))
        if len(ids) > 1:
            return BasicMatch(None, "multi", ids)
        return BasicMatch(ids[0], "toneless" if toneless else "exact", ids)

    def by_id(self, cid: str, how: MatchHow) -> BasicMatch:
        if cid not in self.contexts:
            raise KeyError(f"unknown basic context {cid!r}")
        return BasicMatch(cid, how, (cid,))

    def profile_of(self, cid: str) -> np.ndarray:
        """D^def of one context; an unrated context is the neutral demand 0."""
        p = self.contexts[cid].profile
        return np.zeros(len(STYLES)) if p is None else p.copy()

    def prior(self, m: BasicMatch) -> np.ndarray:
        """D^def for a match, shape (12,)."""
        if not m.ids:
            return self.generic.copy()
        return np.mean([self.profile_of(c) for c in m.ids], axis=0)

    def is_rated(self, m: BasicMatch) -> bool:
        if not m.ids:
            return any(e.profile is not None for e in self.contexts.values())
        return all(self.contexts[c].profile is not None for c in m.ids)

    def with_profiles(self, profiles: dict[str, dict]) -> dict:
        """Raw YAML dict of this catalog with profiles filled in (used by offline step O2)."""
        raw = yaml.safe_load(Path(self.path).read_text(encoding="utf-8"))
        for c in raw["contexts"]:
            if c["id"] in profiles:
                c["profile"] = profiles[c["id"]]["profile"]
                c["n_raters"] = profiles[c["id"]]["n_raters"]
        return raw
