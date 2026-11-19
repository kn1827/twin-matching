"""Persona model for Stage 1 (stage1.md, section 1, input table).

A persona carries only what Stage 1 reads: Big Five with a source flag (measured / proxy / none), the
context sentence, the participation goal, 17 interest scores, age, field, gender and four self ratings.
No appearance field exists: any key starting with "attr" is rejected.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gnwt.config import TRAITS

INTERESTS = ("sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming", "clubbing",
             "reading", "tv", "theater", "movies", "concerts", "music", "shopping", "yoga")
SELF_KEYS = ("sinc", "intel", "fun", "amb")
BF_SOURCES = ("measured", "proxy", "none")


def _no_physical_keys(keys: Any, where: str) -> None:
    bad = [k for k in keys if str(k).lower().startswith("attr")]
    if bad:
        raise ValueError(f"{where}: appearance fields are not allowed: {bad}")


class Persona(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    persona_id: str = Field(min_length=1)
    source: Literal["columbia", "speed_dating", "pilot", "handwritten"]
    market: str = Field(min_length=1)            # market = one population whose intercept is shared
    wave: int | None = None
    language: Literal["en", "vi"] = "en"
    gender: Literal["female", "male"] | None = None
    age: float | None = Field(default=None, ge=14, le=100)
    field: str | None = None
    goal: str | None = None                      # discrete goal code of the basic context
    interests: dict[str, float] = Field(default_factory=dict)       # 1-10
    self_ratings: dict[str, float] = Field(default_factory=dict)    # sinc, intel, fun, amb (1-10)
    bigfive: dict[str, float | None] | None = None                  # O, C, E, A, N in [0, 1]
    bf_source: Literal["measured", "proxy", "none"] = "none"
    context_text: str | None = None
    context_source: Literal["user", "event", "market", "none"] = "none"

    @field_validator("interests")
    @classmethod
    def _interests(cls, v: dict[str, float]) -> dict[str, float]:
        _no_physical_keys(v, "interests")
        unknown = set(v) - set(INTERESTS)
        if unknown:
            raise ValueError(f"interests: unknown items {sorted(unknown)}")
        if any(not 1.0 <= s <= 10.0 for s in v.values()):
            raise ValueError("interest scores must be in 1-10")
        return v

    @field_validator("self_ratings")
    @classmethod
    def _self(cls, v: dict[str, float]) -> dict[str, float]:
        _no_physical_keys(v, "self_ratings")
        unknown = set(v) - set(SELF_KEYS)
        if unknown:
            raise ValueError(f"self_ratings accepts only {SELF_KEYS}, got extra {sorted(unknown)}")
        if any(not 1.0 <= s <= 10.0 for s in v.values()):
            raise ValueError("self ratings must be in 1-10")
        return v

    @field_validator("bigfive")
    @classmethod
    def _bf(cls, v: dict[str, float | None] | None) -> dict[str, float | None] | None:
        if v is None:
            return v
        if set(v) != set(TRAITS):
            raise ValueError(f"bigfive must have exactly the keys {TRAITS}")
        if any(x is not None and not 0.0 <= x <= 1.0 for x in v.values()):
            raise ValueError("bigfive values must be in [0, 1]")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> Persona:
        has_bf = self.bigfive is not None and any(x is not None for x in self.bigfive.values())
        if self.bf_source == "none" and has_bf:
            raise ValueError("bf_source='none' but bigfive values are present")
        if self.bf_source != "none" and not has_bf:
            raise ValueError(f"bf_source={self.bf_source!r} but no bigfive value is present")
        if self.context_text is not None and self.context_source == "none":
            raise ValueError("context_text given with context_source='none'")
        return self


def load_handwritten(path: str | Path, market: str = "handwritten") -> Persona:
    """Read one persona written in the rev 110 JSON layout (data/handwritten/) into the Stage 1 model.

    The rev 110 layout stored goals as a weight vector; Stage 1 uses one discrete goal, the argmax.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    t1, t2, t3 = raw.get("tier1", {}), raw.get("tier2", {}), raw.get("tier3", {})
    goals: dict[str, float] = t2.get("goals") or {}
    goal = max(goals, key=lambda k: goals[k]) if goals else None
    goal = _HW_GOALS.get(goal, goal)
    bf = t3.get("bigfive")
    src = {"bfi10_self": "measured", "bfi10_proxy": "proxy"}.get(t3.get("bigfive_source"), "none")
    pid = str(raw["persona_id"])
    if not re.match(r"^hw_\d+$", pid):
        raise ValueError(f"handwritten persona_id must look like hw_01, got {pid!r}")
    return Persona(
        persona_id=pid, source="handwritten", market=market, language=raw.get("language", "en"),
        gender=t1.get("gender"), age=t1.get("age"), field=(t1.get("field") or None),
        goal=goal, interests={k: float(v) for k, v in (t2.get("interests") or {}).items()},
        self_ratings={k: float(v) for k, v in (t2.get("self_ratings") or {}).items()},
        bigfive={k: float(bf[k]) for k in TRAITS} if bf else None,
        bf_source=src if bf else "none",
    )


_HW_GOALS = {"fun_night_out": "fun_night", "meet_new_people": "meet_new", "get_a_date": "get_date",
             "serious_relationship": "serious", "say_i_did_it": "say_did"}


def load_handwritten_dir(directory: str | Path) -> list[Persona]:
    out = [load_handwritten(p) for p in sorted(Path(directory).glob("*.json"))]
    ids = [p.persona_id for p in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate persona_id in handwritten personas")
    return out
