"""Shared fixtures: the shipped lexicon, and a catalog whose profiles are filled with the ILLUSTRATIVE
numbers of tech spec 3.5 (study_group) plus arbitrary test values, so that prior-dependent rules can be
tested before the real rater profiles exist."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from stage1.context.catalog import Catalog
from stage1.context.demand import ReaderArtifacts
from stage1.context.lexicon import Lexicon
from stage1.context.styles import STYLES

ROOT = Path(__file__).resolve().parents[1]
LEX = ROOT / "data" / "lexicon" / "demand_lexicon_v1.yaml"
CAT = ROOT / "data" / "catalog" / "basic_contexts_v1.yaml"

STUDY = {"DEP": 1.5, "DET": 1.5, "ACH": 1.5, "INN": 0.5, "CUR": 1.5, "AMB": 0.5, "SOC": 0.5, "LEA": 0.5,
         "COO": 2.0, "EMP": 1.0, "STR": 1.0, "SCT": 1.0}


@pytest.fixture(scope="session")
def lexicon() -> Lexicon:
    return Lexicon.load(LEX)


@pytest.fixture(scope="session")
def rated_catalog(tmp_path_factory) -> Catalog:
    raw = yaml.safe_load(CAT.read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    for c in raw["contexts"]:
        vals = STUDY if c["id"] == "study_group" else dict(zip(STYLES, np.round(rng.uniform(-1, 2.5, 12), 2)))
        c["profile"] = {s: {"mean": float(vals[s]), "se": 0.2, "icc2k": 0.8} for s in STYLES}
        c["n_raters"] = 3
    p = tmp_path_factory.mktemp("cat") / "catalog.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return Catalog.load(p)


@pytest.fixture(scope="session")
def art(lexicon, rated_catalog) -> ReaderArtifacts:
    return ReaderArtifacts.illustrative(lexicon, rated_catalog)
