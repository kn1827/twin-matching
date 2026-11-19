from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gnwt.config import PROJECT_ROOT
from gnwt.stage1.context.catalog import Catalog
from gnwt.stage1.context.demand import ReaderArtifacts
from gnwt.stage1.context.lexicon import Lexicon

FIX = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="session")
def lexicon() -> Lexicon:
    return Lexicon.load(PROJECT_ROOT / "data" / "lexicon" / "demand_lexicon_v1.yaml")


@pytest.fixture(scope="session")
def catalog_illustrative() -> Catalog:
    return Catalog.load(FIX / "catalog_illustrative.yaml")


@pytest.fixture(scope="session")
def catalog_v1() -> Catalog:
    return Catalog.load(PROJECT_ROOT / "data" / "catalog" / "basic_contexts_v1.yaml")


@pytest.fixture(scope="session")
def reader_illustrative(lexicon: Lexicon, catalog_illustrative: Catalog) -> ReaderArtifacts:
    art = ReaderArtifacts.from_json(FIX / "reader_illustrative.json", lexicon, catalog_illustrative)
    assert art.k is not None and np.allclose(art.k, 0.5)
    return art
