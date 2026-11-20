"""Datasets of the comparison, all returned as `SurveyData` (person x question matrix + texts).

* twin2k     Twin-2K-500 (2,058 persons x 596 questions), no waves: R3 is left empty.
* gss        General Social Survey 2000-2024 (persons x questions, waves = years).
* movielens  MovieLens-1M (users x movies, "question" = movie, waves = rating quarters).
"""

from __future__ import annotations

from pathlib import Path

DATASETS = ("twin2k", "gss", "movielens")


def load_dataset(name: str, root: Path = Path("."), **kw):
    if name == "twin2k":
        from lrmf_twin.data.twin2k import load
        return load(root / "data/raw/twin2k")
    if name == "gss":
        from lrmf_twin.data.gss import load
        return load(root / "data/raw/gss", cache_dir=root / "data/processed", **kw)
    if name == "movielens":
        from lrmf_twin.data.movielens import load
        return load(root / "data/raw/movielens", **kw)
    raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
