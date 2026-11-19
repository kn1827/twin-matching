"""The literature baselines of experiments/compare_literature_baselines.py run end to end on toy data."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from gnwt.config import PROJECT_ROOT

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def cmp():
    path = PROJECT_ROOT / "experiments" / "compare_literature_baselines.py"
    spec = importlib.util.spec_from_file_location("cmp_baselines", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cmp_baselines"] = mod
    spec.loader.exec_module(mod)
    return mod


def _toy(n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"g": rng.choice(["a", "b"], n), "f": rng.choice(["x", "y", "z"], n),
                          "u": rng.normal(size=n), "v": rng.normal(size=n)})
    logit = 1.2 * (frame["g"] == "a") + 0.8 * frame["u"] - 0.5 * frame["v"] * (frame["f"] == "x")
    y = (rng.random(n) < 1 / (1 + np.exp(-(logit - 0.6)))).astype(float)
    return frame, ["g", "f"], ["u", "v"], y


def test_deep_models_learn_signal(cmp):
    frame, cat, num, y = _toy()
    tr, va, te = np.arange(0, 3000), np.arange(3000, 3500), np.arange(3500, 4000)
    enc = cmp.Encoder(frame, cat, num, tr)
    X = enc.transform(frame)
    for name in cmp.DEEP_MODELS:
        p = cmp.train_deep(name, X[tr], y[tr], X[va], y[va], X[te], int(sum(enc.sizes)), 0, 10,
                           y[tr] if name == "biDeepFM" else None, y[va] if name == "biDeepFM" else None)
        p = p[:, 0] if name == "biDeepFM" else p
        assert p.shape == (len(te),) and np.all((p > 0) & (p < 1)), name
        assert cmp.auc(y[te], p) > 0.6, name


def test_tree_models_run(cmp):
    frame, cat, num, y = _toy()
    tr, va, te = np.arange(0, 3000), np.arange(3000, 3500), np.arange(3500, 4000)
    for name in cmp.TREE_MODELS:
        p = cmp.train_tree(name, frame.iloc[tr], y[tr], frame.iloc[va], y[va], frame.iloc[te], cat, 0)
        assert p.shape == (len(te),) and cmp.auc(y[te], p) > 0.6, name
