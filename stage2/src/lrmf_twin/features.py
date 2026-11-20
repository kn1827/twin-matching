"""Person features shared by every model of one fold (information balance, design 6.4).

A[p, j] = the person's own answer to item j as a numeric feature, computed from TRAINING cells only:
ordinal items -> z-scored raw value, nominal items -> centred indicator of the modal option. Masked or
missing cells are 0 (= the training mean). When item j is the target, column j is never used.
"""

from __future__ import annotations

import numpy as np

from lrmf_twin.data.twin2k import SurveyData


def answer_features(data: SurveyData, train: np.ndarray) -> np.ndarray:
    P, J = data.shape
    A = np.zeros((P, J), dtype=np.float32)
    for j, item in enumerate(data.items):
        rows = train[:, j]
        if not rows.any():
            continue
        if item.kind == "nominal":
            x = data.Y[rows, j]
        else:
            x = data.V[rows, j]
        mu, sd = float(np.mean(x)), float(np.std(x))
        A[rows, j] = (x - mu) / (sd if sd > 1e-6 else 1.0)
    return A


def centred_demographics(X_demo: np.ndarray) -> np.ndarray:
    return (X_demo - X_demo.mean(0, keepdims=True)).astype(np.float32)
