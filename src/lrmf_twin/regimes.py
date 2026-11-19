"""T0 fold splitter and regime flags z (design sections 4.2 T0, 4.4 step 1, 4.6 regimes R1-R6).

* R1 dense         every observed cell is assigned to one of K folds at random (stratified within person);
                   fold k predicts its cells from the rest of the matrix: every person stays warm.
* R2 new items     items are split into K folds; fold k's items have no answer at all in training.
* R3 missing wave  whole (item, wave) blocks are held out (GSS: a question in a survey year, Kim & Lee's
                   retrodiction; MovieLens: a movie in a calendar quarter). Needs `wave`; Twin-2K has none.
* R4 sparse        `n_sparse` persons split into K folds; a person in fold k keeps `n_known` answers.
* R5 new persons   like R4 with no known answer at all: only the person's demographics are available.
* R6 cold items    items split into K folds; an item of fold k keeps only `k_cold` random answers.

Every cell of a regime's scope is predicted exactly once, by models that never saw it (out-of-fold).
The flags z of a cell come from the training mask of its fold: n_p (answers of the person), n_j (answers to
the item), new_item.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Fold:
    regime: str
    k: int
    train: np.ndarray            # (P, J) bool: cells the models may fit on
    test: np.ndarray             # (P, J) bool: cells predicted by this fold
    sparse_persons: np.ndarray   # persons made sparse / new in this fold (R4, R5), else empty
    new_items: np.ndarray        # items without training answers (R2) or cold (R6), else empty


REGIMES = {"R1": "dense (random cells)", "R2": "new questions", "R3": "missing wave (item x wave blocks)",
           "R4": "sparse persons", "R5": "new persons (demographics only)", "R6": "cold items (few answers)"}


def supported(regime: str, has_wave: bool) -> bool:
    return regime in REGIMES and (regime != "R3" or has_wave)


def make_folds(observed: np.ndarray, regime: str, n_folds: int = 5, seed: int = 0, n_sparse: int = 200,
               n_known: int = 10, k_cold: int = 5, wave: np.ndarray | None = None) -> list[Fold]:
    rng = np.random.default_rng(seed)
    P, J = observed.shape
    empty = np.array([], dtype=int)
    folds: list[Fold] = []
    if regime == "R1":
        fold_id = np.full((P, J), -1, dtype=np.int8)
        for p in range(P):
            cells = rng.permutation(np.flatnonzero(observed[p]))
            fold_id[p, cells] = np.arange(cells.size) % n_folds
        for k in range(n_folds):
            test = fold_id == k
            folds.append(Fold(regime, k, observed & ~test, test, empty, empty))
    elif regime in ("R4", "R5"):
        keep = n_known if regime == "R4" else 0
        persons = rng.choice(P, size=min(n_sparse, P), replace=False)
        for k, chunk in enumerate(np.array_split(persons, n_folds)):
            test = np.zeros((P, J), dtype=bool)
            for p in chunk:
                cells = np.flatnonzero(observed[p])
                test[p, cells] = True
                if keep:
                    test[p, rng.choice(cells, size=min(keep, cells.size), replace=False)] = False
            folds.append(Fold(regime, k, observed & ~test, test, np.sort(chunk), empty))
    elif regime in ("R2", "R6"):
        items = rng.permutation(J)
        for k, chunk in enumerate(np.array_split(items, n_folds)):
            test = np.zeros((P, J), dtype=bool)
            test[:, chunk] = observed[:, chunk]
            if regime == "R6":
                for j in chunk:
                    rows = np.flatnonzero(observed[:, j])
                    test[rng.choice(rows, size=min(k_cold, rows.size), replace=False), j] = False
            folds.append(Fold(regime, k, observed & ~test, test, empty, np.sort(chunk)))
    elif regime == "R3":
        if wave is None:
            raise ValueError("R3 needs a wave matrix (the dataset has no waves)")
        w = np.where(observed, wave.astype(np.int64), -1)
        n_w = int(w.max()) + 1
        block = np.where(observed, np.arange(J)[None, :] * n_w + w, -1)
        ids = np.unique(block[block >= 0])
        fold_of_block = dict(zip(ids.tolist(), (rng.permutation(ids.size) % n_folds).tolist()))
        lut = np.full(J * n_w, -1, dtype=np.int8)
        lut[ids] = [fold_of_block[b] for b in ids.tolist()]
        fold_id = np.where(block >= 0, lut[np.maximum(block, 0)], -1)
        for k in range(n_folds):
            test = fold_id == k
            folds.append(Fold(regime, k, observed & ~test, test, empty, empty))
    else:
        raise ValueError(f"unknown regime {regime!r}; expected one of {sorted(REGIMES)}")
    return folds


def regime_flags(train: np.ndarray) -> dict[str, np.ndarray]:
    """z for every person / item, from one fold's training mask."""
    n_p = train.sum(1).astype(np.float32)
    n_j = train.sum(0).astype(np.float32)
    return {"n_p": n_p, "n_j": n_j, "new_item": (n_j == 0).astype(np.float32)}
