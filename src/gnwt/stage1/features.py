"""Profile features of a directed pair i -> j (stage1.md, sections 4-6).

Every feature belongs to one of the five exact blocks of (14):

    eta = eta0 + etaP_fit + etaP_resid + etaP_actor + etaG

* eta0       intercept, chooser gender, ages, |age gap|, same field, self ratings of i and j
* etaP_fit   gamma_P * fit(j, c_i)                                  (11a)
* etaP_resid delta^c' P~_j, with delta orthogonal to D~ when the market has one fixed demand vector
* etaP_actor beta_A' P~_i (shifts a0, never the ranking of i's candidates)
* etaG       same goal, interest correlation, Jaccard of top-5 interests

Missing data follow the missing-indicator method (section 6): a block g of a row is replaced by
r_g (x - mu) + omega_g (1 - r_g), where r_g is the presence flag and omega_g the coefficient of the
indicator column `miss_<g>`; omega belongs to the block of g. Person-level blocks (bf, self, age, goal,
interests, field) can be masked at random per person while fitting (block masking).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gnwt.config import DOMAINS, TRAITS
from gnwt.persona import INTERESTS, SELF_KEYS, Persona
from gnwt.stage1.fit import fit_matrix

BLOCKS = ("eta0", "etaP_fit", "etaP_resid", "etaP_actor", "etaG")
PERSON_BLOCKS = ("bf", "self", "age", "goal", "interests", "field")


# ------------------------------------------------------------------------------------ person table


@dataclass
class PersonTable:
    ids: np.ndarray                     # (N,) str
    gender: np.ndarray                  # (N,) 1 = male, 0 = female, nan = unknown
    age: np.ndarray                     # (N,)
    field: np.ndarray                   # (N,) object (None = missing)
    goal: np.ndarray                    # (N,) object
    interests: np.ndarray               # (N, 17) nan = missing
    selfr: np.ndarray                   # (N, 4)
    bf: np.ndarray                      # (N, 5) O C E A N in [0, 1], nan = missing
    context: np.ndarray                 # (N,) basic context id of the person's own sentence (or None)
    D_tilde: np.ndarray                 # (N, 5) demand vector of the person's own context sentence
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = {str(p): k for k, p in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def from_frame(cls, persons: pd.DataFrame, bf: pd.DataFrame | None, demands: dict[str, object] | None,
                   default_context: str | None = None) -> PersonTable:
        ids = persons.index.to_numpy(dtype=str)
        n = len(ids)
        bf_arr = np.full((n, 5), np.nan)
        if bf is not None:
            bf_arr = bf.reindex(ids)[list(TRAITS)].to_numpy(float)
        D = np.zeros((n, 5))
        ctx = np.array([default_context] * n, dtype=object)
        if demands:
            for k, pid in enumerate(ids):
                cd = demands.get(pid)
                if cd is not None:
                    D[k] = cd.D_tilde
                    ctx[k] = cd.match.c_star if cd.match.c_star is not None else "other"
        g = persons["gender"].map({"male": 1.0, "female": 0.0}).to_numpy(float)
        return cls(ids, g, persons["age"].to_numpy(float),
                   persons["field"].where(persons["field"].notna(), None).to_numpy(object),
                   persons["goal"].where(persons["goal"].notna(), None).to_numpy(object),
                   persons[list(INTERESTS)].to_numpy(float),
                   persons[[f"self_{k}" for k in SELF_KEYS]].to_numpy(float), bf_arr, ctx, D)

    @classmethod
    def from_personas(cls, personas: Sequence[Persona], demands: dict[str, object] | None = None) -> PersonTable:
        rows = []
        for p in personas:
            rows.append({"person_id": p.persona_id, "gender": p.gender, "age": p.age, "field": p.field,
                         "goal": p.goal, **{k: p.interests.get(k, np.nan) for k in INTERESTS},
                         **{f"self_{k}": p.self_ratings.get(k, np.nan) for k in SELF_KEYS}})
        df = pd.DataFrame(rows).set_index("person_id")
        bf = pd.DataFrame([{t: (np.nan if not p.bigfive or p.bigfive.get(t) is None else p.bigfive[t])
                            for t in TRAITS} for p in personas], index=df.index)
        return cls.from_frame(df, bf, demands)


# ------------------------------------------------------------------------------------ standardizer


@dataclass(frozen=True)
class Standardizer:
    """Means and SDs computed on the persons of the training fold (z_within_fold)."""

    bf_mean: np.ndarray
    bf_sd: np.ndarray
    age_mean: float
    age_sd: float
    self_mean: np.ndarray
    self_sd: np.ndarray
    dyad_mean: dict[str, float]
    dyad_sd: dict[str, float]

    @staticmethod
    def _ms(x: np.ndarray) -> tuple[float, float]:
        x = x[np.isfinite(x)]
        if len(x) < 2:
            return 0.0, 1.0
        sd = float(x.std())
        return float(x.mean()), sd if sd > 1e-12 else 1.0

    @classmethod
    def fit(cls, T: PersonTable, persons: np.ndarray, dyads: dict[str, np.ndarray]) -> Standardizer:
        bm, bs = zip(*(cls._ms(T.bf[persons, d]) for d in range(5)))
        am, asd = cls._ms(T.age[persons])
        sm, ss = zip(*(cls._ms(T.selfr[persons, d]) for d in range(4)))
        dm, ds = {}, {}
        for k, v in dyads.items():
            dm[k], ds[k] = cls._ms(v)
        return cls(np.array(bm), np.array(bs), am, asd, np.array(sm), np.array(ss), dm, ds)

    def to_json(self) -> dict[str, object]:
        return {"bf_mean": self.bf_mean.tolist(), "bf_sd": self.bf_sd.tolist(), "age_mean": self.age_mean,
                "age_sd": self.age_sd, "self_mean": self.self_mean.tolist(), "self_sd": self.self_sd.tolist(),
                "dyad_mean": self.dyad_mean, "dyad_sd": self.dyad_sd}

    @classmethod
    def from_json(cls, d: dict[str, object]) -> Standardizer:
        return cls(np.array(d["bf_mean"]), np.array(d["bf_sd"]), float(d["age_mean"]), float(d["age_sd"]),
                   np.array(d["self_mean"]), np.array(d["self_sd"]), dict(d["dyad_mean"]), dict(d["dyad_sd"]))


# ---------------------------------------------------------------------------------- design spec


@dataclass(frozen=True)
class DesignSpec:
    actor: tuple[str, ...] = ("gender", "bf", "age", "self_ratings")
    partner: tuple[str, ...] = ("bf", "age", "self_ratings")
    dyad: tuple[str, ...] = ("same_goal", "int_corr", "jaccard_top5", "abs_age_gap", "same_field")
    trait_block_missing_if_ge: int = 3
    int_corr_min_common_items: int = 8
    bf_similarity: bool = False
    delta_mode: str = "orthogonal"          # orthogonal | zero (branch S8) | free
    use_fit: bool = True

    @classmethod
    def from_params(cls, f: object, delta_mode: str = "orthogonal") -> DesignSpec:
        return cls(tuple(f.actor), tuple(f.partner), tuple(f.dyad), f.trait_block_missing_if_ge,
                   f.int_corr_min_common_items, f.bf_similarity, delta_mode)

    def to_json(self) -> dict[str, object]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_json(cls, d: dict[str, object]) -> DesignSpec:
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


@dataclass
class Design:
    X: np.ndarray                       # (n, p)
    names: list[str]
    block_of: list[str]                 # block of each column
    present: dict[str, np.ndarray]      # r_g per row for every maskable block
    fit: np.ndarray                     # (n,) fit(j, c_i) before slope
    fit_parts: np.ndarray               # (n, 5)

    def block_cols(self, block: str) -> np.ndarray:
        return np.array([k for k, b in enumerate(self.block_of) if b == block], dtype=int)


# ---------------------------------------------------------------------------------- dyad pieces


def _top5(T: PersonTable) -> list[frozenset[int]]:
    out = []
    for row in T.interests:
        ok = np.where(np.isfinite(row))[0]
        if len(ok) < 5:
            out.append(frozenset())
            continue
        order = sorted(ok, key=lambda k: (-row[k], k))
        out.append(frozenset(order[:5]))
    return out


def dyad_raw(T: PersonTable, i: np.ndarray, j: np.ndarray, spec: DesignSpec,
             pmask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Raw (unstandardised) dyad features; NaN = missing. pmask: (N, 6) person-block presence."""
    def ok(block: str, idx: np.ndarray) -> np.ndarray:
        return np.ones(len(idx), bool) if pmask is None else pmask[idx, PERSON_BLOCKS.index(block)]

    out: dict[str, np.ndarray] = {}
    gi, gj = T.goal[i], T.goal[j]
    same_goal = np.array([np.nan if a is None or b is None else float(a == b) for a, b in zip(gi, gj)])
    same_goal[~(ok("goal", i) & ok("goal", j))] = np.nan
    out["same_goal"] = same_goal

    Ii, Ij = T.interests[i], T.interests[j]
    common = np.isfinite(Ii) & np.isfinite(Ij)
    n_c = common.sum(1)
    xi, xj = np.where(common, Ii, 0.0), np.where(common, Ij, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mi = xi.sum(1) / n_c
        mj = xj.sum(1) / n_c
        di = np.where(common, xi - mi[:, None], 0.0)
        dj = np.where(common, xj - mj[:, None], 0.0)
        r = (di * dj).sum(1) / np.sqrt((di ** 2).sum(1) * (dj ** 2).sum(1))
    r[(n_c < spec.int_corr_min_common_items) | ~np.isfinite(r)] = np.nan
    ints_ok = ok("interests", i) & ok("interests", j)
    r[~ints_ok] = np.nan
    out["int_corr"] = r

    tops = _top5(T)
    jac = np.array([np.nan if not tops[a] or not tops[b] else len(tops[a] & tops[b]) / len(tops[a] | tops[b])
                    for a, b in zip(i, j)])
    jac[~ints_ok] = np.nan
    out["jaccard_top5"] = jac

    gap = np.abs(T.age[i] - T.age[j])
    gap[~(ok("age", i) & ok("age", j))] = np.nan
    out["abs_age_gap"] = gap

    fi, fj = T.field[i], T.field[j]
    sf = np.array([np.nan if a is None or b is None else float(a == b) for a, b in zip(fi, fj)])
    sf[~(ok("field", i) & ok("field", j))] = np.nan
    out["same_field"] = sf
    return out


def bf_tilde(T: PersonTable, idx: np.ndarray, std: Standardizer, spec: DesignSpec,
             pmask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P~ on O C E A ES (z within fold, ES = -z(N)), trait mask, block presence."""
    raw = T.bf[idx]
    z = (raw - std.bf_mean) / std.bf_sd
    z[:, 4] = -z[:, 4]                                   # ES = -N
    m = np.isfinite(z)
    block = (5 - m.sum(1)) < spec.trait_block_missing_if_ge
    if pmask is not None:
        block &= pmask[idx, PERSON_BLOCKS.index("bf")]
    m &= block[:, None]
    return np.where(m, np.nan_to_num(z), 0.0), m, block


def orthonormal_complement(d: np.ndarray) -> np.ndarray:
    """(5, 4) orthonormal basis of the complement of d; identity (5, 5) when d == 0."""
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        return np.eye(5)
    u = d / nrm
    Q, _ = np.linalg.qr(np.column_stack([u, np.eye(5)]))
    return Q[:, 1:5]


# --------------------------------------------------------------------------------------- design


def build_design(T: PersonTable, i: np.ndarray, j: np.ndarray, std: Standardizer, spec: DesignSpec,
                 delta_basis: dict[str, list[list[float]]] | None = None,
                 pmask: np.ndarray | None = None, D_rows: np.ndarray | None = None) -> Design:
    """Design matrix of the directed rows i[k] -> j[k]. D_rows defaults to the chooser's own D~."""
    i = np.asarray(i, int)
    j = np.asarray(j, int)
    n = len(i)
    cols: list[np.ndarray] = []
    names: list[str] = []
    blocks: list[str] = []
    present: dict[str, np.ndarray] = {}

    def add(name: str, x: np.ndarray, block: str) -> None:
        cols.append(np.asarray(x, float))
        names.append(name)
        blocks.append(block)

    def zcol(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
        return np.where(np.isfinite(x), (x - mean) / sd, 0.0)

    def pm(block: str, idx: np.ndarray) -> np.ndarray:
        return np.ones(len(idx), bool) if pmask is None else pmask[idx, PERSON_BLOCKS.index(block)]

    add("intercept", np.ones(n), "eta0")
    if "gender" in spec.actor:
        g = T.gender[i]
        add("gender_i", np.where(np.isfinite(g), g - 0.5, 0.0), "eta0")

    for side, idx, feats in (("i", i, spec.actor), ("j", j, spec.partner)):
        if "age" in feats:
            a = T.age[idx]
            r = np.isfinite(a) & pm("age", idx)
            present[f"age_{side}"] = r
            add(f"age_{side}", np.where(r, zcol(a, std.age_mean, std.age_sd), 0.0), "eta0")
            add(f"miss_age_{side}", (~r).astype(float), "eta0")
        if "self_ratings" in feats:
            S = T.selfr[idx]
            r = np.isfinite(S).any(1) & pm("self", idx)
            present[f"self_{side}"] = r
            for d, key in enumerate(SELF_KEYS):
                add(f"self_{side}_{key}", np.where(r, zcol(S[:, d], std.self_mean[d], std.self_sd[d]), 0.0), "eta0")
            add(f"miss_self_{side}", (~r).astype(float), "eta0")

    raw = dyad_raw(T, i, j, spec, pmask)
    for key in ("abs_age_gap", "same_field"):
        if key in spec.dyad:
            r = np.isfinite(raw[key])
            present[key] = r
            add(key, np.where(r, zcol(raw[key], std.dyad_mean.get(key, 0.0), std.dyad_sd.get(key, 1.0)), 0.0), "eta0")
            add(f"miss_{key}", (~r).astype(float), "eta0")

    # personality: fit, residual partner effect, actor effect
    D = T.D_tilde[i] if D_rows is None else np.asarray(D_rows, float)
    fit = np.zeros(n)
    parts = np.zeros((n, 5))
    if "bf" in spec.partner:
        Pj, mj, rj = bf_tilde(T, j, std, spec, pmask)
        present["bf_j"] = rj
        parts = D * Pj
        fit = fit_matrix(Pj, mj, D)
        if spec.use_fit:
            add("fit", fit, "etaP_fit")
        if spec.delta_mode != "zero":
            # always 5 columns; with delta orthogonal to D~ the 5th is identically 0 and is dropped at fit
            ctx = T.context[i].astype(str)
            resid = np.zeros((n, 5))
            for c in np.unique(ctx):
                sel = ctx == c
                Q = np.eye(5)
                if spec.delta_mode == "orthogonal" and delta_basis and c in delta_basis:
                    Q = np.array(delta_basis[c], float)
                    if Q.shape[1] < 5:
                        Q = np.column_stack([Q, np.zeros((5, 5 - Q.shape[1]))])
                resid[sel] = Pj[sel] @ Q
            for d in range(5):
                add(f"bf_j_q{d + 1}", resid[:, d], "etaP_resid")
        add("miss_bf_j", (~rj).astype(float), "etaP_resid")
    if "bf" in spec.actor:
        Pi, _, ri = bf_tilde(T, i, std, spec, pmask)
        present["bf_i"] = ri
        for d, dom in enumerate(DOMAINS):
            add(f"bf_i_{dom}", Pi[:, d], "etaP_actor")
        add("miss_bf_i", (~ri).astype(float), "etaP_actor")
    if spec.bf_similarity and "bf" in spec.actor and "bf" in spec.partner:
        Pi, mi, _ = bf_tilde(T, i, std, spec, pmask)
        Pj, mj, _ = bf_tilde(T, j, std, spec, pmask)
        for d, dom in enumerate(DOMAINS):
            add(f"bf_absdiff_{dom}", np.where(mi[:, d] & mj[:, d], np.abs(Pi[:, d] - Pj[:, d]), 0.0), "etaG")

    for key in ("same_goal", "int_corr", "jaccard_top5"):
        if key in spec.dyad:
            r = np.isfinite(raw[key])
            present[key] = r
            add(key, np.where(r, zcol(raw[key], std.dyad_mean.get(key, 0.0), std.dyad_sd.get(key, 1.0)), 0.0), "etaG")
            if key != "jaccard_top5":            # int_corr and jaccard share the interests indicator
                add(f"miss_{key if key == 'same_goal' else 'interests'}", (~r).astype(float), "etaG")

    X = np.column_stack(cols) if cols else np.zeros((n, 0))
    return Design(X, names, blocks, present, fit, parts)


def dyad_columns_for_standardizer(T: PersonTable, i: np.ndarray, j: np.ndarray, spec: DesignSpec) -> dict[str, np.ndarray]:
    raw = dyad_raw(T, i, j, spec)
    return {k: raw[k] for k in ("abs_age_gap", "same_field", "same_goal", "int_corr", "jaccard_top5")}


def delta_basis_for(T: PersonTable, choosers: np.ndarray) -> dict[str, list[list[float]]]:
    """delta orthogonal to D~ for every basic context whose choosers all carry one D~ (section 4)."""
    out: dict[str, list[list[float]]] = {}
    ctx = T.context[choosers].astype(str)
    for c in np.unique(ctx):
        Ds = T.D_tilde[choosers[ctx == c]]
        if np.allclose(Ds, Ds[0], atol=1e-12):
            out[c] = orthonormal_complement(Ds[0]).tolist()
    return out


def person_mask(n_persons: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """(N, 6) presence after random per-person block masking at the given rate."""
    if rate <= 0:
        return np.ones((n_persons, len(PERSON_BLOCKS)), bool)
    return rng.random((n_persons, len(PERSON_BLOCKS))) >= rate


def dumps(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
