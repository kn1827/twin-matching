"""Profile features of a directed pair i -> j (tech spec 4-6).

Blocks and the linear-predictor part they feed (14):

| block        | columns                                         | eta part        |
|--------------|-------------------------------------------------|-----------------|
| (always)     | intercept, actor_male                           | eta0            |
| age          | actor_age, partner_age, abs_age_gap             | eta0            |
| field        | same_field                                      | eta0            |
| self_actor   | actor sinc/intel/fun/amb self-ratings           | eta0            |
| self_partner | partner sinc/intel/fun/amb self-ratings         | eta0            |
| goal         | same_goal                                       | etaG            |
| interests    | int_corr (Pearson, >= 8 common items), Jaccard top-5 | etaG       |
| bf_partner   | fit(j, c_i) and partner Big Five P~_j            | etaP_fit, etaP_resid |
| bf_actor     | chooser Big Five P~_i                            | etaP_actor      |
| miss_*       | 1 - r_g for each block (missing-indicator method) | eta0          |

Missing data (missing-indicator method, Van Ness et al. 2023): block g enters as r_g (x - mu_hat) and a
learned constant omega_g (1 - r_g). Every non-indicator column (binaries included) is z-scored on the
TRAINING persons/pairs of the fold, so one prior width means one width per SD and a masked block
contributes exactly omega_g. No appearance field is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from dataclasses import field as dc_field

import numpy as np

from stage1.context.styles import DOMAINS
from stage1.fit import fit_scores

SELF_KEYS = ("sinc", "intel", "fun", "amb")          # Columbia sinc3_1, intel3_1, fun3_1, amb3_1
N_INTERESTS = 17
BLOCKS = ("age", "field", "self_actor", "self_partner", "goal", "interests", "bf_actor", "bf_partner")
PERSON_BLOCKS = ("age", "field", "self", "goal", "interests", "bf")   # what random masking hides per person

COLUMNS: tuple[str, ...] = (
    ("intercept", "actor_male", "actor_age", "partner_age", "abs_age_gap", "same_field")
    + tuple(f"actor_{k}" for k in SELF_KEYS) + tuple(f"partner_{k}" for k in SELF_KEYS)
    + ("same_goal", "int_corr", "jaccard_top5", "fit")
    + tuple(f"partner_bf_{d}" for d in DOMAINS) + tuple(f"actor_bf_{d}" for d in DOMAINS)
    + tuple(f"miss_{b}" for b in BLOCKS)
)
COL = {c: i for i, c in enumerate(COLUMNS)}
BLOCK_COLUMNS = {
    "age": ("actor_age", "partner_age", "abs_age_gap"), "field": ("same_field",),
    "self_actor": tuple(f"actor_{k}" for k in SELF_KEYS), "self_partner": tuple(f"partner_{k}" for k in SELF_KEYS),
    "goal": ("same_goal",), "interests": ("int_corr", "jaccard_top5"),
    "bf_actor": tuple(f"actor_bf_{d}" for d in DOMAINS),
    "bf_partner": ("fit",) + tuple(f"partner_bf_{d}" for d in DOMAINS),
}
ETA_PART = {c: "eta0" for c in COLUMNS}
ETA_PART.update({"same_goal": "etaG", "int_corr": "etaG", "jaccard_top5": "etaG", "fit": "etaP_fit"})
ETA_PART.update({f"partner_bf_{d}": "etaP_resid" for d in DOMAINS})
ETA_PART.update({f"actor_bf_{d}": "etaP_actor" for d in DOMAINS})
ACTOR_ONLY = ("actor_male", "actor_age") + tuple(f"actor_{k}" for k in SELF_KEYS) \
    + tuple(f"actor_bf_{d}" for d in DOMAINS) + ("miss_self_actor", "miss_bf_actor")


@dataclass
class Persona:
    pid: str
    wave: int | None = None
    gender: int | None = None                  # 1 = male, 0 = female
    age: float = np.nan
    field: str | None = None
    goal: str | None = None
    interests: np.ndarray = dc_field(default_factory=lambda: np.full(N_INTERESTS, np.nan))
    self_ratings: np.ndarray = dc_field(default_factory=lambda: np.full(len(SELF_KEYS), np.nan))
    bf: np.ndarray = dc_field(default_factory=lambda: np.full(5, np.nan))   # O, C, E, A, N in [0, 1]
    bf_source: str = "none"                    # measured | proxy | none
    context_text: str | None = None
    market: str | None = None
    extra: dict = dc_field(default_factory=dict)  # raw pre-date fields kept for baselines


@dataclass
class PersonTable:
    """Column arrays of a list of personas (index = position)."""
    ids: list[str]
    gender: np.ndarray
    age: np.ndarray
    field: np.ndarray
    goal: np.ndarray
    interests: np.ndarray
    self_ratings: np.ndarray
    bf_es: np.ndarray                          # O, C, E, A, ES (= -N) in raw [0,1] units, nan = missing

    @classmethod
    def from_personas(cls, personas: list[Persona]) -> "PersonTable":
        bf = np.array([np.asarray(p.bf, float) for p in personas]).reshape(len(personas), 5)
        bf_es = bf.copy()
        bf_es[:, 4] = -bf[:, 4]
        return cls(
            ids=[p.pid for p in personas],
            gender=np.array([np.nan if p.gender is None else float(p.gender) for p in personas]),
            age=np.array([float(p.age) if p.age is not None else np.nan for p in personas]),
            field=np.array([p.field for p in personas], dtype=object),
            goal=np.array([p.goal for p in personas], dtype=object),
            interests=np.array([np.asarray(p.interests, float) for p in personas]).reshape(len(personas), N_INTERESTS),
            self_ratings=np.array([np.asarray(p.self_ratings, float) for p in personas]).reshape(len(personas), 4),
            bf_es=bf_es,
        )

    def masked(self, rate: float, rng: np.random.Generator, persons: np.ndarray | None = None) -> "PersonTable":
        """Copy with each person-level block hidden independently with probability `rate` (tech spec 6:
        blocks are masked at random per person so that omega is learned under the deployment pattern)."""
        if rate <= 0:
            return self
        n = len(self.ids)
        sel = np.ones(n, bool) if persons is None else np.isin(np.arange(n), persons)
        hide = {b: (rng.random(n) < rate) & sel for b in PERSON_BLOCKS}
        t = replace(self, age=self.age.copy(), field=self.field.copy(), goal=self.goal.copy(),
                    interests=self.interests.copy(), self_ratings=self.self_ratings.copy(), bf_es=self.bf_es.copy())
        t.age[hide["age"]] = np.nan
        t.field[hide["field"]] = None
        t.goal[hide["goal"]] = None
        t.interests[hide["interests"]] = np.nan
        t.self_ratings[hide["self"]] = np.nan
        t.bf_es[hide["bf"]] = np.nan
        return t


@dataclass(frozen=True)
class FeatureConfig:
    trait_block_missing_if_ge: int = 3
    self_block_missing_if_ge: int = 2
    int_corr_min_common_items: int = 8
    top_k_interests: int = 5


def _pair_raw(t: PersonTable, a: np.ndarray, p: np.ndarray, cfg: FeatureConfig) -> dict:
    """Raw (unstandardised) dyad quantities and block presence for actor rows a, partner rows p."""
    out = {}
    ai, pi = t.interests[a], t.interests[p]
    common = ~np.isnan(ai) & ~np.isnan(pi)
    nc = common.sum(1)
    x = np.where(common, ai, 0.0)
    y = np.where(common, pi, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = x.sum(1) / np.maximum(nc, 1)
        my = y.sum(1) / np.maximum(nc, 1)
        dx = np.where(common, x - mx[:, None], 0.0)
        dy = np.where(common, y - my[:, None], 0.0)
        den = np.sqrt((dx ** 2).sum(1) * (dy ** 2).sum(1))
        r = np.where(den > 0, (dx * dy).sum(1) / np.where(den > 0, den, 1), np.nan)
    out["int_corr"] = np.where(nc >= cfg.int_corr_min_common_items, r, np.nan)
    out["jaccard_top5"] = _jaccard_topk(ai, pi, cfg.top_k_interests)
    out["r_interests"] = ~np.isnan(out["int_corr"]) & ~np.isnan(out["jaccard_top5"])
    out["actor_age"], out["partner_age"] = t.age[a], t.age[p]
    out["abs_age_gap"] = np.abs(t.age[a] - t.age[p])
    out["r_age"] = ~np.isnan(out["abs_age_gap"])
    fa, fp = t.field[a], t.field[p]
    out["r_field"] = np.array([u is not None and v is not None for u, v in zip(fa, fp)])
    out["same_field"] = np.array([float(u == v) if ok else np.nan for u, v, ok in zip(fa, fp, out["r_field"])])
    ga, gp = t.goal[a], t.goal[p]
    out["r_goal"] = np.array([u is not None and v is not None for u, v in zip(ga, gp)])
    out["same_goal"] = np.array([float(u == v) if ok else np.nan for u, v, ok in zip(ga, gp, out["r_goal"])])
    sa, sp = t.self_ratings[a], t.self_ratings[p]
    out["self_actor"], out["self_partner"] = sa, sp
    out["r_self_actor"] = np.isnan(sa).sum(1) < cfg.self_block_missing_if_ge
    out["r_self_partner"] = np.isnan(sp).sum(1) < cfg.self_block_missing_if_ge
    ba, bp = t.bf_es[a], t.bf_es[p]
    out["bf_actor"], out["bf_partner"] = ba, bp
    out["r_bf_actor"] = np.isnan(ba).sum(1) < cfg.trait_block_missing_if_ge
    out["r_bf_partner"] = np.isnan(bp).sum(1) < cfg.trait_block_missing_if_ge
    out["actor_male"] = t.gender[a]
    return out


def _jaccard_topk(ai: np.ndarray, pi: np.ndarray, k: int) -> np.ndarray:
    """Jaccard overlap of each person's top-k interests (ties broken by item order)."""
    res = np.full(len(ai), np.nan)
    for r in range(len(ai)):
        x, y = ai[r], pi[r]
        if np.sum(~np.isnan(x)) < k or np.sum(~np.isnan(y)) < k:
            continue
        tx = set(np.argsort(-np.nan_to_num(x, nan=-np.inf), kind="stable")[:k])
        ty = set(np.argsort(-np.nan_to_num(y, nan=-np.inf), kind="stable")[:k])
        res[r] = len(tx & ty) / len(tx | ty)
    return res


@dataclass
class Standardizer:
    """Means and sds from the training fold (z_within_fold). Person-level stats use unique persons."""
    stats: dict

    @classmethod
    def fit(cls, t: PersonTable, actors: np.ndarray, partners: np.ndarray,
            cfg: FeatureConfig = FeatureConfig(), raw: dict | None = None) -> "Standardizer":
        persons = np.unique(np.concatenate([actors, partners]))
        raw = _pair_raw(t, actors, partners, cfg) if raw is None else raw

        def ms(v):
            v = np.asarray(v, float)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                return 0.0, 1.0
            sd = float(v.std())
            return float(v.mean()), (sd if sd > 1e-12 else 1.0)

        st = {"age": ms(t.age[persons]), "male": ms(t.gender[persons])}
        for k in range(4):
            st[f"self_{k}"] = ms(t.self_ratings[persons, k])
        for d in range(5):
            st[f"bf_{d}"] = ms(t.bf_es[persons, d])
        for c in ("abs_age_gap", "int_corr", "jaccard_top5"):
            st[c] = ms(raw[c])
        for c in ("same_field", "same_goal"):          # binaries are z-scored too: one prior width per SD
            st[c] = ms(raw[c])
        return cls(st)

    def z(self, key: str, v: np.ndarray) -> np.ndarray:
        m, s = self.stats[key]
        return (np.asarray(v, float) - m) / s

    def to_json(self) -> dict:
        return {k: [float(a), float(b)] for k, (a, b) in self.stats.items()}

    @classmethod
    def from_json(cls, d: dict) -> "Standardizer":
        return cls({k: (float(v[0]), float(v[1])) for k, v in d.items()})


@dataclass
class Design:
    X: np.ndarray                 # (n, len(COLUMNS))
    fit_parts: np.ndarray         # (n, 5) D~_d * P~_j,d
    P_partner: np.ndarray         # (n, 5) P~_j (0 where masked)
    present: dict                 # block -> (n,) bool


def pair_raw(t: PersonTable, actors: np.ndarray, partners: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> dict:
    """Unstandardised pair quantities; compute once per dataset and slice with `take_raw`."""
    return _pair_raw(t, np.asarray(actors), np.asarray(partners), cfg)


def take_raw(raw: dict, rows: np.ndarray) -> dict:
    return {k: v[rows] for k, v in raw.items()}


def person_hide(rate: float, n_persons: int, rng: np.random.Generator) -> dict:
    """Random per-person block masking at `rate` (tech spec 6): block -> (n_persons,) bool."""
    return {b: rng.random(n_persons) < rate for b in PERSON_BLOCKS}


def pair_hide(ph: dict, actors: np.ndarray, partners: np.ndarray) -> dict:
    """Pair-level hidden blocks implied by hidden person blocks."""
    a, p = np.asarray(actors), np.asarray(partners)
    return {"age": ph["age"][a] | ph["age"][p], "field": ph["field"][a] | ph["field"][p],
            "self_actor": ph["self"][a], "self_partner": ph["self"][p],
            "goal": ph["goal"][a] | ph["goal"][p], "interests": ph["interests"][a] | ph["interests"][p],
            "bf_actor": ph["bf"][a], "bf_partner": ph["bf"][p]}


def design_matrix(t: PersonTable, actors: np.ndarray, partners: np.ndarray, D_tilde: np.ndarray,
                  std: Standardizer, cfg: FeatureConfig = FeatureConfig()) -> Design:
    """Rows of the design for directed pairs actors[k] -> partners[k]. D_tilde is the chooser's context
    demand, (n, 5) or (5,)."""
    return design_from_raw(_pair_raw(t, np.asarray(actors), np.asarray(partners), cfg), D_tilde, std)


_RAW_PRESENT = {"age": "r_age", "field": "r_field", "self_actor": "r_self_actor", "self_partner": "r_self_partner",
                "goal": "r_goal", "interests": "r_interests", "bf_actor": "r_bf_actor", "bf_partner": "r_bf_partner"}


def design_from_raw(raw: dict, D_tilde: np.ndarray, std: Standardizer, hide: dict | None = None) -> Design:
    """Design rows from precomputed pair quantities; `hide` (block -> bool rows) masks blocks on top of
    the observed missingness."""
    if hide:
        raw = dict(raw)
        for b, h in hide.items():
            raw[_RAW_PRESENT[b]] = raw[_RAW_PRESENT[b]] & ~h
    n = len(raw["actor_male"])
    X = np.zeros((n, len(COLUMNS)))
    X[:, COL["intercept"]] = 1.0
    male = raw["actor_male"]
    X[:, COL["actor_male"]] = np.where(np.isnan(male), 0.0, std.z("male", male))

    def put(cols, values, present):
        for c, v in zip(cols, values):
            X[:, COL[c]] = np.where(present, np.nan_to_num(v), 0.0)

    r_age = raw["r_age"]
    put(BLOCK_COLUMNS["age"], [std.z("age", raw["actor_age"]), std.z("age", raw["partner_age"]),
                               std.z("abs_age_gap", raw["abs_age_gap"])], r_age)
    put(("same_field",), [std.z("same_field", raw["same_field"])], raw["r_field"])
    for side in ("actor", "partner"):
        vals = raw[f"self_{side}"]
        z = [np.nan_to_num(std.z(f"self_{k}", vals[:, k])) for k in range(4)]     # partial: missing item = mean
        put(BLOCK_COLUMNS[f"self_{side}"], z, raw[f"r_self_{side}"])
    put(("same_goal",), [std.z("same_goal", raw["same_goal"])], raw["r_goal"])
    put(BLOCK_COLUMNS["interests"], [std.z("int_corr", raw["int_corr"]), std.z("jaccard_top5", raw["jaccard_top5"])],
        raw["r_interests"])

    def ptilde(v, present):
        z = np.column_stack([std.z(f"bf_{d}", v[:, d]) for d in range(5)])
        trait_ok = ~np.isnan(z) & present[:, None]
        return np.where(trait_ok, z, 0.0), trait_ok

    Pj, okj = ptilde(raw["bf_partner"], raw["r_bf_partner"])
    Pi, _ = ptilde(raw["bf_actor"], raw["r_bf_actor"])
    fit, parts = fit_scores(Pj, okj, np.broadcast_to(np.asarray(D_tilde, float), (n, 5)))
    X[:, COL["fit"]] = fit
    for d, dom in enumerate(DOMAINS):
        X[:, COL[f"partner_bf_{dom}"]] = Pj[:, d]
        X[:, COL[f"actor_bf_{dom}"]] = Pi[:, d]
    present = {"age": r_age, "field": raw["r_field"], "self_actor": raw["r_self_actor"],
               "self_partner": raw["r_self_partner"], "goal": raw["r_goal"], "interests": raw["r_interests"],
               "bf_actor": raw["r_bf_actor"], "bf_partner": raw["r_bf_partner"]}
    for b in BLOCKS:
        X[:, COL[f"miss_{b}"]] = (~present[b]).astype(float)
    return Design(X=X, fit_parts=parts, P_partner=Pj, present=present)


@dataclass
class Features:
    x: np.ndarray
    fit: float
    fit_parts: np.ndarray
    mask: dict


def build_features(pi: Persona, pj: Persona, cd_i, std: Standardizer,
                   cfg: FeatureConfig = FeatureConfig()) -> Features:
    """Features of one directed pair i -> j (spec signature); cd_i is the chooser's ContextDemand."""
    t = PersonTable.from_personas([pi, pj])
    d = design_matrix(t, np.array([0]), np.array([1]), cd_i.D_tilde, std, cfg)
    return Features(x=d.X[0], fit=float(d.X[0, COL["fit"]]), fit_parts=d.fit_parts[0],
                    mask={b: bool(~v[0]) for b, v in d.present.items()})
