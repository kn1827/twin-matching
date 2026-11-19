"""Load the two Speed Dating files into one canonical layout (offline, outside stage 1).

* `columbia`: the original file of Fisman, Iyengar, Kamenica & Simonson (2006), 8378 directed rows,
  195 columns, with person ids `iid` / `pid`.
* `speed_dating`: OpenML 40536, the same experiment re-coded (123 columns). It has no person ids and no
  goal, date or go_out column. Person ids are rebuilt here: a chooser is a run of consecutive rows with
  the same chooser-level answers, and the partner of a row is the unique chooser of the same wave whose
  own answers equal the row's partner columns (`age_o`, `race_o`, `pref_o_*`) and whose reverse row
  carries the same `interests_correlate` and the swapped decisions.

Only pre-date fields are kept for persons. Ratings made during or after the date (`attr`, `like`,
`prob`, `*_o` ratings of the partner, `met`) never leave this module. No appearance field is kept.

`Market.persons` has one row per person; `Market.rows` one row per directed decision i -> j.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnwt.config import DEFAULT_PARAMS, load_params, resolve_path
from gnwt.persona import INTERESTS, SELF_KEYS, Persona

GOAL_CODES = {1: "fun_night", 2: "meet_new", 3: "get_date", 4: "serious", 5: "say_did", 6: "other"}
EVENT_TEXT = "speed dating event"          # event description used when the persona wrote no sentence

PERSON_COLS = ("person_id", "wave", "gender", "age", "field", "goal", *INTERESTS,
               *(f"self_{k}" for k in SELF_KEYS), "go_out")
ROW_COLS = ("chooser", "chosen", "wave", "y", "match", "int_corr_given", "raw_index")


@dataclass(frozen=True)
class Market:
    name: str
    persons: pd.DataFrame          # index person_id
    rows: pd.DataFrame             # columns ROW_COLS
    report: dict[str, int]

    @property
    def waves(self) -> list[int]:
        return sorted(int(w) for w in self.rows["wave"].unique())


def _clip_1_10(s: pd.Series) -> pd.Series:
    return s.astype(float).clip(lower=1.0, upper=10.0)


def _first_valid(g: pd.DataFrame) -> pd.Series:
    return g.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else np.nan)


# ------------------------------------------------------------------------------------- columbia


def load_columbia(path: str | Path) -> Market:
    df = pd.read_csv(path, encoding="latin-1").copy()
    n_raw = len(df)
    df = df.assign(raw_index=np.arange(len(df)))
    df = df[df["pid"].notna()].copy()
    df["chooser"] = "col_" + df["iid"].astype(int).astype(str)
    df["chosen"] = "col_" + df["pid"].astype(int).astype(str)

    per = pd.DataFrame({
        "person_id": df["chooser"], "wave": df["wave"].astype(int),
        "gender": df["gender"].map({0: "female", 1: "male"}), "age": df["age"].astype(float),
        "field": df["field_cd"].map(lambda v: f"fcd_{int(v)}" if pd.notna(v) else np.nan),
        "goal": df["goal"].map(lambda v: GOAL_CODES.get(int(v)) if pd.notna(v) else np.nan),
        **{k: _clip_1_10(df[k]) for k in INTERESTS},
        "self_sinc": _clip_1_10(df["sinc3_1"]), "self_intel": _clip_1_10(df["intel3_1"]),
        "self_fun": _clip_1_10(df["fun3_1"]), "self_amb": _clip_1_10(df["amb3_1"]),
        "go_out": df["go_out"].astype(float),
    })
    persons = per.groupby("person_id", sort=False).agg(_first_valid_agg()).reset_index()
    persons = persons.set_index("person_id")

    rows = pd.DataFrame({"chooser": df["chooser"], "chosen": df["chosen"], "wave": df["wave"].astype(int),
                         "y": df["dec"].astype(int), "match": df["match"].astype(int),
                         "int_corr_given": df["int_corr"].astype(float), "raw_index": df["raw_index"]})
    rows = rows[rows["chosen"].isin(persons.index)].reset_index(drop=True)
    persons["raw_first"] = df.groupby("chooser", sort=False)["raw_index"].min().reindex(persons.index)
    return Market("columbia", persons, rows,
                  {"raw_rows": n_raw, "rows": len(rows), "persons": len(persons),
                   "dropped_rows": n_raw - len(rows)})


def _first_valid_agg() -> dict[str, object]:
    def first(c: pd.Series) -> object:
        nn = c.dropna()
        return nn.iloc[0] if len(nn) else np.nan
    return {c: first for c in PERSON_COLS if c != "person_id"}


# --------------------------------------------------------------------------------------- openml

_OML_SELF = {"self_sinc": "sincere", "self_intel": "intelligence", "self_fun": "funny", "self_amb": "ambition"}
_OML_IMPORTANT = ("attractive_important", "sincere_important", "intellicence_important", "funny_important",
                  "ambtition_important", "shared_interests_important")
_OML_PREF_O = ("pref_o_attractive", "pref_o_sincere", "pref_o_intelligence", "pref_o_funny",
               "pref_o_ambitious", "pref_o_shared_interests")
# chooser-level answers (identical on every row of one chooser); `attractive` is read only to tell
# choosers apart and is dropped before the person table is built
_OML_CHOOSER_KEY = ("wave", "gender", "age", "race", "field", "importance_same_race",
                    "importance_same_religion", *_OML_IMPORTANT, "attractive", "sincere", "intelligence",
                    "funny", "ambition", *INTERESTS, "expected_happy_with_sd_people",
                    "expected_num_interested_in_me", "expected_num_matches")


def _read_arff(path: str | Path) -> pd.DataFrame:
    import arff  # liac-arff

    with open(path, encoding="utf-8") as fh:
        d = arff.load(fh)
    df = pd.DataFrame(d["data"], columns=[a[0] for a in d["attributes"]])
    for a in d["attributes"]:
        if a[1] in ("NUMERIC", "REAL", "INTEGER"):
            df[a[0]] = pd.to_numeric(df[a[0]], errors="coerce")
    return df


def _sig(values: tuple[object, ...]) -> tuple[object, ...]:
    return tuple(None if (isinstance(v, float) and np.isnan(v)) or v is None else v for v in values)


def load_openml(path: str | Path) -> Market:
    df = _read_arff(path).reset_index(drop=True)
    n_raw = len(df)
    for c in ("decision", "decision_o", "match"):
        df[c] = df[c].astype(int)
    df["wave"] = df["wave"].astype(int)

    # 1. choosers = runs of consecutive rows with identical chooser-level answers
    keys = [_sig(t) for t in df[list(_OML_CHOOSER_KEY)].itertuples(index=False, name=None)]
    run_id = np.zeros(n_raw, dtype=int)
    cur = 0
    for r in range(1, n_raw):
        if keys[r] != keys[r - 1]:
            cur += 1
        run_id[r] = cur
    df["chooser"] = [f"oml_{i + 1}" for i in run_id]

    # 2. person signature as seen from the partner side
    own_sig = {pid: _sig(tuple(g.iloc[0][c] for c in ("wave", "gender", "age", "race", *_OML_IMPORTANT)))
               for pid, g in df.groupby("chooser", sort=False)}
    other = {"female": "male", "male": "female"}
    df["partner_sig"] = [
        _sig((w, other.get(g, g), a, r, *p))
        for w, g, a, r, p in zip(df["wave"], df["gender"], df["age_o"], df["race_o"],
                                 df[list(_OML_PREF_O)].itertuples(index=False, name=None))
    ]
    by_sig: dict[tuple[object, ...], list[str]] = {}
    for pid, s in own_sig.items():
        by_sig.setdefault(s, []).append(pid)

    # 3. resolve each row's partner, using the reverse row to break ties
    idx_by_chooser: dict[str, list[int]] = {}
    for r, c in enumerate(df["chooser"]):
        idx_by_chooser.setdefault(c, []).append(r)
    chosen: list[str | None] = []
    n_amb = n_none = 0
    ic = df["interests_correlate"].to_numpy(float)
    dec = df["decision"].to_numpy(int)
    dec_o = df["decision_o"].to_numpy(int)
    for r in range(n_raw):
        cands = by_sig.get(df.at[r, "partner_sig"], [])
        me_sig = own_sig[df.at[r, "chooser"]]
        ok = []
        for j in cands:
            for r2 in idx_by_chooser[j]:
                if (df.at[r2, "partner_sig"] == me_sig and dec_o[r2] == dec[r] and dec[r2] == dec_o[r]
                        and ((np.isnan(ic[r]) and np.isnan(ic[r2])) or ic[r] == ic[r2])):
                    ok.append(j)
                    break
        if len(ok) == 1:
            chosen.append(ok[0])
        elif not ok and len(cands) == 1:
            chosen.append(cands[0])        # reverse row absent from the file: trust the unique signature
        else:
            chosen.append(None)
            n_amb += len(ok) > 1
            n_none += not ok
    df["chosen"] = chosen

    per = pd.DataFrame({
        "person_id": df["chooser"], "wave": df["wave"], "gender": df["gender"], "age": df["age"],
        "field": df["field"].map(lambda v: str(v).strip().lower() if isinstance(v, str) else np.nan),
        "goal": np.nan, **{k: _clip_1_10(df[k]) for k in INTERESTS},
        **{k: _clip_1_10(df[v]) for k, v in _OML_SELF.items()}, "go_out": np.nan,
    })
    persons = per.groupby("person_id", sort=False).agg(_first_valid_agg()).reset_index().set_index("person_id")

    keep = df["chosen"].notna()
    rows = pd.DataFrame({"chooser": df.loc[keep, "chooser"], "chosen": df.loc[keep, "chosen"],
                         "wave": df.loc[keep, "wave"], "y": df.loc[keep, "decision"],
                         "match": df.loc[keep, "match"],
                         "int_corr_given": df.loc[keep, "interests_correlate"].astype(float),
                         "raw_index": np.where(keep)[0]}).reset_index(drop=True)
    persons["raw_first"] = pd.Series(np.arange(n_raw)).groupby(df["chooser"].to_numpy()).min().reindex(persons.index)
    return Market("speed_dating", persons, rows,
                  {"raw_rows": n_raw, "rows": len(rows), "persons": len(persons),
                   "dropped_rows": n_raw - len(rows), "ambiguous_partner": n_amb, "no_partner": n_none})


# ---------------------------------------------------------------------------------------- common


def load_market(name: str, params_path: str | Path = DEFAULT_PARAMS) -> Market:
    params = load_params(params_path)
    cfg = params.data.datasets[name]
    path = resolve_path(params.data.raw_dir) / cfg.file
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m gnwt.data.fetch` first")
    m = load_columbia(path) if name == "columbia" else load_openml(path)
    excl = set(params.data.exclude_waves)
    if excl:
        rows = m.rows[~m.rows["wave"].isin(excl)].reset_index(drop=True)
        m = Market(m.name, m.persons[~m.persons["wave"].isin(excl)], rows, m.report)
    return m


def to_personas(m: Market, bf: pd.DataFrame | None = None) -> list[Persona]:
    """Persona objects for the market. `bf` (index person_id, columns O..N in [0, 1]) is the offline proxy."""
    out: list[Persona] = []
    for pid, p in m.persons.iterrows():
        bfd = None
        if bf is not None and pid in bf.index:
            vals = {t: (None if pd.isna(bf.at[pid, t]) else float(bf.at[pid, t])) for t in ("O", "C", "E", "A", "N")}
            bfd = vals if any(v is not None for v in vals.values()) else None
        out.append(Persona(
            persona_id=str(pid), source=m.name, market=m.name, wave=int(p["wave"]),
            gender=p["gender"] if isinstance(p["gender"], str) else None,
            age=None if pd.isna(p["age"]) else float(p["age"]),
            field=None if pd.isna(p["field"]) else str(p["field"]),
            goal=None if pd.isna(p["goal"]) else str(p["goal"]),
            interests={k: float(p[k]) for k in INTERESTS if not pd.isna(p[k])},
            self_ratings={k: float(p[f"self_{k}"]) for k in SELF_KEYS if not pd.isna(p[f"self_{k}"])},
            bigfive=bfd, bf_source="proxy" if bfd else "none",
            context_text=EVENT_TEXT, context_source="event",
        ))
    return out
