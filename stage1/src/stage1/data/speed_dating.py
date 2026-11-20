"""Speed Dating data (Fisman, Iyengar, Kamenica & Simonson 2006; 21 waves, Columbia Business School,
2002-2004), in two forms:

* ``columbia``  the original "Speed Dating Data.csv" (Columbia; Gelman's ARM site): one row per
                directed date with person ids (iid, pid), wave, goal, field code and every survey item.
* ``openml``    OpenML dataset 40536 "SpeedDating": the same dates re-coded without person ids. Persons
                are rebuilt from their constant profile fields within a wave, and every row is linked
                to its partner through the partner fields it carries (age_o, race_o, pref_o_*). Rows
                whose partner cannot be linked keep their partner blocks missing (MIM).

Only PRE-DATE fields are used: no rating made during or after the date (attr_o, like, prob, met,
*_partner, guess_prob_liked, ...) and no appearance field enters Stage 1. The label is the directed
decision (`dec` / `decision`); `match` is both decisions = 1.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stage1.features import N_INTERESTS, Persona, PersonTable

COLUMBIA_URL = "https://www.stat.columbia.edu/~gelman/arm/examples/speed.dating/Speed%20Dating%20Data.csv"
OPENML_URL = "https://www.openml.org/data/download/13153954/speeddating.arff"
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "speed_dating"
FILES = {"columbia": "columbia_speed_dating.csv", "openml": "openml_speeddating.arff"}
URLS = {"columbia": COLUMBIA_URL, "openml": OPENML_URL}

INTERESTS = ("sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming", "clubbing",
             "reading", "tv", "theater", "movies", "concerts", "music", "shopping", "yoga")
GOAL_CODES = ("fun_night", "meet_new", "get_date", "serious", "say_did", "other")     # Columbia goal 1..6
EVENT_TEXT = "speed dating event"
MARKET = "speed_dating"


@dataclass
class DyadicData:
    name: str
    personas: list[Persona]
    pairs: pd.DataFrame                      # wave, chooser, chosen (person row index), y, match
    rich: pd.DataFrame                       # person-level pre-date fields for the baselines (numeric)
    rich_categorical: tuple[str, ...]
    market: str = MARKET
    event_text: str = EVENT_TEXT
    info: dict = field(default_factory=dict)
    sha256: str = ""

    def __post_init__(self):
        self.table = PersonTable.from_personas(self.personas)
        self.index = {p.pid: k for k, p in enumerate(self.personas)}

    @property
    def waves(self) -> np.ndarray:
        return np.sort(self.pairs["wave"].unique())


def ensure_file(name: str, directory: Path = DEFAULT_DIR, download: bool = True) -> Path:
    path = Path(directory) / FILES[name]
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} is missing; download it from {URLS[name]}")
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URLS[name]} -> {path}", flush=True)
        urllib.request.urlretrieve(URLS[name], path)
    return path


def load(name: str, directory: Path = DEFAULT_DIR, download: bool = True) -> DyadicData:
    path = ensure_file(name, directory, download)
    if name == "columbia":
        return load_columbia(path)
    if name == "openml":
        return load_openml(path)
    raise ValueError(f"unknown dataset {name!r}: use 'columbia' or 'openml'")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f


def _keys(frame: pd.DataFrame) -> list[str]:
    """Row keys with an explicit "nan" for missing values (pandas >= 3 keeps NaN as float in astype(str))."""
    return ["|".join("nan" if (not isinstance(v, str) and pd.isna(v)) else str(v) for v in row)
            for row in frame.itertuples(index=False)]


def _norm_weights(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Stated-preference points rescaled to sum to 100 (waves 6-9 used a 1-10 scale)."""
    w = df[cols].astype(float)
    tot = w.sum(axis=1, min_count=1)
    return w.div(tot.where(tot > 0), axis=0) * 100


# ---------------------------------------------------------------------------------------- Columbia
def load_columbia(path: Path) -> DyadicData:
    d = pd.read_csv(path, encoding="latin-1", low_memory=False).copy()
    for c in ("income", "tuition", "mn_sat"):
        if c in d:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce")
    d = d.copy()
    first = d.sort_values(["iid", "round"] if "round" in d else ["iid"]).groupby("iid", as_index=False).first()
    pref = ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]
    opp = ["attr2_1", "sinc2_1", "intel2_1", "fun2_1", "amb2_1", "shar2_1"]
    first[pref] = _norm_weights(first, pref)
    first[opp] = _norm_weights(first, opp)
    personas = []
    for _, r in first.iterrows():
        goal = None if pd.isna(r["goal"]) else GOAL_CODES[int(r["goal"]) - 1]
        personas.append(Persona(
            pid=str(int(r["iid"])), wave=int(r["wave"]), gender=int(r["gender"]), age=_num(r["age"]),
            field=None if pd.isna(r["field_cd"]) else str(int(r["field_cd"])), goal=goal,
            interests=np.array([_num(r[c]) for c in INTERESTS]),
            self_ratings=np.array([_num(r[c]) for c in ("sinc3_1", "intel3_1", "fun3_1", "amb3_1")]),
            bf=np.full(5, np.nan), bf_source="none", context_text=None, market=MARKET,
        ))
    rich_cols_num = (["age", "imprace", "imprelig", "date", "go_out", "exphappy"] + list(INTERESTS) + pref + opp
                     + ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1"])
    rich = first[rich_cols_num].astype(float).copy()
    rich["gender"] = first["gender"].astype(float)
    rich["race"] = first["race"].astype("Int64").astype(str).replace("<NA>", "nan")
    rich["field_cd"] = first["field_cd"].astype("Int64").astype(str).replace("<NA>", "nan")
    rich["goal"] = first["goal"].astype("Int64").astype(str).replace("<NA>", "nan")
    rich["career_c"] = first["career_c"].astype("Int64").astype(str).replace("<NA>", "nan")
    rich.index = range(len(rich))
    idx = {str(int(i)): k for k, i in enumerate(first["iid"])}
    rows = d[d["pid"].notna()].copy()
    rows["pid_s"] = rows["pid"].astype(int).astype(str)
    rows = rows[rows["pid_s"].isin(idx)]
    pairs = pd.DataFrame({
        "wave": rows["wave"].astype(int).values,
        "chooser": rows["iid"].astype(int).astype(str).map(idx).values,
        "chosen": rows["pid_s"].map(idx).values,
        "y": rows["dec"].astype(int).values,
        "match": rows["match"].astype(int).values,
    }).drop_duplicates(["chooser", "chosen"]).reset_index(drop=True)
    info = {"rows": int(len(d)), "pairs": int(len(pairs)), "persons": len(personas),
            "dropped_rows_without_partner": int(len(d) - len(pairs))}
    return DyadicData("columbia", personas, pairs, rich, ("race", "field_cd", "goal", "career_c"),
                      info=info, sha256=_sha(path))


# ---------------------------------------------------------------------------------------- OpenML
def read_arff(path: Path) -> pd.DataFrame:
    names, data_lines, in_data = [], [], False
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if in_data:
            data_lines.append(s)
        elif s.lower().startswith("@attribute"):
            rest = s[len("@attribute"):].strip()
            if rest.startswith("'"):
                names.append(rest[1:rest.index("'", 1)])
            else:
                names.append(rest.split()[0])
        elif s.lower().startswith("@data"):
            in_data = True
    reader = csv.reader(io.StringIO("\n".join(data_lines)), quotechar="'", skipinitialspace=True)
    df = pd.DataFrame(list(reader), columns=names)
    return df.replace("?", np.nan)


OM_PREF = ["attractive_important", "sincere_important", "intellicence_important", "funny_important",
           "ambtition_important", "shared_interests_important"]
OM_PREF_O = ["pref_o_attractive", "pref_o_sincere", "pref_o_intelligence", "pref_o_funny", "pref_o_ambitious",
             "pref_o_shared_interests"]
OM_SELF = ["attractive", "sincere", "intelligence", "funny", "ambition"]
OM_EXPECT = ["expected_happy_with_sd_people", "expected_num_interested_in_me", "expected_num_matches"]


def load_openml(path: Path) -> DyadicData:
    df = read_arff(path)
    num_cols = ["wave", "age", "age_o", "importance_same_race", "importance_same_religion", "decision",
                "decision_o", "match", "interests_correlate"] + OM_PREF + OM_PREF_O + OM_SELF + list(INTERESTS) + OM_EXPECT
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["field_n"] = df["field"].astype(str).str.strip().str.lower().replace("nan", np.nan)
    key_cols = (["wave", "gender", "age", "race", "field_n", "importance_same_race", "importance_same_religion"]
                + OM_PREF + OM_SELF + list(INTERESTS) + OM_EXPECT)
    keys = pd.Series(_keys(df[key_cols]))
    codes, uniq = pd.factorize(keys)
    df["person"] = codes
    persons = df.groupby("person", as_index=False).first()
    # partner linkage: same wave, opposite gender, (age, race, stated preferences) == the row's partner fields
    pkey_person = _keys(persons[["wave", "gender", "age", "race"] + OM_PREF])
    lookup: dict[str, list[int]] = {}
    for pid, k in zip(persons["person"], pkey_person):
        lookup.setdefault(k, []).append(int(pid))
    other = {"female": "male", "male": "female"}
    rk = pd.concat([df["wave"], df["gender"].map(other), df["age_o"], df["race_o"]] + [df[c] for c in OM_PREF_O],
                   axis=1)
    row_keys = _keys(rk)
    cand = [lookup.get(k, []) for k in row_keys]
    partner = np.full(len(df), -1)
    for r, c in enumerate(cand):
        if len(c) == 1:
            partner[r] = c[0]
    # ambiguous rows: keep the candidate that has a row pointing back to this chooser
    links = {(int(a), int(b)) for a, b in zip(df["person"], partner) if b >= 0}
    n_amb_fixed = 0
    for r, c in enumerate(cand):
        if len(c) > 1:
            back = [p for p in c if (p, int(df["person"].iat[r])) in links]
            if len(back) == 1:
                partner[r] = back[0]
                n_amb_fixed += 1
    df["partner"] = partner
    pers_index = {int(p): k for k, p in enumerate(persons["person"])}
    personas = []
    for _, r in persons.iterrows():
        personas.append(Persona(
            pid=f"om{int(r['person'])}", wave=int(r["wave"]), gender=1 if r["gender"] == "male" else 0,
            age=_num(r["age"]), field=None if pd.isna(r["field_n"]) else str(r["field_n"]), goal=None,
            interests=np.array([_num(r[c]) for c in INTERESTS]),
            self_ratings=np.array([_num(r[c]) for c in ("sincere", "intelligence", "funny", "ambition")]),
            bf=np.full(5, np.nan), bf_source="none", market=MARKET,
        ))
    # unlinked partners become placeholder personas with every block missing (MIM)
    n_unlinked = int((partner < 0).sum())
    chosen = []
    for r in range(len(df)):
        if partner[r] >= 0:
            chosen.append(pers_index[int(partner[r])])
        else:
            personas.append(Persona(pid=f"om_unlinked_{r}", wave=int(df["wave"].iat[r]),
                                    gender=0 if df["gender"].iat[r] == "male" else 1,
                                    age=_num(df["age_o"].iat[r]), market=MARKET))
            chosen.append(len(personas) - 1)
    pairs = pd.DataFrame({
        "wave": df["wave"].astype(int).values,
        "chooser": df["person"].map(pers_index).values,
        "chosen": np.array(chosen),
        "y": df["decision"].astype(int).values,
        "match": df["match"].astype(int).values,
    }).drop_duplicates(["chooser", "chosen"]).reset_index(drop=True)
    prefs = persons[OM_PREF].astype(float)
    tot = prefs.sum(axis=1)
    prefs = prefs.div(tot.where(tot > 0), axis=0) * 100
    rich = pd.DataFrame({"age": persons["age"].astype(float), "imprace": persons["importance_same_race"],
                         "imprelig": persons["importance_same_religion"]})
    for c in list(INTERESTS) + OM_EXPECT:
        rich[c] = persons[c].astype(float)
    for c, o in zip(OM_PREF, ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]):
        rich[o] = prefs[c].values
    for c, o in zip(OM_SELF, ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1"]):
        rich[o] = persons[c].astype(float).values
    rich["gender"] = (persons["gender"] == "male").astype(float)
    rich["race"] = persons["race"].astype(str)
    rich["field_cd"] = persons["field_n"].astype(str)
    rich = rich.reset_index(drop=True)
    extra = pd.DataFrame(np.nan, index=range(len(personas) - len(rich)), columns=rich.columns)
    for c in ("race", "field_cd"):
        extra[c] = "nan"
    if len(extra):
        extra["age"] = [p.age for p in personas[len(rich):]]
        extra["gender"] = [float(p.gender) for p in personas[len(rich):]]
    rich = pd.concat([rich, extra], ignore_index=True)
    linked = partner >= 0
    dec_o_ok = None
    if linked.any():
        back = {(int(a), int(b)): int(y) for a, b, y in zip(df["person"], partner, df["decision"]) if b >= 0}
        agree = [back.get((int(b), int(a))) == int(yo) for a, b, yo in zip(df["person"], partner, df["decision_o"])
                 if b >= 0 and (int(b), int(a)) in back and not pd.isna(yo)]
        dec_o_ok = float(np.mean(agree)) if agree else None
    info = {"rows": int(len(df)), "persons_rebuilt": int(len(persons)), "pairs": int(len(pairs)),
            "partner_linked": int(linked.sum()), "partner_unlinked": n_unlinked,
            "ambiguous_resolved_by_reciprocity": n_amb_fixed,
            "decision_o_consistency": dec_o_ok}
    return DyadicData("openml", personas, pairs, rich, ("race", "field_cd"), info=info, sha256=_sha(path))
