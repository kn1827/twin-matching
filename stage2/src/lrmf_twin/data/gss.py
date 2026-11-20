"""General Social Survey 1972-2024 cumulative file (NORC) as a person x question matrix with waves = years.

Source: https://gss.norc.org (Stata cumulative file, Release 3a, July 2026; free download). The same data
feed Kim & Lee (AI-augmented surveys: imputation, retrodiction of (question, year) pairs, unasked
questions) and Chen et al. 2026 (GSS 2016-2024 + WVS-7 benchmark of LLM-simulated respondents).

Selection rules [quy ước], applied once and cached in data/processed/:
* persons: respondents of `year_min`..2024 (default 2000, ~37.6k persons) with >= 20 answers kept;
* questions: variables after `natspac` in file order (the attitude and topical sections; the background
  block before it is skipped), with a Stata value label, 2-10 valid codes, >= `min_answers` answers in the
  window, not a demographic feature and not administrative (weights, ballots, interview data, codes,
  counts, dates, household and network rosters, follow-on eligibility, consent); derived scores that contain
  other targets (wordsum) are dropped, and so are mode variants of a kept question (abanyg, trustv, trustnv:
  same wording, web grid or volunteered-answer version); the `max_items` most answered are kept;
* target: same rules as Twin-2K: 2 codes -> y = 1{first code}; >= 3 codes -> cut point between ordered codes
  with positive rate closest to 0.5; items outside [0.05, 0.95] dropped. Codes "don't know / IAP / no
  answer" are Stata extended missing values in the cumulative file and stay missing;
* wave of a cell = index of the respondent's survey year (R3 holds out whole (question, year) blocks);
* person features: survey year, age band, sex, race, degree, region, marital status, religion, religious
  attendance, party id, political views, real income quintile, labour force status.
Question text = the Stata variable label (short) plus the value labels; the codebook PDF has fuller wording.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from lrmf_twin.data.twin2k import RATE_RANGE, Item, SurveyData, _balanced_cut, clean

URL = "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_stata.zip"
DEFAULT_DIR = Path("data/raw/gss")
FIRST_ATTITUDE_VAR = "natspac"
DEMO_VARS = {"year": "survey year", "age": "age", "sex": "sex", "race": "race", "degree": "degree",
             "region": "region", "marital": "marital status", "relig": "religion", "attend": "religious attendance",
             "partyid": "party id", "polviews": "political views", "realinc": "income", "wrkstat": "work status"}
ADMIN_NAME = re.compile(r"^(id|year|ballot|version|issp|formwt|wt|oversamp|samp|phase|span|mode|date|"
                        r"int(age|sex|ethn|id|yrs|race\d*|hisp|lang)$|feelevel|lngth|cooprtn|comprend|respnum|"
                        r"hhrace|cohort|xnorcsiz|srcbelt|size|vpsu|vstrat|dwell|hefinfo|zodiac)", re.I)
ADMIN_LABEL = re.compile(r"weight|ballot|interview|sampl|respondent id|version|flag|date|number of|how many|"
                         r"\bcode\b|id number|language of|phase|panel|case|psu|stratum|year|household members|"
                         r"presence of others|follow-on|consent|vital status|probability that this|person #", re.I)
NON_QUESTION = re.compile(r"^(form|consent|adminconsent|respond|incuspop|death|saq\w*|whoelse\d+|gender\d+|"
                          r"(spouse|parent|child|othfam|cowork|memgrp|friend|advisor)\d+|\w+_7222|wordsum\w*|"
                          r"raceacs\d+|sexbirth\d*|sexnow\d*|citizen|uscitzn|ethworld\d+|ethregion\d+|"
                          r"\w*eligible|maborn|paborn)$", re.I)
VARIANT = re.compile(r"^(?P<base>[a-z]+?\d*)(nv|v|g|y|x)$")
AGE_BINS = [(18, 29), (30, 44), (45, 59), (60, 74), (75, 120)]


def ensure(dest: Path = DEFAULT_DIR) -> Path:
    """Path of the cumulative .dta, downloading and extracting the zip if needed."""
    dest.mkdir(parents=True, exist_ok=True)
    found = sorted(dest.rglob("gss7224*.dta")) or sorted(dest.rglob("*.dta"))
    if found:
        return found[-1]
    zpath = dest / "GSS_stata.zip"
    if not zpath.exists():
        print("downloading GSS cumulative file (~48 MB) ...", flush=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 lrmf-twin"})
        with urllib.request.urlopen(req, timeout=600) as resp, open(zpath.with_suffix(".part"), "wb") as out:
            shutil.copyfileobj(resp, out)
        zpath.with_suffix(".part").replace(zpath)
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".dta"))
        z.extract(name, dest)
    return dest / name


def _codes(v: np.ndarray) -> np.ndarray:
    u = np.unique(v[~np.isnan(v)])
    return u


def _demographics(df: pd.DataFrame, vals: dict, lbl: dict) -> tuple[np.ndarray, list[str], list[str]]:
    cols, names, parts = [], [], [[] for _ in range(len(df))]
    for var, nice in DEMO_VARS.items():
        if var not in df:
            continue
        x = df[var].to_numpy(dtype=np.float64)
        labels = vals.get(lbl.get(var, ""), {})
        if var == "age":
            groups = [((x >= a) & (x <= b), f"{a}-{b}" if b < 120 else f"{a}+") for a, b in AGE_BINS]
        elif var == "realinc":
            ok = ~np.isnan(x)
            qs = np.nanquantile(x, [0.2, 0.4, 0.6, 0.8]) if ok.any() else np.zeros(4)
            band = np.digitize(x, qs)
            groups = [(ok & (band == k), f"quintile {k + 1}") for k in range(5)]
        elif var == "year":
            groups = [(x == yv, str(int(yv))) for yv in _codes(x)]
        else:
            codes = _codes(x)
            if var == "relig" and codes.size > 6:      # keep the 5 most common religions + other
                top = pd.Series(x).value_counts().index[:5].to_numpy()
                groups = [(x == c, str(labels.get(int(c), int(c)))) for c in top]
                groups.append(((~np.isnan(x)) & ~np.isin(x, top), "other"))
            else:
                groups = [(x == c, str(labels.get(int(c), int(c)))) for c in codes]
        for mask, text in groups:
            cols.append(mask.astype(np.float32))
            names.append(f"{nice}={text}")
            for p in np.flatnonzero(mask):
                parts[p].append(f"{nice}: {clean(text)}")
    return np.stack(cols, axis=1), names, ["; ".join(pp) for pp in parts]


def load(dest: Path = DEFAULT_DIR, year_min: int = 2000, min_answers: int = 2000, max_items: int = 600,
         min_person_answers: int = 20, cache_dir: Path = Path("data/processed")) -> SurveyData:
    key = hashlib.sha256(f"gss|{year_min}|{min_answers}|{max_items}|{min_person_answers}|v3".encode()).hexdigest()[:12]
    cache = cache_dir / f"gss_{key}.npz"
    if cache.exists():
        return SurveyData.load_cached(cache)
    dta = ensure(dest)
    warnings.filterwarnings("ignore", category=UnicodeWarning)
    with pd.io.stata.StataReader(dta) as r:
        var_labels = r.variable_labels()
        vals = r.value_labels()
        lbllist = r._lbllist                                       # value-label set of each variable
    cols = list(var_labels)
    lbl = dict(zip(cols, lbllist))
    start = cols.index(FIRST_ATTITUDE_VAR)
    candidates = []
    for i, c in enumerate(cols):
        n_lab = len(vals.get(lbl[c], {}))
        if (i >= start and lbl[c] and 2 <= n_lab <= 15 and c not in DEMO_VARS and not NON_QUESTION.match(c)
                and not ADMIN_NAME.match(c) and not ADMIN_LABEL.search(var_labels[c] or "")):
            candidates.append(c)
    keep_cols = ["year", "id"] + [c for c in DEMO_VARS if c in cols and c != "year"] + candidates
    print(f"GSS: reading {len(candidates)} candidate questions from {dta.name} (one pass, a few minutes) ...",
          flush=True)
    parts = []
    for ch in pd.read_stata(dta, convert_categoricals=False, chunksize=5000, columns=keep_cols):
        ch = ch[ch["year"] >= year_min]
        if len(ch):
            parts.append(ch.apply(pd.to_numeric, errors="coerce").astype(np.float32))
    df = pd.concat(parts, ignore_index=True)
    del parts

    Xc = df[candidates].to_numpy(dtype=np.float32)
    n_ans = (~np.isnan(Xc)).sum(0)
    n_codes = np.array([_codes(Xc[:, k]).size for k in range(len(candidates))])
    ok = (n_ans >= min_answers) & (n_codes >= 2) & (n_codes <= 10)
    kept_names = {candidates[k] for k in np.flatnonzero(ok)}
    for k in np.flatnonzero(ok):                          # mode variants of a kept question (abany -> abanyg)
        m = VARIANT.match(candidates[k])
        if m and m.group("base") in kept_names and m.group("base") != candidates[k]:
            ok[k] = False
    order = [k for k in np.argsort(-n_ans, kind="stable") if ok[k]]

    items, ys, vs = [], [], []
    for k in order:
        if len(items) == max_items:
            break
        c = candidates[k]
        raw = Xc[:, k].astype(np.float64)
        codes = _codes(raw)
        v = np.where(np.isnan(raw), np.nan, np.searchsorted(codes, np.nan_to_num(raw)) + 1.0)
        labels = vals.get(lbl[c], {})
        options = [clean(str(labels.get(int(code), int(code)))) for code in codes]
        if len(codes) == 2:
            cut, positive = 1.5, [1]
            y = np.where(np.isnan(v), np.nan, (v < cut).astype(np.float32))
        else:
            cut = _balanced_cut(v[~np.isnan(v)], np.arange(1.5, len(codes), 1.0))
            positive = [p for p in range(1, len(codes) + 1) if p > cut]
            y = np.where(np.isnan(v), np.nan, (v > cut).astype(np.float32))
        rate = np.nanmean(y)
        if not (RATE_RANGE[0] <= rate <= RATE_RANGE[1]):
            continue
        stem = clean(var_labels[c]) or c
        items.append(Item(c, c, "GSS", "ordinal", f"{stem} ({c})", None, options, positive, cut))
        ys.append(y)
        vs.append(v)
    Y = np.stack(ys, axis=1).astype(np.float32)
    V = np.stack(vs, axis=1).astype(np.float32)

    keep = (~np.isnan(Y)).sum(1) >= min_person_answers
    df, Y, V = df[keep].reset_index(drop=True), Y[keep], V[keep]
    X_demo, demo_names, demo_text = _demographics(df, vals, lbl)
    years = df["year"].to_numpy().astype(int)
    year_idx = {yv: i for i, yv in enumerate(sorted(set(years)))}
    wave = np.where(np.isnan(Y), -1, np.array([year_idx[yv] for yv in years])[:, None]).astype(np.int16)
    pids = (years * 100000 + df["id"].to_numpy().astype(int)).astype(np.int64)
    source = {"dataset": "GSS 1972-2024 cumulative (NORC, Release 3a)", "file": dta.name, "year_min": year_min,
              "candidates": len(candidates), "items_kept": len(items), "persons": int(keep.sum()),
              "years": sorted(year_idx)}
    data = SurveyData(pids, items, Y, V, X_demo, demo_names, demo_text, source, wave, "gss")
    data.save(cache)
    return data
