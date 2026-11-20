"""MovieLens-1M (GroupLens) as a user x movie matrix: the recommender-system benchmark of the design.

Source: https://files.grouplens.org/datasets/movielens/ml-1m.zip (6,040 users, 3,706 movies, 1,000,209
ratings, 2000-2003). Research use only, no redistribution (GroupLens licence): the script downloads it.
Used, with this binarisation, by Kang et al. 2023 (MF AUC 0.773), KAR, CoLLM, BinLLM, FLIP, LLMRank.

Rules [quy ước]:
* target y = 1{rating >= 4} ("like", the rating > 3 threshold of TALLRec / CoLLM);
* movies with >= `min_ratings` ratings (default 20: 3,043 movies, 99.5% of the ratings);
* the "question" of a movie is its title, year and genres; the answer scale is 1-5 stars;
* wave of a cell = calendar quarter of the rating (R3 holds out whole (movie, quarter) blocks);
* person features: gender, age band, occupation (the only MovieLens release with demographics).
"""

from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from lrmf_twin.data.twin2k import Item, SurveyData

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DEFAULT_DIR = Path("data/raw/movielens")
AGES = {1: "under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
OCCUPATIONS = ["other or not specified", "academic/educator", "artist", "clerical/admin", "college/grad student",
               "customer service", "doctor/health care", "executive/managerial", "farmer", "homemaker",
               "K-12 student", "lawyer", "programmer", "retired", "sales/marketing", "scientist",
               "self-employed", "technician/engineer", "tradesman/craftsman", "unemployed", "writer"]
STARS = ["1 star (worst)", "2 stars", "3 stars", "4 stars", "5 stars (best)"]


def ensure(dest: Path = DEFAULT_DIR) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "ml-1m.zip"
    if not zpath.exists():
        print("downloading MovieLens-1M (~6 MB) ...", flush=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "lrmf-twin"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(zpath.with_suffix(".part"), "wb") as out:
            shutil.copyfileobj(resp, out)
        zpath.with_suffix(".part").replace(zpath)
    return zpath


def _read(z: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    raw = z.read(f"ml-1m/{name}").decode("latin-1")
    return pd.read_csv(io.StringIO(raw), sep="::", engine="python", names=cols)


def load(dest: Path = DEFAULT_DIR, min_ratings: int = 20, max_items: int = 0) -> SurveyData:
    with zipfile.ZipFile(ensure(dest)) as z:
        ratings = _read(z, "ratings.dat", ["user", "movie", "rating", "ts"])
        movies = _read(z, "movies.dat", ["movie", "title", "genres"]).set_index("movie")
        users = _read(z, "users.dat", ["user", "gender", "age", "occupation", "zip"]).set_index("user")
    counts = ratings["movie"].value_counts()
    keep_movies = counts[counts >= min_ratings]
    if max_items:
        keep_movies = keep_movies.iloc[:max_items]
    movie_ids = np.sort(keep_movies.index.to_numpy())
    ratings = ratings[ratings["movie"].isin(movie_ids)]
    user_ids = np.sort(ratings["user"].unique())
    pi = pd.Series(np.arange(len(user_ids)), index=user_ids)
    ji = pd.Series(np.arange(len(movie_ids)), index=movie_ids)
    P, J = len(user_ids), len(movie_ids)
    rows, cols = pi[ratings["user"]].to_numpy(), ji[ratings["movie"]].to_numpy()
    V = np.full((P, J), np.nan, dtype=np.float32)
    V[rows, cols] = ratings["rating"].to_numpy()
    Y = np.where(np.isnan(V), np.nan, (V >= 4).astype(np.float32)).astype(np.float32)
    quarter = pd.to_datetime(ratings["ts"], unit="s").dt.to_period("Q")
    q_idx = {q: i for i, q in enumerate(sorted(quarter.unique()))}
    wave = np.full((P, J), -1, dtype=np.int16)
    wave[rows, cols] = quarter.map(q_idx).to_numpy()

    items = []
    for m in movie_ids:
        title, genres = movies.loc[m, "title"], movies.loc[m, "genres"].replace("|", ", ")
        items.append(Item(f"M{m}", f"M{m}", "MovieLens-1M", "ordinal",
                          f'How would this user rate the movie "{title}" (genres: {genres})?', None,
                          list(STARS), [4, 5], 3.5))
    u = users.loc[user_ids]
    cols_demo, names = [], []
    for g in ("M", "F"):
        cols_demo.append((u["gender"] == g).to_numpy()); names.append(f"gender={g}")
    for a, txt in AGES.items():
        cols_demo.append((u["age"] == a).to_numpy()); names.append(f"age={txt}")
    for k, txt in enumerate(OCCUPATIONS):
        cols_demo.append((u["occupation"] == k).to_numpy()); names.append(f"occupation={txt}")
    X_demo = np.stack(cols_demo, axis=1).astype(np.float32)
    demo_text = [f"gender: {'male' if g == 'M' else 'female'}; age: {AGES.get(a, a)}; "
                 f"occupation: {OCCUPATIONS[o] if 0 <= o < len(OCCUPATIONS) else o}"
                 for g, a, o in zip(u["gender"], u["age"], u["occupation"])]
    source = {"dataset": "MovieLens-1M (GroupLens)", "min_ratings": min_ratings, "users": P, "movies": J,
              "ratings": int((~np.isnan(Y)).sum()), "quarters": [str(q) for q in sorted(q_idx)]}
    return SurveyData(user_ids.astype(np.int64), items, Y, V, X_demo, names, demo_text, source, wave, "movielens")
