"""Plumbing shared by experiments/run_stage1.py and experiments/compare_baselines.py."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from stage1.context.catalog import Catalog
from stage1.context.demand import ReaderArtifacts
from stage1.context.lexicon import Lexicon
from stage1.data import load
from stage1.metrics import EvalSet, dyads
from stage1.offline.estimate import Problem, SRRMSettings, crossfit

STAGE1_DIR = Path(__file__).resolve().parents[2]
CONFIG = STAGE1_DIR / "config" / "stage1.yaml"
LEXICON = STAGE1_DIR / "data" / "lexicon" / "demand_lexicon_v1.yaml"
CATALOG = STAGE1_DIR / "data" / "catalog" / "basic_contexts_v1.yaml"
RESULT = STAGE1_DIR / "result"
ARTIFACTS = STAGE1_DIR / "artifacts" / "stage1"


def load_reader(path: Path | None = None) -> ReaderArtifacts:
    """Estimated reader artifacts if given, else the illustrative ones (flagged in every output).

    On Speed Dating no participant writes a context sentence, every chooser gets the market prior, and
    the reader numbers (theta, k) never touch a0; they only matter for multi-context data.
    """
    lex, cat = Lexicon.load(LEXICON), Catalog.load(CATALOG)
    if path:
        return ReaderArtifacts.load(Path(path), lex, cat)
    return ReaderArtifacts.illustrative(lex, cat)


def settings(quick: bool = False, inner_cv=None) -> SRRMSettings:
    S = SRRMSettings.from_yaml(CONFIG)
    if quick:
        S = replace(S, crossfit_gibbs=replace(S.crossfit_gibbs, iterations=300, burn_in=100),
                    final_gibbs=replace(S.final_gibbs, iterations=400, burn_in=150, chains=2), inner_cv=5)
    if inner_cv is not None:
        S = replace(S, inner_cv=inner_cv)
    return S


def settings_hash(S: SRRMSettings) -> str:
    d = asdict(S)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def crossfit_cached(dataset: str, S: SRRMSettings, art: ReaderArtifacts, refresh: bool = False,
                    verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, Problem]:
    """Stage 1 cross-fit predictions, cached under result/cache by (data, settings, reader, code) hash."""
    data = load(dataset)
    prob = Problem(data, art, S)
    code = hashlib.sha256(b"".join(p.read_bytes() for p in sorted((STAGE1_DIR / "src").rglob("*.py")))).hexdigest()[:8]
    key = f"{dataset}_{data.sha256[:8]}_{settings_hash(S)}_{art.hash[:8]}_{code}"
    cdir = RESULT / "cache"
    fp, fc = cdir / f"crossfit_{key}.csv", cdir / f"crossfit_{key}_waves.csv"
    if fp.exists() and fc.exists() and not refresh:
        if verbose:
            print(f"Stage 1 cross-fit: cached ({fp.name})", flush=True)
        return pd.read_csv(fp), pd.read_csv(fc), prob
    if verbose:
        print(f"Stage 1 cross-fit on {dataset}: nested LOWO over {len(data.waves)} waves", flush=True)
    pred, cfg = crossfit(prob, verbose=verbose)
    cdir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(fp, index=False)
    cfg.to_csv(fc, index=False)
    return pred, cfg, prob


def female_flags(prob: Problem) -> np.ndarray:
    g = prob.data.table.gender
    return np.where(np.isnan(g), False, g == 0)


def eval_set(pred: pd.DataFrame, probs: dict[str, np.ndarray], mutual: dict[str, str], prob: Problem,
             ece_bins: int = 10) -> EvalSet:
    """EvalSet over the rows of `pred` (ordered as pred). `mutual[name]` is "m0" (use pred.m0 for Stage 1),
    "product" (p_ij * p_ji) or "harmonic" (harmonic mean of the two directions, RECON)."""
    female = female_flags(prob)
    base = pred[["chooser", "chosen", "match"]].copy()
    dy = None
    scores = {}
    for name, p in probs.items():
        base["p"] = p
        d = dyads(base, "p", female)
        if dy is None:
            dy = d
        kind = mutual.get(name, "product")
        if kind == "m0":
            base["m0"] = pred["m0"].values
            dm = dyads(base, "m0", female)
            scores[name] = dm["p_ij"].values                    # m0 is symmetric: either direction
        elif kind == "harmonic":
            scores[name] = 2 * d["p_ij"] * d["p_ji"] / np.maximum(d["p_ij"] + d["p_ji"], 1e-12)
        else:
            scores[name] = (d["p_ij"] * d["p_ji"]).values
        scores[name] = np.asarray(scores[name], float)
    return EvalSet(y=pred["y"].to_numpy(), chooser=pred["chooser"].to_numpy(), probs=probs, match_scores=scores,
                   dyad_frame=dy, ece_bins=ece_bins)


def fmt_ci(v, lo, hi, nd=3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Markdown table without the optional `tabulate` dependency."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])
