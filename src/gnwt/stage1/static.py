"""Run-time Stage 1 (stage1.md, section 10.5 "Lúc chạy").

For every persona: normalise, split clauses, match the catalog, detect keywords, classify, (11b)-(11c);
results are cached by (text_hash, artifact_hash). For every directed pair: build_features -> eta_blocks
-> a0 for both directions -> m0 -> `static_scores`. Table lookup and linear algebra only: no language
model, no sampling, no network. This module never imports gnwt.llm, never reads the held-out label
directory and never imports an embedding library unless `fallback_embedding.enabled` (test U11).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gnwt.config import Params, resolve_path, sha256_file
from gnwt.persona import Persona
from gnwt.stage1.context.catalog import Catalog
from gnwt.stage1.context.demand import ContextDemand, ReaderArtifacts, extract_demands
from gnwt.stage1.context.lexicon import Lexicon
from gnwt.stage1.features import PersonTable, build_design
from gnwt.stage1.srrm import LockedSRRM, eta_blocks
from gnwt.stage1.srrm import a0 as a0_fn
from gnwt.stage1.srrm import bvn_cdf


@dataclass(frozen=True)
class Stage1Artifacts:
    reader: ReaderArtifacts
    srrm: dict[str, LockedSRRM]          # market -> locked model
    srrm_hash: dict[str, str]


def load_reader(P: Params, reader_json: str | Path | None = None) -> ReaderArtifacts:
    cr = P.stage1.context_reader
    tl = cr.toneless_match != "off"
    lex = Lexicon.load(resolve_path(cr.lexicon.path), cr.lexicon.sha256, ngram_max=cr.ngram_max, toneless=tl,
                       min_entries_per_style_lang=cr.lexicon.min_entries_per_style_lang)
    cat = Catalog.load(resolve_path(cr.catalog.path), cr.catalog.sha256, ngram_max=cr.ngram_max, toneless=tl)
    if reader_json is not None and Path(reader_json).exists():
        return ReaderArtifacts.from_json(reader_json, lex, cat)
    return ReaderArtifacts.unestimated(lex, cat, cr.style_icc_min)


def load_artifacts(P: Params) -> Stage1Artifacts:
    """Read the lock manifest written by scripts/run_stage1.py (O9) and verify every sha256."""
    man_path = resolve_path(P.stage1.locked.manifest)
    man = json.loads(man_path.read_text(encoding="utf-8"))
    reader_entry = man.get("reader")
    reader_path = None
    if reader_entry:
        reader_path = resolve_path(reader_entry["path"])
        if sha256_file(reader_path) != reader_entry["sha256"]:
            raise ValueError("reader artifact does not match the manifest")
    reader = load_reader(P, reader_path)
    srrm, hashes = {}, {}
    for market, entry in man["srrm"].items():
        srrm[market] = LockedSRRM.load(resolve_path(entry["path"]), entry["sha256"])
        hashes[market] = entry["sha256"]
    return Stage1Artifacts(reader, srrm, hashes)


def persona_demand(p: Persona, reader: ReaderArtifacts, cache: dict[tuple[str, str], ContextDemand] | None = None) -> ContextDemand:
    text = p.context_text if p.context_source == "user" else None
    event = p.context_text if p.context_source == "event" else None
    market_ctx = p.market if p.context_source == "market" else None
    key = (f"{text}|{event}|{market_ctx}", reader.artifact_hash)
    if cache is not None and key in cache:
        return cache[key]
    cd = extract_demands(text, event, reader, market_ctx)
    if cache is not None:
        cache[key] = cd
    return cd


def score_pairs(personas: list[Persona], pairs: list[tuple[str, str, str]], art: Stage1Artifacts,
                market: str) -> tuple[dict[str, ContextDemand], list[dict[str, object]]]:
    """static_scores rows (both directions) for undirected pairs (pair_id, i, j) of one market."""
    model = art.srrm[market]
    cache: dict[tuple[str, str], ContextDemand] = {}
    demands = {p.persona_id: persona_demand(p, art.reader, cache) for p in personas}
    T = PersonTable.from_personas(personas, demands)
    I = np.array([T.index[a] for _, a, b in pairs] + [T.index[b] for _, a, b in pairs], int)
    J = np.array([T.index[b] for _, a, b in pairs] + [T.index[a] for _, a, b in pairs], int)
    n = len(pairs)
    des = build_design(T, I, J, model.std, model.spec, model.delta_basis)
    bl = eta_blocks(des, model)
    eta = bl.eta
    a0v = a0_fn(eta, model.s)
    rev = np.concatenate([np.arange(n, 2 * n), np.arange(n)])
    m0v = bvn_cdf(eta / model.s, eta[rev] / model.s, model.r)
    rows = []
    for k in range(2 * n):
        pid = pairs[k % n][0]
        mask = {g: bool(v[k]) for g, v in des.present.items()}
        rows.append({
            "pair_id": pid, "chooser": str(T.ids[I[k]]), "chosen": str(T.ids[J[k]]),
            "ctx_id": demands[str(T.ids[I[k]])].match.c_star,
            "eta0": float(bl.eta0[k]), "etaP_fit": float(bl.etaP_fit[k]), "etaP_resid": float(bl.etaP_resid[k]),
            "etaP_actor": float(bl.etaP_actor[k]), "etaG": float(bl.etaG[k]), "eta": float(eta[k]),
            "a0": float(a0v[k]), "m0": float(m0v[k]), "fit": float(des.fit[k]),
            "fit_parts": json.dumps([round(float(x), 6) for x in des.fit_parts[k]]),
            "mask": json.dumps(mask, sort_keys=True), "srrm_hash": art.srrm_hash[market],
        })
    return demands, rows


def run_stage1(db: "Repo", P: Params, art: Stage1Artifacts) -> None:  # noqa: F821 - Repo is imported lazily
    """Score every pair stored in `pairs` with the locked artifacts and write `static_scores`."""
    personas = {r["persona_id"]: Persona.model_validate_json(r["json"]) for r in db.query("SELECT persona_id, json FROM personas")}
    by_market: dict[str, list[tuple[str, str, str]]] = {}
    for r in db.query("SELECT pair_id, market, i, j FROM pairs"):
        by_market.setdefault(r["market"], []).append((r["pair_id"], r["i"], r["j"]))
    for market, pairs in by_market.items():
        if market not in art.srrm:
            raise KeyError(f"no locked SRRM for market {market!r}")
        ids = sorted({x for _, a, b in pairs for x in (a, b)})
        ps = [personas[x] for x in ids]
        demands, rows = score_pairs(ps, pairs, art, market)
        db.insert_many("context_demands", [cd.as_row(pid) | {"lexicon_ver": art.reader.lexicon.version,
                                                             "catalog_ver": art.reader.catalog.version}
                                           for pid, cd in demands.items()], replace=True)
        unmatched = [(cd.text_hash, pid) for pid, cd in demands.items() if cd.match.how == "generic"]
        for th, pid in unmatched:
            db.insert("unmatched_contexts", {"text_hash": th, "text": personas[pid].context_text, "n_seen": 1},
                      replace=True)
        db.insert_many("static_scores", rows, replace=True)
