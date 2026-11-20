"""Run-time Stage 1 (tech spec 10.5): table look-ups and linear algebra only -- no language model, no
sampling, no network.

For every persona: normalise, split clauses, match the catalog, find keywords, assign classes, compute
(11b)-(11c); cached by (text_hash, artifact_hash). For every directed pair: features -> eta blocks ->
a0 both ways -> m0 -> `static_scores`.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stage1.context.demand import ContextDemand, ReaderArtifacts, extract_demands
from stage1.context.styles import STYLES
from stage1.features import BLOCKS, COL, PersonTable, design_from_raw, pair_raw
from stage1.srrm import LockedSRRM, a0, eta_blocks, m0

SCHEMA = Path(__file__).with_name("schema.sql")


class Stage1DB:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.executescript(SCHEMA.read_text(encoding="utf-8"))

    def close(self):
        self.con.commit()
        self.con.close()

    def record_artifact(self, name: str, sha256: str, path: str | None):
        self.con.execute("INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?)",
                         (name, sha256, path, datetime.now(timezone.utc).isoformat()))


def _demand_row(pid: str, cd: ContextDemand, art: ReaderArtifacts) -> tuple:
    hits = [{"style": h.style, "cls": h.cls, "span": list(h.span), "clause": h.clause, "toneless": h.toneless}
            for h in cd.hits]
    return (pid, cd.text_hash, cd.match.c_star, cd.match.how, json.dumps(dict(zip(STYLES, cd.D_style.round(6).tolist()))),
            *map(float, cd.D_tilde), json.dumps(dict(zip(STYLES, cd.Z.round(6).tolist()))), json.dumps(hits),
            art.lexicon.version, art.catalog.version, cd.artifact_hash, cd.reader_status, int(cd.rated_prior))


def run_stage1(db: Stage1DB, data, art: ReaderArtifacts, model: LockedSRRM,
               pairs: np.ndarray | None = None) -> dict:
    """Write context demands and static scores of `data` (a DyadicData) with a locked model.

    `pairs` (k, 2) of person row indices; default: every directed pair of the dataset.
    """
    cache: dict[tuple[str, str], ContextDemand] = {}
    demands = []
    con = db.con
    for p in data.personas:
        key = (p.context_text or "", data.event_text, p.market or data.market)
        ck = (repr(key), art.hash)
        if ck not in cache:
            cache[ck] = extract_demands(p.context_text, data.event_text, art, p.market or data.market)
        cd = cache[ck]
        demands.append(cd)
        con.execute("INSERT OR REPLACE INTO personas VALUES (?,?,?,?)", (p.pid, p.wave, p.gender, p.bf_source))
        con.execute("INSERT OR REPLACE INTO context_texts VALUES (?,?,?,?)",
                    (p.pid, p.context_text or (data.event_text if cd.source == "event" else None), "en", cd.source))
        con.execute("INSERT OR REPLACE INTO context_demands VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    _demand_row(p.pid, cd, art))
        if cd.source == "user" and cd.match.how == "generic":
            con.execute("INSERT INTO unmatched_contexts VALUES (?,?,1) ON CONFLICT(text_hash) "
                        "DO UPDATE SET n_seen = n_seen + 1", (cd.text_hash, p.context_text))
    if pairs is None:                                  # each unordered pair once; both directions are scored
        pairs = np.unique(np.sort(data.pairs[["chooser", "chosen"]].to_numpy(), axis=1), axis=0)
    a, b = pairs[:, 0], pairs[:, 1]
    t: PersonTable = data.table
    D = np.array([cd.D_tilde for cd in demands])
    raw_ab = pair_raw(t, a, b, model.feature_cfg)
    raw_ba = pair_raw(t, b, a, model.feature_cfg)
    des_ab = design_from_raw(raw_ab, D[a], model.standardizer)
    des_ba = design_from_raw(raw_ba, D[b], model.standardizer)
    bl_ab, bl_ba = eta_blocks(des_ab.X, model), eta_blocks(des_ba.X, model)
    e_ab, e_ba = bl_ab.eta, bl_ba.eta
    a0_ab, a0_ba = a0(e_ab, model.s), a0(e_ba, model.s)
    m = m0(e_ab, e_ba, model.r, model.s)
    srrm_hash = model.hash
    rows = []
    for k in range(len(a)):
        u, v = data.personas[a[k]].pid, data.personas[b[k]].pid
        pair_id = "|".join(sorted((u, v)))
        for (ch, ce, bl, des, aa) in ((u, v, bl_ab, des_ab, a0_ab), (v, u, bl_ba, des_ba, a0_ba)):
            rows.append((pair_id, ch, ce, demands[data.index[ch]].match.c_star,
                         float(bl.eta0[k]), float(bl.etaP_fit[k]), float(bl.etaP_resid[k]), float(bl.etaP_actor[k]),
                         float(bl.etaG[k]), float(bl.eta[k]), float(aa[k]), float(m[k]), float(des.X[k, COL["fit"]]),
                         json.dumps(np.round(des.fit_parts[k], 6).tolist()),
                         json.dumps({b_: bool(not des.present[b_][k]) for b_ in BLOCKS}), srrm_hash))
    con.executemany("INSERT OR REPLACE INTO static_scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    db.record_artifact("srrm", srrm_hash, model.meta.get("path"))
    db.record_artifact("reader", art.hash, None)
    con.commit()
    return {"personas": len(data.personas), "static_rows": len(rows), "srrm_hash": srrm_hash, "reader_hash": art.hash}
