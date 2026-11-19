"""Offline steps O6-O9 and the section 9 evaluation of Stage 1 on Speed Dating (both files).

    python scripts/run_stage1.py [--markets columbia speed_dating] [--workers 4] [--quick] [--no-slow]

For each market:
  O6  context demands of every persona (event description "speed dating event": no user sentence)
  O7  features, standardised within each training fold
  O8  nested LOWO selection + Hoff Gibbs sampler; cross-fit a0 for every wave (I1); locked model on all
      development waves (4 chains, split-R^ reported)
  O9  artifacts with sha256 in artifacts/stage1/lock.json and in the `artifacts` table
  9   metrics of a0; with --section9 also the branches (floors, B1, learners, S8) and a chooser-cluster bootstrap
      (B = analysis.bootstrap_B) of AUC differences against a0, TOST on +-analysis.delta_auc_min
Outputs: reports/stage1_<market>.json, reports/stage1_<market>.md, reports/predictions_<market>.csv
and the database tables of section 10.2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gnwt.analysis.branches import FAST_BRANCHES, SLOW_BRANCHES, crossfit_branches, srrm_variants
from gnwt.analysis.metrics import Scored, cluster_bootstrap, summarize, tost
from gnwt.config import DEFAULT_PARAMS, PROJECT_ROOT, load_params, resolve_path, sha256_file
from gnwt.data.bfi10 import PROXY_VERSION, build_proxy
from gnwt.data.speed_dating import load_market, to_personas
from gnwt.db import Repo
from gnwt.stage1.features import DesignSpec, PersonTable
from gnwt.stage1.offline.srrm_fit import DyadData, FitConfig, crossfit, estimate_srrm
from gnwt.stage1.static import load_reader, persona_demand

REPORTS = PROJECT_ROOT / "reports"
ART_DIR = PROJECT_ROOT / "artifacts" / "stage1"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_market(market: str, P, args) -> dict:  # noqa: ANN001
    t0 = time.time()
    m = load_market(market)
    _log(f"{market}: {m.report}")
    bf = build_proxy(m.persons, P.bf_proxy.min_items_per_trait)
    personas = to_personas(m, bf)
    reader = load_reader(P)
    cache: dict = {}
    demands = {p.persona_id: persona_demand(p, reader, cache) for p in personas}
    T = PersonTable.from_frame(m.persons, bf, demands, default_context="speed_dating")
    d = DyadData.from_rows(T, m.rows)
    spec = DesignSpec.from_params(P.stage1.features)
    s = P.stage1.core.sampler
    cfg_cf = FitConfig.from_params(P, chains=s.crossfit_chains)
    cfg_lock = FitConfig.from_params(P, chains=s.chains)
    if args.quick:
        cfg_cf = replace(cfg_cf, iters=400, burn=200, pilot_iters=200)
        cfg_lock = replace(cfg_lock, iters=600, burn=300, pilot_iters=200)

    # O8: cross-fit a0 (LOWO) and the locked model
    _log(f"{market}: cross-fit a0 over {len(np.unique(d.wave))} waves")
    pred, fold_models = crossfit(d, spec, cfg_cf, workers=args.workers)
    _log(f"{market}: locked SRRM on all development waves ({cfg_lock.chains} chains)")
    locked = estimate_srrm(d, spec, cfg_lock)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    digest = locked.sha256()
    lock_path = ART_DIR / f"srrm_{market}_{digest[:12]}.json"
    locked.save(lock_path)

    # section 9 branches: opt-in (--section9). The default run fits a0 only; the comparison with the
    # published baselines lives in experiments/compare_literature_baselines.py.
    obs = d.observed
    y = d.y
    branches: list[Scored] = []
    bp = pd.DataFrame({"random__score": np.zeros(len(y))})
    if args.section9:
        names = FAST_BRANCHES + (() if args.no_slow else SLOW_BRANCHES)
        _log(f"{market}: branches {names}")
        bp = crossfit_branches(d, names, P.global_seed, workers=args.workers)
        for nme in names:
            src = bp["random__score"] if nme == "random" else bp[nme]
            sc = src.to_numpy()
            branches.append(Scored(nme, bp[nme].to_numpy(), sc * sc[d.rev]))
        var_names = ("S8_delta0", "SRRM_no_dyad", "GLMM_crossed")
        cfg_var = replace(cfg_cf, chains=1) if not args.quick else cfg_cf
        _log(f"{market}: SRRM variants {var_names}")
        for nme, vp in srrm_variants(d, spec, cfg_var, args.workers, var_names).items():
            branches.append(Scored(nme, vp["a0"].to_numpy(), vp["m0"].to_numpy()))
    ref = Scored("a0_SRRM", pred["a0"].to_numpy(), pred["m0"].to_numpy())

    # random branch: AUC from the random score, probabilities = base rate
    rows = np.where(obs)[0]
    first = np.where(obs & (np.arange(len(y)) < d.rev) & obs[d.rev])[0]
    chooser = d.i
    yy = np.nan_to_num(y)
    match = np.nan_to_num(d.match)
    table = {}
    for sc in [ref, *branches]:
        sc_eval = sc
        if sc.name == "random":
            sc_eval = Scored(sc.name, bp["random__score"].to_numpy(), sc.match_score)
        met = summarize(yy[rows], chooser[rows], np.searchsorted(rows, first), match[rows],
                        Scored(sc.name, sc_eval.p[rows], sc.match_score[rows]), P.analysis.ece_bins)
        if sc.name == "random":                         # calibration metrics of the base-rate probability
            cal = summarize(yy[rows], chooser[rows], np.searchsorted(rows, first), match[rows],
                            Scored(sc.name, sc.p[rows], sc.match_score[rows]), P.analysis.ece_bins)
            met.update({k: cal[k] for k in ("brier", "ece", "log_loss", "accuracy")})
        table[sc.name] = met
    B_boot = P.analysis.bootstrap_B if not args.quick else 200
    if branches:
        _log(f"{market}: bootstrap B={B_boot}")
    boot_br = [Scored(b.name, bp["random__score"].to_numpy() if b.name == "random" else b.p, b.match_score)
               for b in branches]
    bs = cluster_bootstrap(yy[rows], chooser[rows], np.searchsorted(rows, first), match[rows],
                           [Scored(b.name, b.p[rows], b.match_score[rows]) for b in boot_br],
                           Scored(ref.name, ref.p[rows], ref.match_score[rows]),
                           B_boot, P.global_seed) if branches else {}
    comp = {k: {"delta_auc": tost(v[:, 0], P.analysis.delta_auc_min, P.analysis.alpha),
                "delta_match_auc": tost(v[:, 1], P.analysis.delta_auc_min, P.analysis.alpha)}
            for k, v in bs.items()}

    report = {
        "market": market, "data_report": m.report, "n_rows": int(obs.sum()), "n_dyads": int(len(first)),
        "base_rate": float(yy[rows].mean()), "match_rate": float(match[first].mean()),
        "bf_proxy": PROXY_VERSION, "reader": reader.provenance,
        "context_flags": sorted({f for cd in demands.values() for f in cd.flags}),
        "locked": {"path": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": digest,
                   "s": locked.s, "r": locked.r, "sigma_a2": locked.sigma_a2, "sigma_b2": locked.sigma_b2,
                   "sigma_ab": locked.sigma_ab, "rho": locked.rho, "meta": {k: v for k, v in locked.meta.items()
                                                                           if k not in ("selection", "beta_sd")},
                   "coef": dict(zip(locked.names, map(float, locked.beta))),
                   "coef_sd": dict(zip(locked.names, locked.meta.get("beta_sd", [])))},
        "crossfit_configs": {int(w): fm.meta.get("chosen") for w, fm in fold_models.items()},
        "crossfit_rhat": {int(w): fm.meta.get("rhat") for w, fm in fold_models.items()},
        "metrics": table, "vs_a0": comp, "seconds": time.time() - t0,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"stage1_{market}.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    (REPORTS / f"stage1_{market}.md").write_text(render_md(report), encoding="utf-8")
    out = pd.DataFrame({"chooser": T.ids[d.i], "chosen": T.ids[d.j], "wave": d.wave, "y": d.y, "match": d.match,
                        "a0": pred["a0"], "m0": pred["m0"], "eta": pred["eta"]})
    out.to_csv(REPORTS / f"predictions_{market}.csv", index=False)

    # database (section 10.2): cross-fitted static scores of the development pairs
    with Repo(PROJECT_ROOT / "data" / "gnwt.db", config_version=P.config_version) as db:
        db.insert_many("personas", [{"persona_id": p.persona_id, "source": p.source, "market": p.market,
                                     "language": p.language,
                                     **{t: (p.bigfive or {}).get(t) for t in ("O", "C", "E", "A", "N")},
                                     "bf_source": p.bf_source, "json": p.model_dump_json()} for p in personas],
                       replace=True)
        db.insert_many("context_texts", [{"persona_id": p.persona_id, "text": p.context_text, "lang": p.language,
                                          "source": p.context_source} for p in personas], replace=True)
        db.insert_many("context_demands", [cd.as_row(pid) | {"lexicon_ver": reader.lexicon.version,
                                                             "catalog_ver": reader.catalog.version}
                                           for pid, cd in demands.items()], replace=True)
        pairs, scores = [], []
        for k in range(len(d.i)):
            a, b = str(T.ids[d.i[k]]), str(T.ids[d.j[k]])
            lo, hi = sorted((a, b))
            pid = f"{market}_w{int(d.wave[k])}_{lo}_{hi}"
            if a == lo:
                pairs.append({"pair_id": pid, "market": market, "wave": int(d.wave[k]), "i": lo, "j": hi,
                              "split": "dev"})
            fm = fold_models[int(d.wave[k])]
            scores.append({"pair_id": pid, "chooser": a, "chosen": b, "ctx_id": "speed_dating",
                           **{c: float(pred[c].iloc[k]) for c in ("eta0", "etaP_fit", "etaP_resid", "etaP_actor",
                                                                  "etaG", "eta", "a0", "m0", "fit")},
                           "fit_parts": None, "mask": None, "srrm_hash": fm.sha256()})
        db.insert_many("pairs", pairs, replace=True)
        db.insert_many("static_scores", scores, replace=True)
        db.insert("artifacts", {"name": f"srrm_{market}", "sha256": digest,
                                "path": str(lock_path.relative_to(PROJECT_ROOT))}, replace=True)
    _log(f"{market}: done in {time.time() - t0:.0f} s")
    return report


def render_md(r: dict) -> str:
    L = [f"# Stage 1 — {r['market']}", "",
         f"Rows {r['n_rows']}, dyads {r['n_dyads']}, base rate {r['base_rate']:.3f}, match rate {r['match_rate']:.3f}. "
         f"Cross-fit leave-one-wave-out; bootstrap by chooser; SESOI ΔAUC = 0.03 (TOST).", "",
         f"Locked SRRM: s = {r['locked']['s']:.3f}, r = {r['locked']['r']:.3f}, σa² = {r['locked']['sigma_a2']:.3f}, "
         f"σb² = {r['locked']['sigma_b2']:.3f}, σab = {r['locked']['sigma_ab']:.3f}, ρ = {r['locked']['rho']:.3f}; "
         f"R̂ = {r['locked']['meta'].get('rhat')}; chosen = {r['locked']['meta'].get('chosen')}", "",
         "| branch | AUC | AUC within | Brier | ECE | log-loss | acc | match AUC | ΔAUC vs a0 [95% CI] | TOST ±0.03 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, mt in r["metrics"].items():
        c = r["vs_a0"].get(name)
        delta = "—" if c is None else f"{c['delta_auc']['mean']:+.3f} [{c['delta_auc']['ci95_lo']:+.3f}, {c['delta_auc']['ci95_hi']:+.3f}]"
        eq = "—" if c is None else ("ngang" if c["delta_auc"]["equivalent"] else ("hơn" if c["delta_auc"]["better"] else ("kém" if c["delta_auc"]["worse"] else "chưa rõ")))
        L.append(f"| {name} | {mt['auc']:.3f} | {mt['auc_within']:.3f} | {mt['brier']:.4f} | {mt['ece']:.3f} | "
                 f"{mt['log_loss']:.4f} | {mt['accuracy']:.3f} | {mt['match_auc']:.3f} | {delta} | {eq} |")
    return "\n".join(L) + "\n"


def write_manifest(P, reports: list[dict]) -> None:  # noqa: ANN001
    man_path = resolve_path(P.stage1.locked.manifest)
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {"srrm": {}}
    cr = P.stage1.context_reader
    man["config_version"] = P.config_version
    man["lexicon"] = {"path": cr.lexicon.path, "sha256": sha256_file(resolve_path(cr.lexicon.path))}
    man["catalog"] = {"path": cr.catalog.path, "sha256": sha256_file(resolve_path(cr.catalog.path))}
    man.setdefault("reader", None)
    for r in reports:
        man["srrm"][r["market"]] = {"path": r["locked"]["path"], "sha256": r["locked"]["sha256"]}
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
    with Repo(PROJECT_ROOT / "data" / "gnwt.db", config_version=P.config_version) as db:
        db.insert("artifacts", {"name": "lock_manifest", "sha256": sha256_file(man_path),
                                "path": str(man_path.relative_to(PROJECT_ROOT))}, replace=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=str(DEFAULT_PARAMS))
    ap.add_argument("--markets", nargs="+", default=["columbia", "speed_dating"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="short chains, B = 200 (smoke run)")
    ap.add_argument("--section9", action="store_true",
                    help="also cross-fit the section 9 branches (floors, B1, learners, S8); off by default")
    ap.add_argument("--no-slow", action="store_true", help="with --section9: skip CatBoost and EBM")
    args = ap.parse_args(argv)
    P = load_params(args.params)
    reports = [run_market(mk, P, args) for mk in args.markets]
    write_manifest(P, reports)
    for r in reports:
        print(render_md(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
