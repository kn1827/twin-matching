"""Stage 1 offline run on Speed Dating (OpenML and/or Columbia CSV).

1. Cross-fit a0 / m0: every wave is scored by a model whose prior width and masking rate were chosen by
   nested leave-one-wave-out CV and that was fitted (Gibbs SRRM) without that wave (tests I1).
2. Evaluate on pre-date fields only: pooled AUC, within-chooser AUC, Brier, ECE, log loss, match AUC of
   m0, with a cluster bootstrap over choosers (B from config).
3. --lock: final fit on all waves (4 chains, R-hat), hashed artifact in artifacts/stage1/, and the
   static_scores table written to a SQLite database.

    python experiments/run_stage1.py --dataset columbia
    python experiments/run_stage1.py --dataset both --quick
    python experiments/run_stage1.py --dataset columbia --lock --db result/stage1.db
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from stage1.metrics import bootstrap  # noqa: E402
from stage1.offline.estimate import estimate_srrm  # noqa: E402
from stage1.runner import (ARTIFACTS, CONFIG, RESULT, crossfit_cached, eval_set, fmt_ci, load_reader,  # noqa: E402
                           md_table, settings)
from stage1.static import Stage1DB, run_stage1  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="columbia", choices=["columbia", "openml", "both"])
    ap.add_argument("--quick", action="store_true", help="short chains, 5 inner folds, B = 200")
    ap.add_argument("--reader", type=Path, default=None, help="estimated reader artifact (json)")
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("--lock", action="store_true", help="final fit on all waves + hashed artifact")
    ap.add_argument("--db", type=Path, default=None, help="write static_scores to this SQLite file")
    ap.add_argument("--refresh", action="store_true", help="ignore the cross-fit cache")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["stage1"]
    B = args.bootstrap or (200 if args.quick else int(cfg["evaluation"]["bootstrap_B"]))
    S = settings(args.quick)
    art = load_reader(args.reader)
    if art.status != "estimated":
        print(f"note: reader artifacts are '{art.status}'. Speed Dating has no context sentence, so they do not "
              "change a0 here.", flush=True)
    out = RESULT / f"{datetime.now():%Y%m%d-%H%M%S}_stage1{'_quick' if args.quick else ''}"
    out.mkdir(parents=True, exist_ok=True)
    datasets = ["openml", "columbia"] if args.dataset == "both" else [args.dataset]
    lines = [f"# Stage 1 on Speed Dating ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"Reader status: {art.status}. Bootstrap B = {B} (clusters = choosers). "
             f"Sampler (cross-fit): {S.crossfit_gibbs.iterations} iterations, burn-in {S.crossfit_gibbs.burn_in}, "
             f"{S.crossfit_gibbs.chains} chain(s); inner CV: {S.inner_cv}.", ""]
    for ds in datasets:
        t0 = time.time()
        pred, waves, prob = crossfit_cached(ds, S, art, refresh=args.refresh)
        es = eval_set(pred, {"stage1_a0": pred["a0"].to_numpy()}, {"stage1_a0": "m0"}, prob,
                      int(cfg["evaluation"]["ece_bins"]))
        res = bootstrap(es, "stage1_a0", B=B, seed=S.seed).iloc[0]
        pred.to_csv(out / f"{ds}_a0_crossfit.csv", index=False)
        waves.to_csv(out / f"{ds}_waves.csv", index=False)
        lines += [f"## {ds}", "", f"Rows {len(pred)}, persons {len(prob.data.personas)}, waves {len(waves)}; "
                  f"data info: `{json.dumps(prob.data.info)}`", "",
                  "| metric | value [95% CI] |", "|---|---|"]
        for k, name in (("auc", "AUC (directed decision)"), ("within_auc", "within-chooser AUC"),
                        ("match_auc", "match AUC (m0)"), ("log_loss", "log loss"), ("brier", "Brier"), ("ece", "ECE")):
            lines.append(f"| {name} | {fmt_ci(res[k], res[k + '_lo'], res[k + '_hi'])} |")
        lines += ["", "Per held-out wave (selected prior width c, masking rate q, variance components):", "",
                  md_table(waves), ""]
        print(f"{ds}: AUC {res['auc']:.3f}, within {res['within_auc']:.3f}, match AUC {res['match_auc']:.3f}, "
              f"log loss {res['log_loss']:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if args.lock:
            print(f"locking: final fit on all {len(np.unique(prob.wave))} waves", flush=True)
            model = estimate_srrm(prob, verbose=True)
            path = model.save(ARTIFACTS / ds)
            model.meta["path"] = str(path)
            rpath = ARTIFACTS / ds / f"reader_{art.hash[:16]}.json"
            art.save(rpath)
            lines += [f"Locked SRRM: `{path.relative_to(ROOT)}` (sha {model.hash[:16]}), s = {model.s:.3f}, "
                      f"r = {model.r:+.3f}, R-hat = {model.meta.get('rhat')}", ""]
            if args.db:
                db = Stage1DB(args.db if args.db.is_absolute() else ROOT / args.db)
                info = run_stage1(db, prob.data, art, model)
                db.close()
                lines += [f"static_scores written: {info}", ""]
                print(f"static_scores -> {args.db}: {info}", flush=True)
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {out / 'report.md'}")


if __name__ == "__main__":
    main()
