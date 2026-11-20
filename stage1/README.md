# Stage 1: profile + context sentence -> directed a0 and mutual m0

Stage 1 answers one question: from the profiles and the context sentence each person typed, how
likely is i to say "yes" to j? It runs entirely in code. **No language model is called at any step**,
whether at run time or during offline estimation.

* **Core**: Hoff's probit social relations regression (SRRM, eq. 10) for directed binary decisions.
  It yields a0(i->j) = Phi(eta_ij / s) and m0_ij = Phi2(eta_ij / s, eta_ji / s; r), plus the exact
  split eta = eta0 + etaP_fit + etaP_resid + etaP_actor + etaG.
* **Context reader** (English only): normalise text, split clauses, run a keyword lexicon with
  intensifier, softener, negation and prohibition rules, and match a basic-context catalog. The
  catalog prior and the keyword evidence are merged by Buhlmann credibility (11b). The 12 O*NET Work
  Styles are then mapped onto the Big Five domains (11c).
* **Trait-context fit** (11a): fit(j, c_i) = sum_d D~_d(c_i) P~_j,d. Its slope gamma_P is learned
  from data, with delta orthogonal to D~ under single-context identification.
* **Missing data**: the missing-indicator method per block, with random per-person block masking
  during fitting.
* **Locked procedure**: the prior width and masking rate are chosen by nested leave-one-wave-out
  log loss with a 1-SE rule. The chosen configuration is fitted with a Gibbs sampler and the result is
  cross-fitted. Artifacts are stored hashed.

## Layout

```
stage1/
  config/stage1.yaml                 procedures, grids and priors only (no coefficient), spec 10.6
  data/lexicon/demand_lexicon_v1.yaml   words + poles + modifier phrases (no weights), English
  data/catalog/basic_contexts_v1.yaml   basic contexts, aliases, goal codes; profiles filled by raters (O2)
  data/raw/speed_dating/             downloaded data (git-ignored)
  src/stage1/
    context/  styles.py text.py lexicon.py catalog.py demand.py fallback_embed.py (optional, off)
    fit.py features.py srrm.py       (11a), blocks + MIM, a0 / m0 / eta_blocks / LockedSRRM
    static.py schema.sql             run-time scoring -> SQLite tables of spec 10.2
    data/speed_dating.py             OpenML 40536 and the Columbia CSV
    offline/  probit.py gibbs.py estimate.py reader.py    (never imported at run time)
    metrics.py runner.py             evaluation, cluster bootstrap, experiment plumbing
  experiments/
    run_stage1.py                    cross-fit + evaluation (+ --lock final artifacts, --db static_scores)
    compare_baselines.py             Stage 1 vs strong profile-matching baselines from the literature
  tests/                             U1-U12 and the estimator tests
  artifacts/stage1/                  locked, hashed SRRM and reader artifacts (written by --lock)
```

## Install

Run every command from `stage1/`. Python 3.10 or later.

```bash
pip install -e ".[dev,baselines]"
```

## Data

Both files download on first use into `data/raw/speed_dating/`:

* **openml**: [OpenML 40536 "SpeedDating"](https://www.openml.org/d/40536). It has no person IDs.
  The loader rebuilds the 551 participants from their constant profile fields within a wave, then
  links each row to its partner through the partner fields (`age_o`, `race_o`, `pref_o_*`). This
  links 99.4% of rows, and `decision_o` agrees 100% with the linked partner's decision. The remaining
  50 rows keep their partner blocks missing.
* **columbia**: the original "Speed Dating Data.csv" (Fisman & Iyengar, Columbia Business School,
  21 waves, 2002-2004) with `iid`, `pid`, `wave`, `goal` and `field_cd`.

Only pre-date fields are used. The label is the directed decision (`dec`), and `match` means both
said yes.

## Run

```bash
python experiments/run_stage1.py --dataset both
```

```bash
python experiments/run_stage1.py --dataset columbia --lock --db result/stage1.db
```

```bash
python experiments/compare_baselines.py --dataset both
```

`--quick` / `--fast` use short chains and smaller grids for a smoke run. Results go to
`result/<time>_*/` (`report.md`, `comparison.md`, per-row predictions, metrics CSV). The Stage 1
cross-fit is cached in `result/cache/`, keyed by the data, settings, reader and code hashes.

## Baseline comparison

`experiments/compare_baselines.py` compares exactly one technique, the full Stage 1, with no
ablation. Its docstring lists each baseline and where it comes from:

* logistic regression and profile cosine similarity (the only published Speed Dating
  profile-matching numbers we found);
* RECON with harmonic aggregation (RecSys);
* random forest (Joel, Eastwick & Finkel 2017, *Psychological Science*);
* XGBoost, CatBoost and EBM (KDD / NeurIPS);
* FM and DeepFM (ICDM / IJCAI);
* optionally TabPFN v2 (*Nature* 2025).

Every method uses the same protocol: leave one wave out, with tuning only on the training waves. The
script reports pooled and within-chooser AUC, match AUC, log loss, Brier and ECE. It also reports a
paired cluster bootstrap over choosers and a TOST test on the AUC difference with SESOI 0.03.

## What is and is not estimated yet

* **Big Five**: Speed Dating has no Big Five. Stage 1 does not build an LLM proxy, so `bf_source` is
  `none`, both Big Five blocks are missing (MIM), and fit / gamma_P cannot be estimated on Speed
  Dating. `Persona.bf` accepts measured (BFI-10) or proxy values when a dataset has them.
* **Catalog profiles (O2) and reader labels (O4-O5)** need human raters. Until then, catalog profiles
  are `null` and read as the neutral demand 0 (flag `rated_prior = false`). The reader uses the
  *illustrative* numbers of spec 3.5 (status `illustrative`). On Speed Dating nobody writes a context
  sentence, so neither changes a0 there. The estimators (`offline/reader.py`: `estimate_catalog`,
  `estimate_reader`, `evaluate_reader`) are implemented and tested on synthetic labels. The label
  format is the `demand_labels` table in `src/stage1/schema.sql`.

Deviations from the spec and conventions chosen here are listed in [CHANGELOG.md](CHANGELOG.md).

## Tests

```bash
python -m pytest
```
