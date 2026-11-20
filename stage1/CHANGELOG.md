# CHANGELOG (Stage 1)

Every place where this implementation departs from the Stage 1 tech spec ("Stage 1 reads the context
request by keywords"), or fixes something the spec leaves open, with the reason. Tags follow the spec:
**[proposal]** design choice, **[convention]** pre-registered procedural choice.

## 1.0.0 (19/11/2026)

### Scope decided with the user
- **English only.** The lexicon, catalog aliases, rule cases (U6) and all code are English, the common
  language of the datasets. The spec's Vietnamese tone-stripping step is kept as accent folding
  (`toneless` flag, e.g. "café" -> "cafe").
- **Datasets.** Speed Dating as OpenML 40536 and as the original Columbia CSV (Fisman & Iyengar).
  OpenML has no person IDs, so persons are rebuilt from constant profile fields per wave (551, the
  same count as Columbia). Partners are linked through `age_o`, `race_o` and `pref_o_*`: 8,328 of
  8,378 rows are linked and `decision_o` is 100% consistent. Unlinked partners keep missing blocks.
  OpenML has no `goal`, so the goal block is never collected there and is dropped (Mercaldo & Blume).
- **No ablation in the comparison script**, per the user. S8 (delta = 0) and the reader branches X0 /
  X1 / X3 are not implemented.

### Not estimable yet (needs human raters or measured traits)
- **Big Five on Speed Dating**: there is none. The LLM proxy of the spec (`data/bfi10.py`) is outside
  Stage 1 and is not built, so `bf_source = none` and both Big Five blocks are masked. Their columns
  have zero variance and are dropped, so fit, gamma_P, delta and beta_A are 0 on Speed Dating.
- **Catalog profiles (O2)** are `null` until >= 3 raters score them. An unrated profile is read as
  the neutral demand 0, and the output carries `rated_prior = false`. `estimate_catalog` computes
  mean, SE and ICC(2,k) once ratings exist.
- **Reader numbers (O4-O5)**: no labelled sentences yet. `ReaderArtifacts.illustrative()` uses the
  spec 3.5 example (theta_neg -0.5, base 2.0, strong 2.7, k 0.5). The spec gives no weak or rev
  value, so 1.0 and -1.0 are placeholders, flagged `illustrative`. Speed Dating has no context
  sentence, so this never changes a0 there.

### Estimation
- **Prior on the marginal scale** [convention]: beta_k / s ~ N(0, c^2), i.e. beta_k ~ N(0, c^2 s^2).
  Selection and sampling therefore shrink the quantity a0 depends on. The intercept sd is 5.
- **Nested LOWO selection with the marginal-probit MAP.** The spec selects with the Bayesian model
  itself. Running Gibbs for 21 outer x 20 inner x 20 grid points is not feasible, so each grid point
  is scored with the Newton MAP of the marginal probit (exactly beta / s under (10)). Only the
  selected (c, q) is run through Gibbs.
- **1-SE rule on paired differences**: a configuration is within 1 SE of the best when its mean
  fold-wise excess log loss is at most the SE of that excess. Unpaired SEs are dominated by wave-to-wave
  differences in yes rate.
- **Gibbs sampler written in numpy** (Hoff's scheme, amen 1.4.5), not amen or NumPyro, to avoid an
  R / JAX dependency.
  - (beta, a, b) are drawn jointly and exactly by perturbation + solve, through the Schur complement
    of the person block. People meet only within a wave, so that block is block-diagonal and sparse.
  - rho and Sigma_ab use random-walk Metropolis, adapted during burn-in.
  - Prior: LKJ(eta = 2) on the correlation and half-Cauchy(0, 1) on sigma_a and sigma_b
    [convention: scale 1 on the latent probit scale]. Inverse-Wishart(4, I) is available as the
    sensitivity option.
- **Chains**: the cross-fit runs 1 chain x 800 iterations (burn-in 300) per held-out wave. The final
  locked fit runs 4 chains x 1,500 iterations, and split-R-hat is reported in the artifact.
- **Hierarchical per-context coefficients** (beta^c = beta_bar + zeta^c) are not implemented. Both
  datasets have a single context.
- **Reader theta**: least squares of (11b) with theta_neg <= theta_weak <= theta_base <= theta_strong,
  solved exactly by bounded least squares on non-negative increments. This equals the PAV solution
  when every sentence carries one cue class. k uses Buhlmann's estimator with rater noise subtracted
  from tau^2.

### Features and rules
- **Every non-indicator column is z-scored** within the fold, binaries included, so one prior width
  means one width per SD. Missing indicators stay 0/1.
- **Blocks**: age, field, self_actor, self_partner, goal, interests, bf_actor, bf_partner. U10 tests
  all 2^8 masks, not 2^7. Self-ratings (sinc, intel, fun, amb; no appearance field) count as missing
  when >= 2 of 4 are missing [convention]; a partly missing item is set to the mean.
- **Chooser gender** (`actor_male`) is part of eta0. The spec lists gender among the core inputs.
- **Modifier rules in English**: a plain negator class (not, no) implements "negation + reverse
  keyword -> base" ("no competition"). Post modifiers ("is not important", "is a must") act on every
  keyword before them in the clause. Clauses split at punctuation and at but / however / although /
  though / whereas, never at "and", and not at "but also".
- **Match cluster for the bootstrap**: each dyad is clustered with its female participant (exactly
  one per dyad). ECE uses 10 equal-mass bins.
- **Artifacts** are JSON (`artifacts/stage1/<dataset>/srrm_<sha>.json`, `reader_<sha>.json`), not
  `.nc`.
