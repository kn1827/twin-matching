"""Stage 1 (a0 / m0) against strong profile-matching baselines from recent literature, on Speed Dating
(OpenML 40536 "SpeedDating") and the original Columbia CSV (Fisman & Iyengar, 21 waves, 2002-2004).

ONE technique is compared: the full Stage 1 of the tech spec (SRRM probit core, keyword context
reader, missing-indicator blocks, prior width and masking rate chosen by nested leave-one-wave-out CV,
Gibbs fit, cross-fitted). There is no ablation branch here.

Protocol (identical for every method)
-------------------------------------
* Pre-date fields only (no rating made during or after the date, no appearance field for Stage 1).
* Outer split: leave one wave out (21 folds); a wave contains both i->j and j->i, so whole waves are held
  out. Every baseline is tuned only on the training waves (inner grouped CV by wave, log loss).
* Targets: the directed decision (pooled AUC, within-chooser AUC, log loss, Brier, ECE) and the match
  (AUC of the mutual score: m0 for Stage 1, p_ij * p_ji for the classifiers, the harmonic mean for RECON).
* Paired cluster bootstrap over choosers (B = 2000), Delta = baseline - Stage 1 with 95% CI, TOST on
  Delta AUC with SESOI 0.03 (tech spec 9).

Baselines and their provenance (literature search, November 2026)
------------------------------------------------------------------
Very few peer-reviewed papers report AUC on Speed Dating from PRE-DATE profiles only. The set below
combines (a) the only published Speed-Dating profile-matching numbers we found, (b) the baselines that
B-or-higher papers use for the same task family (predicting initial attraction / reciprocal
recommendation from profiles), and (c) the strong tabular learners those comparisons rely on.

| name        | method                                         | source used for the choice                                                       |
|-------------|------------------------------------------------|---------------------------------------------------------------------------------|
| lr_b1       | L2 logistic regression on the Stage 1 features  | tech spec B1; Shang, Yan & Liu 2025 (NeurIPS 2025 workshop, below B): LR match AUC 0.61 on Speed Dating (pre-date) |
| lr_rich     | L2 logistic regression on all pre-date fields   | same LR baseline, full profile of both people                                     |
| cosine      | cosine similarity of the two profile vectors    | Shang, Yan & Liu 2025 "Similarity" baseline: match AUC 0.54                       |
| recon       | RECON reciprocal content-based score            | Pizzato et al., RecSys 2010 (CORE A); harmonic-mean aggregation as in Neve & Palomares, RecSys 2019 (CORE A) |
| rf          | random forest on 100+ pre-date predictors       | Joel, Eastwick & Finkel 2017, Psychological Science (Q1)                          |
| xgboost     | gradient-boosted trees                          | Chen & Guestrin, KDD 2016 (CORE A*); trees remain the reference on medium tabular data (Grinsztajn et al., NeurIPS 2022) |
| catboost    | ordered boosting with native categoricals        | Prokhorenkova et al., NeurIPS 2018 (A*); the spec's strong learner               |
| ebm         | explainable boosting machine (GA2M)             | Lou et al., KDD 2013 (A*); Nori et al. 2019 (InterpretML)                         |
| fm          | factorization machine on profile features       | Rendle, ICDM 2010 (A*); FM-family baselines of reciprocal recommenders (biDeepFM, Eng. Sci. Technol. Int. J. 2021, Q1; biLorentzFM 2025 on Speed Dating) |
| deepfm      | DeepFM                                          | Guo et al., IJCAI 2017 (A*); same reciprocal-recommendation baselines             |
| tabpfn      | TabPFN v2 (optional, --tabpfn)                  | Hollmann et al., Nature 2025                                                      |

Adaptations (cold start): test waves contain only new people, so ID embeddings and interaction histories
are unusable. RECON's preference model is built from each chooser's STATED preference weights
(attr1_1 ... shar1_1) applied to the partner's self-ratings (attr3_1 ... amb3_1) and interest correlation,
instead of from messaging history. Unsupervised scores (cosine, RECON) are Platt-calibrated on the
training waves (this changes log loss / Brier / ECE, not AUC).

    python experiments/compare_baselines.py --dataset both
    python experiments/compare_baselines.py --dataset columbia --fast --bootstrap 200
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss as sk_log_loss  # noqa: E402

from stage1.data.speed_dating import INTERESTS  # noqa: E402
from stage1.features import COLUMNS  # noqa: E402
from stage1.metrics import bootstrap  # noqa: E402
from stage1.offline.estimate import Problem  # noqa: E402
from stage1.runner import RESULT, crossfit_cached, eval_set, fmt_ci, load_reader, md_table, settings  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SOURCES = {
    "stage1": ("Stage 1 a0 / m0 (this work)", "-", "-"),
    "lr_b1": ("Logistic regression, Stage 1 features (B1)", "Shang, Yan & Liu 2025, NeurIPS'25 workshop", "match AUC 0.61"),
    "lr_rich": ("Logistic regression, full pre-date profile", "Shang, Yan & Liu 2025, NeurIPS'25 workshop", "match AUC 0.61"),
    "cosine": ("Profile cosine similarity", "Shang, Yan & Liu 2025, NeurIPS'25 workshop", "match AUC 0.54"),
    "recon": ("RECON reciprocal content-based (stated prefs)", "Pizzato et al. 2010, RecSys (A); Neve & Palomares 2019, RecSys (A)", "-"),
    "rf": ("Random forest", "Joel, Eastwick & Finkel 2017, Psychol. Sci. (Q1)", "4-18% actor, 7-27% partner variance"),
    "xgboost": ("XGBoost", "Chen & Guestrin 2016, KDD (A*)", "-"),
    "catboost": ("CatBoost", "Prokhorenkova et al. 2018, NeurIPS (A*)", "-"),
    "ebm": ("Explainable boosting machine (GA2M)", "Lou et al. 2013, KDD (A*)", "-"),
    "fm": ("Factorization machine", "Rendle 2010, ICDM (A*)", "-"),
    "deepfm": ("DeepFM", "Guo et al. 2017, IJCAI (A*)", "-"),
    "tabpfn": ("TabPFN v2", "Hollmann et al. 2025, Nature", "-"),
}
MUTUAL = {"stage1": "m0", "recon": "harmonic"}


# ===================================================================================== features
SELF5 = ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1"]
PREF6 = ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]


class RichFeatures:
    """Full pre-date profile of chooser (i_*) and partner (j_*) plus dyadic fields, per directed row."""

    def __init__(self, prob: Problem, rows: np.ndarray):
        data = prob.data
        R = data.rich.copy()
        cats = [c for c in data.rich_categorical if c in R]
        for c in cats:                              # explicit "nan" level (pandas >= 3 keeps NaN in astype(str))
            R[c] = [("nan" if (not isinstance(v, str) and pd.isna(v)) or v in ("<NA>", "None") else str(v))
                    for v in R[c]]
        nums = [c for c in R.columns if c not in cats]
        a, b = prob.a[rows], prob.p[rows]
        Ri, Rj = R.iloc[a].reset_index(drop=True), R.iloc[b].reset_index(drop=True)
        num = pd.concat([Ri[nums].add_prefix("i_"), Rj[nums].add_prefix("j_")], axis=1).astype(float)
        num["abs_age_gap"] = (Ri["age"] - Rj["age"]).abs()
        num["int_corr"] = prob.raw["int_corr"][rows]
        for c in cats:
            same = (Ri[c] == Rj[c]).astype(float)
            same[(Ri[c] == "nan") | (Rj[c] == "nan")] = np.nan
            num[f"same_{c}"] = same.values
        self.num = num
        self.cat = pd.concat([Ri[cats].add_prefix("i_"), Rj[cats].add_prefix("j_")], axis=1).astype(str)
        # RECON: chooser's stated weights x partner's self-ratings (+ shared interests)
        w = Ri[PREF6].to_numpy(float) / 100.0
        s = np.column_stack([Rj[SELF5].to_numpy(float) / 10.0, (np.nan_to_num(num["int_corr"].to_numpy()) + 1) / 2])
        ok = ~np.isnan(w) & ~np.isnan(s)
        self.recon = np.where(ok.any(1), np.nansum(np.where(ok, w * s, 0), 1) / np.maximum(np.where(ok, w, 0).sum(1), 1e-9),
                              np.nan)
        # cosine: z-scored interests + self-ratings + stated preferences of both people
        prof = INTERESTS + tuple(SELF5) + tuple(PREF6)
        Z = R[list(prof)].astype(float)
        Z = ((Z - Z.mean()) / Z.std()).fillna(0.0).to_numpy()
        zi, zj = Z[a], Z[b]
        den = np.linalg.norm(zi, axis=1) * np.linalg.norm(zj, axis=1)
        self.cosine = np.where(den > 0, (zi * zj).sum(1) / np.where(den > 0, den, 1), 0.0)


class Encoder:
    """Train-fold median imputation + missing indicators, standardisation and one-hot categoricals."""

    def __init__(self, num: pd.DataFrame, cat: pd.DataFrame, scale: bool = True):
        self.med = num.median()
        self.miss = [c for c in num.columns if num[c].isna().any()]
        X = self._num(num)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        self.scale = scale
        self.levels = {c: sorted(cat[c].unique()) for c in cat.columns}

    def _num(self, num):
        X = num.fillna(self.med).fillna(0.0).to_numpy(float)
        M = num[self.miss].isna().to_numpy(float)
        return np.hstack([X, M])

    def __call__(self, num, cat) -> np.ndarray:
        X = self._num(num)
        if self.scale:
            X = (X - self.mu) / self.sd
        oh = [np.column_stack([(cat[c].to_numpy() == lv).astype(float) for lv in lvls]) for c, lvls in self.levels.items()]
        return np.hstack([X] + oh) if oh else X


def onehot_nan(num: pd.DataFrame, cat: pd.DataFrame, levels: dict) -> np.ndarray:
    oh = [np.column_stack([(cat[c].to_numpy() == lv).astype(float) for lv in lvls]) for c, lvls in levels.items()]
    return np.hstack([num.to_numpy(float)] + oh)


# ===================================================================================== models
def inner_folds(waves: np.ndarray, k: int) -> list[np.ndarray]:
    uw = np.unique(waves)
    return [np.isin(waves, g) for g in np.array_split(uw, min(k, len(uw)))]


def tune(fit_predict, grid: list[dict], waves: np.ndarray, y: np.ndarray, k: int) -> dict:
    """Pick the grid point with the lowest inner grouped-CV log loss."""
    if len(grid) == 1:
        return grid[0]
    best, best_loss = grid[0], np.inf
    for g in grid:
        losses = []
        for te in inner_folds(waves, k):
            p = fit_predict(~te, te, g)
            losses.append(sk_log_loss(y[te], np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
        if np.mean(losses) < best_loss:
            best, best_loss = g, float(np.mean(losses))
    return best


class Torch_FM:
    """Factorization machine (Rendle 2010) and DeepFM (Guo et al. 2017) on dense profile features."""

    def __init__(self, deep: bool, k: int = 8, epochs: int = 100, patience: int = 8, seed: int = 0):
        self.deep, self.k, self.epochs, self.patience, self.seed = deep, k, epochs, patience, seed

    def _net(self, d):
        import torch
        from torch import nn
        k, deep = self.k, self.deep

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.w0 = nn.Parameter(torch.zeros(1))
                self.w = nn.Linear(d, 1, bias=False)
                self.V = nn.Parameter(0.01 * torch.randn(d, k))
                if deep:
                    self.mlp = nn.Sequential(nn.Linear(d * k, 128), nn.ReLU(), nn.Dropout(0.2),
                                             nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

            def forward(self, x):
                xv = x @ self.V
                inter = 0.5 * (xv ** 2 - (x ** 2) @ (self.V ** 2)).sum(1)
                out = self.w0 + self.w(x).squeeze(1) + inter
                if deep:
                    out = out + self.mlp((x.unsqueeze(2) * self.V.unsqueeze(0)).flatten(1)).squeeze(1)
                return out
        return Net()

    def fit_predict(self, Xtr, ytr, val_mask, Xte):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        net = self._net(Xtr.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = torch.nn.BCEWithLogitsLoss()
        Xa = torch.tensor(Xtr[~val_mask], dtype=torch.float32)
        ya = torch.tensor(ytr[~val_mask], dtype=torch.float32)
        Xv = torch.tensor(Xtr[val_mask], dtype=torch.float32)
        yv = torch.tensor(ytr[val_mask], dtype=torch.float32)
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(len(Xa), generator=g)
            for s in range(0, len(Xa), 256):
                idx = perm[s:s + 256]
                opt.zero_grad()
                lossf(net(Xa[idx]), ya[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(lossf(net(Xv), yv))
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(Xte, dtype=torch.float32))).numpy()


def platt(score_tr, y_tr, score_te):
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(np.nan_to_num(score_tr)[:, None], y_tr)
    return lr.predict_proba(np.nan_to_num(score_te)[:, None])[:, 1]


def run_fold(name: str, prob: Problem, F: RichFeatures, tr: np.ndarray, te: np.ndarray, args) -> np.ndarray:
    """Train `name` on rows tr (positions in the evaluation order), predict rows te."""
    y = prob.y[F.rows]
    waves = prob.wave[F.rows]
    k_inner = 3                                     # inner grouped CV: 3 folds of training waves
    ytr = y[tr]
    if name == "cosine":
        return platt(F.cosine[tr], ytr, F.cosine[te])
    if name == "recon":
        return platt(F.recon[tr], ytr, F.recon[te])
    if name == "lr_b1":
        from stage1.offline.estimate import _active
        rows_tr, rows_te = F.rows[tr], F.rows[te]
        std = prob.standardizer(rows_tr)
        Xtr, Xte = prob.raw_design(rows_tr, std), prob.raw_design(rows_te, std)
        act = _active(Xtr, list(COLUMNS))
        act[COLUMNS.index("intercept")] = False
        Xtr, Xte = Xtr[:, act], Xte[:, act]

        def fp(a, b, g):
            m = LogisticRegression(C=g["C"], max_iter=2000).fit(Xtr[a], ytr[a])
            return m.predict_proba(Xtr[b])[:, 1]
        g = tune(fp, [{"C": c} for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)], waves[tr], ytr, k_inner)
        return LogisticRegression(C=g["C"], max_iter=2000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    num_tr, num_te = F.num.iloc[tr], F.num.iloc[te]
    cat_tr, cat_te = F.cat.iloc[tr], F.cat.iloc[te]
    if name in ("lr_rich", "rf", "fm", "deepfm", "tabpfn"):
        enc = Encoder(num_tr, cat_tr, scale=name != "rf")
        Xtr, Xte = enc(num_tr, cat_tr), enc(num_te, cat_te)
    if name == "lr_rich":
        def fp(a, b, g):
            return LogisticRegression(C=g["C"], max_iter=3000).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
        g = tune(fp, [{"C": c} for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)], waves[tr], ytr, k_inner)
        return LogisticRegression(C=g["C"], max_iter=3000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    if name == "rf":
        n_est = 200 if args.fast else 300

        def fp(a, b, g):
            m = RandomForestClassifier(n_estimators=n_est, min_samples_leaf=g["leaf"], max_features="sqrt",
                                       n_jobs=-1, random_state=0).fit(Xtr[a], ytr[a])
            return m.predict_proba(Xtr[b])[:, 1]
        g = tune(fp, [{"leaf": v} for v in (5, 20, 50)], waves[tr], ytr, k_inner)
        return fp(np.ones(len(ytr), bool), slice(None), g) if False else \
            RandomForestClassifier(n_estimators=n_est, min_samples_leaf=g["leaf"], max_features="sqrt", n_jobs=-1,
                                   random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    if name == "xgboost":
        import xgboost as xgb
        levels = {c: sorted(F.cat[c].iloc[tr].unique()) for c in F.cat.columns}
        Xtr, Xte = onehot_nan(num_tr, cat_tr, levels), onehot_nan(num_te, cat_te, levels)

        def mk(g):
            return xgb.XGBClassifier(n_estimators=g["n"], max_depth=g["d"], learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, n_jobs=-1,
                                     random_state=0, eval_metric="logloss", tree_method="hist")

        def fp(a, b, g):
            return mk(g).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
        grid = [{"d": 2, "n": 300}, {"d": 3, "n": 300}, {"d": 4, "n": 200}]
        g = tune(fp, grid[:1] if args.fast else grid, waves[tr], ytr, k_inner)
        return mk(g).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    if name == "catboost":
        from catboost import CatBoostClassifier
        Xtr_c = pd.concat([num_tr.reset_index(drop=True), cat_tr.reset_index(drop=True)], axis=1)
        Xte_c = pd.concat([num_te.reset_index(drop=True), cat_te.reset_index(drop=True)], axis=1)
        m = CatBoostClassifier(iterations=300 if args.fast else 500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                               random_seed=0, verbose=0, cat_features=list(F.cat.columns), thread_count=-1)
        return m.fit(Xtr_c, ytr).predict_proba(Xte_c)[:, 1]
    if name == "ebm":
        # native categoricals and missing values; max_rounds bounds the cyclic boosting (about 50 s per fold
        # on 8 cores instead of about 4 min, with no loss of held-out AUC in a timing check)
        from interpret.glassbox import ExplainableBoostingClassifier
        Xtr_e = pd.concat([num_tr.reset_index(drop=True), cat_tr.reset_index(drop=True)], axis=1)
        Xte_e = pd.concat([num_te.reset_index(drop=True), cat_te.reset_index(drop=True)], axis=1)
        m = ExplainableBoostingClassifier(interactions=5 if args.fast else 10, max_rounds=1000, outer_bags=8,
                                          random_state=0, n_jobs=-1)
        return m.fit(Xtr_e, ytr).predict_proba(Xte_e)[:, 1]
    if name in ("fm", "deepfm"):
        val = np.isin(waves[tr], np.unique(waves[tr])[-3:])          # last 3 training waves: early stopping
        return Torch_FM(deep=name == "deepfm", epochs=40 if args.fast else 100).fit_predict(Xtr, ytr, val, Xte)
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device="cpu", random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    raise ValueError(name)


def run_baselines(prob: Problem, order_rows: np.ndarray, names: list[str], args) -> dict[str, np.ndarray]:
    F = RichFeatures(prob, order_rows)
    F.rows = order_rows
    waves = prob.wave[order_rows]
    out = {n: np.full(len(order_rows), np.nan) for n in names}
    times: dict[str, float] = {}
    for k, w in enumerate(np.unique(waves)):
        tr, te = np.flatnonzero(waves != w), np.flatnonzero(waves == w)
        for n in names:
            t0 = time.time()
            out[n][te] = run_fold(n, prob, F, tr, te, args)
            times[n] = time.time() - t0
        print(f"  baselines: wave {w} done ({k + 1}/{len(np.unique(waves))}) "
              + " ".join(f"{n}={t:.0f}s" for n, t in times.items()), flush=True)
    return out


# ===================================================================================== report
def report_table(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in res.iterrows():
        name, src, reported = SOURCES[r["method"]]
        rows.append({
            "method": name, "source (venue, rank)": src, "reported in source": reported,
            "AUC [95% CI]": fmt_ci(r["auc"], r["auc_lo"], r["auc_hi"]),
            "dAUC vs Stage 1 [95% CI]": "-" if r["method"] == "stage1" else fmt_ci(r["d_auc"], r["d_auc_lo"], r["d_auc_hi"]),
            "TOST +-0.03": "-" if r["method"] == "stage1" else ("equivalent" if r["tost_equivalent"] else "not shown"),
            "within-chooser AUC": fmt_ci(r["within_auc"], r["within_auc_lo"], r["within_auc_hi"]),
            "match AUC": fmt_ci(r["match_auc"], r["match_auc_lo"], r["match_auc_hi"]),
            "log loss": f"{r['log_loss']:.4f}", "Brier": f"{r['brier']:.4f}", "ECE": f"{r['ece']:.4f}",
        })
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="both", choices=["columbia", "openml", "both"])
    ap.add_argument("--baselines", default="lr_b1,lr_rich,cosine,recon,rf,xgboost,catboost,ebm,fm,deepfm")
    ap.add_argument("--tabpfn", action="store_true", help="add TabPFN v2 (needs `pip install tabpfn`)")
    ap.add_argument("--fast", action="store_true", help="smaller grids / ensembles and quick Stage 1 chains")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--reader", type=Path, default=None)
    ap.add_argument("--refresh", action="store_true", help="recompute the Stage 1 cross-fit")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    names = [n.strip() for n in args.baselines.split(",") if n.strip()]
    if args.tabpfn:
        names.append("tabpfn")
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown baselines {unknown}; choose from {sorted(SOURCES)}")

    S = settings(quick=args.fast)
    art = load_reader(args.reader)
    out = RESULT / f"{datetime.now():%Y%m%d-%H%M%S}_compare{'_fast' if args.fast else ''}"
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# Stage 1 vs profile-matching baselines ({datetime.now():%Y-%m-%d %H:%M})", "",
          "Leave-one-wave-out on pre-date fields; paired cluster bootstrap over choosers "
          f"(B = {args.bootstrap}); dAUC = baseline - Stage 1; TOST on dAUC with SESOI 0.03. "
          f"Stage 1 reader status: {art.status} (no context sentence in Speed Dating).", ""]
    for ds in (["openml", "columbia"] if args.dataset == "both" else [args.dataset]):
        t0 = time.time()
        pred, _, prob = crossfit_cached(ds, S, art, refresh=args.refresh)
        rows = pred["row"].to_numpy()
        print(f"{ds}: running {len(names)} baselines over {len(np.unique(prob.wave))} held-out waves", flush=True)
        probs = {"stage1": pred["a0"].to_numpy()}
        probs.update(run_baselines(prob, rows, names, args))
        es = eval_set(pred, probs, {n: MUTUAL.get(n, "product") for n in probs}, prob)
        res = bootstrap(es, "stage1", B=args.bootstrap, seed=S.seed)
        res = res.sort_values("auc", ascending=False)
        pd.DataFrame({"row": rows, "wave": pred["wave"], "chooser": pred["chooser"], "chosen": pred["chosen"],
                      "y": pred["y"], "match": pred["match"], **probs, "stage1_m0": pred["m0"]}) \
            .to_csv(out / f"{ds}_predictions.csv", index=False)
        res.to_csv(out / f"{ds}_metrics.csv", index=False)
        tab = report_table(res)
        md += [f"## {ds} ({prob.data.info.get('pairs')} directed pairs, {len(np.unique(prob.wave))} waves)", "",
               md_table(tab), ""]
        print(md_table(tab[["method", "AUC [95% CI]", "dAUC vs Stage 1 [95% CI]", "TOST +-0.03", "match AUC",
                            "log loss"]]), flush=True)
        print(f"{ds} done in {time.time() - t0:.0f} s", flush=True)
    (out / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    print(f"report -> {out / 'comparison.md'}")


if __name__ == "__main__":
    main()
