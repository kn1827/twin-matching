"""Stage 1 a0 against the profile-matching baselines that recent ranked papers compare on Speed Dating.

    python experiments/compare_literature_baselines.py [--markets columbia speed_dating]
        [--protocol lowo|person5] [--featureset stage1|literature] [--models all|LR,FM,...]
        [--seeds 1] [--workers 4] [--epochs 20]

Baselines (chosen with the deep-research skill, quick mode; venues checked on 2026-11-15)
--------------------------------------------------------------------------------------------------------
Every number of a published paper below was read from the paper itself; nothing is copied into the
comparison as if we had measured it. The script RE-IMPLEMENTS each model and trains it under the same
folds, the same rows and (by default) the same profile information as a0.

1. biDeepFM / biLorentzFM family (feature-interaction CTR models used as reciprocal matchers).
   * Yıldırım, Azad & Gündüz-Öğüdücü (2021). biDeepFM: A multi-objective deep factorization machine for
     reciprocal recommendation. Engineering Science and Technology, an International Journal 24,
     1467-1477 (Elsevier; SJR Q1). Two heads on shared DeepFM embeddings, one per side of the match.
   * biLorentzFM (2025), Applied Sciences 15(22), 12340 (MDPI; SJR Q2), Table 13, Speed Dating
     (Columbia file, 8378 decisions, 552 participants), 5-fold CV with participant-level folds, features
     "demographics, self-assessed attributes, partner preferences, activity interests":
         biDeepFM_Base   Part A AUC 0.6823 +- 0.0234 | Part B 0.6791 +- 0.0241 | mutual 0.7105 +- 0.0198
         biLorentzFM_Base           0.7012 +- 0.0189 |        0.6985 +- 0.0203 |        0.7289 +- 0.0176
     Its baseline list (from Yıldırım et al.): PNN (Qu et al., ICDM 2016), DeepFM (Guo et al.,
     IJCAI 2017), DCN (Wang et al., ADKDD 2017; here DCN-V2, Wang et al., WWW 2021), AutoInt (Song et
     al., CIKM 2019), AFM, NFM, FGCNN. We add FM (Rendle, ICDM 2010) and xDeepFM (Lian et al., KDD 2018),
     which RRCN (Knowledge and Information Systems, 2025) uses as reciprocal baselines.
     Hyper-parameters follow biLorentzFM Table 7 / Algorithm 1: d = 8, DNN [256, 128, 64], Adam 1e-3,
     batch 256, at most 20 epochs, early stopping (patience 5) on validation AUC, L2 1e-6.
2. Gradient-boosted trees. McElfresh et al. (2023), "When Do Neural Nets Outperform Boosted Trees on
   Tabular Data?", NeurIPS 2023 Datasets & Benchmarks (CORE A*): on OpenML SpeedDating the three best
   of 19 algorithms are XGBoost, CatBoost, LightGBM (Table 4; OpenML's 10 random folds, target `match`,
   121 features that include ratings made during the date, so their task is easier than Stage 1).
   We train the same three libraries with their default settings and early stopping on validation waves.
3. LR and cosine profile similarity. LFKL (Shang, Yan & Liu, 2025; NeurIPS 2025 workshop on LLM persona
   modeling, NOT a ranked venue, quoted by stage1.md as the best published Stage-1 number): match AUC
   pre-date LR 0.61, Similarity 0.54, LLM observer 0.60. Reported for reference only.

Protocols
---------
* `lowo`   leave-one-wave-out, the Stage 1 protocol (stage1.md section 9). a0 is read from
           reports/predictions_<market>.csv written by scripts/run_stage1.py (same folds).
* `person5` 5-fold GroupKFold by chooser, as in biLorentzFM. a0 is refitted on these folds: the test
           choosers' decisions are treated as missing y in the Gibbs sampler.

Metrics: Part A AUC (the chooser's directed decision), within-chooser AUC, log-loss, Brier, accuracy, and
mutual match AUC with s_mutual = p(i->j) * p(j->i) (biLorentzFM's definition; biDeepFM uses its own two
heads). Differences against a0: chooser-cluster bootstrap and TOST on +-0.03 (stage1.md section 9).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnwt.analysis.metrics import Scored, auc, cluster_bootstrap, summarize, tost  # noqa: E402
from gnwt.config import load_params, resolve_path  # noqa: E402
from gnwt.data.bfi10 import build_proxy  # noqa: E402
from gnwt.data.speed_dating import load_market, to_personas  # noqa: E402
from gnwt.persona import INTERESTS, SELF_KEYS  # noqa: E402
from gnwt.stage1.features import DesignSpec, PersonTable, dyad_raw  # noqa: E402
from gnwt.stage1.offline.srrm_fit import DyadData, FitConfig, fit_srrm, predict  # noqa: E402
from gnwt.stage1.static import load_reader, persona_demand  # noqa: E402

REPORTS = ROOT / "reports"
DEEP_MODELS = ("LR", "FM", "DeepFM", "xDeepFM", "DCN-V2", "AutoInt", "IPNN", "biDeepFM")
TREE_MODELS = ("XGBoost", "LightGBM", "CatBoost")
HEURISTIC = ("Cosine",)
ALL_MODELS = DEEP_MODELS + TREE_MODELS            # ranked-venue baselines only; `--models Cosine` adds LFKL's

PUBLISHED = [
    {"paper": "biLorentzFM, Applied Sciences 15(22):12340 (2025), MDPI, SJR Q2", "model": "biDeepFM_Base",
     "protocol": "5-fold, participant-level", "target": "dec (Part A)", "auc": 0.6823, "sd": 0.0234,
     "mutual_auc": 0.7105},
    {"paper": "biLorentzFM, Applied Sciences 15(22):12340 (2025), MDPI, SJR Q2", "model": "biLorentzFM_Base",
     "protocol": "5-fold, participant-level", "target": "dec (Part A)", "auc": 0.7012, "sd": 0.0189,
     "mutual_auc": 0.7289},
    {"paper": "LFKL, NeurIPS 2025 workshop (not ranked)", "model": "Logistic Regression (pre-date)",
     "protocol": "not stated", "target": "match", "auc": None, "sd": None, "mutual_auc": 0.61},
    {"paper": "LFKL, NeurIPS 2025 workshop (not ranked)", "model": "Similarity (cosine)",
     "protocol": "not stated", "target": "match", "auc": None, "sd": None, "mutual_auc": 0.54},
    {"paper": "McElfresh et al., NeurIPS 2023 D&B (A*)", "model": "XGBoost > CatBoost > LightGBM (rank)",
     "protocol": "OpenML 10 random folds, during-date features", "target": "match", "auc": None,
     "sd": None, "mutual_auc": None},
]


# =================================================================================== features


def _extras_columbia(m, path: Path) -> pd.DataFrame:  # noqa: ANN001
    raw = pd.read_csv(path, encoding="latin-1")
    pref = raw[["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]].astype(float)
    pref = pref.div(pref.sum(axis=1), axis=0) * 100.0          # waves 6-9 used a 1-10 scale: normalise
    ex = pd.DataFrame({"race": raw["race"].astype("Int64").astype(str), "imprace": raw["imprace"],
                       "imprelig": raw["imprelig"], "attr_self": raw["attr3_1"],
                       **{f"pref_{c.split('1_1')[0]}": pref[c] for c in pref}})
    first = m.persons["raw_first"].astype(int)
    return ex.iloc[first.to_numpy()].set_axis(m.persons.index)


def _extras_openml(m, path: Path) -> pd.DataFrame:  # noqa: ANN001
    from gnwt.data.speed_dating import _read_arff

    raw = _read_arff(path)
    cols = {"attractive_important": "pref_attr", "sincere_important": "pref_sinc",
            "intellicence_important": "pref_intel", "funny_important": "pref_fun",
            "ambtition_important": "pref_amb", "shared_interests_important": "pref_shar"}
    ex = pd.DataFrame({"race": raw["race"].astype(str), "imprace": raw["importance_same_race"],
                       "imprelig": raw["importance_same_religion"], "attr_self": raw["attractive"],
                       **{v: raw[k] for k, v in cols.items()}})
    first = m.persons["raw_first"].astype(int)
    return ex.iloc[first.to_numpy()].set_axis(m.persons.index)


def feature_table(d: DyadData, extras: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """One row per directed decision; returns (frame, categorical fields, numeric fields)."""
    T = d.T
    F: dict[str, np.ndarray] = {}
    cat, num = [], []
    for side, idx in (("i", d.i), ("j", d.j)):
        if side == "i":
            F["gender_i"] = T.gender[idx]
            cat.append("gender_i")
        for nm, arr, kind in (("age", T.age[idx], "n"), ("field", T.field[idx], "c"), ("goal", T.goal[idx], "c")):
            F[f"{nm}_{side}"] = arr
            (cat if kind == "c" else num).append(f"{nm}_{side}")
        for k, it in enumerate(INTERESTS):
            F[f"{it}_{side}"] = T.interests[idx, k]
            num.append(f"{it}_{side}")
        for k, sk in enumerate(SELF_KEYS):
            F[f"self_{sk}_{side}"] = T.selfr[idx, k]
            num.append(f"self_{sk}_{side}")
        for k, t in enumerate(("O", "C", "E", "A", "N")):
            if np.isfinite(T.bf[:, k]).any():
                F[f"bf_{t}_{side}"] = T.bf[idx, k]
                num.append(f"bf_{t}_{side}")
        if extras is not None:
            ex = extras.reindex(T.ids[idx])
            F[f"race_{side}"] = ex["race"].to_numpy()
            cat.append(f"race_{side}")
            for c in ("imprace", "imprelig", "attr_self", "pref_attr", "pref_sinc", "pref_intel", "pref_fun",
                      "pref_amb", "pref_shar"):
                F[f"{c}_{side}"] = ex[c].to_numpy(float)
                num.append(f"{c}_{side}")
    raw = dyad_raw(T, d.i, d.j, DesignSpec())
    for k in ("int_corr", "jaccard_top5", "abs_age_gap"):
        F[k] = raw[k]
        num.append(k)
    for k in ("same_field", "same_goal"):
        F[k] = raw[k]
        cat.append(k)
    if extras is not None:
        F["samerace"] = (pd.Series(F["race_i"]) == pd.Series(F["race_j"])).astype(float).to_numpy()
        cat.append("samerace")
    frame = pd.DataFrame(F)
    keep = [c for c in frame if frame[c].notna().any() and frame[c].astype(str).nunique() > 1]
    return frame[keep], [c for c in cat if c in keep], [c for c in num if c in keep]


class Encoder:
    """Fields -> integer ids (numeric fields bucketised in <= 10 training quantile bins, NaN own bin)."""

    def __init__(self, frame: pd.DataFrame, cat: list[str], num: list[str], train: np.ndarray, bins: int = 10) -> None:
        self.fields = cat + num
        self.maps: dict[str, object] = {}
        self.sizes: list[int] = []
        for c in cat:
            vals = np.unique([str(x) for x in frame[c].to_numpy()[train]])
            self.maps[c] = {v: k + 1 for k, v in enumerate(sorted(vals))}
            self.sizes.append(len(vals) + 1)
        for c in num:
            x = frame[c].to_numpy(float)[train]
            x = x[np.isfinite(x)]
            edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])) if len(x) else np.array([])
            self.maps[c] = edges
            self.sizes.append(len(edges) + 2)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]]).astype(int)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(frame), len(self.fields)), dtype=np.int64)
        for k, c in enumerate(self.fields):
            mp = self.maps[c]
            if isinstance(mp, dict):
                out[:, k] = [mp.get(str(v), 0) for v in frame[c].to_numpy()]
            else:
                x = frame[c].to_numpy(float)
                ids = np.searchsorted(mp, x, side="right") + 1
                out[:, k] = np.where(np.isfinite(x), ids, 0)
        return out + self.offsets


# ================================================================================ deep models


def _torch_models():  # noqa: ANN202
    import torch
    from torch import nn

    class Base(nn.Module):
        def __init__(self, n_tokens: int, n_fields: int, d: int, two_heads: bool = False) -> None:
            super().__init__()
            self.emb = nn.Embedding(n_tokens, d)
            self.lin = nn.Embedding(n_tokens, 1)
            self.bias = nn.Parameter(torch.zeros(2 if two_heads else 1))
            nn.init.xavier_uniform_(self.emb.weight)
            nn.init.zeros_(self.lin.weight)
            self.F, self.d, self.two = n_fields, d, two_heads

        def linear(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x).sum(dim=(1, 2))

        @staticmethod
        def fm(e: torch.Tensor) -> torch.Tensor:
            return 0.5 * ((e.sum(1) ** 2) - (e ** 2).sum(1)).sum(1)

    def mlp(i: int, hidden: tuple[int, ...] = (256, 128, 64), out: int = 1) -> nn.Sequential:
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(i, h), nn.ReLU()]
            i = h
        layers.append(nn.Linear(i, out))
        return nn.Sequential(*layers)

    class LR(Base):
        def forward(self, x):  # noqa: ANN001, ANN201
            return self.linear(x) + self.bias

    class FM(Base):
        def forward(self, x):  # noqa: ANN001, ANN201
            return self.linear(x) + self.fm(self.emb(x)) + self.bias

    class DeepFM(Base):
        def __init__(self, n_tokens: int, n_fields: int, d: int, two_heads: bool = False) -> None:
            super().__init__(n_tokens, n_fields, d, two_heads)
            self.dnn = mlp(n_fields * d, out=2 if two_heads else 1)

        def forward(self, x):  # noqa: ANN001, ANN201
            e = self.emb(x)
            shared = self.linear(x) + self.fm(e)
            deep = self.dnn(e.flatten(1))
            if self.two:
                return shared[:, None] + deep + self.bias
            return shared + deep[:, 0] + self.bias

    class xDeepFM(Base):
        def __init__(self, n_tokens: int, n_fields: int, d: int, two_heads: bool = False, cin=(16, 16)) -> None:  # noqa: ANN001
            super().__init__(n_tokens, n_fields, d)
            self.cin = nn.ModuleList()
            prev = n_fields
            for h in cin:
                self.cin.append(nn.Conv1d(prev * n_fields, h, 1))
                prev = h
            self.cin_out = nn.Linear(sum(cin), 1)
            self.dnn = mlp(n_fields * d)

        def forward(self, x):  # noqa: ANN001, ANN201
            e = self.emb(x)                               # (B, F, d)
            h, outs = e, []
            for conv in self.cin:
                z = torch.einsum("bhd,bmd->bhmd", h, e).reshape(e.shape[0], -1, self.d)
                h = torch.relu(conv(z))
                outs.append(h.sum(2))
            return self.linear(x) + self.cin_out(torch.cat(outs, 1))[:, 0] + self.dnn(e.flatten(1))[:, 0] + self.bias

    class DCNV2(Base):
        def __init__(self, n_tokens: int, n_fields: int, d: int, two_heads: bool = False, layers: int = 3) -> None:
            super().__init__(n_tokens, n_fields, d)
            D = n_fields * d
            self.cross = nn.ModuleList([nn.Linear(D, D) for _ in range(layers)])
            self.dnn = mlp(D, out=64)
            self.out = nn.Linear(D + 64, 1)

        def forward(self, x):  # noqa: ANN001, ANN201
            x0 = self.emb(x).flatten(1)
            xl = x0
            for lyr in self.cross:
                xl = x0 * lyr(xl) + xl
            return self.out(torch.cat([xl, torch.relu(self.dnn(x0))], 1))[:, 0] + self.bias

    class AutoInt(Base):
        def __init__(self, n_tokens: int, n_fields: int, d: int, two_heads: bool = False, layers: int = 3,
                     heads: int = 2, att: int = 16) -> None:
            super().__init__(n_tokens, n_fields, d)
            self.proj = nn.Linear(d, att)
            self.att = nn.ModuleList([nn.MultiheadAttention(att, heads, batch_first=True) for _ in range(layers)])
            self.res = nn.ModuleList([nn.Linear(att, att) for _ in range(layers)])
            self.out = nn.Linear(n_fields * att, 1)

        def forward(self, x):  # noqa: ANN001, ANN201
            h = self.proj(self.emb(x))
            for a, r in zip(self.att, self.res):
                h = torch.relu(a(h, h, h, need_weights=False)[0] + r(h))
            return self.linear(x) + self.out(h.flatten(1))[:, 0] + self.bias

    class IPNN(Base):
        def __init__(self, n_tokens: int, n_fields: int, d: int, two_heads: bool = False) -> None:
            super().__init__(n_tokens, n_fields, d)
            self.iu = torch.triu_indices(n_fields, n_fields, 1)
            self.dnn = mlp(n_fields * d + self.iu.shape[1])

        def forward(self, x):  # noqa: ANN001, ANN201
            e = self.emb(x)
            ip = (e[:, self.iu[0]] * e[:, self.iu[1]]).sum(-1)
            return self.dnn(torch.cat([e.flatten(1), ip], 1))[:, 0] + self.bias

    return {"LR": LR, "FM": FM, "DeepFM": DeepFM, "xDeepFM": xDeepFM, "DCN-V2": DCNV2, "AutoInt": AutoInt,
            "IPNN": IPNN, "biDeepFM": DeepFM}


def train_deep(name: str, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, Xte: np.ndarray,
               n_tokens: int, seed: int, epochs: int, ytr_b: np.ndarray | None = None,
               yva_b: np.ndarray | None = None) -> np.ndarray:
    """Returns test probabilities (n,) or, for biDeepFM, (n, 2) = (head A, head B)."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    two = name == "biDeepFM"
    model = _torch_models()[name](n_tokens, Xtr.shape[1], 8, two_heads=two)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    lossf = torch.nn.BCEWithLogitsLoss()
    xt = torch.as_tensor(Xtr)
    yt = torch.as_tensor(ytr, dtype=torch.float32)
    ybt = torch.as_tensor(ytr_b, dtype=torch.float32) if two else None
    best, best_state, bad = -1.0, None, 0

    def pred(X: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(torch.as_tensor(X))).numpy()
        model.train()
        return out

    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for k in range(0, len(xt), 256):
            idx = perm[k:k + 256]
            out = model(xt[idx])
            if two:
                loss = 0.5 * lossf(out[:, 0], yt[idx]) + 0.5 * lossf(out[:, 1], ybt[idx])
            else:
                loss = lossf(out, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv = pred(Xva)
        score = auc(yva, pv[:, 0] if two else pv)
        if score > best + 1e-5:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return pred(Xte)


def train_tree(name: str, Ftr: pd.DataFrame, ytr: np.ndarray, Fva: pd.DataFrame, yva: np.ndarray,
               Fte: pd.DataFrame, cat: list[str], seed: int) -> np.ndarray:
    def enc(F: pd.DataFrame) -> pd.DataFrame:
        G = F.copy()
        for c in cat:
            cats = sorted({str(x) for F_ in (Ftr, Fva, Fte) for x in F_[c]})
            G[c] = pd.Categorical([str(x) for x in G[c]], categories=cats)
        return G

    if name == "XGBoost":
        import xgboost as xgb

        m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6, enable_categorical=True,
                              tree_method="hist", early_stopping_rounds=50, eval_metric="logloss",
                              random_state=seed, n_jobs=2)
        m.fit(enc(Ftr), ytr, eval_set=[(enc(Fva), yva)], verbose=False)
        return m.predict_proba(enc(Fte))[:, 1]
    if name == "LightGBM":
        import lightgbm as lgb

        m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=31, random_state=seed,
                               n_jobs=2, verbose=-1)
        m.fit(enc(Ftr), ytr, eval_set=[(enc(Fva), yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
        return m.predict_proba(enc(Fte))[:, 1]
    if name == "CatBoost":
        from catboost import CatBoostClassifier

        G = [F.assign(**{c: [str(x) for x in F[c]] for c in cat}) for F in (Ftr, Fva, Fte)]
        m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, random_seed=seed, verbose=False,
                               thread_count=2, allow_writing_files=False, od_type="Iter", od_wait=50)
        m.fit(G[0], ytr, cat_features=cat, eval_set=(G[1], yva), use_best_model=True)
        return m.predict_proba(G[2])[:, 1]
    raise KeyError(name)


def cosine_baseline(frame: pd.DataFrame, num: list[str], ytr: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    base = sorted({c[:-2] for c in num if c.endswith("_i") and c[:-2] + "_j" in num})
    A = frame[[c + "_i" for c in base]].to_numpy(float)
    B = frame[[c + "_j" for c in base]].to_numpy(float)
    V = np.vstack([A[tr], B[tr]])
    mu, sd = np.nanmean(V, 0), np.nanstd(V, 0)
    sd[sd == 0] = 1
    A, B = np.nan_to_num((A - mu) / sd), np.nan_to_num((B - mu) / sd)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.nan_to_num((A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)))
    lr = LogisticRegression().fit(cos[tr, None], ytr)
    return lr.predict_proba(cos[te, None])[:, 1]


# ===================================================================================== folds


def _fold_job(args: tuple) -> dict[str, np.ndarray]:
    (frame, cat, num, y, y_rev, groups_val, tr, te, models, seed, epochs, n_seeds) = args
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(2)
    out: dict[str, np.ndarray] = {}
    with threadpool_limits(2), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # validation = 10% of the training groups (waves or choosers), deterministic
        rng = np.random.default_rng(seed)
        g_tr = np.unique(groups_val[tr])
        g_va = rng.choice(g_tr, size=max(1, int(round(0.1 * len(g_tr)))), replace=False)
        va = tr[np.isin(groups_val[tr], g_va)]
        fit = tr[~np.isin(groups_val[tr], g_va)]
        enc = Encoder(frame, cat, num, fit)
        X = enc.transform(frame)
        n_tok = int(sum(enc.sizes))
        for name in models:
            preds = []
            for s in range(n_seeds):
                if name in DEEP_MODELS:
                    p = train_deep(name, X[fit], y[fit], X[va], y[va], X[te], n_tok, seed + 1000 * s, epochs,
                                   y_rev[fit] if name == "biDeepFM" else None, y_rev[va] if name == "biDeepFM" else None)
                elif name in TREE_MODELS:
                    p = train_tree(name, frame.iloc[fit], y[fit], frame.iloc[va], y[va], frame.iloc[te], cat, seed + s)
                else:
                    p = cosine_baseline(frame, num, y[tr], tr, te)
                preds.append(p)
            P = np.mean(preds, axis=0)
            if name == "biDeepFM":
                out[name] = P[:, 0]
                out[name + "__headB"] = P[:, 1]
            else:
                out[name] = P
    return out


def a0_person5(d: DyadData, fold_of_row: np.ndarray, P, workers: int) -> np.ndarray:  # noqa: ANN001
    """a0 and m0 refitted on chooser folds: test choosers' decisions are missing y for the sampler."""
    cfg = FitConfig.from_params(P, chains=P.stage1.core.sampler.crossfit_chains)
    spec = DesignSpec.from_params(P.stage1.features)
    jobs = [(d, fold_of_row, f, spec, cfg) for f in np.unique(fold_of_row)]
    a0 = np.full(len(d.i), np.nan)
    m0 = np.full(len(d.i), np.nan)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f, (te, pa, pm) in zip(np.unique(fold_of_row), ex.map(_a0_fold, jobs)):
            a0[te], m0[te] = pa, pm
    return np.column_stack([a0, m0])


def _a0_fold(args: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d, fold_of_row, f, spec, cfg = args
    from threadpoolctl import threadpool_limits

    te = np.where(fold_of_row == f)[0]
    y = d.y.copy()
    y[te] = np.nan
    dm = replace(d, y=y, observed=np.isfinite(y))
    with threadpool_limits(2):
        m = fit_srrm(dm, spec, cfg, cfg.seed + int(f))
        pr = predict(m, d)
    return te, pr["a0"].to_numpy()[te], pr["m0"].to_numpy()[te]


# ====================================================================================== main


def run(market: str, args: argparse.Namespace) -> dict:
    P = load_params()
    t0 = time.time()
    m = load_market(market)
    bf = build_proxy(m.persons, P.bf_proxy.min_items_per_trait)
    reader = load_reader(P)
    personas = to_personas(m, bf)
    demands = {p.persona_id: persona_demand(p, reader) for p in personas}
    T = PersonTable.from_frame(m.persons, bf, demands, default_context="speed_dating")
    d = DyadData.from_rows(T, m.rows)
    d = d.subset(np.where(d.observed & d.observed[d.rev])[0])    # dyads with both decisions in the file
    extras = None
    if args.featureset == "literature":
        path = resolve_path(P.data.raw_dir) / P.data.datasets[market].file
        extras = _extras_columbia(m, path) if market == "columbia" else _extras_openml(m, path)
    frame, cat, num = feature_table(d, extras)
    y = d.y.astype(float)
    y_rev = y[d.rev]

    if args.protocol == "lowo":
        fold = d.wave.copy()
        groups_val = d.wave
        pr = pd.read_csv(REPORTS / f"predictions_{market}.csv")
        key = pd.Series(np.arange(len(pr)), index=pd.MultiIndex.from_arrays([pr["chooser"], pr["chosen"]]))
        pos = key.reindex(pd.MultiIndex.from_arrays([T.ids[d.i], T.ids[d.j]])).to_numpy()
        if np.isnan(pos.astype(float)).any():
            raise RuntimeError("reports/predictions_<market>.csv does not cover every row; rerun scripts/run_stage1.py")
        a0m = pr[["a0", "m0"]].to_numpy()[pos.astype(int)]
    else:
        from sklearn.model_selection import GroupKFold

        fold = np.zeros(len(y), int)
        for k, (_, te) in enumerate(GroupKFold(n_splits=5).split(frame, y, d.i)):
            fold[te] = k
        groups_val = d.i
        a0m = a0_person5(d, fold, P, args.workers)

    models = ALL_MODELS if args.models == "all" else tuple(args.models.split(","))
    jobs = [(frame, cat, num, y, y_rev, groups_val, np.where(fold != f)[0], np.where(fold == f)[0], models,
             P.global_seed + 17 * k, args.epochs, args.seeds) for k, f in enumerate(np.unique(fold))]
    preds = {nm: np.full(len(y), np.nan) for nm in models}
    preds["biDeepFM__headB"] = np.full(len(y), np.nan)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for job, out in zip(jobs, ex.map(_fold_job, jobs)):
            te = job[7]
            for k, v in out.items():
                preds[k][te] = v

    rows = np.arange(len(y))
    first = np.where(rows < d.rev)[0]
    match = np.nan_to_num(d.match)
    ref = Scored("a0 (Stage 1 SRRM)", a0m[:, 0], a0m[:, 1])
    scored = []
    for nm in models:
        p = preds[nm]
        ms = p * preds["biDeepFM__headB"] if nm == "biDeepFM" else p * p[d.rev]
        scored.append(Scored(nm, p, ms))
    table = {sc.name: summarize(y, d.i, first, match, sc) for sc in [ref, *scored]}
    bs = cluster_bootstrap(y, d.i, first, match, scored, ref, args.boot, P.global_seed)
    comp = {k: {"delta_auc": tost(v[:, 0], P.analysis.delta_auc_min, P.analysis.alpha),
                "delta_match_auc": tost(v[:, 1], P.analysis.delta_auc_min, P.analysis.alpha)}
            for k, v in bs.items()}
    per_fold = {sc.name: [auc(y[fold == f], sc.p[fold == f]) for f in np.unique(fold)] for sc in [ref, *scored]}
    return {"market": market, "protocol": args.protocol, "featureset": args.featureset,
            "n_rows": len(y), "n_fields": {"categorical": len(cat), "numeric": len(num)},
            "metrics": table, "vs_a0": comp,
            "fold_auc_mean_sd": {k: [float(np.nanmean(v)), float(np.nanstd(v))] for k, v in per_fold.items()},
            "seconds": time.time() - t0}


def render(results: list[dict]) -> str:
    L = ["# Stage 1 a0 vs. literature profile-matching baselines", "",
         "Every model below was re-trained here on the same rows and folds as a0; the published numbers are in "
         "the last table and are not comparable one-to-one (different features or protocol).", ""]
    for r in results:
        L += [f"## {r['market']} — protocol `{r['protocol']}`, features `{r['featureset']}` "
              f"({r['n_rows']} decisions, {r['n_fields']['categorical']} categorical + {r['n_fields']['numeric']} numeric fields)", "",
              "| model | Part A AUC | fold AUC mean ± sd | AUC within | log-loss | Brier | acc | mutual AUC | ΔAUC vs a0 [95% CI] | TOST ±0.03 | Δ mutual [95% CI] |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for nm, mt in r["metrics"].items():
            c = r["vs_a0"].get(nm)
            fm = r["fold_auc_mean_sd"][nm]
            if c is None:
                dd = eq = dm = "—"
            else:
                da, dmu = c["delta_auc"], c["delta_match_auc"]
                dd = f"{da['mean']:+.3f} [{da['ci95_lo']:+.3f}, {da['ci95_hi']:+.3f}]"
                eq = "equivalent" if da["equivalent"] else ("better" if da["better"] else ("worse" if da["worse"] else "inconclusive"))
                dm = f"{dmu['mean']:+.3f} [{dmu['ci95_lo']:+.3f}, {dmu['ci95_hi']:+.3f}]"
            L.append(f"| {nm} | {mt['auc']:.4f} | {fm[0]:.4f} ± {fm[1]:.4f} | {mt['auc_within']:.4f} | {mt['log_loss']:.4f} | "
                     f"{mt['brier']:.4f} | {mt['accuracy']:.3f} | {mt['match_auc']:.4f} | {dd} | {eq} | {dm} |")
        L.append("")
    L += ["## Published reference numbers (read from the papers, not re-measured)", "",
          "| paper (venue, rank) | model | protocol | target | AUC | sd | mutual / match AUC |", "|---|---|---|---|---|---|---|"]
    for p in PUBLISHED:
        f = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
        L.append(f"| {p['paper']} | {p['model']} | {p['protocol']} | {p['target']} | {f(p['auc'])} | {f(p['sd'])} | {f(p['mutual_auc'])} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", nargs="+", default=["columbia", "speed_dating"])
    ap.add_argument("--protocol", choices=["lowo", "person5"], default="lowo")
    ap.add_argument("--featureset", choices=["stage1", "literature"], default="stage1")
    ap.add_argument("--models", default="all")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    results = [run(mk, args) for mk in args.markets]
    REPORTS.mkdir(exist_ok=True)
    tag = f"{args.protocol}_{args.featureset}_{'-'.join(args.markets)}"
    (REPORTS / f"literature_comparison_{tag}.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    md = render(results)
    (REPORTS / f"literature_comparison_{tag}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
