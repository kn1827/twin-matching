"""Locked estimation procedure of the SRRM (tech spec 7). Never runs at inference time.

* Prior width c (scale_grid) and block-masking rate q (block_masking_rate_grid) are chosen by
  leave-one-wave-out CV on log loss with the 1-SE rule preferring the stronger shrinkage. Folds are
  whole waves because a wave contains both i->j and j->i. Inside the outer cross-fit the selection is
  NESTED (inner LOWO on the training waves only). The selection criterion is evaluated with the MAP of
  the marginal probit (beta / s), the quantity a0 depends on; see CHANGELOG for this choice.
* The selected (c, q) is then fitted with the Gibbs sampler of the full SRRM (gibbs.py).
* `crossfit` gives every wave a0 from a model that never saw that wave (test I1).
* `estimate_srrm` is the final locked fit on all development waves (4 chains, R-hat reported).
* `laplace_update` folds a new labelled batch into the locked model (update: laplace_batch).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import log_ndtr, ndtr

from stage1.context.demand import ReaderArtifacts, extract_demands
from stage1.data.speed_dating import DyadicData
from stage1.features import (COL, COLUMNS, FeatureConfig, Standardizer, design_from_raw, pair_hide, pair_raw,
                             person_hide, take_raw)
from stage1.offline.gibbs import GibbsConfig, SRRMGibbs
from stage1.offline.probit import probit_map
from stage1.srrm import Identification, LockedSRRM, a0, eta_blocks, m0


@dataclass
class SRRMSettings:
    scale_grid: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0)
    mask_grid: tuple[float, ...] = (0.0, 0.15, 0.3, 0.5)
    intercept_sd: float = 5.0
    ident_mode: str = "orthogonal"
    feature_cfg: FeatureConfig = field(default_factory=FeatureConfig)
    crossfit_gibbs: GibbsConfig = field(default_factory=lambda: GibbsConfig(iterations=800, burn_in=300, chains=1))
    final_gibbs: GibbsConfig = field(default_factory=lambda: GibbsConfig(iterations=1500, burn_in=500, chains=4))
    inner_cv: str | int = "lowo"
    covariance: str = "lkj_eta_2"
    variance_scale: float = 1.0
    seed: int = 20261119

    @classmethod
    def from_yaml(cls, path: Path) -> "SRRMSettings":
        c = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["stage1"]
        core, pri, feat = c["core"], c["priors"], c["features"]
        g, cf = core["sampler"], core.get("crossfit_sampler", {})
        cov = pri.get("covariance", "lkj_eta_2")
        vs = float(pri.get("variance_sd", {}).get("scale", 1.0))
        return cls(scale_grid=tuple(pri["scale_grid"]), mask_grid=tuple(c["block_masking_rate_grid"]),
                   intercept_sd=float(pri.get("intercept_sd", 5.0)),
                   ident_mode="orthogonal" if c["personality_fit"]["single_context_identification"]
                   == "delta_orthogonal_to_demand" else "free",
                   feature_cfg=FeatureConfig(trait_block_missing_if_ge=int(feat["trait_block_missing_if_ge"]),
                                             int_corr_min_common_items=int(feat["int_corr_min_common_items"])),
                   crossfit_gibbs=GibbsConfig(iterations=int(cf.get("iterations", 800)),
                                              burn_in=int(cf.get("burn_in", 300)), chains=int(cf.get("chains", 1)),
                                              covariance=cov, variance_scale=vs),
                   final_gibbs=GibbsConfig(iterations=int(g["iterations"]), burn_in=int(g["burn_in"]),
                                           thin=int(g.get("thin", 1)), chains=int(g["chains"]),
                                           covariance=cov, variance_scale=vs),
                   covariance=cov, variance_scale=vs, seed=int(c.get("seed", 20261119)))


class Problem:
    """Dataset-level precomputation: context demands, raw pair quantities, labels and waves."""

    def __init__(self, data: DyadicData, art: ReaderArtifacts, settings: SRRMSettings):
        self.data, self.art, self.settings = data, art, settings
        self.demands = [extract_demands(p.context_text, data.event_text, art, p.market or data.market)
                        for p in data.personas]
        self.D = np.array([cd.D_tilde for cd in self.demands])
        self.a = data.pairs["chooser"].to_numpy()
        self.p = data.pairs["chosen"].to_numpy()
        self.y = data.pairs["y"].to_numpy().astype(int)
        self.match = data.pairs["match"].to_numpy().astype(int)
        self.wave = data.pairs["wave"].to_numpy()
        self.raw = pair_raw(data.table, self.a, self.p, settings.feature_cfg)
        market = extract_demands(None, None, art, data.market)
        self.ident = Identification(settings.ident_mode, market.D_tilde.copy())
        rev = {(u, v): k for k, (u, v) in enumerate(zip(self.a, self.p))}
        self.reverse = np.array([rev.get((v, u), -1) for u, v in zip(self.a, self.p)])

    def standardizer(self, rows: np.ndarray) -> Standardizer:
        return Standardizer.fit(self.data.table, self.a[rows], self.p[rows], self.settings.feature_cfg,
                                raw=take_raw(self.raw, rows))

    def hide(self, rows: np.ndarray, rate: float, seed) -> dict | None:
        if rate <= 0:
            return None
        ph = person_hide(rate, len(self.data.personas), np.random.default_rng(seed))
        return pair_hide(ph, self.a[rows], self.p[rows])

    def raw_design(self, rows: np.ndarray, std: Standardizer, hide: dict | None = None) -> np.ndarray:
        return design_from_raw(take_raw(self.raw, rows), self.D[self.a[rows]], std, hide).X


def _active(Xm: np.ndarray, cols: list[str]) -> np.ndarray:
    act = Xm.std(axis=0) > 1e-12
    act[cols.index("intercept")] = True
    return act


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ------------------------------------------------------------------------------------ selection
def _inner_folds(waves: np.ndarray, inner_cv) -> list[np.ndarray]:
    uw = np.unique(waves)
    if inner_cv == "lowo" or int(inner_cv) >= len(uw):
        return [np.array([w]) for w in uw]
    return [g for g in np.array_split(uw, int(inner_cv)) if len(g)]


def select_config(prob: Problem, train_rows: np.ndarray, seed: int = 0) -> dict:
    """(scale, mask_rate) by (inner) LOWO log loss, 1-SE rule toward the stronger shrinkage."""
    S = prob.settings
    cols = prob.ident.model_columns()
    folds = _inner_folds(prob.wave[train_rows], S.inner_cv)
    losses = {(c, q): [] for c in S.scale_grid for q in S.mask_grid}
    for k, fw in enumerate(folds):
        tr = train_rows[~np.isin(prob.wave[train_rows], fw)]
        ev = train_rows[np.isin(prob.wave[train_rows], fw)]
        std = prob.standardizer(tr)
        Xev = prob.ident.to_model(prob.raw_design(ev, std))
        for q in S.mask_grid:
            Xtr = prob.ident.to_model(prob.raw_design(tr, std, prob.hide(tr, q, [seed, k, int(q * 1000)])))
            act = _active(Xtr, cols)
            b = None
            for c in sorted(S.scale_grid, reverse=True):          # warm start from the widest prior
                sd = prob.ident.prior_sd(c, S.intercept_sd)[act]
                b, _ = probit_map(Xtr[:, act], prob.y[tr], sd, b0=b)
                losses[(c, q)].append(log_loss(prob.y[ev], ndtr(Xev[:, act] @ b)))
    # 1-SE rule on PAIRED differences: a configuration is "within 1 SE" of the best when its mean fold-wise
    # excess loss is below the SE of that excess (wave-to-wave level differences cancel in the pairing).
    best_key = min(losses, key=lambda key: np.mean(losses[key]))
    ref = np.asarray(losses[best_key])
    rows = []
    for (c, q), v in losses.items():
        d = np.asarray(v) - ref
        rows.append({"scale": c, "mask_rate": q, "log_loss": float(np.mean(v)), "excess": float(d.mean()),
                     "se_excess": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0})
    table = pd.DataFrame(rows)
    best = table.loc[table["log_loss"].idxmin()]
    ok = table[table["excess"] <= table["se_excess"]]
    pick = ok.sort_values(["scale", "log_loss"]).iloc[0]
    return {"scale": float(pick["scale"]), "mask_rate": float(pick["mask_rate"]), "table": table,
            "best_loss": float(best["log_loss"])}


# ------------------------------------------------------------------------------------ fitting
def fit_srrm(prob: Problem, train_rows: np.ndarray, scale: float, mask_rate: float, gibbs: GibbsConfig,
             seed: int = 0, verbose: bool = False) -> LockedSRRM:
    S = prob.settings
    std = prob.standardizer(train_rows)
    Xm = prob.ident.to_model(prob.raw_design(train_rows, std, prob.hide(train_rows, mask_rate, [seed, 99])))
    cols = prob.ident.model_columns()
    act = _active(Xm, cols)
    sd = prob.ident.prior_sd(scale, S.intercept_sd)[act]
    g = SRRMGibbs(Xm[:, act], prob.y[train_rows], prob.a[train_rows], prob.p[train_rows], sd,
                  replace(gibbs, seed=seed, covariance=S.covariance, variance_scale=S.variance_scale))
    res = g.run(verbose=verbose)
    b_model = np.zeros(len(cols))
    b_model[act] = res.mean("beta")
    beta_raw = prob.ident.to_raw_coef(b_model)
    bm = res.draws["beta_marginal"].reshape(-1, act.sum())
    meta = {"scale": scale, "mask_rate": mask_rate, "n_obs": int(len(train_rows)),
            "n_persons": int(g.N), "n_dyad_pairs": int(len(g.first)),
            "waves": sorted(int(w) for w in np.unique(prob.wave[train_rows])),
            "active_model_columns": [c for c, a_ in zip(cols, act) if a_],
            "marginal_mean_model": bm.mean(0).tolist(),
            "marginal_cov_model": np.cov(bm, rowvar=False).tolist() if len(bm) > 2 else None,
            "rhat": res.rhat() if gibbs.chains > 1 else None, "accept": res.accept,
            "sampler": {"iterations": gibbs.iterations, "burn_in": gibbs.burn_in, "chains": gibbs.chains,
                        "covariance": S.covariance},
            "posterior_sd": {"sigma_a2": float(res.draws["sigma_a2"].std()),
                             "sigma_b2": float(res.draws["sigma_b2"].std()),
                             "sigma_ab": float(res.draws["sigma_ab"].std()), "rho": float(res.draws["rho"].std())}}
    return LockedSRRM(beta=beta_raw, sigma_a2=float(res.mean("sigma_a2")), sigma_b2=float(res.mean("sigma_b2")),
                      sigma_ab=float(res.mean("sigma_ab")), rho=float(res.mean("rho")), standardizer=std,
                      ident=prob.ident, feature_cfg=S.feature_cfg, meta=meta)


def predict_rows(model: LockedSRRM, prob: Problem, rows: np.ndarray) -> pd.DataFrame:
    """a0, m0 and the eta blocks for `rows` (m0 uses the reverse row when it is present)."""
    X = prob.raw_design(rows, model.standardizer)
    bl = eta_blocks(X, model)
    eta = bl.eta
    out = pd.DataFrame({"row": rows, "wave": prob.wave[rows], "chooser": prob.a[rows], "chosen": prob.p[rows],
                        "y": prob.y[rows], "match": prob.match[rows], "eta": eta, "a0": a0(eta, model.s),
                        "eta0": bl.eta0, "etaP_fit": bl.etaP_fit, "etaP_resid": bl.etaP_resid,
                        "etaP_actor": bl.etaP_actor, "etaG": bl.etaG, "fit": X[:, COL["fit"]]})
    pos = {r: k for k, r in enumerate(rows)}
    rev = np.array([pos.get(prob.reverse[r], -1) for r in rows])
    eta_rev = np.where(rev >= 0, eta[np.maximum(rev, 0)], np.nan)
    out["eta_rev"] = eta_rev
    out["m0"] = np.where(rev >= 0, m0(eta, np.nan_to_num(eta_rev), model.r, model.s), np.nan)
    return out


def crossfit(prob: Problem, waves: list[int] | None = None, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outer LOWO: every held-out wave is scored by a model selected (nested) and fitted without it."""
    S = prob.settings
    all_w = np.unique(prob.wave)
    waves = list(all_w) if waves is None else waves
    preds, cfgs = [], []
    for k, w in enumerate(waves):
        tr = np.flatnonzero(prob.wave != w)
        te = np.flatnonzero(prob.wave == w)
        sel = select_config(prob, tr, seed=S.seed + int(w))
        model = fit_srrm(prob, tr, sel["scale"], sel["mask_rate"], S.crossfit_gibbs, seed=S.seed + int(w))
        pr = predict_rows(model, prob, te)
        preds.append(pr)
        cfgs.append({"wave": int(w), "scale": sel["scale"], "mask_rate": sel["mask_rate"],
                     "inner_log_loss": sel["best_loss"], "s": model.s, "r": model.r,
                     "sigma_a2": model.sigma_a2, "sigma_b2": model.sigma_b2, "sigma_ab": model.sigma_ab,
                     "rho": model.rho, "n_test": int(len(te))})
        if verbose:
            print(f"  wave {w:>2} ({k + 1}/{len(waves)}): c={sel['scale']}, q={sel['mask_rate']}, "
                  f"s={model.s:.3f}, r={model.r:+.3f}, test log-loss={log_loss(pr['y'].values, pr['a0'].values):.4f}",
                  flush=True)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(cfgs)


def estimate_srrm(prob: Problem, rows: np.ndarray | None = None, verbose: bool = True) -> LockedSRRM:
    """Final locked fit on the development rows (spec signature estimate_srrm(dev_rows, P))."""
    S = prob.settings
    rows = np.arange(len(prob.y)) if rows is None else rows
    sel = select_config(prob, rows, seed=S.seed)
    model = fit_srrm(prob, rows, sel["scale"], sel["mask_rate"], S.final_gibbs, seed=S.seed, verbose=verbose)
    model.meta["selection_table"] = sel["table"].to_dict(orient="records")
    model.meta["data_sha256"] = prob.data.sha256
    model.meta["reader_hash"] = prob.art.hash
    return model


def laplace_update(model: LockedSRRM, X_raw: np.ndarray, y: np.ndarray) -> LockedSRRM:
    """Update the locked marginal coefficients with a new labelled batch (Laplace step, Chapelle & Li 2011):
    Gaussian prior N(mean, cov) from the posterior draws, probit likelihood of the batch."""
    meta = model.meta
    if meta.get("marginal_cov_model") is None:
        raise ValueError("the locked model has no posterior covariance (run more than 2 kept draws)")
    cols = model.ident.model_columns()
    act = np.array([c in meta["active_model_columns"] for c in cols])
    mean = np.asarray(meta["marginal_mean_model"])
    cov = np.atleast_2d(np.asarray(meta["marginal_cov_model"])) + 1e-9 * np.eye(int(act.sum()))
    Xm = model.ident.to_model(np.atleast_2d(X_raw))[:, act]
    b, H = probit_map(Xm, np.asarray(y, int), prior_mean=mean, prior_prec=np.linalg.inv(cov), b0=mean)
    b_model = np.zeros(len(cols))
    b_model[act] = b * model.s
    new_meta = dict(meta, marginal_mean_model=b.tolist(), marginal_cov_model=np.linalg.inv(H).tolist(),
                    laplace_updates=int(meta.get("laplace_updates", 0)) + 1)
    return replace(model, beta=model.ident.to_raw_coef(b_model), meta=new_meta)


def probit_loglik(y: np.ndarray, eta_over_s: np.ndarray) -> float:
    q = 2 * y - 1
    return float(np.sum(log_ndtr(q * eta_over_s)))
