"""Load and validate `config/params.yaml` into a frozen `Params` model (stage1.md, section 10.6).

A wrong type, a missing key or an unknown key fails immediately with `pydantic.ValidationError`.
`Params.config_hash` is the SHA-256 of the normalized content (key-sorted JSON of the validated model),
so comments and whitespace in the YAML do not change it while any value does.
The stage1 block only holds procedures, grids, priors and scale definitions: no coefficient lives here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS = PROJECT_ROOT / "config" / "params.yaml"

TRAITS = ("O", "C", "E", "A", "N")          # Big Five as measured
DOMAINS = ("O", "C", "E", "A", "ES")        # common signed scale, ES = -N (section 3.2)


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# -------------------------------------------------------------------------------------------- data


class DatasetCfg(_M):
    file: str = Field(min_length=1)
    url: str = Field(min_length=1)
    md5: str | None = None


class DataCfg(_M):
    raw_dir: str
    sealed_dir: str
    datasets: dict[str, DatasetCfg]
    test_waves: list[int] = Field(default_factory=list)
    exclude_waves: list[int] = Field(default_factory=list)


class AnalysisCfg(_M):
    bootstrap_B: int = Field(ge=100)
    alpha: float = Field(gt=0.0, lt=1.0)
    delta_auc_min: float = Field(gt=0.0, lt=0.5)
    ece_bins: int = Field(ge=2)


class BfProxyCfg(_M):
    method: Literal["unit_weighted_composite"]
    min_items_per_trait: int = Field(ge=1)


# ------------------------------------------------------------------------------------------ stage1


class SamplerCfg(_M):
    name: Literal["hoff_gibbs"]
    chains: int = Field(ge=1)
    rhat_max: float = Field(gt=1.0)
    iter: int = Field(ge=50)
    burn: int = Field(ge=0)
    thin: int = Field(ge=1)
    crossfit_chains: int = Field(ge=1)

    @model_validator(mode="after")
    def _burn(self) -> SamplerCfg:
        if self.burn >= self.iter:
            raise ValueError("sampler.burn must be < sampler.iter")
        return self


class CoreCfg(_M):
    model: Literal["srrm_probit"]
    multiplicative_rank: Literal[0]
    sampler: SamplerCfg


ACTOR_FEATURES = ("gender", "bf", "age", "self_ratings")
PARTNER_FEATURES = ("bf", "age", "self_ratings")
DYAD_FEATURES = ("same_goal", "int_corr", "jaccard_top5", "abs_age_gap", "same_field")


class FeaturesCfg(_M):
    actor: list[str]
    partner: list[str]
    dyad: list[str]
    bf_similarity: bool
    bf_standardize: Literal["z_within_fold"]
    bf_reverse: list[Literal["O", "C", "E", "A", "N"]]
    trait_block_missing_if_ge: int = Field(ge=1, le=5)
    int_corr_min_common_items: int = Field(ge=2, le=17)

    @model_validator(mode="after")
    def _known(self) -> FeaturesCfg:
        for name, allowed, got in (("actor", ACTOR_FEATURES, self.actor),
                                   ("partner", PARTNER_FEATURES, self.partner),
                                   ("dyad", DYAD_FEATURES, self.dyad)):
            bad = set(got) - set(allowed)
            if bad:
                raise ValueError(f"features.{name}: unknown {sorted(bad)}; allowed {allowed}")
        return self


class PersonalityFitCfg(_M):
    demand_scale: Literal["onet_wi"]
    demand_centering: Literal["none", "z_over_catalog"]
    slope: Literal["shared", "per_trait"]
    single_context_identification: Literal["delta_orthogonal_to_demand", "delta_zero"]


class ContextSdCfg(_M):
    family: Literal["half_cauchy"]
    scale_grid: list[float]


class VarianceScaleCfg(_M):
    family: Literal["half_cauchy"]
    scale: float = Field(gt=0.0)


class PriorsCfg(_M):
    coef_center: Literal[0]
    scale_grid: list[float]
    intercept_sd: float = Field(gt=0.0)
    context_sd: ContextSdCfg
    covariance: Literal["lkj_eta_2", "inverse_wishart"]
    variance_scale: VarianceScaleCfg

    @field_validator("scale_grid")
    @classmethod
    def _grid(cls, v: list[float]) -> list[float]:
        if not v or any(x <= 0 for x in v) or any(b <= a for a, b in zip(v, v[1:])):
            raise ValueError("scale_grid must be positive and strictly increasing")
        return v


class SelectionCfg(_M):
    cv: Literal["nested_lowo"]
    criterion: Literal["log_loss"]
    tie_rule: Literal["one_se_prefer_shrinkage"]
    inner_estimator: Literal["laplace_map"]


class UpdateCfg(_M):
    method: Literal["laplace_batch"]


class LexiconRef(_M):
    path: str
    sha256: str | None = None
    min_entries_per_style_lang: int = Field(ge=1)


class CatalogRef(_M):
    path: str
    sha256: str | None = None
    generic_prior: Literal["mean_of_catalog"]


class CueLevelsCfg(_M):
    classes: list[Literal["neg", "rev", "weak", "base", "strong"]]
    order: list[Literal["neg", "weak", "base", "strong"]]
    estimator: Literal["pav_least_squares"]
    per_style: Literal["select_by_group_cv_1se", "shared", "per_style"]


class PriorStrengthCfg(_M):
    estimator: Literal["buhlmann_nonparametric"]
    per_style: Literal["select_by_group_cv_1se", "shared", "per_style"]
    sensitivity_grid: list[float]


class FallbackEmbeddingCfg(_M):
    enabled: bool
    model: str
    threshold: Literal["conformal"]
    target_error: float = Field(gt=0.0, lt=1.0)


class ContextReaderCfg(_M):
    lexicon: LexiconRef
    catalog: CatalogRef
    ngram_max: int = Field(ge=1, le=8)
    toneless_match: Literal["unique_only", "off"]
    multi_match: Literal["longest_then_mean"]
    cue_levels: CueLevelsCfg
    prior_strength: PriorStrengthCfg
    style_icc_min: float = Field(ge=0.0, le=1.0)
    cv: Literal["group_kfold_by_basic_context"]
    no_text_fallback: Literal["event_then_market"]
    fallback_embedding: FallbackEmbeddingCfg


class LockedCfg(_M):
    manifest: str


class Stage1Cfg(_M):
    core: CoreCfg
    features: FeaturesCfg
    personality_fit: PersonalityFitCfg
    priors: PriorsCfg
    selection: SelectionCfg
    block_masking_rate_grid: list[float]
    update: UpdateCfg
    context_reader: ContextReaderCfg
    locked: LockedCfg

    @field_validator("block_masking_rate_grid")
    @classmethod
    def _rates(cls, v: list[float]) -> list[float]:
        if not v or any(not 0.0 <= x < 1.0 for x in v):
            raise ValueError("block_masking_rate_grid values must be in [0, 1)")
        return v


# -------------------------------------------------------------------------------------------- root


class Params(_M):
    config_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    global_seed: int
    data: DataCfg
    analysis: AnalysisCfg
    bf_proxy: BfProxyCfg
    stage1: Stage1Cfg

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def resolve_path(path: str | Path) -> Path:
    """Relative paths in params.yaml are relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_params(path: str | Path = DEFAULT_PARAMS) -> Params:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return Params.model_validate(raw)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
