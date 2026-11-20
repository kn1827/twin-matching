"""Load and validate `config/params.yaml` into a `Params` model (Tech spec, section 2).

A wrong type, a missing key or an unknown key fails immediately with `pydantic.ValidationError`.
`Params.config_hash` is the SHA-256 of the normalized content (key-sorted JSON of the validated
model), so editing comments or whitespace in the YAML keeps the hash, changing any value changes it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS = PROJECT_ROOT / "config" / "params.yaml"

TRAITS = ("O", "C", "E", "A", "N")
Role = Literal["twin", "judge", "warmth", "bfi_proxy"]
ROLES: tuple[Role, ...] = ("twin", "judge", "warmth", "bfi_proxy")

_SUM_TOL = 1e-9


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_sum_one(values: dict[str, float], what: str) -> None:
    total = math.fsum(values.values())
    if abs(total - 1.0) > _SUM_TOL:
        raise ValueError(f"{what}: must sum to 1, got {total}")


def _check_increasing(values: tuple[float, ...] | list[float], what: str) -> None:
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"{what}: must be strictly increasing, got {list(values)}")


# --------------------------------------------------------------------------------------------- llm


class RoleCfg(_M):
    provider: Literal["ollama", "gemini", "mock"]
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    samples: int = Field(default=1, ge=1)


class MaxOutputTokens(_M):
    module: int = Field(gt=0)
    synthesis: int = Field(gt=0)
    judge: int = Field(gt=0)
    warmth: int = Field(gt=0)
    bfi_proxy: int = Field(gt=0)


class Price(_M):
    """USD per 1M tokens; `null` means unknown (cost is stored as NULL)."""

    input: float | None = Field(default=None, ge=0.0)
    output: float | None = Field(default=None, ge=0.0)


class RetryCfg(_M):
    max_attempts: int = Field(ge=1)
    base_delay_s: float = Field(gt=0)
    max_delay_s: float = Field(gt=0)
    json_repair_attempts: int = Field(ge=0)


class OllamaCfg(_M):
    base_url: str = "http://localhost:11434"
    keep_alive: str = "30m"
    num_ctx: int | None = Field(default=None, gt=0)
    num_gpu: int | None = Field(default=None, ge=0)  # layers offloaded to GPU; 0 = CPU only, None = auto


class LLMCfg(_M):
    twin: RoleCfg
    judge: RoleCfg
    warmth: RoleCfg
    bfi_proxy: RoleCfg
    max_output_tokens: MaxOutputTokens
    price_usd_per_mtok: dict[Role, Price]
    retry: RetryCfg
    concurrency: int = Field(ge=1)
    requests_per_minute: int | None = Field(default=None, gt=0)
    timeout_s: float = Field(default=60.0, gt=0)
    cache_dir: str
    ollama: OllamaCfg = Field(default_factory=OllamaCfg)

    def role(self, role: Role) -> RoleCfg:
        cfg: RoleCfg = getattr(self, role)
        return cfg


# -------------------------------------------------------------------------------------------- core


class RagCfg(_M):
    k: int = Field(ge=1)
    max_l2sq: float = Field(gt=0)
    embedding_model: str


class EpisodicCfg(_M):
    max_snippets: int = Field(ge=0)
    in_prompt: int = Field(ge=0)


class CoreCfg(_M):
    tau_base: float
    tau_slope_O: float
    style_threshold: float = Field(ge=0.0, le=1.0)
    mask_cuts: tuple[float, float]
    d_cuts: tuple[float, float]
    risk_value: dict[Literal["Low", "Medium", "High"], float]
    rag: RagCfg
    episodic: EpisodicCfg
    history_in_prompt: int = Field(ge=1)
    response_max_words: int = Field(ge=1)

    @field_validator("mask_cuts", "d_cuts")
    @classmethod
    def _increasing(cls, v: tuple[float, float]) -> tuple[float, float]:
        _check_increasing(v, "cut points")
        return v

    @field_validator("risk_value")
    @classmethod
    def _all_levels(cls, v: dict[str, float]) -> dict[str, float]:
        if set(v) != {"Low", "Medium", "High"}:
            raise ValueError("risk_value needs Low, Medium and High")
        return v


class SessionCfg(_M):
    rounds_T: int = Field(ge=1)
    opener: Literal["seeded_random"]
    language: Literal["en", "vi"]
    modes: list[Literal["free", "full"]] = Field(min_length=1)


# ---------------------------------------------------------------------------------------- affinity

EtaKey = Literal["aff", "dir", "rec", "coh", "safe"]


class AffinityCfg(_M):
    eta: dict[EtaKey, float]
    kappa_plus: float = Field(gt=0)
    kappa_minus: float = Field(gt=0)
    b_E: float
    b_N: float
    b_A: float
    kappa_clip: tuple[float, float]
    e_bar: float = Field(ge=0.0, le=1.0)
    mask_center_A: float = Field(ge=0.0, le=1.0)
    mask_slope_s: float = Field(gt=0)
    nu_init: float = Field(ge=0.0, le=1.0)
    eps_a: float = Field(gt=0)
    bands: list[float] = Field(min_length=1)

    @field_validator("eta")
    @classmethod
    def _eta(cls, v: dict[str, float]) -> dict[str, float]:
        if set(v) != {"aff", "dir", "rec", "coh", "safe"}:
            raise ValueError("eta needs aff, dir, rec, coh and safe")
        if any(w < 0 for w in v.values()):
            raise ValueError("eta weights must be non-negative")
        _check_sum_one(v, "affinity.eta")
        return v

    @field_validator("kappa_clip")
    @classmethod
    def _clip(cls, v: tuple[float, float]) -> tuple[float, float]:
        if not 0.0 < v[0] < v[1] <= 1.0:
            raise ValueError(f"kappa_clip must satisfy 0 < lo < hi <= 1, got {v}")
        return v

    @field_validator("bands")
    @classmethod
    def _bands(cls, v: list[float]) -> list[float]:
        _check_increasing(v, "affinity.bands")
        if not all(0.0 < b < 1.0 for b in v):
            raise ValueError("affinity.bands must lie in (0, 1)")
        return v


# ------------------------------------------------------------------------------------------ stages


class Stage1Cfg(_M):
    mu: float = Field(ge=0.0, le=1.0)
    kappa1: float
    kappa2: float
    rho_P: float = Field(gt=0)
    rho_G: float = Field(gt=0)
    pi_by_context: dict[str, dict[Literal["O", "C", "E", "A", "N"], float]]

    @field_validator("pi_by_context")
    @classmethod
    def _rows(cls, v: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        for ctx, row in v.items():
            if set(row) != set(TRAITS):
                raise ValueError(f"pi_by_context.{ctx} needs O, C, E, A and N")
            _check_sum_one(row, f"stage1.pi_by_context.{ctx}")
        return v


class HalfLife(_M):
    P: float = Field(gt=0)
    G: float = Field(gt=0)
    I: float = Field(gt=0)  # noqa: E741 - symbol name from the proposal
    B: float = Field(gt=0)
    f: float = Field(gt=0)


class Stage3Cfg(_M):
    rho_I: float = Field(gt=0)
    rho_B: float = Field(gt=0)
    half_life_days: HalfLife
    sigma_u: float = Field(gt=0)
    T_ref: int = Field(ge=1)
    field_groups: dict[Literal["demographics", "goals", "interests", "self_ratings", "narrative"], float]

    @field_validator("field_groups")
    @classmethod
    def _groups(cls, v: dict[str, float]) -> dict[str, float]:
        if len(v) != 5:
            raise ValueError("field_groups needs all five groups")
        _check_sum_one(v, "stage3.field_groups")
        return v


class Stage4Cfg(_M):
    p: float
    z: float
    theta: float
    z_sigma: float
    kappa_T0: float
    b_default: int = Field(ge=1)
    ilp_time_limit_s: float = Field(gt=0)


class DataCfg(_M):
    raw_csv: str
    sealed_dir: str
    calib_dates: tuple[int, int]
    test_dates: tuple[int, int]
    exclude_waves: list[int]

    @field_validator("calib_dates", "test_dates")
    @classmethod
    def _range(cls, v: tuple[int, int]) -> tuple[int, int]:
        if not 0 < v[0] <= v[1]:
            raise ValueError(f"invalid pair-count range: {v}")
        return v


class AnalysisCfg(_M):
    bootstrap_B: int = Field(ge=1)
    alpha: float = Field(gt=0.0, lt=1.0)
    delta_auc_min: float = Field(ge=0.0)
    fr_b: int = Field(ge=1)


# ------------------------------------------------------------------------------------------ Params


class Params(_M):
    config_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    global_seed: int
    llm: LLMCfg
    core: CoreCfg
    session: SessionCfg
    affinity: AffinityCfg
    stage1: Stage1Cfg
    stage3: Stage3Cfg
    stage4: Stage4Cfg
    data: DataCfg
    analysis: AnalysisCfg

    @model_validator(mode="after")
    def _twin_has_price(self) -> Params:
        # The spec allows a role without a price (cost stored as NULL), but never the twin.
        if "twin" not in self.llm.price_usd_per_mtok:
            raise ValueError("llm.price_usd_per_mtok needs a price for the twin role")
        return self

    def canonical_json(self) -> str:
        """Normalized content: key-sorted JSON without extra whitespace."""
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
    """Read and validate params.yaml; any type error, missing or unknown key raises `ValidationError`."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: content must be a YAML mapping")
    return Params.model_validate(raw)
