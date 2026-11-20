"""Central twin (report "nang cap twin", part B): a fine-tuned LLM is the predictor; LR / MF are its inputs.

    logit(p, j) = head( LLM_LoRA( [proj_u(u_p), proj_v(v_j), tokens(persona + own related answers + question)] ) )

* Person / question soft tokens: the MF factors u_p, v_j of the fold's B2 fit, mapped to the LLM embedding
  space by learned linear projections and prepended to the text (CoLLM: collaborative embeddings as tokens of
  a LoRA-tuned LLM; Kim & Lee: person / question embeddings inside a fine-tuned LLM, GSS AUC 0.86 / 0.73).
* Text: demographics, top-k related answers of the same person (T4), the question and its target answers.
* Output: a linear head on the last hidden state, trained with log-loss on labels (the LLM is a supervised
  person model, not a zero-shot guesser; tuning-free persona prompts lose to LR, design section 1-2).
* Training episodes mimic the regime: the regime generator is applied to the TRAINING cells of outer fold 0
  (`make_folds(train0, regime)`), the twin learns to predict the inner held-out cells from the inner
  training cells only (evidence, MF factors), as DropoutNet trains on simulated cold starts. Persons of the
  episode set are split 90 / 10 into training / validation.
* Everything that is not an architecture choice is learned on validation log-loss: the number of steps
  (early stopping with patience), the calibration (Platt on the validation cells). LoRA rank / learning rate
  are fixed defaults of QLoRA practice (Dettmers et al. 2023) unless a grid is given.
* Evaluation: `n_eval` cells of the outer fold-0 test set, context from the outer fold-0 training cells; the
  comparison with LR and the stack (twin + LR, convex weights cross-fitted by person) is on these cells.

QLoRA (4-bit NF4 + LoRA) when CUDA and bitsandbytes are available; `model="tiny-random"` builds a random
2-layer Qwen2 with a byte tokenizer (tests, CPU dry runs; not a language model).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from lrmf_twin.data.twin2k import SurveyData
from lrmf_twin.features import answer_features, centred_demographics
from lrmf_twin.regimes import make_folds
from lrmf_twin.t4_retrieval import evidence_lines, normalise, retrieve, short_question
from lrmf_twin.tuning import fit_b2, fit_lr


@dataclass
class FTConfig:
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    quant: str = "auto"             # auto: 4-bit when CUDA + bitsandbytes, else full precision
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 2e-4
    batch_size: int = 8
    max_steps: int = 400
    eval_every: int = 50
    patience: int = 3
    max_len: int = 384
    n_train: int = 4000
    n_val: int = 500
    n_eval: int = 800
    soft_tokens: bool = True
    top_k: int = 5
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


# -- text -----------------------------------------------------------------------------------------------------
def build_text(data: SurveyData, p: int, j: int, evidence: list[int]) -> str:
    item = data.items[j]
    lines = [f"Respondent: {data.demo_text[p]}"]
    if evidence:
        lines.append("Their answers to related questions:")
        lines += evidence_lines(data, p, evidence)
    lines.append(f"Question: {short_question(data, j, stem_chars=300)}")
    opts = "; ".join(o if len(o) <= 80 else o[:80] + "..." for o in item.options)
    target = " or ".join(item.positive_labels())
    lines.append(f"Options: {opts}")
    lines.append(f"Does this respondent answer: {target[:200]}? Answer:")
    return "\n".join(lines)


# -- model ----------------------------------------------------------------------------------------------------
def _tiny_backbone():
    from transformers import ByT5Tokenizer, Qwen2Config, Qwen2Model
    tok = ByT5Tokenizer()
    cfg = Qwen2Config(vocab_size=tok.vocab_size + 10, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=2048)
    torch.manual_seed(0)
    return tok, Qwen2Model(cfg)


def load_backbone(cfg: FTConfig):
    """Tokenizer + LoRA-wrapped backbone (no LM head)."""
    from peft import LoraConfig, get_peft_model
    if cfg.model == "tiny-random":
        tok, model = _tiny_backbone()
        quant = False
    else:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.model)
        quant = cfg.quant in ("auto", "4bit") and torch.cuda.is_available()
        kw = {}
        if quant:
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401
                kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_compute_dtype=torch.float16,
                                                               bnb_4bit_use_double_quant=True)
                kw["device_map"] = {"": 0}
            except ImportError:
                quant = False
        if not quant:
            kw["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModel.from_pretrained(cfg.model, **kw)
        if quant:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        elif torch.cuda.is_available():
            model = model.to("cuda")
            model.gradient_checkpointing_enable()
    tok.padding_side = "right"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules="all-linear", bias="none")
    model = get_peft_model(model, lora)
    return tok, model, quant


class TwinModel(torch.nn.Module):
    def __init__(self, backbone, hidden: int, rank: int, soft: bool):
        super().__init__()
        self.backbone = backbone
        self.soft = soft and rank > 0
        if self.soft:
            self.proj_u = torch.nn.Linear(rank, hidden)
            self.proj_v = torch.nn.Linear(rank, hidden)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, u=None, v=None):
        emb_layer = self.backbone.get_input_embeddings()
        emb = emb_layer(input_ids)
        mask = attention_mask
        if self.soft:
            tok = torch.stack([self.proj_u(u), self.proj_v(v)], dim=1).to(emb.dtype)
            emb = torch.cat([tok, emb], dim=1)
            mask = torch.cat([torch.ones(mask.shape[0], 2, dtype=mask.dtype, device=mask.device), mask], dim=1)
        out = self.backbone(inputs_embeds=emb, attention_mask=mask)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        last = mask.sum(1) - 1
        h = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        return self.head(h.float()).squeeze(-1)


# -- episodes -------------------------------------------------------------------------------------------------
def _examples(data, cells, train_mask, En, U, V, top_k):
    out = []
    for p, j in cells:
        ev = retrieve(int(p), int(j), train_mask, En, k=top_k)
        out.append({"p": int(p), "j": int(j), "y": float(data.Y[p, j]), "text": build_text(data, int(p), int(j), ev),
                    "u": U[p].astype(np.float32), "v": V[j].astype(np.float32)})
    return out


def _sample(mask: np.ndarray, n: int, rng) -> np.ndarray:
    cells = np.argwhere(mask)
    if len(cells) > n:
        cells = cells[rng.choice(len(cells), n, replace=False)]
    return cells


def build_episodes(data: SurveyData, E: np.ndarray, regime: str, ctx: dict, tuned: dict, base_cfg, cfg: FTConfig,
                   n_sparse: int, n_known: int, k_cold: int, seed: int, log=print) -> dict:
    rng = np.random.default_rng(seed + 17)
    En = normalise(E)
    Xd = centred_demographics(data.X_demo)
    inner = make_folds(ctx["train"], regime, n_folds=5, seed=seed + 211, n_sparse=max(20, n_sparse),
                       n_known=n_known, k_cold=k_cold, wave=data.wave)[0]
    A = answer_features(data, inner.train)
    lr = fit_lr(data.Y, inner.train, Xd, A, tuned["lr_C"], base_cfg)
    b2 = fit_b2(data.Y, inner.train, Xd, A, tuned["lr_C"], tuned["mf"], base_cfg, lr)
    comp = b2.components()
    b2.release()
    if b2 is not lr:
        lr.release()
    cells = _sample(inner.test, cfg.n_train + cfg.n_val, rng)
    persons = np.unique(cells[:, 0])
    val_p = set(rng.choice(persons, max(1, int(round(0.1 * persons.size))), replace=False).tolist())
    is_val = np.array([p in val_p for p in cells[:, 0]])
    train_ex = _examples(data, cells[~is_val], inner.train, En, comp["U"], comp["V"], cfg.top_k)
    val_ex = _examples(data, cells[is_val], inner.train, En, comp["U"], comp["V"], cfg.top_k)
    eval_cells = _sample(ctx["test"], cfg.n_eval, rng)
    eval_ex = _examples(data, eval_cells, ctx["train"], En, ctx["U"], ctx["V"], cfg.top_k)
    log(f"    twin-FT {regime}: {len(train_ex)} training / {len(val_ex)} validation episodes "
        f"(inner {regime} split of fold-0 training cells), {len(eval_ex)} evaluation cells (fold-0 test)")
    return {"train": train_ex, "val": val_ex, "eval": eval_ex, "rank": comp["V"].shape[1],
            "eval_cells": eval_cells}


# -- training -------------------------------------------------------------------------------------------------
def _batches(examples, size, rng=None):
    order = np.arange(len(examples)) if rng is None else rng.permutation(len(examples))
    for s in range(0, len(order), size):
        yield [examples[i] for i in order[s:s + size]]


def _encode(tok, batch, max_len, device):
    enc = tok([b["text"] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    u = torch.as_tensor(np.stack([b["u"] for b in batch]), device=device) if batch[0]["u"].size else None
    v = torch.as_tensor(np.stack([b["v"] for b in batch]), device=device) if batch[0]["v"].size else None
    y = torch.as_tensor([b["y"] for b in batch], dtype=torch.float32, device=device)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), u, v, y


@torch.no_grad()
def _predict(model, tok, examples, cfg, device):
    model.eval()
    out = []
    for batch in _batches(examples, cfg.batch_size * 2):
        ids, mask, u, v, _ = _encode(tok, batch, cfg.max_len, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(ids, mask, u, v).float().cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros(0)


def _ll(y, z):
    z = np.clip(z, -30, 30)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def train_twin(episodes: dict, cfg: FTConfig, log=print) -> dict:
    from lrmf_twin.combine import fit_platt
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    tok, backbone, quant = load_backbone(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = backbone.get_input_embeddings().weight.shape[1]
    model = TwinModel(backbone, hidden, episodes["rank"], cfg.soft_tokens)
    for mod in (getattr(model, "proj_u", None), getattr(model, "proj_v", None), model.head):
        if mod is not None:
            mod.to(device)
    if not quant and cfg.model != "tiny-random":
        model.backbone.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    y_val = np.array([e["y"] for e in episodes["val"]])
    best = {"ll": math.inf, "step": 0, "state": None}
    curve, bad, step = [], 0, 0
    model.train()
    while step < cfg.max_steps and bad < cfg.patience:
        for batch in _batches(episodes["train"], cfg.batch_size, rng):
            ids, mask, u, v, y = _encode(tok, batch, cfg.max_len, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                z = model(ids, mask, u, v)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % cfg.eval_every == 0 or step >= cfg.max_steps:
                zv = _predict(model, tok, episodes["val"], cfg, device)
                ll = _ll(y_val, zv) if len(zv) else math.inf
                curve.append({"step": step, "val_logloss": ll, "train_loss": float(loss.detach())})
                log(f"      step {step}: train loss {float(loss.detach()):.4f}, validation log-loss {ll:.4f}")
                if ll < best["ll"] - 1e-4:
                    best = {"ll": ll, "step": step,
                            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                                      if any(t in k for t in ("lora_", "proj_u", "proj_v", "head"))}}
                    bad = 0
                else:
                    bad += 1
            if step >= cfg.max_steps or bad >= cfg.patience:
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)
    zv = _predict(model, tok, episodes["val"], cfg, device)
    platt = fit_platt(1 / (1 + np.exp(-zv)), y_val) if len(zv) else np.array([0.0, 1.0])
    ze = _predict(model, tok, episodes["eval"], cfg, device)
    z_eval = platt[0] + platt[1] * ze
    del model, backbone, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"z_eval": z_eval, "best_step": best["step"], "best_val_logloss": best["ll"], "curve": curve,
            "platt": [float(platt[0]), float(platt[1])], "seconds": time.time() - t0, "quantized": quant}


def run_twin_ft(data: SurveyData, E: np.ndarray, regime: str, ctx: dict, tuned: dict, base_cfg, cfg: FTConfig,
                n_sparse: int, n_known: int, k_cold: int, seed: int, log=print) -> dict:
    episodes = build_episodes(data, E, regime, ctx, tuned, base_cfg, cfg, n_sparse, n_known, k_cold, seed, log)
    fit = train_twin(episodes, cfg, log)
    cells = episodes["eval_cells"]
    rows = []
    for (p, j), z in zip(cells, fit["z_eval"]):
        rows.append({"regime": regime, "person": int(p), "pid": int(data.pids[p]), "item": int(j),
                     "y": float(data.Y[p, j]), "z_twin": float(z),
                     **{f"p_{m}": float(ctx["preds"][m][p, j]) for m in ctx["preds"]}})
    log(f"    twin-FT {regime}: best step {fit['best_step']}, validation log-loss {fit['best_val_logloss']:.4f}, "
        f"{fit['seconds']:.0f}s{' (4-bit)' if fit['quantized'] else ''}")
    return {"rows": rows, "fit": {k: v for k, v in fit.items() if k != "z_eval"}, "config": cfg.as_dict()}
