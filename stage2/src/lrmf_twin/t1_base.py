"""T1 frozen LR + MF base, with the T2 semantic item heads and the T3 sparse-person refit.

Logit of cell (p, j), dense over the whole person x item matrix:

    eta[p, j] = b_j + x_p . W_j + sum_{k != j} a_pk Theta_kj      <- LR: item fixed effect + item-specific
                                                                      slopes on demographics x_p and on the
                                                                      person's other answers a_p (features.py)
              + c_p + u_p . v_j                                    <- MF: person bias + rank-r factors

T2 (design V2): with `semantic=True` the item parameters are "learned part + semantic part",
(b_j, v_j) = free_j + g(e_j) (and W_j too with sem_targets="bwv"), where e_j is the item-text embedding and
g a small MLP (non-linear adapter, as in KAR). The free part is penalised free_mult times harder, so the
text explains what it can. An item without any training answer (R2) has free_j = 0: it is predicted from its
text only.

T3 (design V3): `refit_persons` re-estimates (c_p, u_p) of sparse persons with the item parameters frozen,
adding LLM pseudo-answers on anchor items as soft labels weighted by w(n_p) = n0 / (n0 + n_p): the person
factor shrinks toward the LLM prior instead of toward 0, less so as the person's own answers accumulate.

Everything is fitted by full-batch L-BFGS on the mean penalised log-loss of the fold's training cells, on the
GPU when one is available. With `rank=0, person_bias=False` the model is exactly one L2 logistic regression
per question (the objective is separable by column), which `baselines.torch_lr` uses as the scalable LR.
When there are more than `max_feat_items` questions, the "other answers" features of Theta are restricted to
the `max_feat_items` questions with most training answers (MovieLens: the most rated movies) [quy ước].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

torch.set_default_dtype(torch.float32)


@dataclass
class BaseConfig:
    rank: int = 16
    use_theta: bool = True
    semantic: bool = False
    lam_b: float = 1.0
    lam_W: float = 100.0         # tuned on an inner split of R1 fold-0 training cells
    lam_theta: float = 200.0     # = 1/C, sklearn per-item LR best C ~ 0.005
    lam_c: float = 10.0
    lam_U: float = 10.0
    lam_V: float = 10.0
    sem_targets: str = "bv"         # T2 heads: "bv" = item effect + MF factor (design T2), "bwv" adds W
    free_mult: float = 3.0          # with T2: the free part of the predicted params gets free_mult x penalty
    lam_mlp: float = 10.0
    sem_hidden: int = 64
    max_iter: int = 400
    seed: int = 0
    person_bias: bool = True
    max_feat_items: int = 800
    device: str = "auto"            # "auto" = cuda when available

    def as_dict(self) -> dict:
        return asdict(self)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _t(x: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device)


class LRMFBase:
    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg
        self.fitted = False

    # -- parameters -------------------------------------------------------------------------------------
    def _item_params(self):
        b, W, V = self.b, self.W, self.V
        if self.cfg.semantic:
            g = self.mlp(self.E)                                      # (J, 1 [+ D] + r)
            b = b + g[:, 0]
            V = V + g[:, -V.shape[1]:]
            if self.cfg.sem_targets == "bwv":
                W = W + g[:, 1:1 + W.shape[0]].T
        return b, W, V

    def _lr_part(self, Xd: torch.Tensor, A: torch.Tensor, b, W) -> torch.Tensor:
        Z = b[None, :] + Xd @ W
        if self.cfg.use_theta:
            Z = Z + A[:, self.feat_idx] @ (self.Theta * self.offdiag)
        return Z

    def _logits(self) -> torch.Tensor:
        b, W, V = self._item_params()
        return self._lr_part(self.Xd, self.A, b, W) + self.c[:, None] + self.U @ V.T

    # -- fit --------------------------------------------------------------------------------------------
    def fit(self, Y: np.ndarray, train: np.ndarray, Xd: np.ndarray, A: np.ndarray,
            E: np.ndarray | None = None) -> "LRMFBase":
        cfg = self.cfg
        dev = self.dev = _device(cfg.device)
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        P, J = Y.shape
        D = Xd.shape[1]
        r = cfg.rank
        self.Xd, self.A = _t(Xd, dev), _t(A, dev)
        self.y = _t(np.nan_to_num(Y, nan=0.0), dev)
        self.m = _t(train.astype(np.float32), dev)
        n = max(float(train.sum()), 1.0)

        self.b = torch.zeros(J, device=dev, requires_grad=True)
        self.W = torch.zeros(D, J, device=dev, requires_grad=True)
        self.c = torch.zeros(P, device=dev, requires_grad=cfg.person_bias)
        self.U = (0.1 * torch.randn(P, r, generator=gen)).to(dev).requires_grad_(True)
        self.V = (0.1 * torch.randn(J, r, generator=gen)).to(dev).requires_grad_(True)
        params = [self.b, self.W, self.U, self.V] + ([self.c] if cfg.person_bias else [])
        if cfg.use_theta:
            n_j = train.sum(0)
            feat = np.arange(J) if J <= cfg.max_feat_items else np.sort(np.argsort(-n_j, kind="stable")[:cfg.max_feat_items])
            self.feat_idx = torch.as_tensor(feat, device=dev)
            self.Theta = torch.zeros(len(feat), J, device=dev, requires_grad=True)
            self.offdiag = torch.ones(len(feat), J, device=dev)
            self.offdiag[torch.arange(len(feat), device=dev), self.feat_idx] = 0.0
            params.append(self.Theta)
        if cfg.semantic:
            if E is None:
                raise ValueError("semantic=True needs item embeddings E")
            self.E = _t(E, dev)
            self.mlp = torch.nn.Sequential(torch.nn.Linear(E.shape[1], cfg.sem_hidden), torch.nn.Tanh(),
                                           torch.nn.Linear(cfg.sem_hidden, 1 + r + (D if cfg.sem_targets == "bwv" else 0))).to(dev)
            with torch.no_grad():
                self.mlp[2].weight.mul_(0.1)
                self.mlp[2].bias.zero_()
            params += list(self.mlp.parameters())

        bce = torch.nn.BCEWithLogitsLoss(weight=self.m, reduction="sum")
        k = cfg.free_mult if cfg.semantic else 1.0
        kW = k if cfg.sem_targets == "bwv" else 1.0

        def objective() -> torch.Tensor:
            loss = bce(self._logits(), self.y)
            reg = (cfg.lam_c * self.c.pow(2).sum() + cfg.lam_U * self.U.pow(2).sum()
                   + k * cfg.lam_b * self.b.pow(2).sum()
                   + kW * cfg.lam_W * self.W.pow(2).sum()
                   + k * cfg.lam_V * self.V.pow(2).sum())
            if cfg.use_theta:
                reg = reg + cfg.lam_theta * self.Theta.pow(2).sum()
            if cfg.semantic:
                reg = reg + cfg.lam_mlp * sum(p.pow(2).sum() for p in self.mlp.parameters())
            return (loss + 0.5 * reg) / n

        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg.max_iter, history_size=20,
                                tolerance_grad=1e-7, tolerance_change=1e-10, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            value = objective()
            value.backward()
            return value

        opt.step(closure)
        with torch.no_grad():
            self.final_loss = float(objective())
        self.fitted = True
        return self

    # -- outputs ----------------------------------------------------------------------------------------
    @torch.no_grad()
    def logits(self) -> np.ndarray:
        return self._logits().cpu().numpy()

    @torch.no_grad()
    def item_factors(self) -> np.ndarray:
        return self._item_params()[2].cpu().numpy()

    def release(self) -> None:
        """Free the fold's tensors (GPU memory) once its predictions have been read."""
        for name in ("Xd", "A", "y", "m", "Theta", "offdiag", "E", "U", "V", "W", "b", "c", "mlp"):
            if hasattr(self, name):
                delattr(self, name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- T3 ---------------------------------------------------------------------------------------------
    def refit_persons(self, persons: np.ndarray, pseudo: dict[int, tuple[np.ndarray, np.ndarray]],
                      n0: float = 10.0) -> np.ndarray:
        """Re-estimate (c_p, u_p) of `persons` with LLM soft labels; returns their new logit rows (len, J).

        pseudo[p] = (item indices, probabilities in [0, 1]) produced by T3 for person p.
        """
        cfg = self.cfg
        persons = np.asarray(persons, dtype=int)
        if persons.size == 0:
            return np.zeros((0, self.y.shape[1]), dtype=np.float32)
        with torch.no_grad():
            b, W, V = self._item_params()
            idx = torch.as_tensor(persons, device=self.dev)
            fixed = self._lr_part(self.Xd[idx], self.A[idx], b, W)
            V = V.detach()
        y, m = self.y[idx], self.m[idx]
        n_p = m.sum(1)
        q = torch.zeros_like(y)
        w = torch.zeros_like(y)
        for row, p in enumerate(persons):
            if p in pseudo:
                items, probs = pseudo[p]
                q[row, torch.as_tensor(items, device=self.dev)] = _t(np.asarray(probs, dtype=np.float32), self.dev)
                w[row, torch.as_tensor(items, device=self.dev)] = float(n0 / (n0 + float(n_p[row])))
        c = self.c.detach()[idx].clone().requires_grad_(True)
        U = self.U.detach()[idx].clone().requires_grad_(True)
        bce = torch.nn.functional.binary_cross_entropy_with_logits

        def objective():
            Z = fixed + c[:, None] + U @ V.T
            loss = (bce(Z, y, weight=m, reduction="sum") + bce(Z, q, weight=w, reduction="sum"))
            reg = cfg.lam_c * c.pow(2).sum() + cfg.lam_U * U.pow(2).sum()
            return (loss + 0.5 * reg) / max(float(m.sum() + w.sum()), 1.0)

        opt = torch.optim.LBFGS([c, U], lr=1.0, max_iter=200, history_size=10, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            value = objective()
            value.backward()
            return value

        opt.step(closure)
        with torch.no_grad():
            return (fixed + c[:, None] + U @ V.T).cpu().numpy()
