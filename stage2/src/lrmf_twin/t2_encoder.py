"""T2 question encoder: item text -> embedding (offline, cached). The MLP adapter lives in t1_base.py.

Embedder specs:
* `st:<model>`      sentence-transformers model (default all-MiniLM-L6-v2, works offline once cached)
* `ollama:<model>`  Ollama /api/embed (e.g. `ollama:qwen2.5:3b`), the "LLM embedding" of the design
* `hash`            bag-of-words hashing, no model needed (tests / no network); not a semantic encoder

Ablation C3 (design 6.2): `shuffle_texts=True` permutes the texts across items before embedding, so any
gain that survives cannot come from question semantics.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

DEFAULT_EMBEDDER = "st:sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path("data/cache")


def _hash_embed(texts: list[str], dim: int = 256) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in re.findall(r"[a-z0-9$%]+", t.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            out[i, h % dim] += 1.0 if (h >> 20) & 1 else -1.0
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def _ollama_embed(texts: list[str], model: str, host: str) -> np.ndarray:
    vecs = []
    for start in range(0, len(texts), 32):
        body = json.dumps({"model": model, "input": texts[start:start + 32]}).encode()
        req = urllib.request.Request(f"{host}/api/embed", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            vecs.extend(json.loads(resp.read())["embeddings"])
    return np.asarray(vecs, dtype=np.float32)


def embed_items(texts: list[str], spec: str = DEFAULT_EMBEDDER, cache_dir: Path = CACHE_DIR,
                shuffle_texts: bool = False, seed: int = 0, ollama_host: str = "http://localhost:11434") -> np.ndarray:
    if shuffle_texts:
        perm = np.random.default_rng(seed).permutation(len(texts))
        texts = [texts[i] for i in perm]
    key = hashlib.sha256((spec + "\n" + "\n".join(texts)).encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"lrmf_twin_emb_{key}.npy"
    if path.exists():
        return np.load(path)
    if spec == "hash":
        E = _hash_embed(texts)
    elif spec.startswith("st:"):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(spec[3:], device="cpu")
        E = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    elif spec.startswith("ollama:"):
        E = _ollama_embed(texts, spec[len("ollama:"):], ollama_host)
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    else:
        raise ValueError(f"unknown embedder spec {spec!r}")
    E = np.asarray(E, dtype=np.float32)
    np.save(path, E)
    return E


# -- T2 v0.3: item-parameter generator + Buhlmann shrinkage (report "phuong an sua", A2) --------------------
class ItemGenerator:
    """Predicts an item's fitted parameters (b_j, v_j) from its text embedding.

    Non-linear adapter (RBF kernel ridge; KAR uses MLP adapters) trained only on items with enough answers,
    weighted by n_j / (n_j + 10); alpha and gamma are chosen by 5-fold cross-validation over ITEMS (the honest
    question "how well does text predict the parameters of an item it has not seen"). Same idea as the
    Meta-Embedding generator of Pan et al. (SIGIR 2019): learn to produce a new item's embedding from content.
    """

    def __init__(self, min_answers: int = 20, seed: int = 0):
        self.min_answers = min_answers
        self.seed = seed
        self.model = None
        self.mean = None

    def fit(self, E: np.ndarray, targets: np.ndarray, n_j: np.ndarray) -> "ItemGenerator":
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import GridSearchCV, KFold
        ok = n_j >= self.min_answers
        w = n_j[ok] / (n_j[ok] + 10.0)
        self.mean = np.average(targets[ok], axis=0, weights=w) if ok.any() else targets.mean(0)
        if ok.sum() < 10:
            return self
        grid = GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": [0.1, 1.0, 10.0], "gamma": [0.5, 1.0, 2.0]},
                            cv=KFold(5, shuffle=True, random_state=self.seed), scoring="neg_mean_squared_error")
        grid.fit(E[ok], targets[ok] - self.mean, sample_weight=w)
        self.model = grid.best_estimator_
        self.cv_params = grid.best_params_
        return self

    def predict(self, E: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.broadcast_to(self.mean, (E.shape[0], len(self.mean))).copy()
        return self.model.predict(E) + self.mean


def shrink_item_params(free: np.ndarray, generated: np.ndarray, n_j: np.ndarray, kappa: float) -> np.ndarray:
    """rho_j * learned + (1 - rho_j) * generated, rho_j = n_j / (n_j + kappa) (credibility / Buhlmann)."""
    rho = (n_j / (n_j + kappa))[:, None]
    return rho * free + (1.0 - rho) * generated
