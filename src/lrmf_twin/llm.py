"""LLM client for T3 and T5: one call, T = 0, JSON output, no chain of thought, every answer cached on disk.

Providers (spec string):
* `ollama:<model>`   local Ollama server (default `ollama:qwen2.5:3b`, host from OLLAMA_HOST or localhost:11434).
                     The model is kept loaded (keep_alive 30m). LRMF_OLLAMA_NUM_GPU=0 forces CPU; after a
                     server error (e.g. the GPU runner crashing on a low-memory laptop) the client retries on
                     CPU for the rest of the run and records it.
* `openai:<model>`   any OpenAI-compatible /v1/chat/completions endpoint; base URL from LRMF_LLM_BASE_URL
                     (default https://api.openai.com/v1), key from LRMF_LLM_API_KEY or OPENAI_API_KEY
* `hf:<model>`       Hugging Face transformers in this process (e.g. `hf:Qwen/Qwen2.5-7B-Instruct` or a local
                     path such as a Kaggle model folder), fp16, greedy decoding, prompts generated in batches of
                     LRMF_HF_BATCH (default 8) through `complete_many`; device_map="auto" spreads it over GPUs.
* `mock`             deterministic stand-in that returns hash-based distributions. NOT an LLM: it only
                     exercises the pipeline (tests, dry runs). Results labelled as such.
* `none`             LLM modules switched off.

The cache key is sha256(provider, model, options, system, prompt), so changing the prompt or the model
re-queries, and re-running the same experiment costs no call. The model digest (Ollama) is recorded in the
run manifest so a result can be tied to the exact weights (design 6.4, drift rule D7).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LLM = "ollama:qwen2.5:3b"
DEFAULT_CACHE = Path("data/cache/lrmf_twin_llm.sqlite")
SYSTEM = ("You predict how one specific survey respondent answered. Answer with JSON only, no explanation.")


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"calls": self.calls, "cache_hits": self.cache_hits, "failures": self.failures,
                "seconds": round(self.seconds, 1), "errors": list(dict.fromkeys(self.errors))[:5]}


class LLMClient:
    def __init__(self, spec: str = DEFAULT_LLM, cache_path: Path = DEFAULT_CACHE, max_tokens: int = 160,
                 timeout: float = 600.0, seed: int = 0):
        self.spec = spec
        self.provider, _, self.model = spec.partition(":")
        if self.provider not in ("ollama", "openai", "hf", "mock", "none"):
            raise ValueError(f"unknown LLM spec {spec!r}")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.seed = seed
        self.stats = LLMStats()
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.base_url = os.environ.get("LRMF_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        num_gpu = os.environ.get("LRMF_OLLAMA_NUM_GPU")
        self.num_gpu: int | None = int(num_gpu) if num_gpu not in (None, "") else None
        self.cpu_fallback = False
        self._hf = None
        self.hf_batch = int(os.environ.get("LRMF_HF_BATCH", "8"))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(cache_path))
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT, model TEXT, "
                        "created REAL)")
        self.db.commit()

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    # -- environment --------------------------------------------------------------------------------
    def describe(self) -> dict:
        info = {"spec": self.spec, "provider": self.provider, "model": self.model,
                "temperature": 0, "seed": self.seed, "max_tokens": self.max_tokens}
        if self.provider == "ollama":
            try:
                tags = self._post_json(f"{self.host}/api/tags", None, timeout=10)
                for m in tags.get("models", []):
                    if m.get("name") == self.model or m.get("model") == self.model:
                        info["digest"] = m.get("digest")
                        info["quantization"] = (m.get("details") or {}).get("quantization_level")
            except Exception as exc:  # noqa: BLE001
                info["digest_error"] = str(exc)
        if self.provider == "ollama":
            info["num_gpu"] = self.num_gpu
            info["cpu_fallback"] = self.cpu_fallback
        if self.provider == "mock":
            info["warning"] = "mock provider: hash-based outputs, not a language model"
        return info

    def check(self) -> None:
        """Raise LLMUnavailable with an actionable message if the provider cannot answer."""
        if self.provider == "hf":
            try:
                import transformers  # noqa: F401
            except ImportError as exc:
                raise LLMUnavailable("Provider hf can cai transformers: pip install transformers accelerate") from exc
        if self.provider == "ollama":
            try:
                tags = self._post_json(f"{self.host}/api/tags", None, timeout=10)
            except Exception as exc:  # noqa: BLE001
                raise LLMUnavailable(f"Ollama khong tra loi o {self.host} ({exc}). Chay `ollama serve` "
                                     f"hoac dung --llm none.") from exc
            names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
            if self.model not in names:
                raise LLMUnavailable(f"Ollama chua co model {self.model!r}; chay `ollama pull {self.model}`.")
        elif self.provider == "openai":
            if not (os.environ.get("LRMF_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")):
                raise LLMUnavailable("Thieu LRMF_LLM_API_KEY / OPENAI_API_KEY cho provider openai.")

    # -- calls --------------------------------------------------------------------------------------
    def _post_json(self, url: str, body: dict | None, headers: dict | None = None, timeout: float | None = None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read())

    def _raw(self, prompt: str) -> str:
        if self.provider == "mock":
            h = hashlib.sha256(prompt.encode()).digest()
            n = int(re.findall(r"exactly (\d+) numbers", prompt)[-1]) if "exactly" in prompt else 1
            w = [1 + h[i % len(h)] for i in range(n)]
            return json.dumps({"p": [round(x / sum(w), 4) for x in w]})
        if self.provider == "ollama":
            options = {"temperature": 0, "seed": self.seed, "num_predict": self.max_tokens, "num_ctx": 4096}
            if self.num_gpu is not None:
                options["num_gpu"] = self.num_gpu
            body = {"model": self.model, "stream": False, "format": "json", "keep_alive": "30m",
                    "options": options,
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
            return self._post_json(f"{self.host}/api/chat", body)["message"]["content"]
        if self.provider == "openai":
            key = os.environ.get("LRMF_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
            body = {"model": self.model, "temperature": 0, "seed": self.seed, "max_tokens": self.max_tokens,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
            out = self._post_json(f"{self.base_url}/chat/completions", body, {"Authorization": f"Bearer {key}"})
            return out["choices"][0]["message"]["content"]
        raise LLMUnavailable("LLM disabled (--llm none)")

    # -- transformers ---------------------------------------------------------------------------------
    def _hf_load(self):
        if self._hf is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.model, padding_side="left")
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            kw = {"torch_dtype": dtype, "device_map": "auto" if torch.cuda.is_available() else None}
            quant = os.environ.get("LRMF_HF_QUANT", "").lower()      # "", "8bit", "4bit" (needs bitsandbytes)
            if quant in ("8bit", "4bit"):
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = (BitsAndBytesConfig(load_in_8bit=True) if quant == "8bit" else
                                             BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype))
            mdl = AutoModelForCausalLM.from_pretrained(self.model, **kw)
            mdl.eval()
            self._hf = (tok, mdl)
        return self._hf

    def _hf_generate(self, prompts: list[str]) -> list[str]:
        import torch
        tok, mdl = self._hf_load()
        texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**enc, max_new_tokens=self.max_tokens, do_sample=False, temperature=None,
                               top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

    def _key(self, prompt: str) -> str:
        parts = [self.provider, self.model, self.seed, self.max_tokens, SYSTEM, prompt]
        quant = os.environ.get("LRMF_HF_QUANT", "") if self.provider == "hf" else ""
        if quant:                                   # a quantised model answers differently: separate cache
            parts.append(quant)
        return hashlib.sha256(json.dumps(parts).encode()).hexdigest()

    def complete_many(self, prompts: list[str], progress=None) -> list[str | None]:
        """Answers for many prompts: cached ones from disk, the rest in batches (hf) or one by one."""
        out: list[str | None] = [None] * len(prompts)
        todo = []
        for i, pr in enumerate(prompts):
            row = self.db.execute("SELECT response FROM cache WHERE key = ?", (self._key(pr),)).fetchone()
            if row is not None:
                self.stats.cache_hits += 1
                out[i] = row[0]
            else:
                todo.append(i)
        if self.provider != "hf":
            for n, i in enumerate(todo):
                out[i] = self.complete(prompts[i])
                if progress:
                    progress(n + 1, len(todo))
            return out
        todo.sort(key=lambda i: len(prompts[i]))                  # similar lengths -> less padding
        for start in range(0, len(todo), self.hf_batch):
            batch = todo[start:start + self.hf_batch]
            t0 = time.time()
            try:
                texts = self._hf_generate([prompts[i] for i in batch])
            except Exception as exc:  # noqa: BLE001  (OOM etc.: count, keep going)
                self.stats.failures += len(batch)
                self.stats.errors.append(str(exc)[:200])
                continue
            self.stats.calls += len(batch)
            self.stats.seconds += time.time() - t0
            for i, text in zip(batch, texts):
                out[i] = text
                self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?)",
                                (self._key(prompts[i]), text, self.spec, time.time()))
            self.db.commit()
            if progress:
                progress(min(start + self.hf_batch, len(todo)), len(todo))
        return out

    def complete(self, prompt: str) -> str | None:
        if self.provider == "hf":
            return self.complete_many([prompt])[0]
        key = self._key(prompt)
        row = self.db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.stats.cache_hits += 1
            return row[0]
        t0 = time.time()
        for attempt in range(3):
            try:
                text = self._raw(prompt)
                break
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                server_error = isinstance(exc, urllib.error.HTTPError) and exc.code >= 500
                if server_error and self.provider == "ollama" and self.num_gpu != 0:
                    self.num_gpu, self.cpu_fallback = 0, True      # GPU runner crashed: continue on CPU
                    self.stats.errors.append(f"{exc} -> switched Ollama to CPU (num_gpu=0)")
                    continue
                if attempt == 2:
                    self.stats.failures += 1
                    self.stats.errors.append(str(exc))
                    return None
                time.sleep(2.0 * (attempt + 1))
        self.stats.calls += 1
        self.stats.seconds += time.time() - t0
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?)", (key, text, self.spec, time.time()))
        self.db.commit()
        return text

    def cached(self, prompt: str) -> bool:
        return self.db.execute("SELECT 1 FROM cache WHERE key = ?", (self._key(prompt),)).fetchone() is not None


def parse_list(text: str | None, n: int) -> list[float] | None:
    """Parse {"p": [...]} into n probabilities clipped to [0, 1], without renormalising (T3)."""
    raw = _extract(text, n)
    if raw is None:
        return None
    return [min(max(v, 0.0), 1.0) for v in raw]


def parse_probs(text: str | None, n: int) -> list[float] | None:
    """Parse {"p": [...]} (or a {"1": .., "2": ..} mapping) into a distribution over n options (T5)."""
    raw = _extract(text, n)
    if raw is None:
        return None
    x = [max(v, 0.0) for v in raw]
    s = sum(x)
    return None if s <= 0 else [v / s for v in x]


def _extract(text: str | None, n: int) -> list[float] | None:
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    vals = None
    if isinstance(obj, list):
        vals = obj
    elif isinstance(obj, dict):
        for key in ("p", "probabilities", "probs", "distribution"):
            if key in obj:
                vals = obj[key]
                break
        if vals is None:
            vals = obj
        if isinstance(vals, dict):
            ordered = []
            for k in range(1, n + 1):
                v = vals.get(str(k), vals.get(k))
                ordered.append(v)
            vals = ordered
    if not isinstance(vals, list) or len(vals) != n:
        return None
    try:
        x = [float(v) if v is not None else 0.0 for v in vals]
    except (TypeError, ValueError):
        return None
    if any(v != v for v in x):        # NaN
        return None
    # some models answer in percent
    if 1.5 < max(x) <= 100.0:
        x = [v / 100.0 for v in x]
    return x
