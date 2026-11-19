"""U11: gnwt/stage1 never imports an LLM module, never imports an embedding library outside the opt-in
fallback, never touches data/sealed, never opens a network connection."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from gnwt.config import PROJECT_ROOT

STAGE1 = PROJECT_ROOT / "src" / "gnwt" / "stage1"
FORBIDDEN = ("gnwt.llm", "llm", "sentence_transformers", "transformers", "torch", "openai", "anthropic",
             "httpx", "requests", "urllib", "socket", "chromadb")


def _imports(path: Path) -> list[tuple[str, int, bool]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        top_level = node in tree.body
        for n in names:
            out.append((n, node.lineno, top_level))
    return out


def test_u11_no_forbidden_imports():
    bad = []
    for py in STAGE1.rglob("*.py"):
        for name, line, top in _imports(py):
            hit = any(name == f or name.startswith(f + ".") for f in FORBIDDEN)
            if hit and not (py.name == "fallback_embed.py" and not top):
                bad.append(f"{py.relative_to(PROJECT_ROOT)}:{line} imports {name}")
    assert not bad, bad


def test_u11_no_sealed_access():
    bad = [str(p) for p in STAGE1.rglob("*.py") if "sealed" in p.read_text(encoding="utf-8")]
    assert not bad, bad


def test_u11_runtime_import_has_no_embedding_library():
    code = ("import sys; import gnwt.stage1.static, gnwt.stage1.srrm, gnwt.stage1.context.demand; "
            "bad=[m for m in ('sentence_transformers','torch','transformers','gnwt.llm') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT / "src")})
    assert r.returncode == 0, r.stdout + r.stderr
