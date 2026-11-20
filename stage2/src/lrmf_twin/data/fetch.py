"""Download the Twin-2K-500 files the twin needs (question catalog + numeric response CSVs).

    python -m lrmf_twin.data.fetch [--dest data/raw/twin2k] [--force]

Source: HuggingFace dataset LLM-Digital-Twin/Twin-2K-500 (CC-BY-4.0), folder
`question_catalog_and_human_response_csv/`. Only data files are fetched; the generator script in that
folder is never downloaded or executed. The sha256 of every file is written to `manifest.json` so a run
can be tied to the exact bytes it used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "LLM-Digital-Twin/Twin-2K-500"
REVISION = "main"
FOLDER = "question_catalog_and_human_response_csv"
FILES = ("question_catalog.json", "wave1_3_response.csv", "wave4_response.csv")
DEFAULT_DEST = Path("data/raw/twin2k")


def _url(name: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FOLDER}/{name}"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dest: Path = DEFAULT_DEST, force: bool = False) -> dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name in FILES:
        target = dest / name
        if force or not target.exists():
            tmp = target.with_suffix(target.suffix + ".part")
            print(f"downloading {name} ...", file=sys.stderr)
            req = urllib.request.Request(_url(name), headers={"User-Agent": "lrmf-twin/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
                shutil.copyfileobj(resp, out)
            tmp.replace(target)
        hashes[name] = sha256_of(target)
    manifest = {
        "repo": REPO,
        "revision": REVISION,
        "folder": FOLDER,
        "license": "CC-BY-4.0",
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": hashes,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return hashes


def ensure(dest: Path = DEFAULT_DEST) -> Path:
    """Return `dest`, downloading the files first if any is missing."""
    if not all((dest / f).exists() for f in FILES):
        fetch(dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for name, digest in fetch(args.dest, args.force).items():
        print(f"{digest}  {name}")


if __name__ == "__main__":
    main()
