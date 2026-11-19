"""Download the two raw Speed Dating files declared in `params.data.datasets` (offline step, never stage 1).

    python -m gnwt.data.fetch [--force]

Each file is saved under `data.raw_dir`; when the config declares an md5 the download is checked against
it, and the sha256 of every file is printed so it can be recorded in the artifact manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from gnwt.config import DEFAULT_PARAMS, load_params, resolve_path, sha256_file


def _md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - integrity check against the publisher's checksum, not security
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, *, params_path: str | Path = DEFAULT_PARAMS, force: bool = False) -> Path:
    params = load_params(params_path)
    cfg = params.data.datasets[name]
    raw_dir = resolve_path(params.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / cfg.file
    if dest.exists() and not force:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(cfg.url, headers={"User-Agent": "gnwt-stage1-fetch/2.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:  # noqa: S310
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    if cfg.md5 is not None and _md5(tmp) != cfg.md5:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{name}: md5 mismatch for {cfg.url}")
    tmp.replace(dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=str(DEFAULT_PARAMS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    params = load_params(args.params)
    for name in params.data.datasets:
        path = fetch(name, params_path=args.params, force=args.force)
        print(f"{name:<13} {path}  {path.stat().st_size:>9} B  sha256={sha256_file(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
