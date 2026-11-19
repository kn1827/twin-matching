"""Read a few context sentences with the ILLUSTRATIVE reader of tests/fixtures (stage1.md section 3.5).

    python scripts/demo_context.py ["câu ngữ cảnh" ...]

Prints c*, the keyword hits with their class, Z, D^ on the 12 styles and D~ on the 5 domains. The levels
theta, k and the study_group profile are the illustration numbers of the spec, not estimates.
"""

from __future__ import annotations

import sys

from gnwt.config import PROJECT_ROOT
from gnwt.stage1.context.catalog import Catalog
from gnwt.stage1.context.demand import ReaderArtifacts, extract_demands
from gnwt.stage1.context.lexicon import Lexicon
from gnwt.stage1.context.styles import DOMAINS, STYLES

FIX = PROJECT_ROOT / "tests" / "fixtures"
DEFAULT = ["nhóm ôn thi xác suất, rất cần kỷ luật, không cần sáng tạo", "học nhóm", "hoc nhom, can ky luat",
           "cần sáng tạo, nhưng không cần tỉ mỉ", "study group, very disciplined, not competitive"]


def main(argv: list[str]) -> int:
    lex = Lexicon.load(PROJECT_ROOT / "data" / "lexicon" / "demand_lexicon_v1.yaml")
    cat = Catalog.load(FIX / "catalog_illustrative.yaml")
    art = ReaderArtifacts.from_json(FIX / "reader_illustrative.json", lex, cat)
    for text in argv or DEFAULT:
        cd = extract_demands(text, None, art)
        print(f"\n> {text}")
        print(f"  c* = {cd.match.c_star} ({cd.match.how}, alias {cd.match.alias!r}); flags {list(cd.flags)}")
        for h in cd.hits:
            print(f"  hit {h.style} {h.cls:<6} '{h.phrase}' clause {h.clause}{' toneless' if h.toneless else ''}")
        print("  D^ " + " ".join(f"{s}={v:+.2f}" for s, v in zip(STYLES, cd.D_style)))
        print("  D~ " + " ".join(f"{d}={v:+.2f}" for d, v in zip(DOMAINS, cd.D_tilde)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
