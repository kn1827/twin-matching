"""Sinh kaggle/kaggle_run_full.py: kaggle_template.py + toàn bộ mã nguồn cần thiết nhúng dạng zip base64.

    python kaggle/build_kaggle_script.py

Chạy lại lệnh này mỗi khi sửa src/lrmf_twin hoặc experiments/compare_twin_vs_lr.py. Chỉ đóng gói mã; dữ liệu
và mô hình được tải trên Kaggle (Internet = On).
"""

from __future__ import annotations

import base64
import hashlib
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(p for p in (ROOT / "src" / "lrmf_twin").rglob("*.py")) + [ROOT / "experiments" / "compare_twin_vs_lr.py"]


def build() -> Path:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(f, f.relative_to(ROOT).as_posix())
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    template = (ROOT / "kaggle" / "kaggle_template.py").read_text(encoding="utf-8")
    assert template.count('"__BUNDLE__"') == 1
    digest = hashlib.sha256(buf.getvalue()).hexdigest()[:12]
    out = template.replace('"__BUNDLE__"', f'"{b64}"  # bundle sha256 {digest}, {len(FILES)} files')
    path = ROOT / "kaggle" / "kaggle_run_full.py"
    path.write_text(out, encoding="utf-8")
    print(f"{path} ({path.stat().st_size / 1024:.0f} KB, {len(FILES)} files, bundle {digest})")
    return path


if __name__ == "__main__":
    build()
