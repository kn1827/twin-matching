"""Sinh kaggle/twin_lrmf_kaggle.ipynb: notebook import vào Kaggle, git clone repo rồi chạy full.

    python kaggle/make_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/kn1827/twin-matching.git"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Twin sửa phần dư của nền LR + MF so với Logistic Regression — chạy full

Ba bộ dữ liệu (Twin-2K-500, GSS 2000–2024, MovieLens-1M) × sáu chế độ R1–R6; LLM Qwen2.5-7B-Instruct qua
transformers cho T3/T5.

**Trước khi chạy**: Settings → Accelerator = **GPU T4 x2**, Internet = **On**. Tuỳ chọn: Add Input → Models →
Qwen2.5 (transformers, 7b-instruct) để khỏi tải ~15 GB từ Hugging Face.

Kết quả: `/kaggle/working/result/` (mỗi bộ dữ liệu một thư mục, bảng tổng hợp `overall_*.md|csv`) và
`/kaggle/working/result_bundle.zip`. Ước 3–5 giờ với cấu hình mặc định; nên dùng **Save & Run All (Commit)**.
"""),
    code(f"""
# 1. Lấy mã nguồn
!rm -rf /kaggle/working/twin-matching
!git clone --depth 1 {REPO} /kaggle/working/twin-matching
!cd /kaggle/working/twin-matching && git log -1 --oneline
"""),
    code("""
# 2. Thư viện (Kaggle đã có torch / transformers / sklearn; bổ sung phần thiếu)
!pip install -q sentence-transformers accelerate
"""),
    code("""
# 3. Cấu hình: sửa ở đây (để trống = giá trị mặc định trong kaggle/kaggle_run_full.py)
import os
os.environ["LRMF_CODE_DIR"] = "/kaggle/working/twin-matching"
os.environ["LRMF_DATASETS"] = "twin2k,gss,movielens"   # bỏ bớt để chạy nhanh hơn
os.environ["LRMF_REGIMES"] = "R1,R2,R3,R4,R5,R6"
os.environ["LRMF_LLM_MODEL"] = "Qwen/Qwen2.5-7B-Instruct"
os.environ["LRMF_T5"] = "ab"            # a = twin độc lập, b = twin thấy dự đoán của nền
os.environ["LRMF_LLM_CELLS"] = "150"    # số ô gọi twin mỗi chế độ
os.environ["LRMF_N_SPARSE"] = "150"     # số người thưa (R4) / người mới (R5)
os.environ["LRMF_BOOTSTRAP"] = "300"
os.environ["LRMF_HF_BATCH_SIZE"] = "8"
os.environ["LRMF_TIME_BUDGET_H"] = "10.5"
# os.environ["LRMF_KAGGLE_DRYRUN"] = "1"   # bật để thử nhanh cơ chế (LLM mock, dữ liệu nhỏ, vài phút)
"""),
    code("""
# 4. Chạy full (log in trực tiếp; bảng tổng hợp ở cuối)
!python /kaggle/working/twin-matching/kaggle/kaggle_run_full.py
"""),
    code("""
# 5. Xem bảng tổng hợp
import glob
from IPython.display import Markdown, display
for f in sorted(glob.glob("/kaggle/working/result/overall_*.md"))[-1:]:
    display(Markdown(open(f, encoding="utf-8").read()))
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}

if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "twin_lrmf_kaggle.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)
