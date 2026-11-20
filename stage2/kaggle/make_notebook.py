"""Sinh 3 notebook Kaggle (3 tài khoản, mỗi tài khoản một bộ dữ liệu, tham số lớn): git clone repo rồi chạy full.

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
Qwen2.5 (transformers, 14b-instruct) để khỏi tải ~30 GB từ Hugging Face.

Kết quả: `/kaggle/working/result/` (mỗi bộ dữ liệu một thư mục, bảng tổng hợp `overall_*.md|csv`) và
`/kaggle/working/result_bundle.zip`. Mỗi notebook chạy một nửa số chế độ của một bộ dữ liệu, ước 8–11 giờ (Qwen2.5-14B 4-bit); nên dùng **Save & Run All (Commit)**. Gộp 6 notebook: tải `result_bundle.zip` của từng notebook, giải nén, rồi chạy `python kaggle/run_full.py --merge <thư mục 1> <thư mục 2> ...`.
"""),
    code(f"""
# 1. Lấy mã nguồn (repo public; không hỏi mật khẩu nên lỗi mạng là dừng ngay, không treo)
!rm -rf /kaggle/working/twin-matching
!GIT_TERMINAL_PROMPT=0 git clone --depth 1 {REPO} /kaggle/working/twin-matching && cd /kaggle/working/twin-matching && git log -1 --oneline
"""),
    code("""
# 2. Thư viện (Kaggle đã có torch / transformers / sklearn; bổ sung phần thiếu)
!pip install -q sentence-transformers accelerate peft bitsandbytes
"""),
    code("""
# 3. Cấu hình (notebook __NAME__): sửa ở đây (để trống = giá trị mặc định trong kaggle/run_full.py)
import os
os.environ["LRMF_DATASETS"] = "__DATASETS__"           # mỗi notebook: một bộ dữ liệu, một nửa số chế độ
os.environ["LRMF_REGIMES"] = "__REGIMES__"
os.environ["LRMF_LLM_MODEL"] = "Qwen/Qwen2.5-14B-Instruct"   # 4-bit tự bật cho T5 và twin trung tâm
os.environ["LRMF_T5"] = "a"             # a = twin độc lập; "ab" thêm ablation twin thấy nền (neo số)
os.environ["LRMF_FT"] = "1"             # twin trung tâm: Qwen2.5-7B tinh chỉnh QLoRA, token người/câu từ MF
os.environ["LRMF_FT_STEPS"] = "800"      # bước tối đa, dừng sớm theo log-loss kiểm định
os.environ["LRMF_FT_TRAIN"] = "6000"    # episode huấn luyện mỗi chế độ (mô phỏng chế độ trên fold 0)
os.environ["LRMF_FT_EVAL"] = "1500"     # ô đánh giá mỗi chế độ
os.environ["LRMF_LLM_CELLS"] = "500"    # số ô gọi twin mỗi chế độ (càng nhiều, CI tầng 2 càng hẹp)
os.environ["LRMF_N_SPARSE"] = "400"    # số người thưa (R4) / người mới (R5)
os.environ["LRMF_BOOTSTRAP"] = "1000"
os.environ["LRMF_HF_BATCH_SIZE"] = "8"
os.environ["LRMF_FT_BATCH"] = "4"          # 14B QLoRA trên một T4
os.environ["LRMF_TIME_BUDGET_H"] = "11.3"   # quá mốc này: bỏ chế độ / lần tinh chỉnh chưa bắt đầu
# os.environ["LRMF_KAGGLE_DRYRUN"] = "1"   # bật để thử nhanh cơ chế (LLM mock, dữ liệu nhỏ, vài phút)
"""),
    code("""
# 4. Chạy full (log in trực tiếp; bảng tổng hợp ở cuối)
!python /kaggle/working/twin-matching/stage2/kaggle/run_full.py
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

# Ba tài khoản Kaggle, mỗi tài khoản một notebook một bộ dữ liệu (phiên GPU tối đa 12 giờ)
# 6 notebook: mỗi bộ dữ liệu tách 2 phiên 12 giờ (R1-R3, R4-R6); 3 tài khoản x 2 notebook (quota GPU 30 giờ/tuần)
SPLITS = {f"acc{a}_{ds}_{part}": (ds, regs) for a, ds in ((1, "twin2k"), (2, "gss"), (3, "movielens"))
          for part, regs in (("p1", "R1,R2,R3"), ("p2", "R4,R5,R6"))}

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    (here / "twin_lrmf_kaggle.ipynb").unlink(missing_ok=True)
    for old in here.glob("twin_lrmf_acc*.ipynb"):
        old.unlink()
    for name, (ds, regs) in SPLITS.items():
        text = (json.dumps(nb, ensure_ascii=False, indent=1).replace("__DATASETS__", ds).replace("__NAME__", name)
                .replace("__REGIMES__", regs))
        out = here / f"twin_lrmf_{name}.ipynb"
        out.write_text(text, encoding="utf-8")
        print(out.name)
