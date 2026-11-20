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
Qwen2.5 (transformers, 7b-instruct) để khỏi tải ~15 GB từ Hugging Face.

Kết quả: `/kaggle/working/result/` (mỗi bộ dữ liệu một thư mục, bảng tổng hợp `overall_*.md|csv`) và
`/kaggle/working/result_bundle.zip`. Ước 3–5 giờ với cấu hình mặc định; nên dùng **Save & Run All (Commit)**.
"""),
    code(f"""
# 1. Lấy mã nguồn. Repo PRIVATE: thêm Kaggle Secret tên GITHUB_TOKEN (Add-ons -> Secrets; token GitHub chỉ cần
#    quyền đọc repo này) rồi bật secret cho notebook, hoặc chuyển repo sang public.
#    Không bao giờ hỏi mật khẩu: clone lỗi là dừng ngay thay vì treo đến hết 12 giờ.
import os, subprocess
token = ""
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass
url = "{REPO}"
if token:
    url = url.replace("https://", "https://x-access-token:" + token + "@")
subprocess.run(["rm", "-rf", "/kaggle/working/twin-matching"])
r = subprocess.run(["git", "clone", "--depth", "1", url, "/kaggle/working/twin-matching"],
                   env=dict(os.environ, GIT_TERMINAL_PROMPT="0"), capture_output=True, text=True, timeout=300)
print((r.stdout + r.stderr).replace(token, "***") if token else r.stdout + r.stderr)
if r.returncode != 0:
    raise SystemExit("Clone that bai: repo private ma chua co secret GITHUB_TOKEN, hoac token sai/het han.")
"""),
    code("""
# 2. Thư viện (Kaggle đã có torch / transformers / sklearn; bổ sung phần thiếu)
!pip install -q sentence-transformers accelerate
"""),
    code("""
# 3. Cấu hình (notebook __NAME__): sửa ở đây (để trống = giá trị mặc định trong kaggle/run_full.py)
import os
os.environ["LRMF_DATASETS"] = "__DATASETS__"           # mỗi tài khoản Kaggle chạy một bộ
os.environ["LRMF_REGIMES"] = "R1,R2,R3,R4,R5,R6"
os.environ["LRMF_LLM_MODEL"] = "Qwen/Qwen2.5-7B-Instruct"
os.environ["LRMF_T5"] = "ab"            # a = twin độc lập, b = twin thấy dự đoán của nền
os.environ["LRMF_LLM_CELLS"] = "200"    # số ô gọi twin mỗi chế độ (tầng 2)
os.environ["LRMF_N_SPARSE"] = "200"    # số người thưa (R4) / người mới (R5)
os.environ["LRMF_BOOTSTRAP"] = "500"
os.environ["LRMF_HF_BATCH_SIZE"] = "8"
os.environ["LRMF_TIME_BUDGET_H"] = "11"
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
SPLITS = {"acc1_twin2k": "twin2k", "acc2_gss": "gss", "acc3_movielens": "movielens"}

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    (here / "twin_lrmf_kaggle.ipynb").unlink(missing_ok=True)
    for name, ds in SPLITS.items():
        text = json.dumps(nb, ensure_ascii=False, indent=1).replace("__DATASETS__", ds).replace("__NAME__", name)
        out = here / f"twin_lrmf_{name}.ipynb"
        out.write_text(text, encoding="utf-8")
        print(out.name)
