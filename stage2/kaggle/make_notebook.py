"""Sinh notebook Kaggle (một tài khoản, cả ba bộ dữ liệu trong một phiên 12 giờ): git clone repo rồi chạy full.

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
os.environ["LRMF_DATASETS"] = "__DATASETS__"           # cả ba bộ trong một phiên
os.environ["LRMF_REGIMES"] = "R1,R2,R3,R4,R5,R6"
os.environ["LRMF_LLM_MODEL"] = "Qwen/Qwen2.5-7B-Instruct"
os.environ["LRMF_T5"] = "ab"            # a = twin độc lập, b = twin thấy dự đoán của nền
os.environ["LRMF_LLM_CELLS"] = "80"     # số ô gọi twin mỗi chế độ (giảm để 3 bộ vừa 12 giờ)
os.environ["LRMF_N_SPARSE"] = "100"    # số người thưa (R4) / người mới (R5)
os.environ["LRMF_BOOTSTRAP"] = "200"
os.environ["LRMF_HF_BATCH_SIZE"] = "8"
os.environ["LRMF_TIME_BUDGET_H"] = "10"
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

# Một tài khoản Kaggle: một notebook chạy lần lượt cả ba bộ dữ liệu (TIME_BUDGET_H chặn bộ mới khi gần hết giờ)
SPLITS = {"kaggle": "twin2k,gss,movielens"}

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for old in here.glob("twin_lrmf_acc*.ipynb"):
        old.unlink()
    for name, ds in SPLITS.items():
        text = json.dumps(nb, ensure_ascii=False, indent=1).replace("__DATASETS__", ds).replace("__NAME__", name)
        out = here / f"twin_lrmf_{name}.ipynb"
        out.write_text(text, encoding="utf-8")
        print(out.name)
