"""Chạy FULL so sánh twin "sửa phần dư của nền LR + MF" với logistic regression (Kaggle hoặc máy có GPU).

Dùng từ repo đã clone; notebook kaggle/twin_lrmf_kaggle.ipynb làm sẵn các bước:
1. Kaggle Notebook -> Settings: Accelerator = GPU T4 x2, Internet = On (tải dữ liệu + mô hình).
2. (Tuỳ chọn) Add Input -> Models -> "Qwen2.5" transformers 7b-instruct: script tự tìm trong /kaggle/input;
   không có thì tải Qwen/Qwen2.5-7B-Instruct từ Hugging Face (~15 GB).
3. `!git clone https://github.com/kn1827/twin-matching.git` rồi `!python twin-matching/stage2/kaggle/run_full.py`.
4. Kết quả: /kaggle/working/result/ (mỗi bộ dữ liệu một thư mục + overall_*.md|csv) và
   /kaggle/working/result_bundle.zip. Bảng tổng hợp in ở cuối log.

Đổi cấu hình bằng khối CẤU HÌNH hoặc biến môi trường LRMF_* (xem dưới). Mặc định ước 3-5 giờ trên T4 x2
(phiên Kaggle tối đa 12 giờ); TIME_BUDGET_H chặn việc bắt đầu bộ dữ liệu mới khi đã gần hết giờ.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ============================== CẤU HÌNH ==============================
DATASETS = ["twin2k", "gss", "movielens"]          # thứ tự chạy
REGIMES = "R1,R2,R3,R4,R5,R6"                      # R3 tự để trống trên Twin-2K (không có đợt)
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"             # hoặc đường dẫn local; tự dùng Kaggle Model nếu có
T5_VARIANTS = "ab"                                 # a = twin độc lập, b = twin thấy dự đoán của nền
LLM_CELLS = 150                                    # số ô gọi twin T5 mỗi chế độ (tầng 2)
N_SPARSE = 150                                     # số người thưa (R4) / người mới (R5)
BOOTSTRAP = 300                                    # bootstrap theo người cho CI
HF_BATCH = 8                                       # số prompt mỗi lô generate
TIME_BUDGET_H = 10.5                               # không bắt đầu bộ dữ liệu mới sau mốc này
EXTRA_ARGS = {                                     # tham số riêng từng bộ dữ liệu
    "twin2k": [],
    "gss": ["--gss-year-min", "2000", "--gss-max-items", "600"],
    "movielens": ["--ml-min-ratings", "20", "--max-feat-items", "800"],
}
# Biến môi trường ghi đè cấu hình (notebook Kaggle đặt các biến này): LRMF_DATASETS="twin2k,gss",
# LRMF_REGIMES, LRMF_LLM_MODEL, LRMF_T5, LRMF_LLM_CELLS, LRMF_N_SPARSE, LRMF_BOOTSTRAP, LRMF_HF_BATCH_SIZE,
# LRMF_TIME_BUDGET_H.
_env = os.environ.get
DATASETS = [d.strip() for d in _env("LRMF_DATASETS", ",".join(DATASETS)).split(",") if d.strip()]
REGIMES = _env("LRMF_REGIMES", REGIMES)
LLM_MODEL = _env("LRMF_LLM_MODEL", LLM_MODEL)
T5_VARIANTS = _env("LRMF_T5", T5_VARIANTS)
LLM_CELLS = int(_env("LRMF_LLM_CELLS", LLM_CELLS))
N_SPARSE = int(_env("LRMF_N_SPARSE", N_SPARSE))
BOOTSTRAP = int(_env("LRMF_BOOTSTRAP", BOOTSTRAP))
HF_BATCH = int(_env("LRMF_HF_BATCH_SIZE", HF_BATCH))
TIME_BUDGET_H = float(_env("LRMF_TIME_BUDGET_H", TIME_BUDGET_H))
# =======================================================================

# LRMF_KAGGLE_DRYRUN=1: kiểm tra cơ chế của script (chạy, tổng hợp) với LLM mock và dữ liệu nhỏ
DRY_RUN = os.environ.get("LRMF_KAGGLE_DRYRUN") == "1"
DRY_ARGS = ["--max-persons", "200", "--max-items", "80", "--bootstrap", "10", "--max-iter", "30",
            "--llm-cells", "6", "--n-sparse", "20"]

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
if os.environ.get("LRMF_KAGGLE_WORK"):
    WORK = Path(os.environ["LRMF_KAGGLE_WORK"])
CODE = Path(__file__).resolve().parents[1]                 # gốc stage2/ (chứa src/ và experiments/)
RESULT = WORK / "result"


def sh(cmd: list[str], log_path: Path | None = None, env: dict | None = None) -> int:
    """Run a command, stream its output live and keep a copy in log_path."""
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            cwd=str(CODE), env=env)
    fh = open(log_path, "a", encoding="utf-8") if log_path else None
    for line in proc.stdout:
        if "Loading weights" in line or "Loading checkpoint shards" in line:
            continue
        print(line, end="", flush=True)
        if fh:
            fh.write(line)
    proc.wait()
    if fh:
        fh.close()
    return proc.returncode


def ensure_packages() -> None:
    need = []
    for mod, pkg in (("sentence_transformers", "sentence-transformers"), ("accelerate", "accelerate"),
                     ("transformers", "transformers"), ("sklearn", "scikit-learn"), ("scipy", "scipy")):
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    if need:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=False)


def gpu_info() -> tuple[int, float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0, 0.0
        n = torch.cuda.device_count()
        total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 2**30
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {p.name} {p.total_memory / 2**30:.1f} GB", flush=True)
        return n, total
    except Exception as exc:  # noqa: BLE001
        print("torch/GPU check failed:", exc)
        return 0, 0.0


def find_model() -> str:
    """A Kaggle Model input of the same checkpoint if attached, else the Hugging Face id."""
    want = LLM_MODEL.split("/")[-1].lower()                     # qwen2.5-7b-instruct
    size = "7b" if "7b" in want else ("3b" if "3b" in want else "")
    for cfg in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(cfg).lower()
        if "qwen" in d and size in d and "instruct" in d and os.path.exists(os.path.join(os.path.dirname(cfg), "tokenizer_config.json")):
            return os.path.dirname(cfg)
    return LLM_MODEL


def fmt(x, nd=4, sign=False):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def ci(d, key):
    if not d or key not in d:
        return "-"
    lo, hi = d.get(key + "_ci", [None, None])
    return f"{fmt(d[key], 4, True)} [{fmt(lo, 3, True)}, {fmt(hi, 3, True)}]"


def overall_table(runs: dict[str, Path]) -> None:
    regimes = [r.strip() for r in REGIMES.split(",")]
    header = ["dataset", "regime", "cells", "LR AUC", "dAUC B2 (LR+MF) vs LR", "dAUC B3 (+T2)", "dAUC B4 (+T3)",
              "LLM cells", "dAUC B9 (twin final)", "dLogloss B9", "twin alone AUC", "LR-test p T5a", "p T5b"]
    rows = []
    for ds in DATASETS:
        ov = None
        if ds in runs and (runs[ds] / "overview.json").exists():
            ov = json.loads((runs[ds] / "overview.json").read_text(encoding="utf-8"))
        for rg in regimes:
            if ov is None:
                rows.append([ds, rg, "(chua chay)"] + ["-"] * (len(header) - 3))
                continue
            if rg in ov.get("skipped", []):
                rows.append([ds, rg, "khong co"] + [""] * (len(header) - 3))
                continue
            r = ov["regimes"].get(rg)
            if r is None:
                rows.append([ds, rg, "-"] + ["-"] * (len(header) - 3))
                continue
            l1, l2 = r.get("level1", {}), r.get("level2", {})
            lt = ov.get("lr_test", {})
            rows.append([ds, rg, str(r["cells"]), fmt(l1.get("LR", {}).get("auc")),
                         ci(l1.get("B2 LR+MF (T1)"), "d_auc"), ci(l1.get("B3 +T2"), "d_auc"), ci(l1.get("B4 +T3"), "d_auc"),
                         str(l2.get("LR", {}).get("n", "-")), ci(l2.get("B9 super learner (+T9)"), "d_auc"),
                         ci(l2.get("B9 super learner (+T9)"), "d_logloss"), fmt(l2.get("T-b (twin alone)", {}).get("auc")),
                         fmt(lt.get(f"{rg}/T5a", {}).get("p_value")), fmt(lt.get(f"{rg}/T5b", {}).get("p_value"))])
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths))  # noqa: E731
    text = "\n".join([line(header), "  ".join("-" * w for w in widths)] + [line(r) for r in rows])
    md = "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
                   + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    note = ("dAUC = mo hinh - LR, CI 95% bootstrap theo nguoi. Tang 1 (B2-B4): moi o cua che do. Tang 2 (B9, twin): "
            "mau o co goi LLM. 'khong co' = bo du lieu khong co che do nay (Twin-2K khong co dot -> R3 trong).")
    (RESULT / f"overall_{stamp}.md").write_text(f"# Bang tong hop\n\n{md}\n\n{note}\n", encoding="utf-8")
    with open(RESULT / f"overall_{stamp}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("\n================ BANG TONG HOP (moi bo du lieu x che do) ================")
    print(text)
    print(note)


def main() -> None:
    t0 = time.time()
    print(f"code -> {CODE}", flush=True)
    ensure_packages()
    n_gpu, vram = gpu_info()
    model = find_model()
    env = dict(os.environ, LRMF_HF_BATCH=str(HF_BATCH), TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1")
    if n_gpu and vram < 24 and "7b" in model.lower():
        # one 16 GB GPU cannot hold a 7B model in fp16 next to the fits: load it in 4-bit
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=False)
        env["LRMF_HF_QUANT"] = "4bit"
        print("VRAM < 24 GB: 7B model loaded in 4-bit (LRMF_HF_QUANT=4bit)", flush=True)
    if not n_gpu:
        print("CANH BAO: khong co GPU - chay full se rat cham. Bat Accelerator GPU T4 x2.", flush=True)
    print(f"LLM: hf:{model}", flush=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    runs: dict[str, Path] = {}
    for ds in DATASETS:
        if (time.time() - t0) / 3600 > TIME_BUDGET_H:
            print(f"Het ngan sach thoi gian ({TIME_BUDGET_H} h): bo qua {ds}", flush=True)
            continue
        before = set(RESULT.glob(f"*_{ds}_kaggle"))
        cmd = [sys.executable, "experiments/compare_twin_vs_lr.py", "--dataset", ds, "--regimes", REGIMES,
               "--llm", "mock" if DRY_RUN else f"hf:{model}", "--llm-cells", str(LLM_CELLS), "--t5", T5_VARIANTS,
               "--n-sparse", str(N_SPARSE), "--bootstrap", str(BOOTSTRAP), "--out", str(RESULT), "--tag", "kaggle",
               *EXTRA_ARGS.get(ds, []), *(DRY_ARGS if DRY_RUN else [])]
        print(f"\n######## {ds} ########  (da chay {(time.time() - t0) / 60:.0f} phut)", flush=True)
        code = sh(cmd, RESULT / f"log_{ds}.txt", env)
        new = sorted(set(RESULT.glob(f"*_{ds}_kaggle")) - before)
        if new:
            runs[ds] = new[-1]
        if code != 0:
            print(f"CANH BAO: {ds} ket thuc voi ma {code}; xem {RESULT / f'log_{ds}.txt'}", flush=True)
    overall_table(runs)
    shutil.make_archive(str(WORK / "result_bundle"), "zip", RESULT)
    print(f"\nXong sau {(time.time() - t0) / 3600:.2f} h. Ket qua: {RESULT}  |  tai ve: {WORK / 'result_bundle.zip'}")


if __name__ == "__main__":
    main()
