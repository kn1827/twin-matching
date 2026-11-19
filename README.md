# Twin sửa phần dư của nền LR + MF

Dựng theo thiết kế "Twin sửa phần dư của nền LR + MF". Mười mô-đun, chỉ T3 và
T5 gọi LLM. Twin nằm ở tầng nào và vai trò qua chạy thử: [reports/lrmf_twin_vai_tro.md](reports/lrmf_twin_vai_tro.md).

Ba bộ dữ liệu, cùng một giao diện (người × câu + văn bản câu hỏi) và cùng sáu chế độ:

| Chế độ | Twin-2K-500 | GSS 2000–2024 | MovieLens-1M |
|---|---|---|---|
| R1 dày (ô ngẫu nhiên) | có | có | có |
| R2 câu mới (không câu trả lời nào) | có | có | có (phim mới) |
| R3 đợt khuyết (khối câu × đợt) | **trống** (không có đợt) | có (câu × năm) | có (phim × quý) |
| R4 người thưa (10 câu đã biết) | có | có | có |
| R5 người mới (chỉ nhân khẩu) | có | có | có |
| R6 câu lạnh (5 câu trả lời) | có | có | có |

```
data/twin2k.py, data/gss.py,    T0 tải + nhị phân hoá + cache (Twin-2K: HF; GSS: NORC Stata, 36.055 người ×
data/movielens.py               380 câu; MovieLens-1M: 6.040 user × 3.043 phim); regimes.py: fold R1–R6 + cờ z
t1_base.py                      T1 nền LR (hiệu ứng câu + nhân khẩu + câu trả lời khác) + MF, L-BFGS, GPU nếu có;
                                T2 đầu ngữ nghĩa (embedding → MLP → b_j, v_j); T3 refit người thưa/mới
t2_encoder.py                   embedding văn bản câu hỏi (sentence-transformers / Ollama / hash), cache
t3_prior.py, t4_retrieval.py    tiền nghiệm LLM cho người thưa; truy hồi top-k câu của chính người
t5_twin.py, llm.py              twin hiệu chỉnh T5a/T5b; client Ollama / transformers (hf:) / OpenAI / mock
combine.py                      T6 trực giao hoá, T7 cổng đơn điệu, T8 super learner, T9 Platt
metrics.py                      AUC, AUC theo câu, log-loss, Brier, bootstrap theo người, LR test
baselines.py                    LR theo từng câu (torch hoặc sklearn), LR văn bản cho câu mới, trung bình câu
kaggle/                         run_full.py (chạy full ba bộ dữ liệu), twin_lrmf_kaggle.ipynb (notebook import vào Kaggle)
```

## Cài đặt và chạy

```bash
pip install -e ".[dev]"
```

Thêm `.[hf]` nếu dùng provider `hf:` (transformers). Chạy local (kết
quả vào `result/<thời điểm>_<bộ dữ liệu>/`, bảng in ra màn hình):

```bash
ollama serve
```

```bash
python experiments/compare_twin_vs_lr.py --quick
```

```bash
python experiments/compare_twin_vs_lr.py --dataset gss --max-persons 3000 --llm none
```

`--dataset twin2k|gss|movielens`; `--llm none` chỉ chạy phần thống kê; `--llm hf:<model>` dùng transformers;
`--llm openai:<model>` dùng API kiểu OpenAI (`LRMF_LLM_BASE_URL`, `LRMF_LLM_API_KEY`); `--regimes`,
`--llm-cells`, `--t5 a|b|ab`, `--n-sparse`, `--max-persons`, `--max-items` để chỉnh khối lượng. Mọi lời gọi
LLM được cache ở `data/cache/lrmf_twin_llm.sqlite`. Trên laptop 8 GB, qwen2.5:3b mất ~15–20 s/lời gọi T5 và
~25 s/lời gọi T3 khi có GPU, khoảng gấp đôi khi chỉ chạy CPU (`LRMF_OLLAMA_NUM_GPU=0` ép CPU).

## Chạy full trên Kaggle

Import [kaggle/twin_lrmf_kaggle.ipynb](kaggle/twin_lrmf_kaggle.ipynb) vào Kaggle (File → Import Notebook), bật
Accelerator GPU T4 x2 và Internet On, rồi Save & Run All. Notebook clone repo này và chạy
[kaggle/run_full.py](kaggle/run_full.py): ba bộ dữ liệu × R1–R6, LLM Qwen2.5-7B-Instruct qua transformers, kết quả ở
`/kaggle/working/result/` (bảng tổng hợp `overall_*.md|csv`) và `result_bundle.zip`. Sau khi sửa
`kaggle/make_notebook.py`, sinh lại notebook bằng `python kaggle/make_notebook.py`.

## Kiểm thử

```bash
python -m pytest
```

Quy ước và chỗ lệch thiết kế: mục `lrmf_twin` trong [CHANGELOG.md](CHANGELOG.md).
