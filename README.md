# GNWT-TM — Stage 1

Chặng 1 của GNWT-TM, viết lại theo `stage1.md`: hồ sơ + một câu ngữ cảnh → xác suất "có" có hướng
a⁰(i→j), a⁰(j→i) và xác suất cùng đồng ý m⁰, bằng SRRM probit của Hoff, với bộ đọc ngữ cảnh dùng từ điển
song ngữ + luật mức/phủ định, gộp với hồ sơ mặc định bằng credibility Bühlmann. Lúc chạy không gọi LLM,
không lấy mẫu, không gọi mạng. Mọi chỗ lệch đặc tả ghi trong [CHANGELOG.md](CHANGELOG.md).

Dữ liệu: **Columbia** (tệp gốc Fisman et al. 2006) và **Speed Dating** (OpenML 40536, cùng thí nghiệm,
mã hoá lại, mã người được dựng lại).

## Cài đặt

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -e ".[dev,baselines]"
```

## Chạy

Tải hai tệp dữ liệu vào `data/raw/` (kiểm md5 của OpenML):

```bash
python -m gnwt.data.fetch
```

Các bước ngoại tuyến O6–O9: cross-fit a⁰ bỏ-một-đợt và mô hình khoá (thêm `--section9` nếu cần cả các nhánh
sàn/thống kê của Mục 9):

```bash
python scripts/run_stage1.py --workers 4
```

So sánh a⁰ với các baseline profile matching mà các paper xếp hạng B trở lên đã dùng trên Speed Dating
(FM, DeepFM, xDeepFM, DCN-V2, AutoInt, IPNN, biDeepFM, XGBoost, LightGBM, CatBoost):

```bash
python experiments/compare_literature_baselines.py --protocol lowo --featureset stage1
```

```bash
python experiments/compare_literature_baselines.py --protocol person5 --featureset literature
```

Chẩn đoán khoảng cách tới con số công bố: `python experiments/diagnose_gap.py`; tóm tắt ở
`reports/chan_doan_a0_vs_baseline.md`.

Kết quả ở `reports/`: `stage1_<market>.md|json`, `predictions_<market>.csv`,
`literature_comparison_<protocol>_<featureset>.md|json`. Tạo phẩm khoá ở `artifacts/stage1/` cùng
manifest `lock.json`; bảng Mục 10.2 ở `data/gnwt.db`.

## Cấu trúc

```
config/params.yaml                 thủ tục, lưới, tiên nghiệm (không có hệ số)
data/lexicon/demand_lexicon_v1.yaml  từ điển 12 tính chất O*NET, vi/en, chỉ từ và chiều
data/catalog/basic_contexts_v1.yaml  catalog ngữ cảnh cơ bản (hồ sơ do bước O2 ghi)
src/gnwt/config.py                 Params, config_hash
src/gnwt/persona.py                Persona của chặng 1 (cờ nguồn Big Five)
src/gnwt/data/                     fetch.py, speed_dating.py (2 bộ dữ liệu), bfi10.py (proxy ngoại tuyến)
src/gnwt/stage1/context/           styles, text, lexicon, catalog, demand (11b)-(11c), fallback_embed (tắt)
src/gnwt/stage1/fit.py             (11a)
src/gnwt/stage1/features.py        đặc trưng, khối (14), chỉ báo thiếu, δ ⊥ D̃
src/gnwt/stage1/srrm.py            LockedSRRM, eta_blocks, a0 (12), m0 (13), Φ₂ (Genz)
src/gnwt/stage1/static.py          run_stage1 lúc chạy
src/gnwt/stage1/offline/           srrm_fit.py (Gibbs Hoff, LOWO lồng, cross-fit), reader.py (θ, k, ICC)
src/gnwt/analysis/                 metrics.py, branches.py (Mục 9)
experiments/compare_literature_baselines.py   so sánh với baseline từ paper
scripts/run_stage1.py              O6–O9 + đánh giá
tests/unit/                        U1–U12 và kiểm thử bộ ước lượng
```

## Kiểm thử

```bash
python -m pytest
```

## Twin sửa phần dư của nền LR + MF (`src/lrmf_twin/`)

Gói riêng, độc lập với GNWT, dựng theo thiết kế "Twin sửa phần dư của nền LR + MF". Mười mô-đun, chỉ T3 và
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
kaggle/                         kaggle_run_full.py (một file chạy full trên Kaggle), template + script đóng gói
```

Cần thêm extra `embed` (sentence-transformers): `pip install -e ".[dev,baselines,embed]"`. Chạy local (kết
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

**Chạy full trên Kaggle**: dùng một file duy nhất [kaggle/kaggle_run_full.py](kaggle/kaggle_run_full.py)
(Accelerator GPU T4 x2, Internet On; LLM Qwen2.5-7B-Instruct qua transformers; hướng dẫn ở đầu file). Sau khi
sửa mã, sinh lại file đó:

```bash
python kaggle/build_kaggle_script.py
```

Quy ước và chỗ lệch thiết kế: mục `lrmf_twin` trong [CHANGELOG.md](CHANGELOG.md).
