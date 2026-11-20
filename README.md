# GNWT-TM

Lõi GNWT-Agent, lớp động lực thiện cảm và hạ tầng thực nghiệm của đề tài GNWT-TM (Twin-Matching tương hỗ 1-n và
n-n), viết theo tab "Đặc tả kỹ thuật" của đề cương và kế hoạch 06–15/11.

## Repository layout (stages)

| Folder | Stage | Contents |
|---|---|---|
| [`stage1/`](stage1/README.md) | **Stage 1**: profile + context sentence -> a0, m0 | SRRM probit core, keyword context reader, Speed Dating (OpenML + Columbia), comparison with profile-matching baselines; no LLM |
| [`stage2/`](stage2/README.md) | **Stage 2**: twin | LR + MF residual twin (Twin-2K, GSS, MovieLens-1M), Kaggle runner |
| `src/gnwt/`, `config/` | shared GNWT infrastructure | Params, DB schema |

Each stage is self-contained (its own `pyproject.toml`, `tests/`, `data/`, `result/`); run its commands from its folder.

**Đã xong: B1** (Đặc tả Mục 1–4). `config/params.yaml` và model `Params` có `config_hash`; CSDL SQLite WAL đủ 14
bảng; `LLMClient` có cache theo nội dung, giới hạn đồng thời và tốc độ, thử lại có chờ luỹ thừa, một lần gọi sửa
JSON, mỗi lần thử một dòng `api_calls` có chi phí; `MockLLMClient` cho kiểm thử; model persona và 10 hồ sơ viết tay.

LLM tạm thời chạy local qua Ollama (`qwen2.5:3b`); các chỗ lệch đặc tả ghi trong [CHANGELOG.md](CHANGELOG.md).

## Cài đặt

Cần Python ≥ 3.12 và Ollama.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -e ".[dev]"
```

```bash
ollama pull qwen2.5:3b
```

## Chạy

```bash
python scripts/init_db.py
```

```bash
python scripts/smoke_llm.py
```

`init_db.py` tạo `data/gnwt.db` và nạp 10 persona; `smoke_llm.py` gọi mô hình thật hai lần với cùng prompt và kiểm
tra: lần một có đầu ra cấu trúc và dòng `api_calls`, lần hai trúng cache với chi phí 0.

## Cấu trúc

```
config/params.yaml          mọi tham số (Mục 2)
src/gnwt/config.py          Params, config_hash, load_params
src/gnwt/db/                schema.sql, repo.py (Mục 3.2)
src/gnwt/llm/               client.py, cache.py, pricing.py, providers.py (Ollama, fixture), mock.py (Mục 4)
src/gnwt/persona/schema.py  Persona, Tier 1–3 (Mục 3.1)
data/handwritten/           10 persona viết tay
scripts/                    init_db.py, smoke_llm.py
tests/unit/                 kiểm thử B1 (không gọi mạng)
```

Mã nguồn, chú thích và thông báo lỗi viết bằng tiếng Anh; tài liệu viết bằng tiếng Việt.

## Kiểm thử

```bash
python -m pytest
```
