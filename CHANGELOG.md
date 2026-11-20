# CHANGELOG

Ghi mọi chỗ bản hiện thực lệch khỏi một quy tắc **NÊN** của Đặc tả kỹ thuật, kèm lý do (Đặc tả Mục 0).
Chỗ lệch một quy tắc **PHẢI** không được ghi ở đây mà phải sửa đặc tả và đề cương cùng lúc.

## B1 — khung dự án, LLM client, ghi chi phí (07/11/2026)

### Lệch tạm thời, PHẢI trả trước E1

- **LLM chạy local qua Ollama (`qwen2.5:3b`) cho cả bốn vai trò** thay vì Gemini Flash-Lite cho twin và
  một mô hình khác họ cho giám khảo, độ ấm, Big Five proxy (Mục 1, Mục 2). Lý do: chưa có khoá API; dựng
  và kiểm thử lõi trước. Hệ quả:
  - chi phí ghi 0 vì giá trong `params.yaml` là 0 cho mô hình local; số token vẫn ghi đủ trong
    `api_calls`, nên ước lượng lại chi phí theo bảng giá API được bằng một truy vấn;
  - giám khảo cùng họ với twin, nên mọi điểm judge, warmth, bfi_proxy trên cấu hình này chỉ dùng để
    thử luồng, không dùng cho số đo nào của đề cương;
  - độ trễ đo được (khoảng 20 s mỗi lời gọi ngắn khi mô hình đã nạp, máy có GPU 2 GB) không đại diện
    cho API; mốc độ trễ và chi phí của B5 phải đo lại trên mô hình thật.
  - Việc cần làm khi có khoá: cài provider `gemini` trong `src/gnwt/llm/providers.py` (hiện ném
    `NotImplementedError`), đổi `llm.*.provider/model`, điền giá theo bảng giá chính thức, đặt
    `concurrency: 20`, `timeout_s: 60`, `requests_per_minute` theo hạn mức.

### Lệch có chủ ý (NÊN)

- **Mã nguồn viết hoàn toàn bằng tiếng Anh** (chú thích, docstring, thông báo lỗi), theo quyết định của nhóm.
  Đặc tả Mục 0 quy định văn bản giải thích viết tiếng Việt; tài liệu (README, CHANGELOG) vẫn viết tiếng Việt.
- **Bỏ gói `gnwt_tm` cũ** (bản dựng twin trước đề cương) cùng `configs/`, `docs/twin_design.md`,
  `data/data_sample.json`, `data/schema/` và kiểm thử của nó; toàn bộ mã được viết mới trong `src/gnwt`.
- **Python 3.12** thay 3.11 (môi trường sẵn có; không dùng tính năng riêng của 3.12 trong `src/gnwt`).
- **Khoá thêm trong `params.yaml`**: `llm.timeout_s` (đặc tả ghi cứng 60 s; để 180 s cho máy local) và
  `llm.ollama` (`base_url`, `keep_alive`, `num_ctx`). Hai khoá này nằm trong `config_hash`.
- **`llm.ollama.num_gpu: 0`** (chạy thuần CPU): driver GPU 2 GB bị sập khi tải mô hình ("GPU is lost").
  Sau khi khởi động lại máy có thể đặt `null` để Ollama tự đẩy một phần mô hình lên GPU.
- **`concurrency: 1`** khi dùng Ollama: máy chủ local xử lý tuần tự, gửi song song chỉ làm các yêu cầu
  xếp hàng phía máy chủ và quá hạn. Bốn mô-đun vẫn chạy song song ở tầng đồ thị (B2).
- **Dòng `api_calls` trúng cache ghi token 0** (cùng chi phí 0), để tổng token theo `api_calls` bằng
  đúng số token thật đã tiêu.
- **`CallMeta` là tổng của một lời gọi logic**: token, chi phí, độ trễ cộng qua mọi lần thử (kể cả lần
  gọi sửa); `attempt` = số lần đã gửi; `call_id` = dòng `api_calls` cuối cùng. Mỗi lần thử vẫn có
  dòng `api_calls` riêng như đặc tả.
- **Độ trễ `latency_ms`** chỉ tính thời gian gọi API, không tính thời gian chờ phép của bộ giới hạn.
- **Khoá cache** mã hoá bảy thành phần của Mục 4 bằng một mảng JSON thay cho nối chuỗi bằng `|`, để
  ranh giới giữa các thành phần không nhập nhằng. `schema_version` = thuộc tính lớp `SCHEMA_VERSION`
  của lược đồ đầu ra (mặc định "1").
- **Seed của Ollama** suy ra từ khoá cache và `global_seed`, nên cùng prompt cho cùng đầu ra kể cả
  khi xoá cache.
- **`MockLLMClient`** là `LLMClient` với `FixtureProvider`, nên kiểm thử tích hợp đi qua đúng đường
  cache, kiểm tra lược đồ, gọi sửa và ghi `api_calls` như client thật.
- **Purpose `smoke`** (hạn mức token nhóm `module`) chỉ dùng cho `scripts/smoke_llm.py`.
- **Persona**: `tier1.display_name` (Mục 12.3) và `tier2.look_for` ("what you look for", Mục 12.2)
  được thêm vào model của Mục 3.1. Mọi khoá bắt đầu bằng `attr` bị từ chối.
- **Lỗi kết nối tới Ollama** (máy chủ chưa chạy) được coi như lỗi tạm thời và thử lại như 5xx.

### Mâu thuẫn trong đặc tả, cần sửa đặc tả

- Mục 16.4 yêu cầu không tệp nào ngoài `src/gnwt/analysis/` chứa chuỗi `sealed`, nhưng Mục 2 đặt
  khoá `data.sealed_dir` trong `params.yaml`, nên `src/gnwt/config.py` buộc phải chứa chuỗi này; Mục
  12.1 cũng cần `data/speed_dating.py` ghi tệp nhãn niêm phong. Đề xuất: phép quét tĩnh kiểm tra việc
  **đọc** thư mục nhãn, miễn trừ `config.py` (khai báo) và hàm niêm phong trong `data/speed_dating.py` (chỉ ghi).
