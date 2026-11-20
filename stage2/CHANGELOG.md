# CHANGELOG

Ghi mọi chỗ bản hiện thực lệch khỏi thiết kế "Twin sửa phần dư của nền LR + MF", kèm lý do. Nhãn **[đề xuất]**,
**[quy ước]** dùng như trong đặc tả.

## Moved to stage2/ (19/11/2026)

- The whole twin (`src/lrmf_twin`, `experiments/`, `kaggle/`, `reports/`, `tests/unit/`) and its data,
  cache and results moved from the repository root into `stage2/`, next to the new `stage1/`. Paths are
  resolved from `stage2/` (`ROOT = parents[1]` of the scripts), so commands are unchanged when run from
  there. The Kaggle notebook now runs `stage2/kaggle/run_full.py`. No code behaviour changed; the 22
  unit tests pass from `stage2/`.

## lrmf_twin 0.2.0 — GSS + MovieLens-1M, chế độ R3/R5/R6, Kaggle (19/11/2026)

- **Bộ dữ liệu** (chốt với người dùng sau khi tra cứu): GSS 1972–2024 (NORC, Release 3a; Kim & Lee, Chen et al.
  2026, AlignSurvey) và MovieLens-1M (Kang et al., KAR, CoLLM, BinLLM, FLIP, LLMRank). Twin-2K giữ nguyên;
  chế độ nào bộ dữ liệu không có thì để trống (chỉ R3 trên Twin-2K).
- **GSS [quy ước]**: người 2000–2024 có ≥ 20 câu trả lời; câu = biến sau `natspac` theo thứ tự file, có nhãn giá
  trị, 2–10 mã hợp lệ, ≥ 2.000 câu trả lời; bỏ biến hành chính, danh sách hộ/mạng quan hệ, điều kiện khảo sát
  tiếp, biến tổng hợp chứa nhãn (wordsum), biến nhân khẩu đã dùng làm đặc trưng, và bản biến thể theo chế độ
  phỏng vấn của một câu đã giữ (abanyg, trustv, trustnv…; abany/abanyg có 647 người trả lời cả hai). Kết quả
  36.055 người × 380 câu, 3,95 triệu ô. Văn bản câu = nhãn biến Stata (ngắn) + nhãn giá trị. Đợt = năm khảo sát.
- **MovieLens-1M [quy ước]**: y = 1{điểm ≥ 4}; phim có ≥ 20 đánh giá (3.043 phim, 99,5% đánh giá); "câu hỏi" =
  tên + năm + thể loại; đợt = quý của lần đánh giá; đặc trưng người = giới, nhóm tuổi, nghề. Θ ("câu trả lời
  khác") chỉ dùng 800 phim nhiều đánh giá nhất làm đặc trưng (`max_feat_items`).
- **Chế độ mới**: R3 giữ lại nguyên khối (câu, đợt) như retrodiction của Kim & Lee; R5 người mới (0 câu đã
  biết, T3 chỉ dựa vào nhân khẩu); R6 câu lạnh mô phỏng (giữ 5 câu trả lời/câu) cho mọi bộ dữ liệu, thay vì
  câu lạnh tự nhiên, để ba bộ dữ liệu cùng một định nghĩa.
- **LR theo từng câu bằng torch** (mặc định, GPU nếu có): cùng mô hình với sklearn (λ = 1/C, chặn gần như không
  phạt), khớp mọi câu một lần; test kiểm khớp sklearn (tương quan > 0,999). sklearn vẫn dùng được
  (`--lr-engine sklearn`) cho dữ liệu nhỏ.
- **B0** làm trơn bằng 2 câu trả lời giả theo tỉ lệ chung (câu lạnh 5 câu trả lời giống nhau không còn ra 0/1).
- **LLM**: provider `hf:` (transformers trong tiến trình, fp16, greedy, sinh theo lô `LRMF_HF_BATCH`, lượng tử
  hoá tuỳ chọn `LRMF_HF_QUANT=4bit|8bit`, khoá cache tách riêng khi lượng tử hoá); T3/T5 gọi theo lô mỗi fold.
- **Kaggle**: notebook `kaggle/twin_lrmf_kaggle.ipynb` clone repo và chạy `kaggle/run_full.py`: tải dữ liệu, ba bộ dữ liệu × R1–R6
  với Qwen2.5-7B-Instruct, in và lưu bảng tổng hợp `overall_*.md|csv`, nén `result_bundle.zip`.

## lrmf_twin 0.1.0 — twin sửa phần dư của nền LR + MF (18/11/2026)

Gói `src/lrmf_twin/`. Dựng theo "Thiết kế twin lồng LR MF" (mục 4–6) trên
Twin-2K-500. Chỗ thiết kế để ngỏ và chỗ lệch:

- **Nhãn [quy ước]**: thiết kế chỉ nói "nhị phân hoặc từng lựa chọn". Câu có thứ tự (Likert, Bipolar, slider,
  BDI = mức nặng nhất được đánh dấu, MC ngoài phần nhận thức) → y = 1{giá trị > điểm cắt}, điểm cắt cho tỉ lệ
  dương gần 0,5 nhất (chỉ dùng phân phối biên của câu); câu 2 lựa chọn → y = 1{lựa chọn 1}; MC nhận thức ≥ 3
  lựa chọn → y = 1{lựa chọn phổ biến nhất}. Bỏ câu tự điền (TE), DB, Wason (MAHR), câu < 200 trả lời hoặc tỉ lệ
  dương ngoài [0,05; 0,95]. Không dùng wave 4 (lặp lại wave 1–3, sẽ rò nhãn). Nhân khẩu là đặc trưng, không là nhãn.
- **Chế độ**: có R1 (dày), R4 (người thưa, `n_known` câu), R2 (câu mới). R3 (đợt khuyết), R5, R6 không có trên
  Twin-2K. Fold nền theo đúng chế độ (ô ngẫu nhiên / người thưa / câu), còn tầng kết hợp T6–T9 cross-fit theo
  nhóm người, gộp mẫu của mọi chế độ để cổng T7 học từ z.
- **T1**: "LR" trong nền gồm hiệu ứng cố định của câu, hệ số riêng theo câu cho nhân khẩu và cho câu trả lời
  khác của người (đường chéo bị khoá), khớp đồng thời với MF hạng 16. λ chọn trên một tách trong của ô huấn
  luyện fold 0 R1 rồi khoá: λ_Θ = 200 (≈ 1/C của LR sklearn tốt nhất, C ≈ 0,005), λ_W = 100. LR văn bản (R2): C = 0,005
  chọn trên tách câu-mới trong của câu huấn luyện fold 0 R2. Lưu ý: các tách phát triển này lấy từ toàn bộ
  ma trận nên chồng với ô kiểm tra của lần chạy chính; chỉ 2–3 siêu tham số toàn cục, chọn cùng cách cho LR và nền.
- **T2**: đầu ngữ nghĩa chỉ sinh b_j và v_j (đúng "nhân tử câu hỏi kiểu MF và hiệu ứng câu hỏi"); thêm W_j
  (`sem_targets="bwv"`) làm quá khớp trên tách trong. Embedder mặc định all-MiniLM-L6-v2 (không phải LLM);
  `--embedder ollama:qwen2.5:3b` là bản "embedding LLM".
- **T3**: tiền nghiệm đi vào dưới dạng nhãn mềm trên 8 câu neo (nhân tử MF lớn, không cộng tuyến), trọng số
  n0/(n0 + n_p), n0 = 10, chỉ cho người có ≤ 25 câu đã biết [đề xuất].
- **T5**: 1 lời gọi/ô, T = 0, JSON `{"p": [...]}`; P(y = 1) = tổng xác suất các lựa chọn dương; lỗi → D = NaN
  → D~ = 0. Chỉ chạy trên `--llm-cells` ô mỗi chế độ (mặc định 30) vì qwen2.5:3b trên CPU mất ~12 s/lời gọi.
- **T6** hồi quy D trên [1, η, η²] của η_B4. **T7** cổng σ(a0 − softplus·log1p(n_p) − softplus·log1p(n_j) +
  softplus·new_item). **T8** trên {B2, B3, B4, B7, B8}. **T9** Platt toàn cục, co về hiệu chuẩn đồng nhất (a = 0, b = 1) với trọng số 5 quan sát
  giả, vì mẫu tầng 2 nhỏ (chưa theo từng câu). Với vài chục ô, T9 vẫn có thể làm log-loss của B9 xấu đi.
- **Kiểm định**: LR test của β (offset η_B4) + Holm trên chế độ × biến thể; ΔAUC/Δlog-loss bootstrap theo
  người; AUC theo câu tính trên từng (câu × fold) để tránh tạo tác gộp dự đoán từ các fold khác nhau.
  SAGE thay bằng Shapley 2 nhóm {nền, twin} = trung bình "thêm sau" và "thêm trước".
- **Ollama**: client giữ model nạp sẵn (`keep_alive` 30m); khi runner GPU (Vulkan, MX230) crash vì thiếu bộ nhớ,
  client tự chuyển sang CPU (`num_gpu = 0`) và ghi `cpu_fallback` vào `config.json`. Lời gọi lỗi không được cache:
  chạy lại cùng lệnh chỉ gọi lại các ô lỗi.
- **Chưa làm**: O1–O4, C1, C2, C4, C5, số nhãn tương đương, PPI, ΔAUC theo nhóm con.
