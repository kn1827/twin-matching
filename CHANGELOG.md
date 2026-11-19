# CHANGELOG

Ghi mọi chỗ bản hiện thực lệch khỏi đặc tả Stage 1 (`stage1.md`), kèm lý do. Nhãn **[đề xuất]**,
**[quy ước]** dùng như trong đặc tả.

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
- **Kaggle**: `kaggle/kaggle_run_full.py` nhúng toàn bộ mã (zip base64), tải dữ liệu, chạy ba bộ dữ liệu × R1–R6
  với Qwen2.5-7B-Instruct, in và lưu bảng tổng hợp `overall_*.md|csv`, nén `result_bundle.zip`.

## lrmf_twin 0.1.0 — twin sửa phần dư của nền LR + MF (18/11/2026)

Gói mới `src/lrmf_twin/`, không đụng tới mã GNWT. Dựng theo "Thiết kế twin lồng LR MF" (mục 4–6) trên
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

## 2.0.0 — Stage 1 viết lại (15/11/2026)

### Xoá và giữ lại

- **Xoá** toàn bộ mã B1 gắn với twin/LLM: `src/gnwt/llm/` (client, cache, Ollama provider, mock),
  `src/gnwt/persona/` (Tier 1–3), `scripts/init_db.py`, `scripts/smoke_llm.py`, kiểm thử B1, `data/cache/`,
  `data/gnwt.db`. Chặng 1 không được gọi LLM ở bất kỳ bước nào (đặc tả, mở đầu và 10.1).
- **Xoá khỏi `params.yaml`** các khối `llm`, `core`, `session`, `affinity`, `stage3`, `stage4` và khối
  `stage1` cũ (`mu`, `kappa1`, `kappa2`, `rho_P`, `rho_G`, `pi_by_context`): đặc tả 10.6 thay khối này bằng
  thủ tục, lưới và tiên nghiệm; các khối kia thuộc những chặng đã xoá mã.
- **Giữ** mẫu `config.py` (model Pydantic đóng băng + `config_hash`), `db/repo.py` (SQLite WAL, đổi danh
  sách bảng theo 10.2) và 10 persona viết tay `data/handwritten/` (đọc bằng `gnwt.persona.load_handwritten`).

### Dữ liệu

- Hai bộ dữ liệu được hiểu như sau: **`columbia`** = tệp gốc của Fisman, Iyengar, Kamenica & Simonson
  (2006) tại Columbia (có `iid`/`pid`, `goal`, `field_cd`); **`speed_dating`** = OpenML 40536, cùng thí
  nghiệm nhưng mã hoá lại (không có `iid`/`pid`, không có `goal`, `go_out`). Mã người của OpenML được dựng
  lại từ câu trả lời cấp người chọn; đối tác được ghép theo `age_o`, `race_o`, `pref_o_*` và kiểm bằng hàng
  ngược (cùng `interests_correlate`, quyết định hoán đổi): 8360/8378 hàng ghép được.
- Trên `speed_dating` khối mục tiêu không bao giờ được thu thập: số hạng bị bỏ, hệ số chặn hấp thụ phần
  còn lại (Mercaldo & Blume, đúng bảng Mục 6).
- Điểm sở thích ngoài 1–10 trong tệp gốc (0, 13, 14) được cắt về [1, 10] **[quy ước]**.

### Lệch có chủ ý

- **Proxy Big Five không dùng LLM** (`gnwt/data/bfi10.py`, ngoài chặng 1): rev 110 dựng BFI-10 proxy
  bằng LLM. Bản này dùng tổ hợp đơn vị (z-score, trọng số ±1, chỉ dấu lấy từ tài liệu) của sở thích, tự
  đánh giá và tần suất đi chơi. N không có chỉ báo nên luôn thiếu, ES bị che theo luật nét. Độ hợp lệ của
  proxy chưa đo **[chưa kiểm chứng]**; có thể thay proxy LLM ngoại tuyến qua cùng giao diện.
- **Bộ lấy mẫu**: Gibbs của Hoff (đặc tả cho phép "Gibbs của Hoff hoặc NUTS"), viết bằng NumPy vì môi
  trường không có NumPyro. (β, a, b) được rút đồng thời theo từng thành phần liên thông (đợt), dùng bù
  Schur, để hệ số chặn không trộn chậm với hiệu ứng người. Hiệp phương sai mặc định LKJ(2) + half-Cauchy(1)
  qua Metropolis; nhánh độ nhạy `inverse_wishart` là Gibbs liên hợp.
- **Chọn lưới lồng dùng xấp xỉ Laplace/MAP** của probit biên Φ(x'β/s), với s lấy từ một lần chạy Gibbs
  ngắn trên cùng các đợt huấn luyện (`selection.inner_estimator: laplace_map`). Chạy Gibbs cho
  21 × 20 × 20 lần khớp trong không khả thi; mô hình cuối và cross-fit vẫn là Bayes đầy đủ. Bộ chuẩn hoá
  của vòng trong dùng lại bộ chuẩn hoá của vòng ngoài (không dùng nhãn, nên không rò nhãn).
- **Quy tắc 1 SE**: khi hoà, chọn độ rộng tiên nghiệm nhỏ hơn, rồi tỉ lệ che khối nhỏ hơn.
- **Cross-fit dùng 2 chuỗi** (`sampler.crossfit_chains`), mô hình khoá dùng 4 chuỗi; R̂ được ghi cho
  từng đợt và cho mô hình khoá. Các nhánh SRRM phụ (S8, không khối cặp, GLMM) dùng 1 chuỗi.
- **Thêm giới tính người chọn vào η₀** (`features.actor: gender`): bảng đầu vào Mục 1 liệt kê giới là
  trường lõi nhưng bảng Mục 5 không có; Speed Dating dị tính nên giới đối tác trùng giới người chọn.
- **Không có hồ sơ catalog nào được chấm**: `profile: null` cho mọi ngữ cảnh, đúng 10.4 ("không viết
  tay"). Ngữ cảnh chưa chấm có hồ sơ trung tính (0), theo luật "ICC thấp/không biết → D̃_d = 0". Trên
  Speed Dating vì thế D̃_SD = 0, fit = 0 và γ_P không định danh; δ^SD tự do, trùng mô hình có hệ số Big
  Five đối tác tự do (đặc tả đã nói khối ngữ cảnh không đổi dự báo trên Speed Dating). Nhánh S8 (δ = 0)
  khi đó kiểm "Big Five đối tác có giúp không".
- **Chưa có bộ nhãn `demand_labels`**: θ_κ, k chưa ước lượng. `ReaderArtifacts.unestimated` vẫn dò và
  ghi từ khoá nhưng đặt Z = 0 (tức k = ∞), nên D̂ = D^def. Bộ ước lượng `estimate_reader` (PAV/bình
  phương tối thiểu có thứ tự + Bühlmann, CV nhóm, ICC) đã có và được kiểm trên dữ liệu mô phỏng.
  Các số minh hoạ của Mục 3.5 nằm trong `tests/fixtures/` (gắn cờ `illustrative`), chỉ dùng cho kiểm thử.
- **θ theo thứ tự**: khi mỗi hàng chỉ có một lớp từ khoá thì đây đúng là PAV; với hàng nhiều lớp, ràng buộc
  thứ tự được giải bằng bình phương tối thiểu có cận (`scipy.optimize.lsq_linear` trên tham số tích luỹ).
- **Luật từ điển [quy ước] bổ sung**: "không"/"not"/"no" là phủ định đứng trước (cần cho "không cạnh
  tranh" → base); bổ ngữ đứng trước mà sau nó không còn từ khoá thì tác dụng ngược về trước
  ("kỷ luật là bắt buộc" → strong, "kỷ luật không cần" → neg); cấm + từ khoá đúng chiều → rev; mạnh/yếu +
  từ khoá ngược chiều → rev; "but also" không tách mệnh đề; khớp không dấu chỉ áp lên âm tiết gõ không dấu
  (để "hỏi" không khớp "hơi"); chuẩn hoá vị trí dấu ("hoà" = "hòa"). Danh sách `ignore` (ví dụ "không
  khí") chặn phủ định giả.
- **Nhánh chưa chạy**: TabPFN (cần checkpoint ghim, tải từ mạng) và A0-cũ (công thức rev 110 không có
  trong đặc tả mới). GLMM được xấp xỉ bằng SRRM probit với ρ = 0 và Σ chéo (đặc tả ghi logit).
- **`static_scores` của các đợt phát triển** lấy từ cross-fit (I1), `srrm_hash` là băm của mô hình đợt.
- **O9** ghi băm vào `artifacts/stage1/lock.json` và bảng `artifacts` thay vì sửa `params.yaml` tự động
  (giữ chú thích của YAML); `params.yaml` trỏ tới manifest qua `stage1.locked.manifest`.
- `demand_centering: z_over_catalog` (độ nhạy S11) và `slope: per_trait` mới có khoá cấu hình, chưa có
  mã: cả hai cần ít nhất hai ngữ cảnh có nhãn.
- **Hệ số phân cấp theo ngữ cảnh** (β^c = β̄ + ζ^c, half-Cauchy) và **cập nhật Laplace theo lô E4** chưa
  hiện thực: cả hai bộ dữ liệu chỉ có một ngữ cảnh, nên β^SD = β̄ + ζ^SD không tách được và không đổi dự
  báo. Phần δ ⊥ D̃ đã viết tổng quát theo từng ngữ cảnh cơ bản.
