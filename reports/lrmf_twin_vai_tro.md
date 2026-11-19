# Twin LR + MF: twin nằm ở tầng nào và vai trò thể hiện qua chạy thử

## 1. Twin được triển khai ở tầng nào

Twin **không phải bộ dự đoán chính**. Nền thống kê LR + MF (T1) chấm điểm cho mọi ô. LLM chỉ vào hệ thống qua
ba cửa mà thiết kế cho phép, ở hai tầng khác nhau: tầng **tham số của nền** và tầng **kết hợp đầu ra**.

```
 Tầng 0  Dữ liệu + fold + cờ chế độ z (T0)                         không LLM
            │
 Tầng 1  NỀN đóng băng ngoài fold: LR + MF (T1)  ──► η_B2            không LLM
            ▲            ▲
            │            └── T3 tiền nghiệm người thưa (LLM, 1 lời gọi / người)   ┐ cửa vào THAM SỐ của nền
            └── T2 mã hoá câu hỏi (embedder → MLP → b_j, v_j)                      ┘ → η_B3, η_B4
            │
 Tầng 2  TWIN hiệu chỉnh: T4 truy hồi top-k câu của chính người
                          → T5 LLM (T = 0, không CoT) → phân phối D       cửa PHẦN DƯ, chạy song song với nền
            │
 Tầng 3  KẾT HỢP (thống kê): T6 trực giao hoá D̃ = D − m̂(η)
                             T7 cổng w(z) đơn điệu giảm theo lượng dữ liệu
                             T8 super learner {B2, B3, B4, B7, B8}   ← "nền đứng riêng" luôn là ứng viên
                             T9 hiệu chuẩn
            │
         Dự đoán cuối B9
```

| Thang so sánh | Thêm gì so với bậc trước | Tầng | Có gọi LLM? |
|---|---|---|---|
| LR (baseline) | Hồi quy logistic theo từng câu: nhân khẩu + câu trả lời khác của người | — | Không |
| B2 = T1 | + MF (nhân tử người × câu), khớp chung với LR | Nền | Không |
| B3 = + T2 | Tham số câu hỏi sinh từ văn bản câu hỏi; câu mới vẫn dự đoán được | Tham số nền | Embedder |
| B4 = + T3 | Nhân tử của người thưa/người mới co về tiền nghiệm do LLM sinh | Tham số nền | Có |
| B5–B8 | + tín hiệu twin D (thô / trực giao / có cổng / twin thấy nền) | Kết hợp | Có |
| B9 | Super learner + hiệu chuẩn trên các bậc trên | Kết hợp | — |

Hệ quả của vị trí này:

- Ở chế độ dữ liệu dày, nền đã nắm phần lớn tín hiệu. Twin chỉ được ghi công cho phần nền không đoán được (D̃).
- Ở chế độ nền thiếu tín hiệu, cửa T2/T3 và cổng w(z) mở. Đó là câu mới (R2), người thưa/người mới (R4/R5), câu lạnh (R6) và đợt khuyết (R3).
- Super learner giữ "nền đứng riêng" làm ứng viên. Vì vậy, *khi đủ dữ liệu để ước lượng*, mô hình cuối không kém nền quá mức nhiễu.

## 2. Vai trò thể hiện qua lượt chạy thử

Lượt chạy: `result/20261119-000631_quick_demo`, Twin-2K-500 với 600 người được lấy mẫu, Ollama qwen2.5:3b trên CPU.
Mỗi chế độ có 8 ô gọi twin T5a; R4 có 20 người thưa. Đây là lượt chạy thử: tầng 1 đủ lớn để kết luận, còn tầng 2 thì chưa.

### Tầng 1 (mọi ô, CI bootstrap theo người)

| Chế độ | Kết quả so với LR | Đọc ra vai trò |
|---|---|---|
| R1 dày (341.603 ô) | B2 (LR+MF): ΔAUC **+0,0025** [+0,0022; +0,0027], Δlog-loss −0,0031. B3 (+T2): ΔAUC +0,0014, Δlog-loss −0,0041 | Nền làm gần như toàn bộ việc. MF thêm một chút. Văn bản câu hỏi không thêm gì đáng kể, đúng dự đoán "0 đến +0,005" của thiết kế. |
| R2 câu mới (341.603 ô) | Không có văn bản câu hỏi, B2 chỉ đạt AUC 0,558. Có T2, B3 đạt **0,651**, tức +0,093 so với B2. So với LR dùng văn bản câu hỏi (0,657): ΔAUC −0,006 [−0,008; −0,004] | Cửa T2 là thứ biến câu hỏi chưa có dữ liệu thành dự đoán được. Tuy vậy, với embedder MiniLM thì nó chưa hơn một LR dùng văn bản. |
| R4 người thưa (20 người, 11.184 ô) | B2: ΔAUC **+0,0059** [+0,0031; +0,0093]. B3: Δlog-loss +0,015 [−0,009; +0,051]. B4 (+T3 qwen 3B): Δlog-loss +0,021 [−0,006; +0,052] | MF giúp người thưa. T2 và tiền nghiệm từ LLM 3B chưa giúp (CI chứa 0, nghiêng về xấu hơn). |

### Tầng 2 (twin T5, 8 ô mỗi chế độ, gộp 24 ô)

- **Twin đứng riêng yếu:** AUC 0,07 (R1), 0,63 (R4), 0,07 (R2). Trên 8 ô, các con số này gần như ngẫu nhiên, nhưng không có dấu hiệu twin 3B tự đoán được.
- **LR test β (offset η_B4):** gộp p = 0,061, β = −0,33. Hệ số âm cho thấy phần dư mà qwen 3B đưa ra có xu hướng ngược chiều phần dư thật: Corr(D̃, Y − p_B4) = −0,37 [−0,67; 0,04]. Kết quả không có ý nghĩa sau hiệu chỉnh Holm.
- **Cơ chế phòng thủ có hoạt động:**
  - D thô đưa vào stacking tuyến tính, không cổng (B5), là cấu hình hại nhất: Δlog-loss +0,74 ở R1 và +0,58 ở R2. Điều này khớp cảnh báo của thiết kế (KAR, S-LLMR).
  - Bản trực giao + cổng (B6/B7) ít hại hơn hẳn.
  - Super learner đặt trọng số chủ yếu vào nền: B2 0,48, B3 0,35, B7 0,17.
- **Giới hạn của lượt chạy:** với 24 ô, bảo đảm "không kém nền" của super learner chưa có hiệu lực, vì đó là bảo đảm tiệm cận. B9 vẫn có lúc kém LR (R2: Δlog-loss +0,19, CI chứa 0).

### Kết luận tạm thời

1. Twin đúng là một **bộ sửa phần dư**, không thay nền. Ở dữ liệu dày, phần đóng góp gần 0, đúng như thiết kế dự đoán.
2. Vai trò rõ nhất hiện thấy là **cửa T2 cho câu mới**: AUC +0,09 so với nền không có văn bản.
3. Vai trò của LLM (T3, T5) **chưa đo được** với qwen2.5:3b và 8 ô/chế độ. Cần bản chạy full trên Kaggle (Qwen2.5-7B, 150 ô/chế độ, ba bộ dữ liệu Twin-2K, GSS, MovieLens-1M) để có LR test và CI đủ hẹp.
4. Chi phí nằm gần như hoàn toàn ở lời gọi LLM. Trên CPU laptop là 12–50 giây/lời gọi; trên GPU Kaggle, sinh theo lô sẽ nhanh hơn nhiều.
