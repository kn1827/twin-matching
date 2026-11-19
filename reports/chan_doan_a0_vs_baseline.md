# Vì sao a⁰ có vẻ kém các baseline đã công bố

Chỉ so **một mô hình của mình (a⁰, SRRM probit của Stage 1)** với các baseline profile matching lấy từ paper
hạng B trở lên: FM, DeepFM, xDeepFM, DCN-V2, AutoInt, IPNN, biDeepFM (biDeepFM, JESTECH 2021, Q1; biLorentzFM,
Applied Sciences 2025, Q2) và XGBoost, LightGBM, CatBoost (McElfresh et al., NeurIPS 2023 D&B, A*). Mọi mô
hình được huấn luyện lại trên cùng hàng và cùng fold với a⁰. Thước đo chính: Part A AUC, tức AUC của quyết
định có hướng `dec`.

## 1. Kết luận ngắn

a⁰ chỉ kém con số **công bố** (0,68–0,70). Khi chạy trên **cùng bài toán** của Stage 1 (đoán cho người mới,
chia bỏ-một-đợt), a⁰ **không thua baseline nào**. Khoảng cách tới con số công bố gồm ba phần, xếp theo độ lớn:

| Nguồn chênh (Columbia, Part A AUC) | Cỡ | Có phải lỗi của a⁰? |
|---|---|---|
| Giao thức chia của paper rò danh tính đối tác | ≈ +0,09 | Không: a⁰ không được phép dùng |
| Paper dùng thêm trường a⁰ không đọc (tự chấm ngoại hình, trọng số sở thích, chủng tộc) | ≈ +0,02 | Một phần là quy ước của đặc tả |
| a⁰ nén 17 điểm sở thích thành độ giống (r^int, Jaccard), mất mức sở thích riêng của từng người | ≈ +0,03 | **Có**: sửa được |
| Độ co hệ số, probit thay logit | ≤ 0,007, không ý nghĩa | Không |

## 2. Cùng giao thức của Stage 1 (bỏ-một-đợt, đặc trưng của a⁰)

| Mô hình | Columbia | Speed Dating (OpenML) |
|---|---|---|
| **a⁰ (SRRM)** | **0,572** | **0,564** |
| LR (embedding) | 0,558 (ngang) | 0,555 (ngang) |
| AutoInt | 0,560 (ngang) | 0,542 (kém) |
| xDeepFM / DCN-V2 | 0,549 (kém) | 0,549–0,550 (chưa rõ) |
| DeepFM / biDeepFM / IPNN / FM | 0,536–0,545 (kém) | 0,535–0,545 (kém hoặc chưa rõ) |
| XGBoost / LightGBM / CatBoost | 0,536–0,545 (kém) | 0,485–0,512 (kém, có lúc dưới ngẫu nhiên) |

Khoảng tin cậy 95% lấy từ bootstrap theo người chọn (B = 2000); "ngang" là TOST trong ±0,03. Trên cả hai bộ
dữ liệu, a⁰ không thua mô hình nào, và hơn có ý nghĩa thống kê ở 9/11 (Columbia) và 7/11 (Speed Dating). a⁰
còn hiệu chuẩn tốt nhất: log-loss 0,672 so với 0,679–0,907 của các baseline sâu. Bảng đầy đủ:
`literature_comparison_lowo_stage1_columbia.md` và `literature_comparison_lowo_stage1_speed_dating.md`.

## 3. Chẩn đoán từng nguồn chênh

Số liệu lấy từ `diagnosis_gap.md` (sinh bằng `experiments/diagnose_gap.py`).

### 3.1 Giao thức chia của paper rò danh tính đối tác (nguồn lớn nhất)

biLorentzFM chia 5 fold **theo người tham gia**. Khi giữ lại quyết định của người chọn i, đối tác j của i vẫn
xuất hiện trong tập huấn luyện như người được người khác chọn. Có hai điều cho thấy mô hình học được "j được
ưa chuộng đến đâu":

- 98,9% người có vector hồ sơ duy nhất, nên mô hình cây và mô hình sâu nhận ra được từng đối tác.
- Chỉ riêng độ phổ biến của j trong tập huấn luyện đã cho AUC = 0,705, cao hơn cả biDeepFM công bố.

Tái lập trên Columbia với đặc trưng literature:

| Giao thức | LightGBM | biDeepFM | DeepFM | a⁰ |
|---|---|---|---|---|
| 5 fold theo người (như paper) | 0,694 | 0,652 | 0,662 | 0,587 |
| Bỏ-một-đợt (người mới hoàn toàn) | 0,594 | — | — | 0,572 |

Theo giao thức của paper, các baseline cho 0,645–0,694, tức đúng khoảng 0,68 đã công bố; biDeepFM_Base công bố
là 0,682. Đổi sang người mới, cùng LightGBM rơi xuống 0,594. Phần AUC này là **nhớ đối tác**, không phải
dự báo từ hồ sơ. a⁰ được định nghĩa cho người mới (a_i = b_j = 0), nên theo đúng đặc tả nó không được dùng
phần này.

### 3.2 Trường dữ liệu a⁰ không đọc

Hồi quy logistic L2 trên trường thô, chia bỏ-một-đợt:

| Đặc trưng | Columbia | Speed Dating |
|---|---|---|
| Trường của Stage 1 | 0,607 | 0,588 |
| + tự chấm ngoại hình (`attr3_1`) | 0,611 (+0,004) | 0,593 (+0,005) |
| + chủng tộc, cùng chủng tộc, mức coi trọng chủng tộc/tôn giáo, 6 trọng số sở thích | 0,626 (+0,019) | 0,612 (+0,024) |

Trường ngoại hình bị đặc tả cấm, và nó chỉ góp +0,004. Phần lớn hơn đến từ trọng số sở thích và chủng tộc.
Hai nhóm này không bị cấm, nhưng persona của Stage 1 không có.

### 3.3 Cách a⁰ mã hoá sở thích (phần a⁰ tự để mất)

Trên cùng thông tin, hồi quy logistic thường đạt 0,607, trong khi chính ma trận thiết kế của a⁰ chỉ đạt
0,578–0,579 dù đổi C từ 0,01 tới 100. Ablation trên Columbia:

| Biến thể | AUC |
|---|---|
| Logit, mọi trường thô | 0,607 |
| − 34 điểm sở thích thô (chỉ giữ r^int, Jaccard) | 0,587 (−0,020) |
| − one-hot ngành, mục tiêu (chỉ giữ cùng ngành, cùng mục tiêu) | 0,602 (−0,005) |
| − proxy Big Five | 0,604 (−0,004) |
| Ma trận thiết kế của a⁰ | 0,579 |

Theo Mục 5 của đặc tả, sở thích chỉ vào η_G dưới dạng *độ giống* giữa hai người. Mức sở thích riêng của đối
tác (ví dụ người thích clubbing được chọn nhiều hơn) và của người chọn (độ kén) bị bỏ. Đó là hiệu ứng chính
cấp đối tác/tác nhân, không phải so khớp chéo nét. Lưu ý: đặc tả chỉ cấm ma trận W chéo nét, không cấm hiệu
ứng chính này.

### 3.4 Không phải do độ co hoặc do probit

- Vòng chọn lưới luôn chọn độ rộng tiên nghiệm 0,05 (log-loss 0,6713 so với 0,6737 ở 1,0).
- AUC gần như không đổi theo độ co: 0,578–0,579 với mọi C.
- a⁰ so với B1 (logit, cùng đặc trưng): ΔAUC = +0,007, khoảng tin cậy chứa 0.

### 3.5 Trần dữ liệu

- Phần dự báo được từ hồ sơ chỉ giải thích R² ≈ 2,4% phương sai tiềm ẩn (1,8% trên Speed Dating). Phần còn
  lại nằm ở hiệu ứng người: σa² ≈ 0,72, σb² ≈ 0,53.
- Nếu biết mọi a_i, b_j (tác động cố định, trong mẫu), AUC lên tới 0,896.

Hồ sơ trước buổi hẹn chỉ chạm được một phần nhỏ của "ai thích ai". Điều này khớp với Joel, Eastwick & Finkel
(2017), mà đặc tả đã nêu ở Mục 9.

### 3.6 Mô hình linh hoạt hơn không giúp với người mới

Dưới giao thức bỏ-một-đợt, LightGBM kém logit (0,554 so với 0,607 trên Columbia). Trên Speed Dating, các mô
hình cây có lúc xuống dưới 0,5, vì trường ngành ở dạng chữ tự do (259 nhãn) bị học thuộc. Cây và mạng sâu chỉ
thắng khi có rò danh tính.

## 4. Hướng sửa

1. Khi báo cáo, chỉ so với con số công bố kèm ghi chú giao thức. Con số 0,68–0,70 không phải dự báo cho người
   mới.
2. Để lấy lại ≈ 0,03 mà a⁰ tự để mất (mục 3.3), thêm mức sở thích thô làm hiệu ứng chính, và one-hot ngành,
   mục tiêu:
   - 17 điểm sở thích của đối tác vào khối đối tác (η₀ phần j);
   - 17 điểm của người chọn vào khối tác nhân.
   Việc này đổi Mục 5 của đặc tả nên cần nhóm đồng ý. Dự kiến a⁰ về khoảng logit 0,60.
3. Nếu muốn đọc thêm trọng số sở thích và chủng tộc (+0,02), cần bổ sung trường này vào persona. Ngoại hình
   vẫn để ngoài.
