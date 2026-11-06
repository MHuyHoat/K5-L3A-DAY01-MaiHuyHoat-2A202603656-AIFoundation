# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | v|
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | v|
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | v |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
<img width="1700" height="564" alt="image" src="https://github.com/user-attachments/assets/2f4d0b69-16ab-4658-b43a-f6fe4724fb6b" />
## 3. Dự đoán và thực tế

Ba dự đoán `predict(...)` có độ lệch đáng chú ý so với thực tế thí nghiệm:

1. **Dự đoán F1 của Machine Learning (`ml_f1`)**:
   * **Bạn đoán gì:** F1 khoảng 0.85, độ chính xác (precision) cao hơn độ bao phủ (recall).
   * **Thực tế ra gì:** F1 trên tập test là **0.814** (với precision = 0.958 nhưng recall tụt xuống **0.708**)[cite: 3].
   * **Vì sao lệch:** Mô hình Bag-of-Words tuyến tính quá đơn giản, chỉ dựa vào sự xuất hiện đơn lẻ của từ khóa mà bỏ qua cấu trúc ngữ cảnh. Khi gặp các tin nhắn spam dùng từ ngữ mới hoặc lảng tránh từ khóa quen thuộc ở tập test, mô hình dễ bị bỏ sót (khiến recall thấp hơn dự kiến).

2. **Dự đoán đường cong & hiệu năng Deep Learning (`dl_curve`)**:
   * **Bạn đoán gì:** Mạng MLP sẽ cho điểm F1 xấp xỉ hoặc nhỉnh hơn không đáng kể so với Logistic Regression vì dữ liệu nhỏ và biểu diễn Bag-of-Words còn hạn chế.
   * **Thực tế ra gì:** MLP đạt F1 = **0.874** (vượt trội hơn hẳn mốc 0.814 của LogReg ở Phần 1), đặc biệt Recall tăng mạnh lên **0.833**[cite: 3].
   * **Vì sao lệch:** Đã đánh giá thấp khả năng của lớp ẩn (hidden layer 64 chiều kết hợp hàm kích hoạt ReLU) trong việc học các tổ hợp đặc trưng phi tuyến từ tập từ vựng, giúp mô hình bắt được nhiều mẫu spam phức tạp hơn.

3. **Dự đoán tìm kiếm theo nghĩa Embedding (`emb_search`)**:
   * **Bạn đoán gì:** Câu truy vấn không chứa từ khóa trực diện nhưng mang ý nghĩa rủ rê nhận quà sẽ cho kết quả láng giềng gần nhất là spam.
   * **Thực tế ra gì:** Các câu láng giềng trả về đều chính xác thuộc nhóm spam với độ tương đồng cosine rất cao (>0.8)[cite: 3].
   * **Vì sao lệch:** Dự đoán khá sát thực tế, tuy nhiên mô hình Encoder (MiniLM) thể hiện khả năng hiểu ngữ nghĩa (semantics) sắc bén hơn dự kiến, vượt qua giới hạn của từ khóa bề mặt[cite: 3].

---

## 4. Phân tích 50 case khó

* **Model tụt phong độ nhiều nhất:** Mô hình Machine Learning truyền thống (BoW + LogReg) và Deep Learning đơn giản (MLP) dựa trên đặc trưng từ vựng thô.
* **Nhóm case bị sai nhiều nhất:** (1) Tin nhắn spam cố tình ngụy trang, lách từ khóa nhạy cảm (như thay "free", "win" bằng từ đồng nghĩa) và (2) Tin nhắn bình thường (ham) nhưng chứa các từ khóa hay gặp ở tin quảng cáo (ví dụ từ "call", "win" trong ngữ cảnh công việc hoặc thể thao).
* **2 case cụ thể và cơ chế khiến model sai:**
  1. *Case 1 (Tin Ham bị nhận nhầm là Spam):* *"I won the match yesterday, call me when you can"*[cite: 3].
     * *Cơ chế sai:* Mô hình BoW/LogReg ghi nhận các từ "won" và "call" có trọng số dương rất cao hướng về phía spam, khiến mô hình nhầm lẫn ngữ cảnh chiến thắng thể thao cá nhân thành thông báo trúng thưởng rác.
  2. *Case 2 (Tin Spam bị bỏ sót, đoán nhầm là Ham):* Các tin nhắn lách luật bằng ký tự đặc biệt hoặc từ lóng mới chưa có trong từ điển `vocab`[cite: 3].
     * *Cơ chế sai:* Do vector đặc trưng `x` không kích hoạt các chiều từ khóa đặc trưng (giá trị bằng 0 ở các chiều quan trọng), tích vô hướng $X \cdot w + b$ không đủ lớn để hàm sigmoid vượt ngưỡng 0.5, dẫn đến phán đoán nhầm thành tin thường.

---

## 5. Thử thách cá nhân

* **Đề theo SEED (`SEED = 1999`):** Xây dựng và kiểm chứng các hàm vector hóa, huấn luyện và đánh giá trên không gian dữ liệu được chia tách ngẫu nhiên đặc thù cho mã sinh viên[cite: 3].
* **Cách làm:** Tuân thủ chặt chẽ các vùng tự code (không dùng thư viện ngoài như `sklearn` trong các hàm cốt lõi), tự cài đặt hàm tính xác suất, hàm mất mát BCE và gradient descent[cite: 3].
* **Kết quả & Giải thích:** 
  * Kết quả tính tay #1: Dựa trên phân phối dữ liệu sau khi lọc trùng (tổng 5169 tin, trong đó có 4516 tin ham và 653 tin spam)[cite: 3], một mô hình lười biếng luôn đoán `ham` sẽ đạt độ chính xác (Accuracy):
    $$\text{Accuracy} = \frac{4516}{5169} \approx 87.37\%$$
  * Seed 1999 đảm bảo tính ngẫu nhiên nhưng nhất quán cho việc chia tập train/val/test, giúp quy trình thực nghiệm tái lập chính xác trên hệ thống cá nhân[cite: 3].

---

## 6. Phương án cho sản phẩm thật

* **Yêu cầu kỹ thuật:** $p95 < 50\text{ ms/tin}$ cho $10\text{ triệu SMS/ngày}$[cite: 3].
* **Chọn phương án:** **3b. Encoder (MiniLM) + LogReg** (hoặc kết hợp đầu phân loại tuyến tính trên vector embedding cố định)[cite: 3].
* **Dẫn số liệu từ bảng benchmark:**
  * Điểm F1 đạt mức xuất sắc: **0.968** (vượt trội so với 0.814 của BoW thuần)[cite: 3].
  * Độ trễ thời gian thực cực tốt: **p50 = 28.07 ms** và **p95 = 45.72 ms**, hoàn toàn thỏa mãn ràng buộc nghiêm ngặt $p95 < 50\text{ ms}$[cite: 3].
* **Chấp nhận đánh đổi:** 
  * Chi phí tài nguyên tính toán ban đầu cao hơn cho việc chạy Encoder (MiniLM) để sinh embedding văn bản[cite: 3]. 
  * Đổi lại, hệ thống vừa đạt độ chính xác nhận diện tin rác cực cao, vừa đảm bảo tốc độ phản hồi đáp ứng tiêu chuẩn sản phẩm thương mại lớn.

---

## 7. AI log và điều đã học

* **AI log:**
  * *Hỏi gì:* Nhờ AI hỗ trợ cấu trúc toán học tối ưu hóa ma trận cosine similarity trong NumPy (tránh dùng vòng lặp qua 3600 dòng) và cách viết 5 dòng vòng lặp huấn luyện chuẩn trong PyTorch (`zero_grad`, `forward`, `loss`, `backward`, `step`)[cite: 3].
  * *Nhận gì:* Công thức chuẩn hóa ma trận `M_norm` và cú pháp ma trận tích vô hướng `dot`[cite: 3].
  * *Bạn sửa gì:* Tự bổ sung cơ chế kiểm tra điều kiện biên (chia cho 0, xử lý vector rỗng) và thêm hàm `np.clip` vào hàm sigmoid để ngăn chặn triệt để lỗi tràn số (overflow/NaN) trong quá trình tính gradient[cite: 3].

* **Điều bạn hiểu ra hômాం mà trước đó chưa hiểu:**
  Việc chuyển dịch từ mô hình đếm từ thô (Bag-of-Words) sang không gian biểu diễn ngữ nghĩa (Embedding) cho thấy AI hiện đại mạnh lên không chỉ nhờ số lượng tham số mà nhờ khả năng khái quát hóa bản chất câu chữ. Đồng thời, qua bài lab, tôi thấm thía rằng trong các bài toán thực tế có dữ liệu lệch nặng (như tin rác), việc tối ưu Accuracy là vô nghĩa nếu bỏ qua các chỉ số cân bằng như F1-score[cite: 3].
