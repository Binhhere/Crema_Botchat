# Báo cáo Thực nghiệm Giai đoạn 4 — Crema HR & HSE Policy RAG

## 1. Hiệu chuẩn Ngưỡng Rủi ro Bất đối xứng (Calibration Set — 36 câu)
- **Ngưỡng được chọn:** `TAU_COSINE_REJECT = 0.69`, `TAU_COSINE_FLOOR = 0.62`
- **Calibration Positive Recall (Gate):** `16/16` (`100.0%`) | Retrieval Hit@5: `100.0%`
- **Calibration Negative FAR (Gate):** `0/20` (`0.0%`, ràng buộc `<= 5.0%`)

## 2. Kết quả Độc lập trên Tập Kiểm chuẩn Giữ riêng (Held-out Test Set — 60 câu)

| Lát cắt | Số câu (Dương/Âm) | Hit@5 | Verified Cite | Answer Correctness | FAR | FRR | P50 (ms) | P95 (ms) | API/Turn |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tổng thể (Held-out)** | 60 (34/26) | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 13291 | 38742 | 1.93 |
| Có dấu | 20 (10/10) | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 12270 | 38424 | 1.55 |
| Không dấu | 20 (10/10) | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 23198 | 34867 | 2.60 |
| Đa lượt | 8 (8/0) | 100.0% | 100.0% | 100.0% | N/A | 0.0% | 25404 | 69894 | 2.62 |
| Near-miss & Injection | 12 (6/6) | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 6101 | 13689 | 1.00 |

## 3. Phân tích các ca lệch (0 ca)
- Không có ca sai lệch nào.
