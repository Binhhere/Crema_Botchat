# Crema HR & HSE Policy RAG (iLD Coffee — ICPC 200)

Trợ lý AI Hỏi đáp Chính sách Nhân sự & An toàn lao động (Greenfield PoC) dành cho kỹ sư nhà máy, nhân sự vận hành và phòng Nhân sự (iLD Coffee Vietnam). Hệ thống được thiết kế theo nguyên tắc **không bịa đặt số liệu**, **trích dẫn xác thực từ văn bản gốc** và **bảo vệ danh tính người hỏi ($k$-anonymity)**.

---

## 1. Kiến trúc 5 Lớp Chống Bịa Đặt & Bảo Mật

1. **Nạp & Cắt đoạn Pháp lý 3 Tầng (`src/loader.py`, `src/chunker.py`):**
   - Tự động trích xuất metadata hiệu lực từ văn bản hợp nhất (`04/VBHN-BNV`, `06/VBHN-BNV`), chú thích chân trang từng điều khoản (`clause_effective_date`), giữ số trang vật lý cho `.pdf` (`page_number = None` cho `.docx/.txt/.md`), và chặn PDF scan không có lớp text (`ScannedPDFError`).
   - Cắt đoạn theo 3 chiến lược: `Điều/Khoản/Điểm` $\rightarrow$ `Mục La Mã/Thập phân` $\rightarrow$ `Sliding Window` có overlap.
2. **Viết lại Truy vấn Có điều kiện `0 ms` (`src/rewriter.py`):**
   - Kiểm tra bằng Python thuần (`0 ms`): câu đơn lẻ có dấu đi thẳng không gọi LLM; chỉ gọi Flash model khi câu tiếng Việt không dấu ($\ge 3$ từ) hoặc câu hỏi nối tiếp đa lượt (`< 7` từ hoặc chứa từ quy chiếu).
3. **Truy xuất Lai & Cổng Tiền LLM (`src/indexer.py`, `src/retriever.py`):**
   - Vector `gemini-embedding-001` (`768` chiều, chuẩn hóa L2, cache đĩa theo khóa `SHA-256`) kết hợp `BM25Okapi` trên văn bản bỏ dấu (uni-gram + bi-gram âm tiết) qua RRF (`k = 60`).
   - Cổng từ chối trước LLM dùng ngưỡng hiệu chuẩn `TAU_COSINE_REJECT = 0.69` và `TAU_COSINE_FLOOR = 0.62`, kèm **Cổng cứu hộ Lexical** cho câu hỏi nêu chính xác số `Điều`, `Phụ lục`, mã văn bản hoặc thuật ngữ HSE.
4. **Phòng thủ Ý định & Khóa Cấu trúc JSON (`src/generator.py`, `src/models.py`):**
   - Phân loại nhạy cảm theo **ý định** (không chặn bằng từ khóa đơn lẻ như `"lương"`): tách câu hỏi quy chế chung khỏi câu hỏi hồ sơ cá nhân/khiếu nại (chuyển hướng tạo Phiếu Nhân sự kèm Mã tra cứu yêu cầu `#TKT-001`) và từ chối Prompt Injection qua `response_schema` JSON nghiêm ngặt.
5. **Deterministic Verifier & Backlog $k$-Anonymity (`src/verifier.py`, `src/storage.py`, `src/server.py`):**
   - **Verifier 3 điều kiện cứng:** (1) `chunk_id` thuộc Top-5, (2) `verbatim_quote` $\ge 8$ từ là chuỗi con nguyên văn (chuẩn hóa NFC, nối `-\n`, quy đổi `1,5` $\leftrightarrow$ `1.5`), (3) **Numerical Grounding Check** (mọi con số/`%`/ngày/tháng/tiền trong câu trả lời phải tồn tại trong chunk được trích hoặc câu hỏi gốc).
   - **Backlog & RBAC:** Bảng `crema_faq_backlog` chỉ lưu ngày (`YYYY-MM-DD`), khử SĐT/email/mã nhân sự/tên người/bộ phận, chỉ hiển thị chủ đề đạt `hit_count >= 3`; khóa `/v1/ingest` và `/v1/backlog` bằng `HR_API_KEY` / `ADMIN_API_KEY`.

---

## 2. Cài đặt & Khởi chạy Cục bộ

### Yêu cầu môi trường
- Python 3.12+ (đã cấu hình sẵn trong `.venv` cục bộ)

### Thiết lập `.venv` và `.env`
```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
Copy-Item .env.example .env
```
Điền `GEMINI_API_KEY`, `HR_API_KEY`, `ADMIN_API_KEY` và `DB_BACKEND=sqlite` vào tệp `.env` (tệp `.env` đã được chặn bởi `.gitignore`).

### Chạy Server & Giao diện Web
```powershell
.\.venv\Scripts\uvicorn.exe src.server:app --host 127.0.0.1 --port 8000
```
Mở trình duyệt tại `http://127.0.0.1:8000` (tài liệu API Swagger tại `http://127.0.0.1:8000/docs`).

### Chạy Kiểm thử & Đánh giá Thực nghiệm
```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe -q
.\.venv\Scripts\python.exe eval\run_eval.py
```

---

## 3. Kết quả Thực nghiệm (Trích từ `eval/eval_report.md`)

### Hiệu chuẩn trên Calibration Set (36 câu — Constrained Risk Selection)
- **Ngưỡng được chọn:** `TAU_COSINE_REJECT = 0.69`, `TAU_COSINE_FLOOR = 0.62`
- **Calibration Positive Recall (Gate):** `16/16` (`100.0%`) | **Retrieval Hit@5:** `100.0%`
- **Calibration Negative FAR (Gate):** `0/20` (`0.0%`, ràng buộc `<= 5.0%`)

### Đo độc lập trên Held-out Test Set (60 câu — 4 lát cắt)

| Lát cắt | Số câu (Dương/Âm) | Hit@5 | Verified Cite | Answer Correctness | FAR | FRR | P50 (ms) | P95 (ms) | API/Turn |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tổng thể (Held-out)** | 60 (34/26) | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 13291 | 38742 | 1.93 |
| Có dấu | 20 (10/10) | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 12270 | 38424 | 1.55 |
| Không dấu | 20 (10/10) | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 23198 | 34867 | 2.60 |
| Đa lượt | 8 (8/0) | 100.0% | 100.0% | 100.0% | N/A | 0.0% | 25404 | 69894 | 2.62 |
| Near-miss & Injection | 12 (6/6) | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 6101 | 13689 | 1.00 |
