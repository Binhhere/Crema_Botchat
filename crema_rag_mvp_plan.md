# Kế hoạch Xây dựng Mới (Greenfield): Trợ lý AI Crema — Chính sách Nhân sự & An toàn (v3)

> [!IMPORTANT]
> **Nguyên tắc nền tảng:** Xây dựng mới 100% từ đầu (Greenfield), không kế thừa bất kỳ dòng code nào từ bộ `crena_rag` cũ. Mọi tuyên bố về độ chính xác, tỷ lệ từ chối và độ trễ đều phải được đo đạc thực nghiệm trên tập kiểm thử độc lập (Held-out Test Set), không dùng con số ước lượng cảm tính.

---

## 1. Kho Văn bản Chuẩn Hóa & Kiểm chứng Hiệu lực Pháp lý Thực tế

Kho văn bản PoC được chuẩn hóa đồng bộ với **05 tệp tài liệu thực tế** (04 tệp nền trong `data/raw/` và 01 tệp kiểm thử nạp nóng trong `eval/`):

1. **`data/raw/01_nd88_2020_hop_nhat_04_vbhn_bnv.pdf`** (`doc_nature = "consolidated_law"` — Nhãn UI: **`[Văn bản pháp luật hợp nhất]`**):
   - **Nguồn:** Văn bản hợp nhất số `04/VBHN-BNV` của Bộ Nội vụ (xác thực ngày `02/02/2026` tại trang 30), hợp nhất **Nghị định `88/2020/NĐ-CP`** (hiệu lực gốc `15/09/2020`) và phần sửa đổi, bổ sung tại **Nghị định `158/2025/NĐ-CP`** (hiệu lực sửa đổi `01/07/2025`).
   - **Quy chuẩn xử lý:** Tự trích xuất `original_effective_date`, `amended_by` từ trang đầu; tự đọc `consolidated_date` & số hiệu `VBHN-BNV` từ khối chữ ký trang 30; đọc chú thích chân trang `[1]..[6]` gắn `clause_effective_date`. Chỉ index trang 1–30 (Điều 1–47); bỏ qua Phụ lục biểu mẫu điền tay (trang 31–63).
2. **`data/raw/02_tt28_2021_hop_nhat_06_vbhn_bnv.pdf`** (`doc_nature = "consolidated_law"` — Nhãn UI: **`[Văn bản pháp luật hợp nhất]`**):
   - **Nguồn:** Văn bản hợp nhất số `06/VBHN-BNV` của Bộ Nội vụ (xác thực ngày `11/02/2026` tại trang 24), hợp nhất **Thông tư `28/2021/TT-BLĐTBXH`** (hiệu lực gốc `01/03/2022`) và **Thông tư `25/2025/TT-BNV`** (hiệu lực `01/01/2026`).
   - **Quy chuẩn xử lý:** Tự đọc metadata từ trang đầu, khối xác thực trang 24 và chú thích chân trang; index phần quy phạm (trang 1–24, Điều 1–18) và giữ `Phụ lục I` (trang 25–28), `Phụ lục V` (trang 36–37) với `section_type = "appendix"`; bỏ qua biểu mẫu điền tay `Phụ lục II, III, IV`.
3. **`data/raw/03_nd145_2020_dieu_kien_lao_dong.docx`** (`doc_nature = "legal_decree"` — Nhãn UI: **`[Nghị định pháp luật]`**):
   - **Nguồn:** Bản chuyển thể cấu trúc hóa từ **Nghị định `145/2020/NĐ-CP`** (hiệu lực `01/02/2021`, Điều 1–115; thay thế cho bản PDF scan ảnh 122 trang không có lớp text).
   - **Quy chuẩn xử lý:** Chỉ index phần quy phạm Điều 1–115 (`page_number = None`), bỏ qua phần tóm tắt Phụ lục (`[APP]`). Gắn cờ cảnh báo `partial_expiry_warning = True` riêng cho **Điều 4** và **Điều 31** (các điều khoản báo cáo hành chính đã chịu tác động sửa đổi một phần).
4. **`data/raw/04_nike_cls_2020_an_toan_lao_dong.pdf`** (`doc_nature = "enterprise_reference"` — Nhãn UI: **`[Tiêu chuẩn doanh nghiệp tham chiếu]`**):
   - **Nguồn:** Bộ Quy tắc Chuẩn mực Lãnh đạo của Nike (Nike CLS, 138 trang PDF).
   - **Quy chuẩn xử lý:** Làm sạch tiêu đề đầu trang/chân trang lặp lại (`BỘ QUY TẮC CHUẨN MỰC LÃNH ĐẠO CỦA NIKE` và `Về đầu trang | Trang X / 135`) trước khi chunk nhưng bảo toàn `page_number` vật lý thật; chunk theo mục thập phân (`1 -> 1.2 -> 1.2.1`) và giữ cả phần `Định nghĩa` (`section_type = "glossary"`).
5. **`eval/extra_05_so_tay_nha_thau.docx`** (`doc_nature = "synthetic_policy"` — Nhãn UI: **`[Văn bản mẫu mô phỏng]`**):
   - **Nguồn:** Sổ tay An toàn và Quản lý Nhà thầu tại Khuôn viên Nhà máy iLD (`CREMA-HSE-NT-2026`, hiệu lực `15/11/2026`).
   - **Quy chuẩn xử lý:** Giữ riêng ngoài `data/raw/` (KHÔNG index lúc khởi tạo) để chạy bài test nạp nóng; `page_number = None`; chunk theo mục La Mã/thập phân (`I. -> 1.1 -> 1.2.1`, không có chữ `"Điều"`).

> [!WARNING]
> **Quy tắc trích xuất ngày tháng & hiệu lực (Không gõ tay từ trí nhớ):**
> `loader.py` phải tự đọc trực tiếp từ header, khối chữ ký xác thực bằng tọa độ từ (`get_text("words")`), bảng metadata `.docx` và chú thích chân trang `[1], [2]...` để điền `original_effective_date`, `amended_by`, `consolidated_date` và `clause_effective_date`.

---

## 2. Đặc tả Kỹ thuật Đóng Kín 12 Điểm Rà Soát

| Thành phần | Thiết kế kỹ thuật chi tiết | Cơ sở kỹ thuật (Vì sao thiết kế này đúng) |
| :--- | :--- | :--- |
| **1. Conditional Query Rewriter (`src/rewriter.py`)** | **Không gọi LLM ở mọi lượt hỏi.** Kiểm tra điều kiện bằng Python ($0\text{ ms}$): chỉ gọi bước viết lại/phục hồi dấu khi **(a)** câu hỏi gõ tiếng Việt không dấu ($\ge 3$ từ, $0$ ký tự có dấu), hoặc **(b)** có lịch sử hội thoại VÀ câu mới ngắn ($< 7$ từ) hoặc chứa từ quy chiếu (*"đó", "này", "trên", "còn", "thế", "vậy", "bao nhiêu"*). Các câu đơn lẻ có dấu đầy đủ đi thẳng sang bước nhúng vector. | Cắt giảm từ 3 cuộc gọi API xuống còn 2 cuộc gọi API cho đa số lượt hỏi thông thường, giảm độ trễ P50/P95 và tránh chạm trần RPM của gói Gemini Free Tier. |
| **2. Chống Rớt Câu Không Dấu 4 Lớp (`src/retriever.py`)** | 1. Phục hồi dấu tại `rewriter.py` khi phát hiện câu không dấu.<br>2. Gọi `gemini-embedding-001` với `task_type="RETRIEVAL_DOCUMENT"` (kèm `title`) khi lập chỉ mục và `task_type="RETRIEVAL_QUERY"` khi truy vấn.<br>3. **BM25 chỉ lập chỉ mục trên văn bản đã bỏ dấu (uni-gram + bi-gram âm tiết)**, không trộn lẫn bản có dấu.<br>4. **Cổng cứu hộ Lexical:** Cho qua cổng nếu `top1_cosine >= TAU_COSINE_REJECT` **HOẶC** câu hỏi khớp chính xác số hiệu Điều/mã văn bản **HOẶC** điểm BM25 vượt ngưỡng mạnh kèm `top1_cosine >= TAU_COSINE_FLOOR`. | Lập chỉ mục BM25 cả có dấu lẫn không dấu sẽ làm nhân đôi độ dài tài liệu và làm méo trọng số IDF; trong khi đó vector tiếng Việt không dấu thường bị tụt cosine nên cần cơ chế khôi phục dấu và cổng cứu hộ từ BM25. |
| **3. Cache Nhúng Bất Biến (`src/indexer.py`)** | Lưu cache vector trên đĩa tại `data/cache/` với khóa kiểm tra `SHA-256(raw_chunk_text + "|" + EMBEDDING_MODEL + "|" + str(OUTPUT_DIM) + "|" + CHUNKER_VERSION)`. Chỉ gọi API nhúng cho những chunk có chữ ký thay đổi hoặc mới thêm. | Tránh nhúng lại từ đầu mỗi khi khởi động server, đồng thời tự động làm mới vector nếu thay đổi tên model, số chiều hoặc logic cắt đoạn. |
| **4. Phân loại Nhạy cảm theo Ý định & Phòng thủ Cấu trúc** | • **Bỏ lọc nhạy cảm bằng từ khóa đơn lẻ** (như `"lương"`). Tách rõ ý định: hỏi quy định chung (*"cách tính lương làm thêm giờ"*) $\rightarrow$ hợp lệ; hỏi dữ liệu riêng cá nhân hoặc khiếu nại sự vụ (*"lương tháng này của tôi"*, *"tố giác quản lý"*) $\rightarrow$ chuyển sang hướng dẫn tạo Phiếu Nhân sự.<br>• **Chống Prompt Injection bằng cấu trúc:** Khóa đầu ra bằng `response_schema` JSON nghiêm ngặt + hậu kiểm tất định tại `verifier.py`. | Danh sách từ khóa luôn bị chặn nhầm ở các cặp câu gần giống nhau (near-miss) và dễ dàng bị lách; ngược lại, dù kẻ tấn công lừa được LLM sinh văn bản tự do thì cũng không thể vượt qua kiểm tra trích dẫn nguyên văn và bảo toàn con số của Verifier. |
| **5. Deterministic Verifier Kiểm cả Quote & Số liệu (`src/verifier.py`)** | Trước khi so khớp, chuẩn hóa cả hai phía: Unicode `NFC`, nối từ bị gạch ngang ngắt dòng PDF (`-\n`), chuẩn hóa khoảng trắng và quy đổi dấu thập phân/phân cách nghìn (`1,5` $\leftrightarrow$ `1.5`). Câu trả lời **chỉ hợp lệ** khi thỏa mãn đồng thời:<br>1. `chunk_id` thuộc Top-5 đã cấp.<br>2. `verbatim_quote` **dài tối thiểu $\ge 8$ từ** và là `substring` nguyên văn của `chunk` đó.<br>3. **Numerical Grounding Check:** Mọi con số, tỷ lệ `%`, số ngày/tháng, số tiền xuất hiện trong `answer_friendly` bắt buộc phải tồn tại trong các `chunk` được trích (hoặc trong câu hỏi gốc). Sai bất kỳ điều kiện nào $\rightarrow$ hủy câu trả lời và buộc chuyển sang trạng thái từ chối an toàn. | Trích dẫn ngắn 2–3 từ luôn khớp ngẫu nhiên, và việc có trích dẫn thật vẫn không ngăn được lỗi LLM đọc nhầm con số từ khoản khác hoặc tự tính nhẩm sai; kiểm tra bảo toàn tập số chặn đứng lỗi sai lệch định mức chính sách. |
| **6. Thẻ Nguồn Trung Thực trên UI (`src/server.py` + UI)** | • Chỉ hiển thị `Trang X` khi nguồn là `.pdf` (`page_number != None`); với `.docx`/`.txt`/`.md` chỉ hiển thị `Điều/Khoản` hoặc `Mục`.<br>• Nối trực tiếp `doc_nature` lên giao diện theo 4 nhãn chuẩn (`[Văn bản pháp luật hợp nhất]`, `[Nghị định pháp luật]`, `[Tiêu chuẩn doanh nghiệp tham chiếu]`, `[Văn bản mẫu mô phỏng]`) kèm số hiệu, ngày hiệu lực và cảnh báo nếu chạm Điều 4 / Điều 31 NĐ 145. | Đảm bảo người xem demo và Hội đồng phân biệt tức thì đâu là văn bản hợp nhất, nghị định, tiêu chuẩn tham chiếu hay tài liệu mô phỏng. |
| **7. Bảo mật Backlog $k$-Anonymity & Phân quyền API (`src/storage.py`)** | • Hỗ trợ SQLite cục bộ hoặc Supabase qua biến `DB_BACKEND`.<br>• Bảng `crema_faq_backlog` **chỉ lưu ngày (`asked_date DATE`), không lưu giờ/phút/giây** và không lưu `user_id`.<br>• Bộ lọc khử định danh xóa SĐT, email, mã `ILD-xxx`, **tên người, tên tổ chuyền/phân xưởng/bộ phận**.<br>• **Ngưỡng ẩn danh $k$-anonymity:** Chỉ hiển thị cho Nhân sự những chủ đề đạt **`hit_count >= 3`**.<br>• **RBAC trên API:** Khóa `/v1/ingest` (chỉ `ADMIN_API_KEY`) và `/v1/backlog` (chỉ `HR_API_KEY` / `ADMIN_API_KEY`). | Ở quy mô doanh nghiệp nhỏ (~110 nhân sự), lưu timestamp chi tiết hoặc hiển thị ngay câu hỏi đầu tiên (`hit_count = 1`) kèm tên tổ/bộ phận sẽ làm lộ người hỏi dù không lưu `user_id`. |

---

## 3. Cấu trúc Thư mục Dự án Mới (Greenfield)

```text
crema_rag/
├── AGENTS.md                     # Quy tắc làm việc cho AI Agent (< 120 dòng)
├── CONTEXT.md                    # Nhật ký Checkpoint 5 Giai đoạn (< 40 dòng)
├── crema_rag_mvp_plan.md         # Tài liệu đặc tả kiến trúc chuẩn
├── .gitignore                    # Chặn .env, .venv/, data/cache/, *.db, __pycache__/
├── .env.example                  # Mẫu GEMINI_API_KEY, HR_API_KEY, ADMIN_API_KEY, DB_BACKEND
├── requirements.txt              # Danh sách gói gọn nhẹ (google-genai, pymupdf, python-docx, ...)
├── data/
│   ├── raw/                      # 4 tài liệu khởi tạo (01..04)
│   └── cache/                    # Embedding & index cache trên đĩa
├── eval/
│   ├── extra_05_so_tay_nha_thau.docx  # Tệp thứ 5 giữ ngoài để test nạp nóng
│   ├── calibration_set.json      # 36 câu chọn ngưỡng (16 hợp lệ + 20 âm)
│   ├── heldout_test_set.json     # 60 câu kiểm chuẩn giữ riêng (4 lát cắt)
│   └── run_eval.py               # Script hiệu chuẩn & xuất báo cáo thực nghiệm
├── src/
│   ├── config.py                 # Cấu hình model, hằng số chunker & ngưỡng hiệu chuẩn
│   ├── loader.py                 # Đọc PDF/DOCX + Scan Guard + tách metadata & chú thích chân trang
│   ├── chunker.py                # Chunker 3 tầng (Điều/Khoản -> Mục La Mã/Thập phân -> Sliding Window)
│   ├── indexer.py                # Embedding cache (SHA-256) & xây chỉ mục Hybrid
│   ├── rewriter.py               # Conditional Query Rewriter (0ms gate + LLM khi cần)
│   ├── retriever.py              # Dense Cosine + Unaccented Bi-gram BM25 + Cổng cứu hộ Lexical
│   ├── verifier.py               # Deterministic Verifier (chunk_id + quote >=8 từ + Numerical Grounding)
│   ├── storage.py                # Backlog k-anonymity (SQLite / Supabase)
│   └── server.py                 # FastAPI (/v1/ask, /v1/ingest, /v1/backlog, /v1/health) & UI
└── tests/                        # Bộ pytest kiểm chứng cho từng Giai đoạn 1..5
```

---

## 4. Thiết kế Đánh giá Thực nghiệm Nghiêm ngặt (Giai đoạn 4)

1. **`eval/calibration_set.json` (36 câu — Chỉ dùng chọn ngưỡng):** 16 câu hợp lệ + 20 câu âm ngoài phạm vi. Chọn ngưỡng `TAU_COSINE_REJECT` nhỏ nhất thỏa mãn ràng buộc **`False Answer Rate (FAR) <= 5%`** trên 20 câu âm.
2. **`eval/heldout_test_set.json` (60 câu — Đóng băng đo điểm cuối):** 20 câu hợp lệ (gồm 10 câu đời thường chia đều có dấu/không dấu) + 20 câu âm + 8 đa lượt + 12 câu cặp `Near-Miss` & Injection.
3. **Chỉ số đo theo 4 lát cắt (`Có dấu`, `Không dấu`, `Đa lượt`, `Near-miss & Injection`):** `Retrieval Hit@5`, `Verified Citation Accuracy`, `Answer Correctness (must_contain)`, `FAR`, `FRR`, `Latency P50/P95` và `API Calls/Turn`.
