# AGENTS.md — Crema HR & HSE Policy RAG (Greenfield)

## 1. Cấu trúc Repository
- `src/`: Mã nguồn lõi (`config.py`, `loader.py`, `chunker.py`, `indexer.py`, `rewriter.py`, `retriever.py`, `verifier.py`, `storage.py`, `server.py`).
- `data/raw/`: 4 văn bản khởi tạo (`01`..`04`). Không bỏ `extra_05_so_tay_nha_thau.docx` vào đây.
- `data/cache/`: Chỉ mục vector & cache cục bộ (đã gitignore).
- `eval/`: Tệp nạp nóng `extra_05_so_tay_nha_thau.docx`, tập Calibration/Held-out và script `run_eval.py`.
- `tests/`: Bộ kiểm thử `pytest` theo từng giai đoạn.

## 2. Môi trường & Lệnh Kiểm tra
- Kích hoạt `.venv` (PowerShell): `.\.venv\Scripts\Activate.ps1`
- Kiểm tra lint: `.\.venv\Scripts\ruff.exe check .`
- Chạy test: `.\.venv\Scripts\pytest.exe -q`

## 3. Quy ước Code Python
- Python 3.12 thuần gọn sạch; tuyệt đối KHÔNG cài LangChain, LlamaIndex, ChromaDB hay vector DB nặng.
- Luôn dùng `pathlib.Path`, type hints đầy đủ, Pydantic v2 (`BaseModel`), và `encoding="utf-8"` khi đọc/ghi file.
- Dùng `import pymupdf` (không dùng `import fitz` cũ).

## 4. Ràng buộc Cứng (Hard Rules)
- KHÔNG đọc, in hoặc commit `.env` (chỉ đọc qua `os.getenv` / `python-dotenv`).
- KHÔNG tham chiếu, import hoặc kế thừa bất kỳ code nào từ `crena_rag` cũ.
- KHÔNG tự bịa ngày hiệu lực, số hiệu hợp nhất, số trang hay con số pháp lý từ trí nhớ.
- KHÔNG lọc câu hỏi nhạy cảm bằng danh sách từ khóa đơn lẻ (như `"lương"`).
- Chế độ Checkpoint: chỉ làm đúng 1 Giai đoạn mỗi lượt, chạy `ruff check .` và `pytest -q` pass 100%, cập nhật `CONTEXT.md`, rồi DỪNG chờ người dùng xác nhận.
