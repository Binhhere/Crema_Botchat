# CONTEXT.md — Checkpoint Log (Crema HR & HSE RAG)

## Trạng thái 5 Giai đoạn
- [x] Giai đoạn 1: Setup nền, `src/loader.py` & `src/chunker.py` (`ruff check .` 0 lỗi, `pytest -q` 6/6 pass)
- [ ] Giai đoạn 2: `src/indexer.py`, `src/rewriter.py`, `src/retriever.py`
- [ ] Giai đoạn 3: `src/verifier.py`, `response_schema`, phân loại ý định
- [ ] Giai đoạn 4: Tập Calibration (36) / Held-out (60) & `eval/run_eval.py`
- [ ] Giai đoạn 5: `src/storage.py` (k-anonymity), `src/server.py` (RBAC) & UI

## Thống kê Chunk Thực tế (Giai đoạn 1 — Tổng nền `data/raw/`: 585 chunks)
- `01_nd88_2020_hop_nhat_04_vbhn_bnv.pdf`: 55 chunks (`04/VBHN-BNV`, `2026-02-02`, gốc `2020-09-15`, sửa đổi `158/2025/NĐ-CP` `2025-07-01`)
- `02_tt28_2021_hop_nhat_06_vbhn_bnv.pdf`: 38 chunks (`06/VBHN-BNV`, `2026-02-11`, gốc `2022-03-01`, sửa đổi `25/2025/TT-BNV` `2026-01-01`, gồm Phụ lục I & V)
- `03_nd145_2020_dieu_kien_lao_dong.docx`: 77 chunks (`145/2020/NĐ-CP`, `2021-02-01`, `page_number=None`, cảnh báo Điều 4 & 31)
- `04_nike_cls_2020_an_toan_lao_dong.pdf`: 415 chunks (394 body + 21 glossary, sạch header/footer, giữ `page_number`)
- `eval/extra_05_so_tay_nha_thau.docx` (giữ ngoài tập nền): 32 chunks (`CREMA-HSE-NT-2026`, `2026-11-15`, `page_number=None`)

## Ngưỡng hiệu chuẩn (Giai đoạn 4)
- `TAU_COSINE_REJECT`: TBD
- `TAU_COSINE_FLOOR`: TBD

## Lưu ý mở
- Chưa tạo `.env`: cần bổ sung `GEMINI_API_KEY` trước khi chạy thử API thật ở Giai đoạn 2.
- Không nạp `data and eval/03_nd145_2020_dieu_kien_lao_dong.pdf` (PDF scan 100%); dùng bản `.docx` trong `data/raw/`.
