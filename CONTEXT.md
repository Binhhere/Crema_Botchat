# CONTEXT.md — Checkpoint Log (Crema HR & HSE RAG)

## Trạng thái 5 Giai đoạn
- [x] Giai đoạn 1: Setup nền, `src/loader.py` & `src/chunker.py` (pushed `origin/main` `89a6e67`, `ruff` 0 lỗi, `pytest` pass)
- [x] Giai đoạn 2: `src/indexer.py`, `src/rewriter.py`, `src/retriever.py` (`ruff` 0 lỗi, `pytest -q` 10/10 pass, cache SHA-256 768-dim)
- [ ] Giai đoạn 3: `src/verifier.py`, `response_schema`, phân loại ý định
- [ ] Giai đoạn 4: Tập Calibration (36) / Held-out (60) & `eval/run_eval.py`
- [ ] Giai đoạn 5: `src/storage.py` (k-anonymity), `src/server.py` (RBAC) & UI

## Thống kê Chunk Thực tế (Nền `data/raw/`: 590 chunks, cache hit 590/590)
- `01_nd88_2020_hop_nhat_04_vbhn_bnv.pdf`: 56 chunks (`04/VBHN-BNV`, `2026-02-02`, gốc `2020-09-15`, sửa đổi `158/2025/NĐ-CP` `2025-07-01`)
- `02_tt28_2021_hop_nhat_06_vbhn_bnv.pdf`: 42 chunks (`06/VBHN-BNV`, `2026-02-11`, gốc `2022-03-01`, sửa đổi `25/2025/TT-BNV` `2026-01-01`; `Điều 5 khoản 2 điểm e` verified `2026-01-01`)
- `03_nd145_2020_dieu_kien_lao_dong.docx`: 77 chunks (`145/2020/NĐ-CP`, `2021-02-01`, `page_number=None`, cảnh báo Điều 4 & 31)
- `04_nike_cls_2020_an_toan_lao_dong.pdf`: 415 chunks (394 body + 21 glossary, sạch header/footer, giữ `page_number`)
- `eval/extra_05_so_tay_nha_thau.docx` (giữ ngoài tập nền): 32 chunks (`CREMA-HSE-NT-2026`, `2026-11-15`, `page_number=None`)

## Ngưỡng & Cấu hình hiện tại (Chờ hiệu chuẩn chính thức ở Giai đoạn 4)
- `EMBEDDING_MODEL`: `gemini-embedding-001` (`768` chiều, L2-normalized, `CHUNKER_VERSION="1.0.1"`)
- `REWRITER_MODEL` / `GENERATION_MODEL`: `gemini-3.6-flash` (fallback `gemini-3.5-flash` -> `gemini-3.1-flash-lite`; `gemini-2.5-flash` trả `404` với tài khoản mới)
- `TAU_COSINE_REJECT` (khởi tạo): `0.65` | `TAU_COSINE_FLOOR` (khởi tạo): `0.58` (BM25 strong rescue: `bm25_top1 >= 28.0` & `z >= 5.0`)
- Đã tạo `.env` cục bộ (`.gitignore` chặn 100%, không commit).
