# CONTEXT.md — Checkpoint Log (Crema HR & HSE RAG)

## Trạng thái 5 Giai đoạn
- [x] Giai đoạn 1: Setup nền, `src/loader.py` & `src/chunker.py` (pushed `origin/main` `89a6e67`)
- [x] Giai đoạn 2: `src/indexer.py`, `src/rewriter.py`, `src/retriever.py` (pushed `origin/main` `503f1cb`)
- [x] Giai đoạn 3: `src/verifier.py`, `src/generator.py`, `response_schema`, phân loại ý định (`ruff` 0 lỗi, `pytest -q` 16/16 pass)
- [ ] Giai đoạn 4: Tập Calibration (36) / Held-out (60) & `eval/run_eval.py`
- [ ] Giai đoạn 5: `src/storage.py` (k-anonymity), `src/server.py` (RBAC) & UI

## Thống kê Chunk Thực tế (Nền `data/raw/`: 590 chunks, cache hit 590/590)
- `01_nd88_2020_hop_nhat_04_vbhn_bnv.pdf`: 56 chunks (`04/VBHN-BNV`, `2026-02-02`)
- `02_tt28_2021_hop_nhat_06_vbhn_bnv.pdf`: 42 chunks (`06/VBHN-BNV`, `2026-02-11`, `Điều 5 khoản 2 điểm e`: `2026-01-01`)
- `03_nd145_2020_dieu_kien_lao_dong.docx`: 77 chunks (`145/2020/NĐ-CP`, `2021-02-01`, `page_number=None`, cảnh báo Điều 4 & 31)
- `04_nike_cls_2020_an_toan_lao_dong.pdf`: 415 chunks (394 body + 21 glossary, giữ `page_number`)
- `eval/extra_05_so_tay_nha_thau.docx` (giữ ngoài tập nền): 32 chunks (`CREMA-HSE-NT-2026`, `2026-11-15`)

## Ngưỡng & Cấu hình hiện tại (Chờ hiệu chuẩn chính thức ở Giai đoạn 4)
- `EMBEDDING_MODEL`: `gemini-embedding-001` (`768` chiều, L2-normalized, `CHUNKER_VERSION="1.0.1"`)
- `REWRITER_MODEL` / `GENERATION_MODEL`: `gemini-3.6-flash` (fallback `gemini-3.5-flash` -> `gemini-3.1-flash-lite`)
- `TAU_COSINE_REJECT` / `TAU_COSINE_FLOOR` (khởi tạo): `0.65` / `0.58`
- Verifier Giai đoạn 3: bắt buộc `chunk_id` trong Top-5, `verbatim_quote >= 8` từ (NFC + nối `-\n` + `1,5`<->`1.5`), và Numerical Grounding Check.
