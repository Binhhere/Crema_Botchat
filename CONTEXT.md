# CONTEXT.md — Checkpoint Log (Crema HR & HSE RAG)

## Trạng thái 5 Giai đoạn
- [x] Giai đoạn 1: Setup nền, `src/loader.py` & `src/chunker.py` (pushed `origin/main` `89a6e67`)
- [x] Giai đoạn 2: `src/indexer.py`, `src/rewriter.py`, `src/retriever.py` (pushed `origin/main` `503f1cb`)
- [x] Giai đoạn 3: `src/verifier.py`, `src/generator.py`, `response_schema`, ý định (pushed `origin/main` `45c2e5a`)
- [x] Giai đoạn 4: Calibration (36), Held-out (60), nạp nóng & `eval/run_eval.py` (`ruff` 0 lỗi, `pytest` 19/19 pass)
- [ ] Giai đoạn 5: `src/storage.py` (k-anonymity), `src/server.py` (RBAC) & UI

## Kết quả Hiệu chuẩn & Đo đạc Thực nghiệm (Giai đoạn 4)
- **Ngưỡng hiệu chuẩn trên `calibration_set.json` (36 câu):** `TAU_COSINE_REJECT = 0.69`, `TAU_COSINE_FLOOR = 0.62` (Pos Recall `16/16 = 100%`, Neg FAR `0/20 = 0.0%`).
- **Held-out tổng thể (60 câu — 34 dương / 26 âm):** Hit@5 `100.0%`, Verified Cite `100.0%`, Answer Correctness `100.0%`, FAR `0.0%`, FRR `0.0%`, P50 `13291 ms`, P95 `38742 ms` (gồm retry RPM Free tier), API/Turn `1.93`.
- **Lát cắt `Không dấu` (20 câu):** Hit@5 `100.0%`, Answer Correctness `100.0%`, FAR `0.0%`, **FRR `0.0%`**.
- **Nạp nóng `eval/extra_05_so_tay_nha_thau.docx` (32 chunks):** Nhúng bổ sung tức thì thành công (`622` chunks), nhãn `[Văn bản mẫu mô phỏng]`, `page_number=None`, pass Verifier 4/4 truy vấn.

## Cấu hình hiện tại
- `EMBEDDING_MODEL`: `gemini-embedding-001` (`768` chiều, L2-normalized, `CHUNKER_VERSION="1.0.1"`)
- `REWRITER_MODEL` / `GENERATION_MODEL`: `gemini-3.6-flash` (fallback `gemini-3.5-flash` -> `gemini-3.1-flash-lite`)
- `TAU_COSINE_REJECT = 0.69` | `TAU_COSINE_FLOOR = 0.62`
