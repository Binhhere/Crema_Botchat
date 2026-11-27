# CONTEXT.md — Checkpoint Log (Crema HR & HSE RAG)

## Trạng thái 5 Giai đoạn
- [x] Giai đoạn 1: Setup nền, `src/loader.py` & `src/chunker.py` (pushed `origin/main` `89a6e67`)
- [x] Giai đoạn 2: `src/indexer.py`, `src/rewriter.py`, `src/retriever.py` (pushed `origin/main` `503f1cb`)
- [x] Giai đoạn 3: `src/verifier.py`, `src/generator.py`, `response_schema`, ý định (pushed `origin/main` `45c2e5a`)
- [x] Giai đoạn 4: Calibration (36), Held-out (60), nạp nóng & `eval/run_eval.py` (pushed `origin/main` `eecdaa4`)
- [x] Giai đoạn 5: `src/storage.py` (k-anonymity), `src/server.py` (RBAC) & Web UI (`ruff` 0 lỗi, `pytest` 23/23 pass)

## Kết quả Hiệu chuẩn & Thực nghiệm (Giai đoạn 4)
- `TAU_COSINE_REJECT = 0.69` | `TAU_COSINE_FLOOR = 0.62` (Calibration 36 câu: Pos Recall `100%`, Neg FAR `0.0%`)
- Held-out (60 câu): Hit@5 `100.0%`, Verified Cite `100.0%`, Answer Correctness `100.0%`, FAR `0.0%`, FRR `0.0%`, API/Turn `1.93`

## API & Lưu trữ Cục bộ (Giai đoạn 5)
- `DB_BACKEND=sqlite` (`data/crema_backlog.db`): chỉ lưu `asked_date YYYY-MM-DD`, khử SĐT/email/mã nhân sự/tên người/bộ phận, ngưỡng `hit_count >= 3`.
- FastAPI (`src/server.py` + `src/static/index.html`): `/`, `/health`, `/v1/chat`, RBAC `/v1/ingest` & `/v1/backlog` (`401/403` khi thiếu/sai key).
- Chờ kiểm tra thủ công trên trình duyệt trước khi commit Giai đoạn 5.
