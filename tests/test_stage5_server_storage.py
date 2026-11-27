"""Stage 5 tests verifying PII scrubbing, k-anonymity, RBAC, and E2E FastAPI workflows."""

from __future__ import annotations

import re
import shutil
import sqlite3
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

from src.config import DATA_CACHE_DIR, DATA_RAW_DIR, EVAL_DIR, get_admin_api_key, get_hr_api_key
from src.indexer import get_genai_client
from src.server import create_app
from src.storage import SQLiteBacklogStore, scrub_pii_and_identity


def test_scrubber_removes_pii_names_units_and_stores_date_only(tmp_path: Path) -> None:
    raw = (
        "Tôi tên là Nguyễn Văn Hùng, mã nhân viên ILD-042, sđt 0912.345.678, "
        "email hung.nguyen@ildcoffee.vn thuộc phân xưởng sấy lạnh 2 tổ chuyền 3, "
        "hỏi anh Minh ở bộ phận cơ điện về hỗ trợ mua xe máy điện?"
    )
    scrubbed = scrub_pii_and_identity(raw)
    for forbidden_pii in [
        "Nguyễn Văn Hùng",
        "ILD-042",
        "0912.345.678",
        "hung.nguyen@ildcoffee.vn",
        "anh Minh",
        "phân xưởng sấy lạnh 2",
        "tổ chuyền 3",
        "bộ phận cơ điện",
    ]:
        assert forbidden_pii.lower() not in scrubbed.lower(), scrubbed
    assert "[EMAIL]" in scrubbed
    assert "[SDT]" in scrubbed
    assert "[MA_NV]" in scrubbed
    assert "[TEN_NGUOI]" in scrubbed
    assert "[BO_PHAN]" in scrubbed

    db_path = tmp_path / "backlog.db"
    store = SQLiteBacklogStore(db_path=db_path)
    entry = store.record_unanswered(raw, status="OUT_OF_SCOPE", asked_date=date(2026, 11, 27))
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry.asked_date)
    assert len(entry.asked_date) == 10

    with sqlite3.connect(str(db_path)) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(crema_faq_backlog)").fetchall()]
        assert cols == ["topic_key", "scrubbed_question", "asked_date", "hit_count", "sample_status"]
        assert "user_id" not in cols


def test_k_anonymity_hides_below_3_and_merges_variants(tmp_path: Path) -> None:
    store = SQLiteBacklogStore(db_path=tmp_path / "kanon.db")

    # Topic A asked twice -> MUST remain hidden even if caller requests min_hits=1
    store.record_unanswered("Chính sách thưởng cổ phiếu ESOP cho kỹ sư là gì?")
    store.record_unanswered("Chính sách thưởng cổ phiếu ESOP cho kỹ sư là gì?")
    assert store.list_public_backlog(min_hits=1) == []

    # Topic B asked 3 times (including unaccented & different PII) -> MUST surface at hit_count=3
    store.record_unanswered("Công ty có hỗ trợ tiền thuê nhà trọ không?")
    store.record_unanswered("Em là Lan ở phân xưởng 1, cho hỏi công ty có hỗ trợ tiền thuê nhà trọ không?")
    store.record_unanswered("cong ty co ho tro tien thue nha tro khong")

    visible = store.list_public_backlog()
    assert len(visible) == 1
    assert visible[0].hit_count == 3
    assert "Lan" not in visible[0].scrubbed_question
    assert "phân xưởng" not in visible[0].scrubbed_question.lower()


def test_rbac_rejects_unauthorized_on_ingest_and_backlog(tmp_path: Path) -> None:
    app = create_app(
        raw_dir=DATA_RAW_DIR,
        cache_dir=tmp_path / "cache",
        ingested_dir=tmp_path / "ingested",
        sqlite_path=tmp_path / "rbac.db",
        load_persisted_ingested=False,
    )
    tc = TestClient(app)

    # Missing API key -> 401
    assert tc.get("/v1/backlog").status_code == 401
    assert tc.post("/v1/ingest?preset=extra_05").status_code == 401

    # Wrong API key -> 403
    bad_headers = {"X-API-Key": "wrong_key_123"}
    assert tc.get("/v1/backlog", headers=bad_headers).status_code == 403
    assert tc.post("/v1/ingest?preset=extra_05", headers=bad_headers).status_code == 403

    # Valid HR_API_KEY / ADMIN_API_KEY -> 200 on backlog
    hr_key = get_hr_api_key()
    admin_key = get_admin_api_key()
    assert hr_key and admin_key
    r_hr = tc.get("/v1/backlog", headers={"X-API-Key": hr_key})
    assert r_hr.status_code == 200
    assert r_hr.json()["k_anonymity_min_hits"] == 3
    r_adm = tc.get("/v1/backlog", headers={"Authorization": f"Bearer {admin_key}"})
    assert r_adm.status_code == 200


def test_e2e_server_chat_backlog_and_hot_ingest_extra_05(tmp_path: Path) -> None:
    tmp_cache = tmp_path / "cache"
    tmp_cache.mkdir(parents=True, exist_ok=True)
    for fn in ("embeddings_meta.json", "embeddings_vectors.npy"):
        shutil.copy2(DATA_CACHE_DIR / fn, tmp_cache / fn)

    app = create_app(
        raw_dir=DATA_RAW_DIR,
        cache_dir=tmp_cache,
        ingested_dir=tmp_path / "ingested",
        sqlite_path=tmp_path / "e2e_backlog.db",
        client=get_genai_client(),
        load_persisted_ingested=False,
    )
    tc = TestClient(app)

    # 1. UI and /health check
    home = tc.get("/")
    assert home.status_code == 200
    html_lower = home.text.lower()
    for banned in ("công nhân", "kiosk", "mã băm", "ca làm", "trưởng ca"):
        assert banned not in html_lower

    health = tc.get("/health")
    assert health.status_code == 200
    assert health.json()["total_chunks"] == 590

    # 2. Send 3 out-of-scope chat queries with PII -> verify backlog scrubbing & k-anonymity >= 3
    oos_variants = [
        "Tôi là Trần Minh Tuấn ở phân xưởng sấy lạnh 2 sđt 0987654321 hỏi công ty có hỗ trợ gói vay mua nhà không?",
        "Cho em hỏi công ty có hỗ trợ gói vay mua nhà không? Mã ILD-108",
        "cong ty co ho tro goi vay mua nha khong",
    ]
    for q in oos_variants:
        resp = tc.post("/v1/chat", json={"query": q, "history": []})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "OUT_OF_SCOPE"
        assert payload["backlog_recorded"] is True

    admin_headers = {"X-API-Key": get_admin_api_key() or ""}
    backlog_res = tc.get("/v1/backlog", headers=admin_headers)
    assert backlog_res.status_code == 200
    items = backlog_res.json()["items"]
    assert len(items) == 1
    assert items[0]["hit_count"] == 3
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", items[0]["asked_date"])
    assert "Trần Minh Tuấn" not in items[0]["scrubbed_question"]
    assert "0987654321" not in items[0]["scrubbed_question"]
    assert "ILD-108" not in items[0]["scrubbed_question"]

    # 3. Hot-ingest eval/extra_05_so_tay_nha_thau.docx via multipart upload -> ask via /v1/chat
    extra_docx = EVAL_DIR / "extra_05_so_tay_nha_thau.docx"
    with extra_docx.open("rb") as fh:
        ingest_res = tc.post(
            "/v1/ingest",
            headers=admin_headers,
            files={
                "file": (
                    extra_docx.name,
                    fh,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
    assert ingest_res.status_code == 200
    ing = ingest_res.json()
    assert ing["chunks_added"] == 32
    assert ing["total_chunks"] == 622
    assert ing["doc_nature"] == "synthetic_policy"
    assert ing["ui_source_label"] == "[Văn bản mẫu mô phỏng]"

    chat_extra = tc.post(
        "/v1/chat",
        json={
            "query": "Ngưỡng nồng độ khí Oxy (O2) an toàn bắt buộc khi nhà thầu vào không gian kín là từ bao nhiêu đến bao nhiêu phần trăm?",
            "history": [],
        },
    )
    assert chat_extra.status_code == 200
    extra_ans = chat_extra.json()
    assert extra_ans["status"] == "ANSWERED"
    assert extra_ans["citations"]
    assert any(c["doc_id"] == "extra_05_so_tay_nha_thau" for c in extra_ans["citations"])
    for c in extra_ans["citations"]:
        if c["doc_id"] == "extra_05_so_tay_nha_thau":
            assert c["ui_source_label"] == "[Văn bản mẫu mô phỏng]"
            assert c["page_number"] is None
            assert "Trang" not in c["display_label"]

    # 4. Partial expiry warning check on Điều 4 NĐ 145/2020/NĐ-CP
    chat_d4 = tc.post(
        "/v1/chat",
        json={
            "query": "Theo Điều 4 Nghị định 145/2020/NĐ-CP, doanh nghiệp phải báo cáo định kỳ thay đổi lao động trước ngày nào của tháng 6 và tháng 12?",
            "history": [],
        },
    )
    assert chat_d4.status_code == 200
    d4_ans = chat_d4.json()
    assert d4_ans["status"] == "ANSWERED"
    assert d4_ans["has_partial_expiry_warning"] is True
