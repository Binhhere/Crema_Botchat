"""Stage 4 tests: Calibration/Held-out dataset integrity, Constrained Risk Selection, and Hot-Ingest."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from eval.run_eval import (
    CALIBRATION_PATH,
    HELDOUT_PATH,
    check_must_contain,
    select_calibrated_threshold,
)
from src.chunker import chunk_document
from src.config import DATA_CACHE_DIR, DATA_RAW_DIR, EVAL_DIR
from src.generator import ask_policy
from src.indexer import build_or_load_index, get_genai_client, load_corpus_chunks
from src.loader import load_document
from src.retriever import HybridRetriever
from src.rewriter import has_vietnamese_diacritics


def test_calibration_and_heldout_dataset_integrity() -> None:
    calib = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))

    assert len(calib) == 36
    assert sum(1 for x in calib if x["expected_action"] == "ANSWERED") == 16
    assert sum(1 for x in calib if x["expected_action"] == "OUT_OF_SCOPE") == 20

    assert len(heldout) == 60
    assert sum(1 for x in heldout if x["category"] == "valid_policy") == 20
    assert sum(1 for x in heldout if x["category"] == "negative_oos") == 20
    assert sum(1 for x in heldout if x["category"] == "multi_turn") == 8
    assert sum(1 for x in heldout if x["slice"] == "near_miss_injection") == 12
    assert sum(1 for x in heldout if x["category"] == "near_miss_policy") == 6
    assert sum(1 for x in heldout if x["category"] == "near_miss_blocked") == 6

    # Verify strict separation (zero duplicate queries) between calibration & held-out
    cal_queries = {x["query"].strip().lower() for x in calib}
    ho_queries = {x["query"].strip().lower() for x in heldout}
    assert cal_queries.isdisjoint(ho_queries)

    # Verify all unaccented queries have 0 Vietnamese diacritics
    for x in calib:
        if x["is_unaccented"]:
            assert not has_vietnamese_diacritics(x["query"]), x["id"]
    for x in heldout:
        if x["slice"] == "unaccented":
            assert not has_vietnamese_diacritics(x["query"]), x["id"]

    # Verify every (gold_doc_id, gold_clause) exists verbatim in the 590 base corpus chunks
    corpus = load_corpus_chunks(DATA_RAW_DIR)
    doc_clauses = {(c.doc_id, c.section_code) for c in corpus}
    for row in [*calib, *heldout]:
        if row["gold_doc_id"] and row["gold_clause"]:
            assert (row["gold_doc_id"], row["gold_clause"]) in doc_clauses, row["id"]


def test_constrained_risk_selection_prefers_far_bound_over_f1() -> None:
    # Construct synthetic calibration scores where tau=0.58 yields high F1 but FAR=15% (3/20),
    # whereas tau=0.66 enforces FAR=0% (0/20) with 14/16 positive recall (87.5% >= 85%).
    synth: list[dict] = []
    for i in range(14):
        synth.append(
            {"id": f"P{i}", "query": "q", "is_positive": True, "top1_cosine": 0.72, "bm25_top1_score": 30.0, "bm25_zscore": 5.0}
        )
    for i in range(2):
        synth.append(
            {"id": f"PL{i}", "query": "q", "is_positive": True, "top1_cosine": 0.60, "bm25_top1_score": 10.0, "bm25_zscore": 1.5}
        )
    for i in range(3):
        synth.append(
            {"id": f"NH{i}", "query": "q", "is_positive": False, "top1_cosine": 0.64, "bm25_top1_score": 12.0, "bm25_zscore": 2.0}
        )
    for i in range(17):
        synth.append(
            {"id": f"NL{i}", "query": "q", "is_positive": False, "top1_cosine": 0.52, "bm25_top1_score": 10.0, "bm25_zscore": 1.5}
        )

    res = select_calibrated_threshold(synth, max_far=0.05)
    assert res["chosen"]["neg_far"] == 0.0
    assert res["chosen"]["tau_reject"] == 0.65


def test_hot_ingest_extra_05_contractor_handbook(tmp_path: Path) -> None:
    extra_file = EVAL_DIR / "extra_05_so_tay_nha_thau.docx"
    assert extra_file.exists()
    assert not (DATA_RAW_DIR / extra_file.name).exists()

    # Copy existing base cache into isolated tmp_path so data/raw & data/cache remain untouched
    tmp_cache = tmp_path / "cache"
    tmp_cache.mkdir(parents=True, exist_ok=True)
    for fn in ("embeddings_meta.json", "embeddings_vectors.npy"):
        shutil.copy2(DATA_CACHE_DIR / fn, tmp_cache / fn)

    client = get_genai_client()
    base_index = build_or_load_index(raw_dir=DATA_RAW_DIR, cache_dir=tmp_cache, client=client)
    assert len(base_index.chunks) == 590

    loaded_extra = load_document(extra_file)
    extra_chunks = chunk_document(loaded_extra)
    assert len(extra_chunks) == 32

    stats = base_index.add_chunks(extra_chunks, client=client, batch_size=64)
    assert stats.total_chunks == 622
    assert stats.cache_hits == 590
    assert stats.newly_embedded == 32

    retriever = HybridRetriever(base_index)

    checks = [
        (
            "Giới hạn tốc độ tối đa của xe cơ giới nhà thầu trên đường trục chính và khu vực cửa xưởng trong Nhà máy iLD là bao nhiêu km/giờ?",
            ["15", "5"],
        ),
        (
            "Thời hạn hiệu lực gốc của mỗi Giấy phép thi công PTW và thời gian gia hạn tối đa bao nhiêu giờ?",
            ["8", "4"],
        ),
        (
            "Ngưỡng nồng độ khí Oxy (O2) an toàn bắt buộc khi vào không gian kín theo Sổ tay Nhà thầu là từ bao nhiêu đến bao nhiêu phần trăm?",
            ["19,5", "23,5"],
        ),
        (
            "Khung chế tài khấu trừ phạt tiền đối với nhà thầu cho vi phạm mức nhẹ, trung bình và nghiêm trọng là bao nhiêu đồng?",
            ["1.000.000", "5.000.000", "20.000.000"],
        ),
    ]
    for query, expected_nums in checks:
        ans = ask_policy(query=query, retriever=retriever, client=client)
        assert ans.status == "ANSWERED"
        assert ans.verification is not None and ans.verification.is_valid is True
        assert check_must_contain(ans.answer, expected_nums), ans.answer
        assert any(c.doc_id == "extra_05_so_tay_nha_thau" for c in ans.citations)
        for c in ans.citations:
            if c.doc_id == "extra_05_so_tay_nha_thau":
                assert c.doc_nature == "synthetic_policy"
                assert c.ui_source_label == "[Văn bản mẫu mô phỏng]"
                assert c.page_number is None
                assert "Trang" not in c.display_label
