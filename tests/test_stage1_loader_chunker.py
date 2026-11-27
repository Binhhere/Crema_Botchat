"""Stage 1 tests verifying forensic loading and multi-strategy chunking on all 5 real files."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pymupdf
import pytest

from src.chunker import chunk_document
from src.config import DATA_RAW_DIR, EVAL_DIR
from src.loader import load_document
from src.models import DocumentMetadata, LoadedBlock, LoadedDocument, ScannedPDFError

FILE_01 = DATA_RAW_DIR / "01_nd88_2020_hop_nhat_04_vbhn_bnv.pdf"
FILE_02 = DATA_RAW_DIR / "02_tt28_2021_hop_nhat_06_vbhn_bnv.pdf"
FILE_03 = DATA_RAW_DIR / "03_nd145_2020_dieu_kien_lao_dong.docx"
FILE_04 = DATA_RAW_DIR / "04_nike_cls_2020_an_toan_lao_dong.pdf"
FILE_05 = EVAL_DIR / "extra_05_so_tay_nha_thau.docx"


def _assert_common_chunk_invariants(chunks: list, expect_pdf: bool) -> None:
    assert len(chunks) > 0
    seen_ids: set[str] = set()
    for ch in chunks:
        assert ch.text.strip() != ""
        assert len(ch.text.strip()) >= 20
        assert ch.text == unicodedata.normalize("NFC", ch.text)
        assert ch.chunk_id not in seen_ids
        seen_ids.add(ch.chunk_id)
        if expect_pdf:
            assert ch.page_number is not None and ch.page_number >= 1
        else:
            assert ch.page_number is None


def test_file_01_vbhn_04_metadata_footnotes_and_scope() -> None:
    loaded = load_document(FILE_01)
    chunks = chunk_document(loaded)
    meta = loaded.metadata

    assert meta.doc_nature == "consolidated_law"
    assert meta.ui_source_label == "[Văn bản pháp luật hợp nhất]"
    assert meta.document_code == "88/2020/NĐ-CP"
    assert meta.consolidated_code == "04/VBHN-BNV"
    assert meta.original_effective_date == "2020-09-15"
    assert meta.consolidated_date == "2026-02-02"
    assert any("158/2025/NĐ-CP" in item for item in meta.amended_by)

    _assert_common_chunk_invariants(chunks, expect_pdf=True)
    assert all(1 <= ch.page_number <= 30 for ch in chunks if ch.page_number is not None)
    assert all(ch.section_type == "body" for ch in chunks)
    assert not any("ĐƠN ĐỀ NGHỊ" in ch.text for ch in chunks)

    dieu_codes = {ch.section_code for ch in chunks}
    assert {"Điều 1", "Điều 2", "Điều 11", "Điều 39", "Điều 45", "Điều 47"}.issubset(dieu_codes)
    # Footnote 6 quoting Điều 45 of NĐ 158/2025 must NOT create a duplicate Điều 45
    assert sum(1 for ch in chunks if ch.section_code == "Điều 45") == 1

    # Inline superscript "2" must not corrupt "1. Người lao động" into "1.2 Người lao động"
    d2_chunks = [ch for ch in chunks if ch.section_code == "Điều 2"]
    assert d2_chunks and "1.2 Người lao động" not in d2_chunks[0].text
    assert "1. Người lao động" in d2_chunks[0].text
    assert d2_chunks[0].clause_effective_date == "2025-07-01"

    d11_chunks = [ch for ch in chunks if ch.section_code == "Điều 11"]
    assert any(ch.clause_effective_date == "2025-07-01" for ch in d11_chunks)

    d39_chunks = [ch for ch in chunks if ch.section_code == "Điều 39"]
    assert d39_chunks and d39_chunks[0].clause_effective_date == "2025-07-01"

    d1_chunks = [ch for ch in chunks if ch.section_code == "Điều 1"]
    assert d1_chunks and d1_chunks[0].clause_effective_date == "2020-09-15"


def test_file_02_vbhn_06_font_fix_footnotes_and_appendices() -> None:
    loaded = load_document(FILE_02)
    chunks = chunk_document(loaded)
    meta = loaded.metadata

    assert meta.doc_nature == "consolidated_law"
    assert meta.ui_source_label == "[Văn bản pháp luật hợp nhất]"
    assert meta.document_code == "28/2021/TT-BLĐTBXH"
    assert meta.consolidated_code == "06/VBHN-BNV"
    assert meta.original_effective_date == "2022-03-01"
    assert meta.consolidated_date == "2026-02-11"
    assert meta.amended_by == ["25/2025/TT-BNV (2026-01-01)"]

    _assert_common_chunk_invariants(chunks, expect_pdf=True)

    # Zero corrupted legacy Ƣ/ƣ glyphs anywhere
    for ch in chunks:
        assert "\u01a2" not in ch.text and "\u01a3" not in ch.text

    d5_chunks = [ch for ch in chunks if ch.section_code == "Điều 5"]
    assert any(ch.clause_effective_date == "2026-01-01" for ch in d5_chunks)
    d5_e_chunks = [ch for ch in d5_chunks if "Đối với hạ sĩ quan, binh sĩ quân đội nhân dân" in ch.text]
    assert len(d5_e_chunks) == 1
    assert "cơ yếu đang" in d5_e_chunks[0].text
    assert d5_e_chunks[0].clause_effective_date == "2026-01-01"
    d3_chunks = [ch for ch in chunks if ch.section_code == "Điều 3"]
    assert d3_chunks and all(ch.clause_effective_date == "2022-03-01" for ch in d3_chunks)

    appendix_codes = {ch.section_code for ch in chunks if ch.section_type == "appendix"}
    assert appendix_codes == {"Phụ lục I", "Phụ lục V"}
    app1_text = "\n".join(ch.text for ch in chunks if ch.section_code == "Phụ lục I")
    app5_text = "\n".join(ch.text for ch in chunks if ch.section_code == "Phụ lục V")
    assert "3,50" in app1_text and "1,40" in app1_text
    assert "2.800.000" in app5_text


def test_file_03_nd145_docx_articles_and_partial_expiry_warning() -> None:
    loaded = load_document(FILE_03)
    chunks = chunk_document(loaded)
    meta = loaded.metadata

    assert meta.doc_nature == "legal_decree"
    assert meta.ui_source_label == "[Nghị định pháp luật]"
    assert meta.document_code == "145/2020/NĐ-CP"
    assert meta.original_effective_date == "2021-02-01"

    _assert_common_chunk_invariants(chunks, expect_pdf=False)

    dieu_codes = {ch.section_code for ch in chunks}
    assert {"Điều 1", "Điều 4", "Điều 9", "Điều 31", "Điều 55", "Điều 92-97", "Điều 115"}.issubset(
        dieu_codes
    )
    assert not any("Mẫu số 01/PLIV" in ch.text for ch in chunks)

    warned = {ch.section_code for ch in chunks if ch.partial_expiry_warning}
    assert warned == {"Điều 4", "Điều 31"}


def test_file_04_nike_cls_clean_headers_glossary_and_decimal_sections() -> None:
    loaded = load_document(FILE_04)
    chunks = chunk_document(loaded)
    meta = loaded.metadata

    assert meta.doc_nature == "enterprise_reference"
    assert meta.ui_source_label == "[Tiêu chuẩn doanh nghiệp tham chiếu]"

    _assert_common_chunk_invariants(chunks, expect_pdf=True)

    for ch in chunks:
        assert "BỘ QUY TẮC CHUẨN MỰC LÃNH ĐẠO CỦA NIKE" not in ch.text
        assert "Về đầu trang | Trang" not in ch.text

    assert any(ch.section_type == "glossary" for ch in chunks)
    section_codes = {ch.section_code for ch in chunks}
    assert {"1.1", "13.2.2", "50.2.7"}.issubset(section_codes)


def test_file_05_extra_docx_isolated_for_hot_ingest() -> None:
    assert not (DATA_RAW_DIR / FILE_05.name).exists()
    loaded = load_document(FILE_05)
    chunks = chunk_document(loaded)
    meta = loaded.metadata

    assert meta.doc_nature == "synthetic_policy"
    assert meta.ui_source_label == "[Văn bản mẫu mô phỏng]"
    assert meta.document_code == "CREMA-HSE-NT-2026"
    assert meta.original_effective_date == "2026-11-15"

    _assert_common_chunk_invariants(chunks, expect_pdf=False)
    section_codes = {ch.section_code for ch in chunks}
    assert {"0.2", "1.2.3", "3.2.3", "4.2.3", "4.4"}.issubset(section_codes)


def test_sliding_window_fallback_and_scanned_pdf_guard(tmp_path: Path) -> None:
    # Strategy 3 sliding window fallback on unstructured prose
    paras = [
        LoadedBlock(text=f"Đoạn văn bản số {i}. " + ("Nội dung quy định nội bộ công ty. " * 25))
        for i in range(6)
    ]
    plain_doc = LoadedDocument(
        metadata=DocumentMetadata(
            doc_id="plain_prose",
            filename="plain_prose.txt",
            doc_title="Tài liệu tự do",
            doc_nature="synthetic_policy",
            ui_source_label="[Văn bản mẫu mô phỏng]",
        ),
        blocks=paras,
    )
    chunks = chunk_document(plain_doc, max_words=220, overlap_words=60)
    assert len(chunks) >= 3
    assert chunks[0].section_code == "Đoạn 1"

    # Scanned PDF guard on blank/image-only multi-page PDF
    scan_pdf = tmp_path / "scanned.pdf"
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page()
    doc.save(scan_pdf)
    doc.close()

    with pytest.raises(ScannedPDFError, match="UNSUPPORTED_SCANNED_PDF"):
        load_document(scan_pdf)
