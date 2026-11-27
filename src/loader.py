"""Forensic document loaders for PDF, DOCX, TXT, and MD policy documents."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import docx
import pymupdf
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.config import DOC_NATURE_LABELS
from src.models import (
    DocNature,
    DocumentMetadata,
    LoadedBlock,
    LoadedDocument,
    ScannedPDFError,
    SectionType,
)

_VN_LONG_DATE_RE = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE)
_VN_SLASH_DATE_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_DOC_CODE_RE = re.compile(
    r"(?:Nghị định|Thông tư)\s+(?:số\s+)?([0-9]+/[0-9]{4}/[A-ZĐƯ\-]+)",
    re.IGNORECASE,
)


def normalize_source_text(text: str) -> str:
    """Normalize Unicode to NFC and fix legacy font glyph artifacts (e.g. Ƣ/ƣ -> Ư/ư)."""
    cleaned = text.replace("\u01a2", "Ư").replace("\u01a3", "ư")
    cleaned = cleaned.replace("\u00ad", "").replace("\r", "")
    cleaned = unicodedata.normalize("NFC", cleaned)
    return cleaned


def _parse_vn_date(text: str) -> str | None:
    """Parse 'ngày DD tháng MM năm YYYY' or 'DD/MM/YYYY' into ISO YYYY-MM-DD."""
    m = _VN_LONG_DATE_RE.search(text)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"
    m2 = _VN_SLASH_DATE_RE.search(text)
    if m2:
        day, month, year = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _check_pdf_text_density(pages: list[pymupdf.Page], filename: str) -> None:
    """Guard against scanned image-only PDFs lacking an extractable text layer."""
    if not pages:
        raise ScannedPDFError(f"UNSUPPORTED_SCANNED_PDF: {filename} has 0 target pages.")
    total_chars = sum(
        len(re.sub(r"\s+", "", pg.get_text("text")))
        for pg in pages
    )
    avg_chars = total_chars / len(pages)
    if avg_chars < 30.0:
        raise ScannedPDFError(
            f"UNSUPPORTED_SCANNED_PDF: {filename} averages {avg_chars:.1f} chars/page (< 30)."
        )


def _group_words_into_lines(page: pymupdf.Page, y_tol: float = 4.0) -> list[tuple[float, str]]:
    """Group PyMuPDF word tuples into horizontally sorted lines by y0 proximity."""
    words = page.get_text("words")
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))
    buckets: list[tuple[float, list[tuple[float, str]]]] = []
    for w in words_sorted:
        x0, y0, _, _, token = w[0], w[1], w[2], w[3], normalize_source_text(w[4])
        if not token.strip():
            continue
        if buckets and abs(y0 - buckets[-1][0]) <= y_tol:
            buckets[-1][1].append((x0, token))
        else:
            buckets.append((y0, [(x0, token)]))
    result: list[tuple[float, str]] = []
    for y_val, row in buckets:
        row_sorted = sorted(row, key=lambda item: item[0])
        result.append((y_val, " ".join(tok for _, tok in row_sorted)))
    return result


def _extract_vbhn_signature_info(page: pymupdf.Page) -> tuple[str | None, str | None]:
    """Extract VBHN code (e.g. 04/VBHN-BNV) and consolidated date via spatial word coords."""
    words = [(w[0], w[1], normalize_source_text(w[4])) for w in page.get_text("words")]
    vbhn_word = next((w for w in words if "/VBHN-" in w[2].upper()), None)
    if not vbhn_word:
        return None, None
    x_vbhn, y_vbhn, vbhn_tok = vbhn_word

    band = [w for w in words if abs(w[1] - y_vbhn) <= 15.0]
    so_word = next((w for w in band if w[2].rstrip(":") == "Số"), None)
    x_so = so_word[0] if so_word else x_vbhn - 120.0

    num_tokens = [
        w[2] for w in sorted(band, key=lambda item: item[0])
        if x_so < w[0] < x_vbhn and w[2].isdigit()
    ]
    consolidated_code: str | None = None
    if num_tokens:
        num_str = num_tokens[-1].zfill(2)
        consolidated_code = f"{num_str}{vbhn_tok}" if vbhn_tok.startswith("/") else vbhn_tok

    ngay_w = next((w for w in band if w[0] > x_vbhn and w[2].lower() == "ngày"), None)
    thang_w = next((w for w in band if w[0] > x_vbhn and w[2].lower() == "tháng"), None)
    nam_w = next((w for w in band if w[0] > x_vbhn and w[2].lower() == "năm"), None)

    consolidated_date: str | None = None
    if ngay_w and thang_w and nam_w:
        days = [w[2] for w in band if ngay_w[0] < w[0] < thang_w[0] and w[2].isdigit()]
        months = [w[2] for w in band if thang_w[0] < w[0] < nam_w[0] and w[2].isdigit()]
        years = [w[2] for w in band if w[0] > nam_w[0] and w[2].isdigit() and len(w[2]) == 4]
        if days and months and years:
            consolidated_date = f"{int(years[0]):04d}-{int(months[0]):02d}-{int(days[0]):02d}"

    return consolidated_code, consolidated_date


def _extract_vbhn_preamble(page1: pymupdf.Page) -> dict[str, object]:
    """Parse base code, original effective date, and amending docs from VBHN page 1."""
    lines_sorted = _group_words_into_lines(page1, y_tol=5.0)
    full_text = "\n".join(txt for _, txt in lines_sorted)
    full_text = re.sub(r"([A-ZĐ])-\s*\n\s*([A-ZĐ]+)", r"\1-\2", full_text)

    # Title block: NGHỊ ĐỊNH / THÔNG TƯ + subtitle lines
    title_lines: list[str] = []
    capture_title = False
    for _, line_txt in lines_sorted:
        stripped = line_txt.strip()
        if re.match(r"^(NGHỊ ĐỊNH|THÔNG TƯ)$", stripped):
            capture_title = True
            title_lines.append(stripped)
            continue
        if capture_title:
            if any(
                stripped.startswith(p)
                for p in ("Thông tư số", "Nghị định số", "Căn cứ ", "Chương ")
            ):
                break
            title_lines.append(stripped)
            if len(title_lines) >= 6:
                break

    doc_title = " ".join(title_lines).strip()

    # Split preamble at "được sửa đổi, bổ sung bởi:"
    parts = re.split(
        r"được sửa đổi,\s*bổ sung bởi\s*:",
        full_text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    base_part = parts[0]
    amend_part = parts[1].split("Căn cứ", 1)[0] if len(parts) > 1 else ""

    base_code_m = _DOC_CODE_RE.search(base_part)
    document_code = base_code_m.group(1) if base_code_m else None

    eff_m = re.search(
        r"có hiệu lực kể từ\s+(ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
        base_part,
        re.IGNORECASE,
    )
    original_effective_date = _parse_vn_date(eff_m.group(1)) if eff_m else None

    amended_by: list[str] = []
    amend_date_map: dict[str, str] = {}
    if amend_part:
        # Split amending clauses by sentence boundary (".\n" or ";\n") rather than target references
        clauses = [c.strip() for c in re.split(r"[;\.]\s*\n", amend_part) if c.strip()]
        for seg in clauses:
            cm = _DOC_CODE_RE.search(seg)
            if not cm:
                continue
            code = cm.group(1)
            if document_code and code.upper() == document_code.upper():
                continue
            dm = re.search(
                r"có hiệu lực kể từ\s+(ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
                seg,
                re.IGNORECASE,
            )
            eff_iso = _parse_vn_date(dm.group(1)) if dm else None
            label = f"{code} ({eff_iso})" if eff_iso else code
            if label not in amended_by:
                amended_by.append(label)
            if eff_iso:
                amend_date_map[code.upper()] = eff_iso

    if doc_title and document_code:
        doc_title = f"{doc_title} (Hợp nhất {document_code})"

    return {
        "doc_title": doc_title or (document_code or "Văn bản hợp nhất"),
        "document_code": document_code,
        "original_effective_date": original_effective_date,
        "amended_by": amended_by,
        "amend_date_map": amend_date_map,
    }


def _extract_vbhn_page_dict(
    page: pymupdf.Page,
    is_sig_page: bool,
) -> tuple[list[tuple[str, list[int]]], list[str]]:
    """Separate main body lines (with superscript footnote IDs) from small bottom footnotes."""
    raw_dict = page.get_text("dict")
    all_lines: list[tuple[float, list[dict[str, object]]]] = []
    for block in raw_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            raw_spans = line.get("spans", [])
            if not any(str(s.get("text", "")).strip() for s in raw_spans):
                continue
            spans = [s for s in raw_spans if str(s.get("text", "")) != ""]
            all_lines.append((float(line["bbox"][1]), spans))

    all_lines.sort(key=lambda item: item[0])

    body_lines: list[tuple[str, list[int]]] = []
    footnote_lines: list[str] = []
    stop_body = False

    for y0, spans in all_lines:
        raw_joined = normalize_source_text("".join(str(s["text"]) for s in spans)).strip()
        if not raw_joined:
            continue

        # Skip standalone top page numbers
        if y0 < 65.0 and raw_joined.isdigit():
            continue

        max_size = max(
            float(s["size"]) for s in spans if str(s.get("text", "")).strip()
        )

        # Bottom footnote zone in VBHN PDFs uses <= 10.5pt font
        if max_size <= 10.5:
            footnote_lines.append(raw_joined)
            continue

        if is_sig_page and (
            raw_joined.startswith(("BỘ NỘI VỤ", "Nơi nhận:"))
            or "XÁC THỰC VĂN BẢN HỢP NHẤT" in raw_joined
        ):
            stop_body = True
        if stop_body:
            continue

        # Build clean body line while capturing inline superscript footnote references
        parts: list[str] = []
        fn_ids: list[int] = []
        for s in spans:
            stext = normalize_source_text(str(s["text"]))
            flags = int(s["flags"])
            size = float(s["size"])
            stripped = stext.strip()
            if (flags & 1) != 0 and size <= 10.5 and stripped.isdigit():
                fn_num = int(stripped)
                if 1 <= fn_num <= 50:
                    fn_ids.append(fn_num)
                parts.append(" ")
                continue
            parts.append(stext)

        clean_line = re.sub(r"\s+", " ", "".join(parts)).strip()
        clean_line = re.sub(r"\s+([,.;:)!?])", r"\1", clean_line)
        if clean_line:
            body_lines.append((clean_line, fn_ids))

    return body_lines, footnote_lines


def _parse_footnote_effective_dates(
    footnote_lines: list[str],
    amend_date_map: dict[str, str],
) -> dict[int, str]:
    """Map footnote numbers [1..N] to amending document effective dates."""
    grouped: list[tuple[int, str]] = []
    for line in footnote_lines:
        m = re.match(r"^(\d{1,2})\s*(.+)$", line)
        if m and (
            any(
                m.group(2).startswith(prefix)
                for prefix in ("Khoản", "Điều", "Điểm", "Nghị định", "Thông tư", "Cụm từ", "Bãi bỏ")
            )
        ):
            grouped.append((int(m.group(1)), m.group(2)))
        elif grouped:
            fn_id, prev = grouped[-1]
            grouped[-1] = (fn_id, f"{prev} {line}")

    fn_dates: dict[int, str] = {}
    for fn_id, fn_text in grouped:
        for amend_code, eff_date in amend_date_map.items():
            if amend_code in fn_text.upper():
                fn_dates[fn_id] = eff_date
                break
        if fn_id not in fn_dates:
            dt = _parse_vn_date(fn_text)
            if dt and "có hiệu lực" in fn_text.lower():
                fn_dates[fn_id] = dt
    return fn_dates


def _load_vbhn_pdf(
    path: Path,
    pdf: pymupdf.Document,
    doc_nature: DocNature,
) -> LoadedDocument:
    """Load a Vietnamese consolidated legal PDF (VBHN-BNV) with footnotes & selected appendices."""
    sig_page_idx: int | None = None
    for idx in range(len(pdf)):
        txt = normalize_source_text(pdf[idx].get_text("text"))
        if "XÁC THỰC VĂN BẢN HỢP NHẤT" in txt:
            sig_page_idx = idx
            break
    if sig_page_idx is None:
        sig_page_idx = len(pdf) - 1

    target_pages = [pdf[i] for i in range(sig_page_idx + 1)]
    _check_pdf_text_density(target_pages, path.name)

    preamble = _extract_vbhn_preamble(pdf[0])
    consolidated_code, consolidated_date = _extract_vbhn_signature_info(pdf[sig_page_idx])

    blocks: list[LoadedBlock] = []
    all_footnotes: list[str] = []

    for pno in range(sig_page_idx + 1):
        body_lines, fn_lines = _extract_vbhn_page_dict(pdf[pno], is_sig_page=(pno == sig_page_idx))
        all_footnotes.extend(fn_lines)
        for line_txt, fn_ids in body_lines:
            heading_level: int | None = None
            if re.match(r"^Chương\s+[IVXLCDM]+\b", line_txt):
                heading_level = 1
            elif re.match(r"^Mục\s+\d+\b", line_txt):
                heading_level = 2
            elif re.match(r"^Điều\s+\d+[\.\:]", line_txt):
                heading_level = 3
            blocks.append(
                LoadedBlock(
                    text=line_txt,
                    page_number=pno + 1,
                    heading_level=heading_level,
                    footnote_ids=fn_ids,
                    section_type="body",
                )
            )

    # Inspect post-signature appendices (keep Phụ lục I and Phụ lục V for Thông tư 28/2021)
    doc_code = str(preamble.get("document_code") or "")
    keep_appendices = {"I", "V"} if "28/2021" in doc_code else set()
    if keep_appendices:
        current_app: str | None = None
        for pno in range(sig_page_idx + 1, len(pdf)):
            lines = _group_words_into_lines(pdf[pno], y_tol=4.0)
            cleaned_lines = [
                txt for y_val, txt in lines
                if not (y_val < 55.0 and txt.strip().isdigit())
            ]
            if not cleaned_lines:
                continue
            head_join = " ".join(cleaned_lines[:3])
            app_m = re.search(r"PHỤ\s+LỤC\s+([IVXLCDM]+)\b", head_join)
            if app_m:
                current_app = app_m.group(1)
            if current_app in keep_appendices:
                for line_txt in cleaned_lines:
                    h_lvl = 2 if re.match(r"^PHỤ\s+LỤC\s+[IVXLCDM]+\b", line_txt) else None
                    blocks.append(
                        LoadedBlock(
                            text=line_txt,
                            page_number=pno + 1,
                            heading_level=h_lvl,
                            is_table=True,
                            section_type="appendix",
                        )
                    )

    amend_date_map: dict[str, str] = preamble["amend_date_map"]  # type: ignore[assignment]
    fn_effective_dates = _parse_footnote_effective_dates(all_footnotes, amend_date_map)

    metadata = DocumentMetadata(
        doc_id=path.stem,
        filename=path.name,
        doc_title=str(preamble["doc_title"]),
        doc_nature=doc_nature,
        ui_source_label=DOC_NATURE_LABELS[doc_nature],
        document_code=preamble["document_code"],  # type: ignore[arg-type]
        consolidated_code=consolidated_code,
        original_effective_date=preamble["original_effective_date"],  # type: ignore[arg-type]
        amended_by=preamble["amended_by"],  # type: ignore[arg-type]
        consolidated_date=consolidated_date,
        footnote_effective_dates=fn_effective_dates,
    )
    return LoadedDocument(metadata=metadata, blocks=blocks)


def _strip_nike_header_footer(raw_lines: list[str]) -> list[str]:
    """Remove repeating header/footer chrome from Nike CLS pages."""
    lines = [normalize_source_text(ln).strip() for ln in raw_lines]
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if not cur:
            i += 1
            continue
        # Lookahead for the running 2-line header: "<Chapter Title>\nBỘ QUY TẮC CHUẨN MỰC..."
        if i + 1 < len(lines) and lines[i + 1].strip() == "BỘ QUY TẮC CHUẨN MỰC LÃNH ĐẠO CỦA NIKE":
            i += 2
            continue
        if cur == "BỘ QUY TẮC CHUẨN MỰC LÃNH ĐẠO CỦA NIKE":
            i += 1
            continue
        if re.match(r"^Về đầu trang\s*\|\s*Trang\s*\d+\s*/\s*\d+$", cur):
            i += 1
            continue
        cleaned.append(cur)
        i += 1
    return cleaned


def _merge_cls_split_headings(lines: list[str]) -> list[tuple[str, int | None]]:
    """Merge split numeric heading markers (e.g. '1.1' + 'TIÊU CHUẨN') into single headings."""
    merged: list[tuple[str, int | None]] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        # Top-level CLS chapter split across two lines: "1" then "Bộ quy tắc được triển khai đầy đủ"
        if re.match(r"^\d{1,2}$", cur) and i + 1 < len(lines):
            nxt = lines[i + 1]
            if nxt and not re.match(r"^\d", nxt) and len(nxt) <= 95:
                merged.append((f"{cur} {nxt}", 1))
                i += 2
                continue
        # Two-level heading split: "1.1" then "TIÊU CHUẨN"
        if re.match(r"^\d{1,2}\.\d{1,2}$", cur) and i + 1 < len(lines):
            nxt = lines[i + 1]
            if nxt and nxt.upper() == nxt and len(nxt) <= 80:
                merged.append((f"{cur} {nxt}", 2))
                i += 2
                continue
        # Single-line 3-level heading: "13.2.2 CHÍNH SÁCH VÀ THỦ TỤC"
        if re.match(r"^\d{1,2}\.\d{1,2}\.\d{1,2}\s+\S", cur):
            merged.append((cur, 3))
            i += 1
            continue
        # Single-line 2-level heading: "1.1 TIÊU CHUẨN"
        if re.match(r"^\d{1,2}\.\d{1,2}\s+[A-ZÀ-Ỹ]", cur):
            merged.append((cur, 2))
            i += 1
            continue
        # Single-line 1-level chapter heading: "1 Bộ quy tắc..."
        if re.match(r"^\d{1,2}\s+[A-ZÀ-Ỹ]", cur) and len(cur) <= 95 and not cur.endswith("."):
            merged.append((cur, 1))
            i += 1
            continue
        merged.append((cur, None))
        i += 1
    return merged


def _load_enterprise_cls_pdf(
    path: Path,
    pdf: pymupdf.Document,
    doc_nature: DocNature,
) -> LoadedDocument:
    """Load Nike CLS PDF preserving physical page numbers, glossary, and decimal headings."""
    _check_pdf_text_density([pdf[i] for i in range(len(pdf))], path.name)

    blocks: list[LoadedBlock] = []
    in_glossary = False

    for pno in range(len(pdf)):
        raw_text = pdf[pno].get_text("text")
        if len(raw_text.strip()) < 15:
            continue
        stripped_lines = _strip_nike_header_footer(raw_text.splitlines())
        if not stripped_lines:
            continue
        # Skip Table of Contents pages
        dot_leader_count = sum(1 for ln in stripped_lines if "....." in ln)
        if stripped_lines[0] == "Mục lục" or dot_leader_count >= 3:
            continue

        merged_lines = _merge_cls_split_headings(stripped_lines)
        for line_txt, h_lvl in merged_lines:
            if line_txt == "Định nghĩa":
                in_glossary = True
                blocks.append(
                    LoadedBlock(
                        text="Định nghĩa",
                        page_number=pno + 1,
                        heading_level=1,
                        section_type="glossary",
                    )
                )
                continue
            if line_txt == "Có hiệu lực":
                in_glossary = False
                continue
            if in_glossary and h_lvl == 1 and re.match(r"^1\s+Bộ quy tắc", line_txt):
                in_glossary = False

            sec_type: SectionType = "glossary" if in_glossary else "body"
            if line_txt in {"Giới thiệu về Tài liệu này", "Tổng quát"} and h_lvl is None:
                h_lvl = 1
            blocks.append(
                LoadedBlock(
                    text=line_txt,
                    page_number=pno + 1,
                    heading_level=h_lvl,
                    section_type=sec_type,
                )
            )

    pdf_title = normalize_source_text(pdf.metadata.get("title") or "").strip()
    doc_title = (
        f"Bộ Quy tắc Chuẩn mực Lãnh đạo của Nike ({pdf_title})"
        if pdf_title
        else "Bộ Quy tắc Chuẩn mực Lãnh đạo của Nike"
    )
    metadata = DocumentMetadata(
        doc_id=path.stem,
        filename=path.name,
        doc_title=doc_title,
        doc_nature=doc_nature,
        ui_source_label=DOC_NATURE_LABELS[doc_nature],
        document_code="NIKE-CLS-2020",
        original_effective_date=None,
    )
    return LoadedDocument(metadata=metadata, blocks=blocks)


def _load_generic_pdf(
    path: Path,
    pdf: pymupdf.Document,
    doc_nature: DocNature,
) -> LoadedDocument:
    """Fallback generic PDF loader with scan guard."""
    _check_pdf_text_density([pdf[i] for i in range(len(pdf))], path.name)
    blocks: list[LoadedBlock] = []
    for pno in range(len(pdf)):
        raw = normalize_source_text(pdf[pno].get_text("text"))
        for line in raw.splitlines():
            ls = line.strip()
            if ls:
                blocks.append(LoadedBlock(text=ls, page_number=pno + 1))
    metadata = DocumentMetadata(
        doc_id=path.stem,
        filename=path.name,
        doc_title=path.stem.replace("_", " ").title(),
        doc_nature=doc_nature,
        ui_source_label=DOC_NATURE_LABELS[doc_nature],
    )
    return LoadedDocument(metadata=metadata, blocks=blocks)


def _clean_art_prefix(text: str) -> str:
    """Normalize '[ART-009] Điều 9...' markers while keeping 'Điều 9...' clean."""
    cleaned = re.sub(r"^\[ART-\d+\](?:[–-]\[ART-\d+\])?\s*", "", text)
    return cleaned.strip()


def _load_docx(path: Path, doc_nature: DocNature | None = None) -> LoadedDocument:
    """Load DOCX documents in XML body order (paragraphs + tables), strictly page_number=None."""
    doc = docx.Document(path)
    blocks: list[LoadedBlock] = []

    extracted_code: str | None = None
    extracted_eff_date: str | None = None
    inferred_nature: DocNature = doc_nature or "synthetic_policy"

    skip_index_table = False
    stop_at_appendix = False
    first_heading_seen: str | None = None

    for child in doc.element.body.iterchildren():
        if child.tag.endswith("p"):
            para = Paragraph(child, doc)
            txt = normalize_source_text(para.text).strip()
            if not txt:
                continue
            style_name = (para.style.name or "").lower()
            if style_name.startswith("toc") or txt.upper() == "MỤC LỤC":
                continue

            if txt.startswith("[APP]"):
                stop_at_appendix = True
                break
            if txt.startswith("[META-000]"):
                skip_index_table = False
                continue
            if txt.startswith("[INDEX-001]"):
                skip_index_table = True
                continue
            if txt.startswith("Lưu ý: Tài liệu này là bản chuyển thể"):
                continue

            skip_index_table = False

            if first_heading_seen is None and "heading 1" in style_name:
                first_heading_seen = txt

            # Map heading levels
            h_lvl: int | None = None
            if "heading 1" in style_name:
                h_lvl = 1
            elif "heading 2" in style_name:
                h_lvl = 2
            elif "heading 3" in style_name:
                h_lvl = 3

            # Strip structural tags like [CH-01], [SEC-03.1], [ART-001] cleanly
            art_range_m = re.match(r"^\[ART-(\d+)\]\s*[–-]\s*\[ART-(\d+)\]\s*(.+)$", txt)
            if art_range_m and not art_range_m.group(3).startswith("Điều"):
                a1, a2 = int(art_range_m.group(1)), int(art_range_m.group(2))
                clean_txt = f"Điều {a1}-{a2}. {art_range_m.group(3)}"
            else:
                clean_txt = re.sub(r"^\[(?:CH|SEC|ART)-\d+(?:\.\d+)?(?:[A-Z])?\]\s*", "", txt)
                clean_txt = _clean_art_prefix(clean_txt)
            clean_txt = re.sub(r"\s*\[FORMULA-\d+\]", "", clean_txt)

            if re.match(r"^CHƯƠNG\s+[IVXLCDM0-9]+", clean_txt, re.IGNORECASE):
                h_lvl = 1
            elif re.match(r"^Mục\s+\d+\.", clean_txt):
                h_lvl = 2
            elif re.match(r"^Điều\s+\d+", clean_txt):
                h_lvl = 3

            if h_lvl == 1 and "NGHỊ ĐỊNH 145/2020/NĐ-CP" in clean_txt:
                inferred_nature = doc_nature or "legal_decree"
                continue

            blocks.append(LoadedBlock(text=clean_txt, page_number=None, heading_level=h_lvl))

        elif child.tag.endswith("tbl"):
            if stop_at_appendix:
                break
            table = Table(child, doc)
            rows_text = [
                [normalize_source_text(cell.text).strip().replace("\n", " ") for cell in row.cells]
                for row in table.rows
            ]
            if not rows_text:
                continue

            flat_table_str = " ".join(" ".join(r) for r in rows_text)
            # Parse key-value metadata tables in ND145 docx or Sổ tay nhà thầu docx
            if any("Số hiệu" in r[0] or "Mã tài liệu" in r[0] for r in rows_text if r):
                for r in rows_text:
                    for ci in range(0, len(r) - 1, 2):
                        k, v = r[ci].strip(), r[ci + 1].strip()
                        if k in ("Số hiệu", "Mã tài liệu"):
                            extracted_code = v
                        elif k in ("Ngày hiệu lực", "Ngày có hiệu lực"):
                            extracted_eff_date = _parse_vn_date(v)
                        elif k == "Nhãn hệ thống" and v in DOC_NATURE_LABELS:
                            inferred_nature = doc_nature or v  # type: ignore[assignment]
                continue

            if skip_index_table or ("Chủ đề / Điều khoản" in flat_table_str):
                skip_index_table = False
                continue

            # Check if table encodes distinct legal articles (e.g. Điều 9-11, Điều 31-36 in ND145)
            header = rows_text[0]
            data_rows = rows_text[1:] if len(rows_text) > 1 else rows_text
            if any(re.search(r"\[ART-\d+\]|^Điều\s+\d+", r[0]) for r in data_rows if r):
                for r in data_rows:
                    art_col = _clean_art_prefix(r[0])
                    rest = " — ".join(
                        f"{header[idx]}: {r[idx]}" if idx < len(header) and header[idx] else r[idx]
                        for idx in range(1, len(r))
                        if r[idx]
                    )
                    art_line = f"{art_col}. {rest}" if rest else art_col
                    blocks.append(
                        LoadedBlock(
                            text=art_line,
                            page_number=None,
                            heading_level=3,
                            is_table=True,
                        )
                    )
            else:
                table_lines = [" | ".join(r) for r in rows_text if any(c for c in r)]
                if table_lines:
                    blocks.append(
                        LoadedBlock(
                            text="\n".join(table_lines),
                            page_number=None,
                            is_table=True,
                        )
                    )

    if extracted_code == "145/2020/NĐ-CP":
        doc_title = (
            "Nghị định số 145/2020/NĐ-CP quy định chi tiết và hướng dẫn thi hành một số điều "
            "của Bộ luật Lao động về điều kiện lao động và quan hệ lao động"
        )
    elif extracted_code == "CREMA-HSE-NT-2026":
        doc_title = "Sổ tay An toàn và Quản lý Nhà thầu tại Khuôn viên Nhà máy iLD"
    else:
        doc_title = first_heading_seen or path.stem.replace("_", " ").title()

    metadata = DocumentMetadata(
        doc_id=path.stem,
        filename=path.name,
        doc_title=doc_title,
        doc_nature=inferred_nature,
        ui_source_label=DOC_NATURE_LABELS[inferred_nature],
        document_code=extracted_code,
        original_effective_date=extracted_eff_date,
    )
    return LoadedDocument(metadata=metadata, blocks=blocks)


def _load_plain_text(path: Path, doc_nature: DocNature | None = None) -> LoadedDocument:
    """Load UTF-8 TXT/MD documents cleanly with page_number=None."""
    raw = normalize_source_text(path.read_text(encoding="utf-8"))
    blocks: list[LoadedBlock] = []
    for para in re.split(r"\n\s*\n", raw):
        p = para.strip()
        if not p:
            continue
        h_lvl: int | None = None
        if p.startswith("### "):
            h_lvl, p = 3, p[4:].strip()
        elif p.startswith("## "):
            h_lvl, p = 2, p[3:].strip()
        elif p.startswith("# "):
            h_lvl, p = 1, p[2:].strip()
        blocks.append(LoadedBlock(text=p, page_number=None, heading_level=h_lvl))

    nature: DocNature = doc_nature or "synthetic_policy"
    metadata = DocumentMetadata(
        doc_id=path.stem,
        filename=path.name,
        doc_title=path.stem.replace("_", " ").title(),
        doc_nature=nature,
        ui_source_label=DOC_NATURE_LABELS[nature],
    )
    return LoadedDocument(metadata=metadata, blocks=blocks)


def load_document(path: Path, doc_nature: DocNature | None = None) -> LoadedDocument:
    """Dispatch document loading by file extension and detected structure."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with pymupdf.open(path) as pdf:
            sample_head = normalize_source_text(
                "\n".join(pdf[i].get_text("text") for i in range(min(5, len(pdf))))
            )
            has_vbhn_marker = any(
                "XÁC THỰC VĂN BẢN HỢP NHẤT" in normalize_source_text(pdf[i].get_text("text"))
                for i in range(len(pdf))
            )
            if has_vbhn_marker:
                return _load_vbhn_pdf(path, pdf, doc_nature or "consolidated_law")
            if "BỘ QUY TẮC CHUẨN MỰC LÃNH ĐẠO CỦA NIKE" in sample_head or "Nike" in sample_head:
                return _load_enterprise_cls_pdf(path, pdf, doc_nature or "enterprise_reference")
            return _load_generic_pdf(path, pdf, doc_nature or "legal_decree")
    if suffix == ".docx":
        return _load_docx(path, doc_nature=doc_nature)
    if suffix in {".txt", ".md"}:
        return _load_plain_text(path, doc_nature=doc_nature)
    raise ValueError(f"Unsupported file format: {path.suffix}")
