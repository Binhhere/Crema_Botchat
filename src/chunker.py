"""Multi-strategy document chunker (Legal Điều/Khoản -> Hierarchical Headings -> Sliding Window)."""

from __future__ import annotations

import re

from src.models import Chunk, DocumentMetadata, LoadedBlock, LoadedDocument, SectionType

_CHAPTER_RE = re.compile(r"^Chương\s+[IVXLCDM0-9]+\b", re.IGNORECASE)
_SEC_MUC_RE = re.compile(r"^Mục\s+\d+\.", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"^Điều\s+(\d+)(?:\s*[–-]\s*(?:Điều\s*)?(\d+))?[\.\:\s]")
_APPENDIX_RE = re.compile(r"^PHỤ\s+LỤC\s+([IVXLCDM]+)\b")
_CLAUSE_SPLIT_RE = re.compile(r"^(?:\d+\.|[a-zđ]\))\s+")

_ROMAN_H1_RE = re.compile(r"^[IVXLCDM]+\.\s+\S")
_DECIMAL_H2_RE = re.compile(r"^(\d{1,2}\.\d{1,2})\.?(?:\s+|$)")
_DECIMAL_H3_RE = re.compile(r"^(\d{1,2}\.\d{1,2}\.\d{1,2})\.?(?:\s+|$)")
_CHAPTER_NUM_H1_RE = re.compile(r"^(\d{1,2})\s+\S")


def _word_count(text: str) -> int:
    return len(text.split())


def _is_legal_article_doc(blocks: list[LoadedBlock]) -> bool:
    article_hits = sum(1 for b in blocks if _ARTICLE_RE.match(b.text.strip()))
    return article_hits >= 3


def _is_hierarchical_doc(blocks: list[LoadedBlock]) -> bool:
    heading_hits = sum(
        1
        for b in blocks
        if b.heading_level is not None
        or _ROMAN_H1_RE.match(b.text.strip())
        or _DECIMAL_H2_RE.match(b.text.strip())
        or _DECIMAL_H3_RE.match(b.text.strip())
    )
    return heading_hits >= 3


def _build_chunk(
    meta: DocumentMetadata,
    idx: int,
    section_code: str,
    section_title: str,
    section_type: SectionType,
    body_blocks: list[LoadedBlock],
    body_text: str,
) -> Chunk:
    page_nums = [b.page_number for b in body_blocks if b.page_number is not None]
    page_start = min(page_nums) if page_nums else None
    page_end = max(page_nums) if page_nums else None

    fn_ids = [fid for b in body_blocks for fid in b.footnote_ids]
    clause_eff = meta.original_effective_date
    for fid in fn_ids:
        if fid in meta.footnote_effective_dates:
            clause_eff = meta.footnote_effective_dates[fid]
            break

    partial_warn = (
        meta.document_code == "145/2020/NĐ-CP"
        and section_code in {"Điều 4", "Điều 31"}
    )

    header_prefix = f"[{meta.doc_title} | {section_title}]"
    full_text = f"{header_prefix}\n{body_text.strip()}".strip()

    return Chunk(
        chunk_id=f"{meta.doc_id}__c{idx:04d}",
        doc_id=meta.doc_id,
        doc_title=meta.doc_title,
        doc_nature=meta.doc_nature,
        ui_source_label=meta.ui_source_label,
        section_code=section_code,
        section_title=section_title,
        section_type=section_type,
        page_number=page_start,
        page_end=page_end,
        original_effective_date=meta.original_effective_date,
        consolidated_code=meta.consolidated_code,
        consolidated_date=meta.consolidated_date,
        amended_by=meta.amended_by,
        clause_effective_date=clause_eff,
        partial_expiry_warning=partial_warn,
        text=full_text,
    )


def _split_blocks_by_size(
    blocks: list[LoadedBlock],
    max_words: int,
    clause_boundary: bool = False,
) -> list[tuple[list[LoadedBlock], str]]:
    """Split a long section's blocks into <= max_words sub-groups preserving page metadata."""
    groups: list[tuple[list[LoadedBlock], str]] = []
    cur_blocks: list[LoadedBlock] = []
    cur_words = 0

    for blk in blocks:
        blk_words = _word_count(blk.text)
        is_split_point = clause_boundary and bool(_CLAUSE_SPLIT_RE.match(blk.text.strip()))
        should_flush = cur_blocks and (
            cur_words + blk_words > max_words
            and (not clause_boundary or is_split_point or cur_words >= max_words // 2)
        )
        if should_flush:
            if clause_boundary and not is_split_point:
                split_idx: int | None = None
                for idx in range(len(cur_blocks) - 1, 0, -1):
                    if _CLAUSE_SPLIT_RE.match(cur_blocks[idx].text.strip()):
                        prefix_words = sum(_word_count(b.text) for b in cur_blocks[:idx])
                        if prefix_words >= max_words // 3:
                            split_idx = idx
                            break
                if split_idx is not None:
                    flushed = cur_blocks[:split_idx]
                    groups.append((flushed, "\n".join(b.text for b in flushed)))
                    cur_blocks = cur_blocks[split_idx:]
                    cur_words = sum(_word_count(b.text) for b in cur_blocks)
                    cur_blocks.append(blk)
                    cur_words += blk_words
                    continue

            joined = "\n".join(b.text for b in cur_blocks)
            groups.append((cur_blocks, joined))
            cur_blocks = []
            cur_words = 0

        cur_blocks.append(blk)
        cur_words += blk_words

    if cur_blocks:
        groups.append((cur_blocks, "\n".join(b.text for b in cur_blocks)))
    return groups


def _chunk_legal_articles(loaded: LoadedDocument, max_words: int) -> list[Chunk]:
    """Strategy 1: Group blocks by Chương / Mục / Điều / Phụ lục."""
    meta = loaded.metadata
    chunks: list[Chunk] = []

    current_chapter = ""
    current_muc = ""
    current_code: str | None = None
    current_title = ""
    current_type: SectionType = "body"
    current_blocks: list[LoadedBlock] = []
    pending_chapter_lines: list[str] = []

    def flush() -> None:
        nonlocal current_blocks
        if current_code is None or not current_blocks:
            current_blocks = []
            return
        meaningful = [b for b in current_blocks if b.text.strip()]
        if not meaningful:
            current_blocks = []
            return
        sub_groups = _split_blocks_by_size(meaningful, max_words=max_words, clause_boundary=True)
        for sub_blks, sub_txt in sub_groups:
            if len(sub_txt.strip()) < 15:
                continue
            chunks.append(
                _build_chunk(
                    meta=meta,
                    idx=len(chunks) + 1,
                    section_code=current_code,
                    section_title=current_title,
                    section_type=current_type,
                    body_blocks=sub_blks,
                    body_text=sub_txt,
                )
            )
        current_blocks = []

    for blk in loaded.blocks:
        txt = blk.text.strip()
        if not txt:
            continue

        if _CHAPTER_RE.match(txt):
            flush()
            current_code = None
            pending_chapter_lines = [txt]
            current_chapter = txt
            current_muc = ""
            continue

        # Capture uppercase chapter subtitle line right after "Chương I"
        if (
            pending_chapter_lines
            and len(pending_chapter_lines) == 1
            and not _ARTICLE_RE.match(txt)
            and not _SEC_MUC_RE.match(txt)
            and txt == txt.upper()
            and len(txt) <= 120
        ):
            current_chapter = f"{pending_chapter_lines[0]}: {txt}"
            pending_chapter_lines = []
            continue
        pending_chapter_lines = []

        if _SEC_MUC_RE.match(txt):
            flush()
            current_code = None
            current_muc = txt
            continue

        app_m = _APPENDIX_RE.match(txt)
        if app_m:
            flush()
            current_code = f"Phụ lục {app_m.group(1)}"
            current_title = txt
            current_type = "appendix"
            current_blocks = [blk]
            continue

        art_m = _ARTICLE_RE.match(txt)
        if art_m:
            flush()
            art_start = art_m.group(1)
            art_end = art_m.group(2)
            current_code = f"Điều {art_start}-{art_end}" if art_end else f"Điều {art_start}"
            crumbs = [c for c in (current_chapter, current_muc, txt.split("\n")[0][:120]) if c]
            current_title = " > ".join(crumbs)
            current_type = blk.section_type
            current_blocks = [blk]
            continue

        if current_code is not None:
            current_blocks.append(blk)

    flush()
    return chunks


def _extract_section_code(heading_text: str) -> str:
    m3 = _DECIMAL_H3_RE.match(heading_text)
    if m3:
        return m3.group(1)
    m2 = _DECIMAL_H2_RE.match(heading_text)
    if m2:
        return m2.group(1)
    m1 = _CHAPTER_NUM_H1_RE.match(heading_text)
    if m1:
        return f"Mục {m1.group(1)}"
    if _ROMAN_H1_RE.match(heading_text):
        return heading_text.split(".")[0].strip()
    return heading_text[:40]


def _chunk_hierarchical(loaded: LoadedDocument, max_words: int) -> list[Chunk]:
    """Strategy 2: Hierarchical Roman / Decimal / DOCX Heading chunker (zero empty chunks)."""
    meta = loaded.metadata
    chunks: list[Chunk] = []

    h1 = ""
    h2 = ""
    h3 = ""
    cur_type: SectionType = "body"
    cur_blocks: list[LoadedBlock] = []

    def current_breadcrumbs() -> tuple[str, str]:
        active = [x for x in (h1, h2, h3) if x]
        if not active:
            return "Tổng quan", "Tổng quan"
        code = _extract_section_code(active[-1])
        return code, " > ".join(active)

    def flush() -> None:
        nonlocal cur_blocks
        meaningful = [b for b in cur_blocks if b.text.strip()]
        if not meaningful:
            cur_blocks = []
            return
        sec_code, sec_title = current_breadcrumbs()
        sub_groups = _split_blocks_by_size(meaningful, max_words=max_words, clause_boundary=False)
        for sub_blks, sub_txt in sub_groups:
            if len(sub_txt.strip()) < 20:
                continue
            chunks.append(
                _build_chunk(
                    meta=meta,
                    idx=len(chunks) + 1,
                    section_code=sec_code,
                    section_title=sec_title,
                    section_type=cur_type,
                    body_blocks=sub_blks,
                    body_text=sub_txt,
                )
            )
        cur_blocks = []

    for blk in loaded.blocks:
        txt = blk.text.strip()
        if not txt:
            continue

        lvl = blk.heading_level
        if lvl is None:
            if _DECIMAL_H3_RE.match(txt):
                lvl = 3
            elif _DECIMAL_H2_RE.match(txt):
                lvl = 2
            elif _ROMAN_H1_RE.match(txt):
                lvl = 1

        if lvl == 1:
            flush()
            h1, h2, h3 = txt, "", ""
            cur_type = blk.section_type
            continue
        if lvl == 2:
            flush()
            h2, h3 = txt, ""
            cur_type = blk.section_type
            continue
        if lvl == 3:
            flush()
            h3 = txt
            cur_type = blk.section_type
            continue

        cur_blocks.append(blk)
        cur_type = blk.section_type

    flush()
    return chunks


def _chunk_sliding_window(
    loaded: LoadedDocument,
    max_words: int = 400,
    overlap_words: int = 80,
) -> list[Chunk]:
    """Strategy 3: Sliding-window paragraph chunker with overlap for unstructured prose."""
    meta = loaded.metadata
    chunks: list[Chunk] = []

    cur_blocks: list[LoadedBlock] = []
    cur_words = 0

    def emit(blks: list[LoadedBlock]) -> None:
        txt = "\n".join(b.text for b in blks).strip()
        if len(txt) < 20:
            return
        sec_idx = len(chunks) + 1
        chunks.append(
            _build_chunk(
                meta=meta,
                idx=sec_idx,
                section_code=f"Đoạn {sec_idx}",
                section_title=f"Nội dung {sec_idx}",
                section_type=blks[0].section_type,
                body_blocks=blks,
                body_text=txt,
            )
        )

    for blk in loaded.blocks:
        txt = blk.text.strip()
        if not txt:
            continue
        w_len = _word_count(txt)
        if cur_blocks and cur_words + w_len > max_words:
            emit(cur_blocks)
            # Keep trailing blocks up to overlap_words
            overlap_blks: list[LoadedBlock] = []
            ov_words = 0
            for prev in reversed(cur_blocks):
                pw = _word_count(prev.text)
                if overlap_blks and ov_words + pw > overlap_words:
                    break
                overlap_blks.insert(0, prev)
                ov_words += pw
            cur_blocks = overlap_blks
            cur_words = ov_words

        cur_blocks.append(blk)
        cur_words += w_len

    if cur_blocks:
        emit(cur_blocks)
    return chunks


def chunk_document(
    loaded: LoadedDocument,
    max_words: int = 450,
    overlap_words: int = 80,
) -> list[Chunk]:
    """Chunk a LoadedDocument automatically using the 3-tier multi-strategy pipeline."""
    if not loaded.blocks:
        return []
    if _is_legal_article_doc(loaded.blocks):
        chunks = _chunk_legal_articles(loaded, max_words=max_words)
    elif _is_hierarchical_doc(loaded.blocks):
        chunks = _chunk_hierarchical(loaded, max_words=max_words)
    else:
        chunks = _chunk_sliding_window(loaded, max_words=min(max_words, 400), overlap_words=overlap_words)

    return [c for c in chunks if c.text.strip() and len(c.text.strip()) >= 20]
