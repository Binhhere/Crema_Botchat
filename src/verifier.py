"""Deterministic post-generation verifier (chunk_id membership, >=8-word quote substring, numerical grounding)."""

from __future__ import annotations

import re
import unicodedata

from src.models import (
    Chunk,
    LLMGenerationOutput,
    RetrievedCandidate,
    VerificationReport,
    VerifiedCitation,
)

MIN_QUOTE_WORDS = 8

_UPPER_HYPHEN_BREAK_RE = re.compile(r"([A-ZĐƯ0-9])-\s*\n\s*([A-ZĐƯ0-9])")
_LOWER_HYPHEN_BREAK_RE = re.compile(r"([a-zà-ỹ])-\s*\n\s*([a-zà-ỹ])")
_DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_TOKEN_RE = re.compile(r"\w+(?:[.,/-]\w+)*", flags=re.UNICODE)
_NUMBER_TOKEN_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)")
_THOUSAND_GROUP_RE = re.compile(r"^[1-9]\d{0,2}(?:[.,]\d{3})+$")
_MULTI_DOT_SECTION_RE = re.compile(r"^\d+(?:\.\d+){2,}$")
_SCALED_UNIT_RE = re.compile(
    r"(?<!\w)(\d+(?:[.,]\d+)?)\s*(nghìn|ngàn|triệu|tỷ)\b",
    flags=re.IGNORECASE,
)
_UNIT_MULTIPLIERS: dict[str, int] = {
    "nghìn": 1_000,
    "ngàn": 1_000,
    "triệu": 1_000_000,
    "tỷ": 1_000_000_000,
}


def _strip_quote_affixes(text: str) -> str:
    cleaned = text.strip()
    while True:
        prev = cleaned
        cleaned = cleaned.strip().strip('"\'“”‘’«»').strip()
        if cleaned.startswith("..."):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("..."):
            cleaned = cleaned[:-3].strip()
        if cleaned.startswith("…"):
            cleaned = cleaned[1:].strip()
        if cleaned.endswith("…"):
            cleaned = cleaned[:-1].strip()
        if cleaned == prev:
            break
    return cleaned


def normalize_for_quote_match(text: str, join_hyphen_tight: bool = False) -> str:
    """Normalize NFC, legacy glyphs, dashes, PDF line-break hyphens, decimals, and spaces."""
    s = (
        text.replace("\u01a2", "Ư")
        .replace("\u01a3", "ư")
        .replace("\u00ad", "")
        .replace("\u200b", "")
        .replace("\r", "")
    )
    s = unicodedata.normalize("NFC", s)
    s = (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("–", "-")
        .replace("—", "-")
    )
    s = _UPPER_HYPHEN_BREAK_RE.sub(r"\1-\2", s)
    s = _LOWER_HYPHEN_BREAK_RE.sub(r"\1\2" if join_hyphen_tight else r"\1 \2", s)
    s = _DECIMAL_COMMA_RE.sub(".", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def count_quote_words(quote: str) -> int:
    """Count word/compound tokens in quote after stripping surrounding quotes/ellipsis."""
    cleaned = unicodedata.normalize("NFC", _strip_quote_affixes(quote))
    if not cleaned:
        return 0
    return len(_WORD_TOKEN_RE.findall(cleaned))


def _canonicalize_numeric_token(tok: str) -> set[str]:
    """Canonicalize integer, decimal (1,5 == 1.5), thousand-grouped, and section codes."""
    if _THOUSAND_GROUP_RE.match(tok):
        digits = tok.replace(".", "").replace(",", "")
        return {str(int(digits))}

    if _MULTI_DOT_SECTION_RE.match(tok):
        parts = [str(int(p)) for p in tok.split(".")]
        return {tok, *parts}

    if "," in tok or "." in tok:
        normalized = tok.replace(",", ".")
        if normalized.count(".") == 1:
            int_part, frac_part = normalized.split(".")
            int_clean = str(int(int_part))
            frac_clean = frac_part.rstrip("0")
            return {f"{int_clean}.{frac_clean}" if frac_clean else int_clean}
        parts = [str(int(p)) for p in normalized.split(".")]
        return {normalized, *parts}

    return {str(int(tok))}


def extract_canonical_numbers(text: str, include_unscaled: bool = False) -> set[str]:
    """Extract canonical numeric representations (incl. Vietnamese unit scales) for grounding."""
    nfc = unicodedata.normalize("NFC", text)
    results: set[str] = set()

    def _replace_scaled(match: re.Match[str]) -> str:
        raw_num = match.group(1)
        unit = match.group(2).lower()
        mult = _UNIT_MULTIPLIERS[unit]
        if _THOUSAND_GROUP_RE.match(raw_num):
            val = float(raw_num.replace(".", "").replace(",", ""))
        else:
            val = float(raw_num.replace(",", "."))
        scaled_int = round(val * mult)
        results.add(str(scaled_int))
        if include_unscaled:
            results.update(_canonicalize_numeric_token(raw_num))
        return " "

    remainder = _SCALED_UNIT_RE.sub(_replace_scaled, nfc)
    for tok in _NUMBER_TOKEN_RE.findall(remainder):
        results.update(_canonicalize_numeric_token(tok))
    return results


def build_verified_citation(chunk: Chunk, cleaned_quote: str) -> VerifiedCitation:
    """Map a verified chunk deterministically into UI-honest citation metadata."""
    eff_date = chunk.clause_effective_date or chunk.original_effective_date or chunk.consolidated_date
    if chunk.page_number is not None:
        display_label = (
            f"{chunk.ui_source_label} {chunk.doc_title} — {chunk.section_code} "
            f"(Trang {chunk.page_number})"
        )
    else:
        display_label = f"{chunk.ui_source_label} {chunk.doc_title} — {chunk.section_code}"

    return VerifiedCitation(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        doc_title=chunk.doc_title,
        doc_nature=chunk.doc_nature,
        ui_source_label=chunk.ui_source_label,
        section_code=chunk.section_code,
        section_title=chunk.section_title,
        page_number=chunk.page_number,
        effective_date=eff_date,
        partial_expiry_warning=chunk.partial_expiry_warning,
        display_label=display_label,
        verbatim_quote=cleaned_quote,
    )


def _chunk_grounding_text(chunk: Chunk) -> str:
    parts = [
        chunk.text,
        chunk.doc_title,
        chunk.section_code,
        chunk.consolidated_code or "",
        chunk.original_effective_date or "",
        chunk.consolidated_date or "",
        chunk.clause_effective_date or "",
        " ".join(chunk.amended_by),
    ]
    if chunk.page_number is not None:
        parts.append(str(chunk.page_number))
    if chunk.page_end is not None:
        parts.append(str(chunk.page_end))
    return "\n".join(p for p in parts if p)


def verify_llm_output(
    llm_output: LLMGenerationOutput,
    top_candidates: list[RetrievedCandidate],
    original_query: str,
) -> tuple[VerificationReport, list[VerifiedCitation]]:
    """Verify chunk_id membership, >=8-word verbatim substring, and numerical grounding."""
    if not llm_output.citations:
        return (
            VerificationReport(is_valid=False, failure_reason="missing_citations"),
            [],
        )

    chunk_map: dict[str, Chunk] = {c.chunk.chunk_id: c.chunk for c in top_candidates}

    # Condition 1: Every cited chunk_id must belong to Top-k candidates provided
    invalid_ids = [item.chunk_id for item in llm_output.citations if item.chunk_id not in chunk_map]
    if invalid_ids:
        return (
            VerificationReport(
                is_valid=False,
                failure_reason="chunk_id_not_in_top5",
                invalid_chunk_ids=invalid_ids,
            ),
            [],
        )

    # Condition 2: Every quote must have >= 8 words and match as normalized substring
    verified_citations: list[VerifiedCitation] = []
    seen_citations: set[tuple[str, str]] = set()
    cited_chunks: list[Chunk] = []

    for item in llm_output.citations:
        stripped_quote = _strip_quote_affixes(item.verbatim_quote)
        if count_quote_words(stripped_quote) < MIN_QUOTE_WORDS:
            return (
                VerificationReport(is_valid=False, failure_reason="quote_too_short"),
                [],
            )

        chunk = chunk_map[item.chunk_id]
        q_norm_space = normalize_for_quote_match(stripped_quote, join_hyphen_tight=False)
        q_norm_tight = normalize_for_quote_match(stripped_quote, join_hyphen_tight=True)
        c_norm_space = normalize_for_quote_match(chunk.text, join_hyphen_tight=False)
        c_norm_tight = normalize_for_quote_match(chunk.text, join_hyphen_tight=True)

        matched = (
            bool(q_norm_space)
            and (
                q_norm_space in c_norm_space
                or q_norm_space in c_norm_tight
                or q_norm_tight in c_norm_space
                or q_norm_tight in c_norm_tight
            )
        )
        if not matched:
            return (
                VerificationReport(is_valid=False, failure_reason="quote_not_substring"),
                [],
            )

        dedup_key = (chunk.chunk_id, q_norm_space)
        if dedup_key not in seen_citations:
            seen_citations.add(dedup_key)
            cited_chunks.append(chunk)
            verified_citations.append(build_verified_citation(chunk, stripped_quote))

    # Condition 3: Numerical Grounding Check against cited chunks + original question
    answer_nums = extract_canonical_numbers(llm_output.answer_friendly, include_unscaled=False)
    if answer_nums:
        allowed_sources = [original_query, *(_chunk_grounding_text(ch) for ch in cited_chunks)]
        allowed_nums = extract_canonical_numbers("\n".join(allowed_sources), include_unscaled=True)
        ungrounded = sorted(answer_nums - allowed_nums)
        if ungrounded:
            return (
                VerificationReport(
                    is_valid=False,
                    failure_reason="ungrounded_numbers",
                    ungrounded_numbers=ungrounded,
                ),
                [],
            )

    return VerificationReport(is_valid=True), verified_citations
