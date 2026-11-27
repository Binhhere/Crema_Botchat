"""Pydantic v2 data models and exceptions for Crema HR & HSE Policy RAG."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DocNature = Literal[
    "consolidated_law",
    "legal_decree",
    "enterprise_reference",
    "synthetic_policy",
]

SectionType = Literal["body", "appendix", "glossary"]

RewriteReason = Literal["standalone_clear", "unaccented_vietnamese", "contextual_followup"]

GateReason = Literal[
    "cosine_pass",
    "lexical_code_rescue",
    "bm25_strong_rescue",
    "below_threshold",
    "empty_index",
]


class ScannedPDFError(ValueError):
    """Raised when a PDF lacks a usable text layer (< 30 chars/page)."""


class LoadedBlock(BaseModel):
    """Normalized structural block extracted from a PDF, DOCX, TXT, or MD file."""

    text: str
    page_number: int | None = None
    heading_level: int | None = None
    is_table: bool = False
    footnote_ids: list[int] = Field(default_factory=list)
    section_type: SectionType = "body"


class DocumentMetadata(BaseModel):
    """Document-level provenance and version metadata extracted from source files."""

    doc_id: str
    filename: str
    doc_title: str
    doc_nature: DocNature
    ui_source_label: str
    document_code: str | None = None
    consolidated_code: str | None = None
    original_effective_date: str | None = None
    amended_by: list[str] = Field(default_factory=list)
    consolidated_date: str | None = None
    footnote_effective_dates: dict[int, str] = Field(default_factory=dict)


class LoadedDocument(BaseModel):
    """Container pairing extracted document metadata with ordered structural blocks."""

    metadata: DocumentMetadata
    blocks: list[LoadedBlock]


class Chunk(BaseModel):
    """Self-contained retrieval unit with provenance and legal effective-date metadata."""

    chunk_id: str
    doc_id: str
    doc_title: str
    doc_nature: DocNature
    ui_source_label: str
    section_code: str
    section_title: str
    section_type: SectionType = "body"
    page_number: int | None = None
    page_end: int | None = None
    original_effective_date: str | None = None
    consolidated_code: str | None = None
    consolidated_date: str | None = None
    amended_by: list[str] = Field(default_factory=list)
    clause_effective_date: str | None = None
    partial_expiry_warning: bool = False
    text: str


class RewriteResult(BaseModel):
    """Result from conditional 0ms/LLM query rewriting."""

    original_query: str
    rewritten_query: str
    was_rewritten: bool
    reason: RewriteReason
    api_calls: int = 0


class IndexBuildStats(BaseModel):
    """Statistics describing cache reuse vs. new embedding calls."""

    total_chunks: int
    cache_hits: int
    newly_embedded: int
    api_batches_called: int


class RetrievedCandidate(BaseModel):
    """Single retrieved chunk with dense, BM25, and RRF ranking signals."""

    chunk: Chunk
    dense_score: float
    dense_rank: int
    bm25_score: float
    bm25_rank: int
    rrf_score: float


class RetrievalResult(BaseModel):
    """Full Stage 2 retrieval output including pre-LLM confidence gate decision."""

    query_used: str
    rewrite_result: RewriteResult | None = None
    candidates: list[RetrievedCandidate]
    top1_cosine: float
    bm25_top1_score: float
    bm25_zscore: float
    passed_gate: bool
    gate_reason: GateReason
    api_calls: int = 0
