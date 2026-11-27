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
