"""Hybrid retriever combining RETRIEVAL_QUERY dense cosine, unaccented bi-gram BM25, RRF, and lexical rescue gate."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np
from google.genai import types
from rank_bm25 import BM25Okapi

from src.config import TAU_COSINE_FLOOR, TAU_COSINE_REJECT
from src.indexer import VectorIndex, call_gemini_with_backoff, get_genai_client, l2_normalize
from src.models import Chunk, GateReason, RetrievalResult, RetrievedCandidate
from src.rewriter import rewrite_query

_ALNUM_TOKEN_RE = re.compile(r"[a-z0-9]+")
_EXPLICIT_ARTICLE_RE = re.compile(r"\bdieu\s+(\d+)\b")
_EXPLICIT_APPENDIX_RE = re.compile(r"\bphu\s+luc\s+([ivxlcdm]+)\b")
_EXPLICIT_DECIMAL_SEC_RE = re.compile(r"\b(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\b")
_EXPLICIT_DOC_CODE_RE = re.compile(
    r"\b(\d{2,3}/\d{4}(?:/[a-z0-9\-]+)?|\d{2}/vbhn(?:-bnv)?|crema-[a-z0-9\-]+)\b"
)
_EXPLICIT_HSE_CODE_RE = re.compile(r"\b(loto|ptw|ppe)\b")
_ARTICLE_RANGE_RE = re.compile(r"^Điều\s+(\d+)(?:\s*[–-]\s*(\d+))?$")


def strip_vietnamese_accents(text: str) -> str:
    """Convert Vietnamese text into lowercase unaccented ASCII-compatible string."""
    nfc = unicodedata.normalize("NFC", text).replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", nfc)
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return stripped.lower()


def tokenize_unaccented_bigrams(text: str) -> list[str]:
    """Tokenize strictly on unaccented syllable unigrams + adjacent bigrams for BM25."""
    unigrams = _ALNUM_TOKEN_RE.findall(strip_vietnamese_accents(text))
    bigrams = [f"{unigrams[i]}_{unigrams[i + 1]}" for i in range(len(unigrams) - 1)]
    return unigrams + bigrams


def _chunk_matches_article(section_code: str, article_num: int) -> bool:
    m = _ARTICLE_RANGE_RE.match(section_code.strip())
    if not m:
        return False
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else start
    return start <= article_num <= end


def _check_lexical_code_rescue(
    queries: list[str],
    candidates: list[RetrievedCandidate],
    top1_cosine: float,
    tau_floor: float,
) -> bool:
    """Rescue exact legal article, decimal section, or document/HSE code matches in Top-k."""
    if not candidates:
        return False

    unaccented_queries = [strip_vietnamese_accents(q) for q in queries if q.strip()]

    # 1. Exact article ("Điều N") or "Phụ lục X" or decimal section ("13.2.2")
    if top1_cosine >= tau_floor * 0.85:
        for uq in unaccented_queries:
            for art_m in _EXPLICIT_ARTICLE_RE.finditer(uq):
                target_num = int(art_m.group(1))
                if any(_chunk_matches_article(c.chunk.section_code, target_num) for c in candidates):
                    return True

            for app_m in _EXPLICIT_APPENDIX_RE.finditer(uq):
                target_app = f"phu luc {app_m.group(1)}"
                if any(strip_vietnamese_accents(c.chunk.section_code) == target_app for c in candidates):
                    return True

            for dec_m in _EXPLICIT_DECIMAL_SEC_RE.finditer(uq):
                target_sec = dec_m.group(1)
                if any(c.chunk.section_code == target_sec for c in candidates):
                    return True

    # 2. Explicit document code or HSE acronym, paired with tau_floor guard
    if top1_cosine >= tau_floor:
        for uq in unaccented_queries:
            for code_m in _EXPLICIT_DOC_CODE_RE.finditer(uq):
                code = code_m.group(1)
                for cand in candidates:
                    ch: Chunk = cand.chunk
                    haystack = strip_vietnamese_accents(
                        f"{ch.doc_title} {ch.consolidated_code or ''} "
                        f"{' '.join(ch.amended_by)} {ch.text[:350]}"
                    )
                    if code in haystack:
                        return True
            for hse_m in _EXPLICIT_HSE_CODE_RE.finditer(uq):
                hse_tok = hse_m.group(1)
                for cand in candidates:
                    if hse_tok in strip_vietnamese_accents(cand.chunk.text):
                        return True

    return False


class HybridRetriever:
    """Hybrid Dense Cosine + Unaccented Bi-gram BM25 retriever with RRF and lexical gate."""

    def __init__(self, vector_index: VectorIndex, rrf_k: int = 60) -> None:
        self.index = vector_index
        self.rrf_k = rrf_k
        self._chunk_tokens: list[list[str]] = [
            tokenize_unaccented_bigrams(c.text) or ["empty"]
            for c in self.index.chunks
        ]
        self.bm25: BM25Okapi | None = (
            BM25Okapi(self._chunk_tokens) if self._chunk_tokens else None
        )

    def embed_query(self, query: str, client: Any | None = None) -> np.ndarray:
        """Embed a retrieval query using task_type=RETRIEVAL_QUERY and L2 normalization."""
        resolved_client = client if client is not None else get_genai_client()
        cfg = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self.index.output_dim,
        )
        resp = call_gemini_with_backoff(
            lambda: resolved_client.models.embed_content(
                model=self.index.embedding_model,
                contents=query,
                config=cfg,
            )
        )
        vec = np.asarray(resp.embeddings[0].values, dtype=np.float32)
        return l2_normalize(vec)

    def retrieve(
        self,
        query: str,
        original_query: str | None = None,
        top_k: int = 5,
        client: Any | None = None,
        tau_reject: float = TAU_COSINE_REJECT,
        tau_floor: float = TAU_COSINE_FLOOR,
    ) -> RetrievalResult:
        """Retrieve Top-k candidates via RRF and evaluate the pre-LLM confidence gate."""
        if not self.index.chunks or self.bm25 is None:
            return RetrievalResult(
                query_used=query,
                candidates=[],
                top1_cosine=0.0,
                bm25_top1_score=0.0,
                bm25_zscore=0.0,
                passed_gate=False,
                gate_reason="empty_index",
                api_calls=0,
            )

        q_vec = self.embed_query(query, client=client)
        dense_scores = self.index.embeddings @ q_vec
        q_tokens = tokenize_unaccented_bigrams(query)
        bm25_scores = (
            np.asarray(self.bm25.get_scores(q_tokens), dtype=np.float64)
            if q_tokens
            else np.zeros(len(self.index.chunks), dtype=np.float64)
        )

        dense_order = np.argsort(-dense_scores, kind="mergesort")
        bm25_order = np.argsort(-bm25_scores, kind="mergesort")

        dense_ranks = np.empty(len(self.index.chunks), dtype=np.int64)
        bm25_ranks = np.empty(len(self.index.chunks), dtype=np.int64)
        dense_ranks[dense_order] = np.arange(1, len(self.index.chunks) + 1)
        bm25_ranks[bm25_order] = np.arange(1, len(self.index.chunks) + 1)

        rrf_scores = np.zeros(len(self.index.chunks), dtype=np.float64)
        for i in range(len(self.index.chunks)):
            score = 1.0 / (self.rrf_k + int(dense_ranks[i]))
            if bm25_scores[i] > 0.0:
                score += 1.0 / (self.rrf_k + int(bm25_ranks[i]))
            rrf_scores[i] = score

        # Sort deterministically by RRF descending, then dense cosine descending
        sorted_indices = sorted(
            range(len(self.index.chunks)),
            key=lambda idx: (float(rrf_scores[idx]), float(dense_scores[idx])),
            reverse=True,
        )[:top_k]

        candidates = [
            RetrievedCandidate(
                chunk=self.index.chunks[idx],
                dense_score=float(dense_scores[idx]),
                dense_rank=int(dense_ranks[idx]),
                bm25_score=float(bm25_scores[idx]),
                bm25_rank=int(bm25_ranks[idx]),
                rrf_score=float(rrf_scores[idx]),
            )
            for idx in sorted_indices
        ]

        top1_cosine = float(np.max(dense_scores))
        bm25_top1 = float(np.max(bm25_scores))
        bm25_std = float(np.std(bm25_scores))
        bm25_zscore = (
            float((bm25_top1 - float(np.mean(bm25_scores))) / bm25_std)
            if bm25_std > 1e-9
            else 0.0
        )

        raw_queries = [original_query or query, query]
        passed_gate = False
        gate_reason: GateReason = "below_threshold"

        if top1_cosine >= tau_reject:
            passed_gate = True
            gate_reason = "cosine_pass"
        elif _check_lexical_code_rescue(raw_queries, candidates, top1_cosine, tau_floor):
            passed_gate = True
            gate_reason = "lexical_code_rescue"
        elif top1_cosine >= tau_floor and bm25_top1 >= 28.0 and bm25_zscore >= 5.0:
            passed_gate = True
            gate_reason = "bm25_strong_rescue"

        return RetrievalResult(
            query_used=query,
            candidates=candidates,
            top1_cosine=top1_cosine,
            bm25_top1_score=bm25_top1,
            bm25_zscore=bm25_zscore,
            passed_gate=passed_gate,
            gate_reason=gate_reason,
            api_calls=1,
        )

    def search(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        top_k: int = 5,
        use_rewriter: bool = True,
        client: Any | None = None,
        tau_reject: float = TAU_COSINE_REJECT,
        tau_floor: float = TAU_COSINE_FLOOR,
    ) -> RetrievalResult:
        """Run optional 0ms/LLM query rewriting followed by hybrid retrieval and gate check."""
        rewrite_res = (
            rewrite_query(query=query, history=history, client=client)
            if use_rewriter
            else None
        )
        effective_query = rewrite_res.rewritten_query if rewrite_res else query.strip()

        result = self.retrieve(
            query=effective_query,
            original_query=query,
            top_k=top_k,
            client=client,
            tau_reject=tau_reject,
            tau_floor=tau_floor,
        )
        result.rewrite_result = rewrite_res
        result.api_calls += rewrite_res.api_calls if rewrite_res else 0
        return result
