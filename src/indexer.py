"""Embedding generation, retry/backoff, and immutable SHA-256 disk caching."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.chunker import chunk_document
from src.config import (
    CHUNKER_VERSION,
    DATA_CACHE_DIR,
    DATA_RAW_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_OUTPUT_DIM,
    get_gemini_api_key,
)
from src.loader import load_document
from src.models import Chunk, IndexBuildStats

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
_RETRY_IN_RE = re.compile(r"retry\s+(?:in|after)\s+([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)
_META_FILENAME = "embeddings_meta.json"
_VECTORS_FILENAME = "embeddings_vectors.npy"


def compute_chunk_cache_key(
    raw_chunk_text: str,
    embedding_model: str = EMBEDDING_MODEL,
    output_dim: int = EMBEDDING_OUTPUT_DIM,
    chunker_version: str = CHUNKER_VERSION,
) -> str:
    """Compute immutable SHA-256 cache key tied to text, model, dim, and chunker version."""
    payload = f"{raw_chunk_text}|{embedding_model}|{output_dim}|{chunker_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize 1D or 2D embedding array into float32."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.size == 0:
        return arr
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return (arr / max(norm, 1e-12)).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norms, 1e-12)).astype(np.float32)


def get_genai_client(api_key: str | None = None) -> genai.Client:
    """Create a Google GenAI client using the configured GEMINI_API_KEY."""
    resolved = (api_key or get_gemini_api_key() or "").strip()
    if not resolved:
        raise RuntimeError("GEMINI_API_KEY is not configured in environment or .env.")
    return genai.Client(api_key=resolved)


def _extract_retry_delay_seconds(exc: Exception) -> float | None:
    match = _RETRY_IN_RE.search(str(exc))
    if match:
        return min(float(match.group(1)) + 2.0, 90.0)
    return None


def call_gemini_with_backoff[T](
    fn: Callable[[], T],
    max_retries: int = 6,
    initial_delay: float = 4.0,
) -> T:
    """Execute a Gemini SDK call with exponential backoff on 429 and transient errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (genai_errors.APIError, httpx.HTTPError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            if isinstance(exc, genai_errors.APIError):
                code = getattr(exc, "code", None)
                msg_upper = str(exc).upper()
                is_rate_limit = code == 429 or "RESOURCE_EXHAUSTED" in msg_upper or "429" in msg_upper
                is_transient = is_rate_limit or (code is not None and code >= 500)
                if not is_transient and code in (400, 401, 403, 404):
                    raise
            if attempt >= max_retries:
                break
            hinted = _extract_retry_delay_seconds(exc)
            delay = (
                hinted
                if hinted is not None
                else min(initial_delay * (2**attempt) + random.uniform(0.2, 1.0), 65.0)
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def load_corpus_chunks(raw_dir: Path = DATA_RAW_DIR) -> list[Chunk]:
    """Load and chunk all supported documents from raw_dir in deterministic filename order."""
    if not raw_dir.exists():
        return []
    files = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    all_chunks: list[Chunk] = []
    for path in files:
        loaded = load_document(path)
        all_chunks.extend(chunk_document(loaded))
    return all_chunks


def _load_disk_cache(
    cache_dir: Path,
    embedding_model: str,
    output_dim: int,
) -> tuple[list[str], np.ndarray]:
    """Load cached key list and L2-normalized float32 vectors if compatible."""
    meta_path = cache_dir / _META_FILENAME
    vec_path = cache_dir / _VECTORS_FILENAME
    if not meta_path.exists() or not vec_path.exists():
        return [], np.zeros((0, output_dim), dtype=np.float32)

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        keys = meta.get("keys", [])
        if meta.get("model") != embedding_model or meta.get("dim") != output_dim:
            return [], np.zeros((0, output_dim), dtype=np.float32)
        matrix = np.load(vec_path, allow_pickle=False)
        if (
            not isinstance(keys, list)
            or matrix.ndim != 2
            or matrix.shape != (len(keys), output_dim)
        ):
            return [], np.zeros((0, output_dim), dtype=np.float32)
        return [str(k) for k in keys], l2_normalize(matrix)
    except (OSError, ValueError, json.JSONDecodeError):
        return [], np.zeros((0, output_dim), dtype=np.float32)


def _save_disk_cache(
    cache_dir: Path,
    keys: list[str],
    vectors: np.ndarray,
    embedding_model: str,
    output_dim: int,
) -> None:
    """Persist cache keys and vectors atomically under cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_tmp = cache_dir / "embeddings_meta.tmp.json"
    vec_tmp = cache_dir / "embeddings_vectors.tmp.npy"

    meta_payload = {
        "model": embedding_model,
        "dim": output_dim,
        "keys": keys,
    }
    meta_tmp.write_text(json.dumps(meta_payload, ensure_ascii=False), encoding="utf-8")
    np.save(vec_tmp, vectors.astype(np.float32))

    meta_tmp.replace(cache_dir / _META_FILENAME)
    vec_tmp.replace(cache_dir / _VECTORS_FILENAME)


class VectorIndex:
    """In-memory vector matrix paired with chunks and incremental disk cache support."""

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        cache_dir: Path,
        embedding_model: str,
        output_dim: int,
        chunker_version: str,
        stats: IndexBuildStats,
    ) -> None:
        self.chunks = list(chunks)
        self.embeddings = l2_normalize(embeddings) if chunks else np.zeros((0, output_dim), dtype=np.float32)
        self.cache_dir = cache_dir
        self.embedding_model = embedding_model
        self.output_dim = output_dim
        self.chunker_version = chunker_version
        self.stats = stats

    def add_chunks(
        self,
        new_chunks: list[Chunk],
        client: Any | None = None,
        batch_size: int = 64,
    ) -> IndexBuildStats:
        """Append new chunks (reusing disk cache where available) without rebuilding existing rows."""
        existing_ids = {c.chunk_id for c in self.chunks}
        unique_new = [c for c in new_chunks if c.chunk_id not in existing_ids]
        if not unique_new:
            return IndexBuildStats(
                total_chunks=len(self.chunks),
                cache_hits=len(self.chunks),
                newly_embedded=0,
                api_batches_called=0,
            )

        combined_chunks = [*self.chunks, *unique_new]
        updated = build_or_load_index(
            chunks=combined_chunks,
            cache_dir=self.cache_dir,
            embedding_model=self.embedding_model,
            output_dim=self.output_dim,
            chunker_version=self.chunker_version,
            client=client,
            batch_size=batch_size,
        )
        self.chunks = updated.chunks
        self.embeddings = updated.embeddings
        self.stats = updated.stats
        return updated.stats


def _embed_document_batch(
    client: Any,
    texts: list[str],
    doc_title: str,
    embedding_model: str,
    output_dim: int,
) -> np.ndarray:
    """Call Gemini embedding API for a single document-homogeneous batch."""
    cfg = types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        title=doc_title[:200] if doc_title else None,
        output_dimensionality=output_dim,
    )
    resp = call_gemini_with_backoff(
        lambda: client.models.embed_content(
            model=embedding_model,
            contents=texts,
            config=cfg,
        )
    )
    rows = [emb.values for emb in resp.embeddings]
    if len(rows) != len(texts):
        raise RuntimeError(
            f"Embedding response count mismatch: expected {len(texts)}, got {len(rows)}."
        )
    return l2_normalize(np.asarray(rows, dtype=np.float32))


def build_or_load_index(
    chunks: list[Chunk] | None = None,
    raw_dir: Path = DATA_RAW_DIR,
    cache_dir: Path = DATA_CACHE_DIR,
    embedding_model: str = EMBEDDING_MODEL,
    output_dim: int = EMBEDDING_OUTPUT_DIM,
    chunker_version: str = CHUNKER_VERSION,
    client: Any | None = None,
    batch_size: int = 64,
    inter_batch_delay: float = 0.4,
) -> VectorIndex:
    """Build or load cached document embeddings with SHA-256 deduplication and batch backoff."""
    target_chunks = list(chunks) if chunks is not None else load_corpus_chunks(raw_dir)
    if not target_chunks:
        empty_stats = IndexBuildStats(
            total_chunks=0,
            cache_hits=0,
            newly_embedded=0,
            api_batches_called=0,
        )
        return VectorIndex(
            chunks=[],
            embeddings=np.zeros((0, output_dim), dtype=np.float32),
            cache_dir=cache_dir,
            embedding_model=embedding_model,
            output_dim=output_dim,
            chunker_version=chunker_version,
            stats=empty_stats,
        )

    cached_keys, cached_vectors = _load_disk_cache(cache_dir, embedding_model, output_dim)
    key_to_row: dict[str, int] = {k: idx for idx, k in enumerate(cached_keys)}

    chunk_keys = [
        compute_chunk_cache_key(
            raw_chunk_text=c.text,
            embedding_model=embedding_model,
            output_dim=output_dim,
            chunker_version=chunker_version,
        )
        for c in target_chunks
    ]

    missing_items: list[tuple[str, Chunk]] = []
    seen_missing: set[str] = set()
    cache_hits = 0
    for key, chunk in zip(chunk_keys, target_chunks, strict=True):
        if key in key_to_row:
            cache_hits += 1
        elif key not in seen_missing:
            seen_missing.add(key)
            missing_items.append((key, chunk))

    newly_embedded = 0
    api_batches_called = 0

    if missing_items:
        resolved_client = client if client is not None else get_genai_client()
        clamped_batch = max(1, min(batch_size, 100))

        # Group by doc_title so EmbedContentConfig.title matches every chunk in the batch
        by_title: dict[str, list[tuple[str, Chunk]]] = defaultdict(list)
        for key, chunk in missing_items:
            by_title[chunk.doc_title].append((key, chunk))

        work_batches: list[tuple[str, list[tuple[str, Chunk]]]] = []
        for title, items in by_title.items():
            for start in range(0, len(items), clamped_batch):
                work_batches.append((title, items[start : start + clamped_batch]))

        for b_idx, (title, batch_items) in enumerate(work_batches):
            batch_vecs = _embed_document_batch(
                client=resolved_client,
                texts=[ch.text for _, ch in batch_items],
                doc_title=title,
                embedding_model=embedding_model,
                output_dim=output_dim,
            )
            api_batches_called += 1
            newly_embedded += len(batch_items)

            start_row = len(cached_keys)
            cached_vectors = (
                np.vstack([cached_vectors, batch_vecs])
                if cached_vectors.shape[0] > 0
                else batch_vecs
            )
            for offset, (key, _) in enumerate(batch_items):
                key_to_row[key] = start_row + offset
                cached_keys.append(key)

            _save_disk_cache(cache_dir, cached_keys, cached_vectors, embedding_model, output_dim)
            if inter_batch_delay > 0 and b_idx < len(work_batches) - 1:
                time.sleep(inter_batch_delay)

    row_indices = [key_to_row[k] for k in chunk_keys]
    ordered_embeddings = cached_vectors[row_indices]

    stats = IndexBuildStats(
        total_chunks=len(target_chunks),
        cache_hits=cache_hits,
        newly_embedded=newly_embedded,
        api_batches_called=api_batches_called,
    )
    return VectorIndex(
        chunks=target_chunks,
        embeddings=ordered_embeddings,
        cache_dir=cache_dir,
        embedding_model=embedding_model,
        output_dim=output_dim,
        chunker_version=chunker_version,
        stats=stats,
    )
