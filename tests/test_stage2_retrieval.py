"""Stage 2 unit and integration tests for indexer, 0ms rewriter, and hybrid retriever."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import (
    CHUNKER_VERSION,
    DATA_CACHE_DIR,
    DATA_RAW_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_OUTPUT_DIM,
)
from src.indexer import (
    VectorIndex,
    build_or_load_index,
    compute_chunk_cache_key,
    get_genai_client,
)
from src.models import Chunk
from src.retriever import HybridRetriever, strip_vietnamese_accents, tokenize_unaccented_bigrams
from src.rewriter import needs_query_rewrite, rewrite_query


class _ForbiddenClient:
    """Sentinel client that fails immediately if any API call is attempted."""

    class _FailModels:
        def embed_content(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("Unexpected Gemini embed_content API call on warm cache!")

        def generate_content(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("Unexpected Gemini generate_content API call on 0ms bypass!")

    def __init__(self) -> None:
        self.models = self._FailModels()


class _StubEmbedClient:
    """Deterministic local embedder for testing partial SHA-256 cache invalidation."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []
        self.models = self

    def embed_content(
        self,
        *,
        model: str,
        contents: list[str] | str,
        config: object = None,
    ) -> SimpleNamespace:
        texts = [contents] if isinstance(contents, str) else list(contents)
        self.calls.append(texts)
        embs = []
        for idx, txt in enumerate(texts):
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[idx % self.dim] = float(len(txt) + 1)
            embs.append(SimpleNamespace(values=vec.tolist()))
        return SimpleNamespace(embeddings=embs)


@pytest.fixture(scope="module")
def real_vector_index() -> VectorIndex:
    """Build or load the full 585-chunk data/raw index against Gemini API once per module."""
    client = get_genai_client()
    return build_or_load_index(
        raw_dir=DATA_RAW_DIR,
        cache_dir=DATA_CACHE_DIR,
        client=client,
        batch_size=64,
    )


def test_rewriter_zero_ms_gate_pure_python() -> None:
    forbidden = _ForbiddenClient()

    # Standalone accented queries (including "bao nhiêu" or <7 words WITHOUT history) must bypass LLM
    for standalone in [
        "Mức bồi thường tai nạn lao động suy giảm 81% là bao nhiêu tháng lương?",
        "Điều 31 Nghị định 145 quy định gì?",
        "Thời giờ nghỉ ngơi tối thiểu?",
    ]:
        need, reason = needs_query_rewrite(standalone, history=None)
        assert need is False
        assert reason == "standalone_clear"
        res = rewrite_query(standalone, history=None, client=forbidden)
        assert res.was_rewritten is False
        assert res.reason == "standalone_clear"
        assert res.api_calls == 0

    # Unaccented Vietnamese >= 3 words must trigger LLM rewrite
    need_unacc, reason_unacc = needs_query_rewrite("lam them gio ban dem tinh luong the nao")
    assert need_unacc is True
    assert reason_unacc == "unaccented_vietnamese"

    # Multi-turn short or coreferential query must trigger LLM rewrite
    hist = [
        {"role": "user", "content": "Làm thêm giờ ngày lễ hưởng ít nhất bao nhiêu phần trăm?"},
        {"role": "assistant", "content": "Ít nhất bằng 300% theo Điều 55 Nghị định 145/2020/NĐ-CP."},
    ]
    need_short, reason_short = needs_query_rewrite("Còn ngày nghỉ hàng tuần?", history=hist)
    assert need_short is True
    assert reason_short == "contextual_followup"

    need_long_coref, reason_long = needs_query_rewrite(
        "Trong trường hợp làm thêm vào ban đêm đó thì cộng thêm mức nào?",
        history=hist,
    )
    assert need_long_coref is True
    assert reason_long == "contextual_followup"


def test_unaccented_bigram_tokenizer_and_selective_cache_invalidation(tmp_path: Path) -> None:
    tokens = tokenize_unaccented_bigrams("Làm thêm giờ theo Điều 31 Nghị định 145/2020/NĐ-CP")
    assert tokens
    for tok in tokens:
        assert tok == strip_vietnamese_accents(tok)
    assert {"lam", "them", "lam_them", "dieu_31", "145_2020"}.issubset(set(tokens))

    k1 = compute_chunk_cache_key("Nội dung A", EMBEDDING_MODEL, EMBEDDING_OUTPUT_DIM, CHUNKER_VERSION)
    assert k1 != compute_chunk_cache_key("Nội dung B", EMBEDDING_MODEL, EMBEDDING_OUTPUT_DIM, CHUNKER_VERSION)
    assert k1 != compute_chunk_cache_key("Nội dung A", "other-model", EMBEDDING_OUTPUT_DIM, CHUNKER_VERSION)
    assert k1 != compute_chunk_cache_key("Nội dung A", EMBEDDING_MODEL, 256, CHUNKER_VERSION)
    assert k1 != compute_chunk_cache_key("Nội dung A", EMBEDDING_MODEL, EMBEDDING_OUTPUT_DIM, "9.9.9")

    base_chunks = [
        Chunk(
            chunk_id="d1__c0001",
            doc_id="d1",
            doc_title="Tài liệu 1",
            doc_nature="synthetic_policy",
            ui_source_label="[Văn bản mẫu mô phỏng]",
            section_code="Điều 1",
            section_title="Điều 1",
            text="Nội dung chính sách thứ nhất.",
        ),
        Chunk(
            chunk_id="d1__c0002",
            doc_id="d1",
            doc_title="Tài liệu 1",
            doc_nature="synthetic_policy",
            ui_source_label="[Văn bản mẫu mô phỏng]",
            section_code="Điều 2",
            section_title="Điều 2",
            text="Nội dung chính sách thứ hai.",
        ),
    ]
    stub = _StubEmbedClient(dim=8)
    idx1 = build_or_load_index(
        chunks=base_chunks,
        cache_dir=tmp_path,
        output_dim=8,
        client=stub,
        inter_batch_delay=0.0,
    )
    assert idx1.stats.newly_embedded == 2
    assert idx1.stats.api_batches_called == 1

    # Modify only chunk 2 -> only 1 chunk should be re-embedded
    modified = [
        base_chunks[0],
        base_chunks[1].model_copy(update={"text": "Nội dung chính sách thứ hai đã cập nhật."}),
    ]
    stub2 = _StubEmbedClient(dim=8)
    idx2 = build_or_load_index(
        chunks=modified,
        cache_dir=tmp_path,
        output_dim=8,
        client=stub2,
        inter_batch_delay=0.0,
    )
    assert idx2.stats.cache_hits == 1
    assert idx2.stats.newly_embedded == 1
    assert len(stub2.calls) == 1 and len(stub2.calls[0]) == 1


def test_real_corpus_index_build_and_warm_cache_zero_api_calls(real_vector_index: VectorIndex) -> None:
    assert len(real_vector_index.chunks) >= 580
    assert real_vector_index.embeddings.shape == (len(real_vector_index.chunks), EMBEDDING_OUTPUT_DIM)

    norms = np.linalg.norm(real_vector_index.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    # Second load on DATA_CACHE_DIR with _ForbiddenClient must achieve 100% cache hit and 0 API calls
    warm = build_or_load_index(
        chunks=real_vector_index.chunks,
        cache_dir=DATA_CACHE_DIR,
        client=_ForbiddenClient(),
    )
    assert warm.stats.cache_hits == len(real_vector_index.chunks)
    assert warm.stats.newly_embedded == 0
    assert warm.stats.api_batches_called == 0


def test_hybrid_retrieval_accented_unaccented_article_and_negative(
    real_vector_index: VectorIndex,
) -> None:
    client = get_genai_client()
    retriever = HybridRetriever(real_vector_index)

    # 1. Accented standalone query -> 0ms rewriter skip (1 API call total)
    q_acc = "Người lao động bị tai nạn lao động suy giảm khả năng lao động 81% được bồi thường ít nhất bao nhiêu tháng tiền lương?"
    res_acc = retriever.search(q_acc, client=client)
    assert res_acc.rewrite_result is not None
    assert res_acc.rewrite_result.was_rewritten is False
    assert res_acc.api_calls == 1
    assert res_acc.passed_gate is True
    assert any(
        c.chunk.doc_id.startswith("02_tt28_2021") and c.chunk.section_code == "Điều 3"
        for c in res_acc.candidates
    )

    # 2. Unaccented phone-style query -> LLM restores accents (2 API calls) and retrieves Điều 55
    q_unacc = "lam them gio vao ngay nghi le tet duoc huong it nhat bao nhieu phan tram"
    res_unacc = retriever.search(q_unacc, client=client)
    assert res_unacc.rewrite_result is not None
    assert res_unacc.rewrite_result.was_rewritten is True
    assert res_unacc.rewrite_result.reason == "unaccented_vietnamese"
    assert res_unacc.api_calls == 2
    assert res_unacc.passed_gate is True
    assert any(
        c.chunk.doc_id.startswith("03_nd145_2020") and c.chunk.section_code == "Điều 55"
        for c in res_unacc.candidates
    )

    # 3a. Explicit article query (Điều 31 NĐ 145 -> partial_expiry_warning=True)
    res_d31 = retriever.search("Điều 31 Nghị định 145/2020/NĐ-CP quy định gì?", client=client)
    assert res_d31.api_calls == 1
    assert res_d31.passed_gate is True
    assert any(
        c.chunk.section_code == "Điều 31" and c.chunk.partial_expiry_warning is True
        for c in res_d31.candidates
    )

    # 3b. Explicit article query (Điểm e khoản 2 Điều 5 TT 28/2021 -> 2026-01-01)
    res_d5e = retriever.search(
        "Điểm e khoản 2 Điều 5 Thông tư 28/2021 quy định mức tiền lương đối với hạ sĩ quan, binh sĩ thế nào?",
        client=client,
    )
    assert res_d5e.passed_gate is True
    assert any(
        c.chunk.section_code == "Điều 5"
        and c.chunk.clause_effective_date == "2026-01-01"
        and "hạ sĩ quan" in c.chunk.text
        for c in res_d5e.candidates
    )

    # 4. Multi-turn follow-up -> triggers contextual rewrite
    hist = [
        {"role": "user", "content": "Làm thêm giờ ban ngày ngày thường hưởng bao nhiêu phần trăm?"},
        {"role": "assistant", "content": "Theo Điều 55 Nghị định 145/2020/NĐ-CP, ít nhất 150%."},
    ]
    res_mt = retriever.search(
        "Còn làm vào ban đêm thì cộng thêm mấy phần trăm?",
        history=hist,
        client=client,
    )
    assert res_mt.rewrite_result is not None
    assert res_mt.rewrite_result.was_rewritten is True
    assert res_mt.rewrite_result.reason == "contextual_followup"
    assert res_mt.api_calls == 2
    assert res_mt.passed_gate is True
    assert any(
        c.chunk.doc_id.startswith("03_nd145_2020") and c.chunk.section_code in {"Điều 56", "Điều 57"}
        for c in res_mt.candidates
    )

    # 5. Out-of-scope negative query -> rejected at pre-LLM gate
    res_neg = retriever.search(
        "Cách nấu bún bò Huế chuẩn vị truyền thống gồm những nguyên liệu gì?",
        client=client,
    )
    assert res_neg.passed_gate is False
    assert res_neg.gate_reason == "below_threshold"
