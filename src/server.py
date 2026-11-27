"""FastAPI service with RBAC endpoints (/v1/chat, /v1/ingest, /v1/backlog) and Web UI."""

from __future__ import annotations

import secrets
import shutil
import threading
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, status
from fastapi.responses import HTMLResponse

from src.chunker import chunk_document
from src.config import (
    DATA_CACHE_DIR,
    DATA_RAW_DIR,
    DB_BACKEND,
    EMBEDDING_MODEL,
    EVAL_DIR,
    GENERATION_MODEL,
    INGESTED_DIR,
    K_ANONYMITY_MIN_HITS,
    REWRITER_MODEL,
    SQLITE_DB_PATH,
    TAU_COSINE_FLOOR,
    TAU_COSINE_REJECT,
    get_admin_api_key,
    get_hr_api_key,
)
from src.generator import ask_policy
from src.indexer import SUPPORTED_EXTENSIONS, VectorIndex, build_or_load_index, get_genai_client
from src.loader import load_document
from src.models import ChatRequest, ChatResponse, IngestResponse, ScannedPDFError
from src.retriever import HybridRetriever
from src.storage import BacklogStore, create_backlog_store

STATIC_INDEX_PATH = Path(__file__).resolve().parent / "static" / "index.html"


class _RuntimeContainer:
    """Thread-safe lazy holder for index, hybrid retriever, client, and backlog store."""

    def __init__(
        self,
        *,
        raw_dir: Path,
        cache_dir: Path,
        ingested_dir: Path,
        sqlite_path: Path,
        client: Any | None = None,
        vector_index: VectorIndex | None = None,
        backlog_store: BacklogStore | None = None,
        load_persisted_ingested: bool = True,
    ) -> None:
        self.raw_dir = raw_dir
        self.cache_dir = cache_dir
        self.ingested_dir = ingested_dir
        self.sqlite_path = sqlite_path
        self.client = client
        self._vector_index = vector_index
        self._retriever: HybridRetriever | None = (
            HybridRetriever(vector_index) if vector_index is not None else None
        )
        self._store = backlog_store
        self._load_persisted_ingested = load_persisted_ingested
        self._lock = threading.Lock()

    def get_client(self) -> Any:
        if self.client is None:
            self.client = get_genai_client()
        return self.client

    def get_store(self) -> BacklogStore:
        if self._store is None:
            self._store = create_backlog_store(sqlite_path=self.sqlite_path)
        return self._store

    def ensure_retriever(self) -> tuple[VectorIndex, HybridRetriever]:
        if self._vector_index is not None and self._retriever is not None:
            return self._vector_index, self._retriever

        with self._lock:
            if self._vector_index is not None and self._retriever is not None:
                return self._vector_index, self._retriever

            idx = build_or_load_index(
                raw_dir=self.raw_dir,
                cache_dir=self.cache_dir,
                client=self.get_client(),
            )
            if self._load_persisted_ingested and self.ingested_dir.exists():
                extra_files = sorted(
                    p for p in self.ingested_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                )
                for ef in extra_files:
                    loaded = load_document(ef)
                    idx.add_chunks(chunk_document(loaded), client=self.get_client())

            self._vector_index = idx
            self._retriever = HybridRetriever(idx)
            return self._vector_index, self._retriever

    def ingest_file(self, file_path: Path) -> IngestResponse:
        loaded = load_document(file_path)
        new_chunks = chunk_document(loaded)
        if not new_chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Tệp không trích xuất được đoạn nội dung hợp lệ nào.",
            )

        with self._lock:
            idx, _ = self.ensure_retriever()
            stats = idx.add_chunks(new_chunks, client=self.get_client())
            self._retriever = HybridRetriever(idx)

        meta = loaded.metadata
        return IngestResponse(
            status="INGESTED",
            doc_id=meta.doc_id,
            doc_title=meta.doc_title,
            doc_nature=meta.doc_nature,
            ui_source_label=meta.ui_source_label,
            chunks_added=len(new_chunks),
            total_chunks=stats.total_chunks,
        )


def _extract_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> str:
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if authorization and authorization.strip().lower().startswith("bearer "):
        token = authorization.strip()[7:].strip()
        if token:
            return token
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Thiếu khóa xác thực API (gửi qua header X-API-Key hoặc Bearer token).",
    )


def require_staff_api_key(provided_key: str = Depends(_extract_api_key)) -> str:
    """Enforce RBAC requiring either HR_API_KEY or ADMIN_API_KEY."""
    allowed_keys = [k for k in (get_hr_api_key(), get_admin_api_key()) if k]
    if not allowed_keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống chưa cấu hình HR_API_KEY / ADMIN_API_KEY.",
        )
    if any(secrets.compare_digest(provided_key, expected) for expected in allowed_keys):
        return provided_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Khóa API không hợp lệ hoặc không đủ quyền truy cập.",
    )


def create_app(
    *,
    raw_dir: Path = DATA_RAW_DIR,
    cache_dir: Path = DATA_CACHE_DIR,
    ingested_dir: Path = INGESTED_DIR,
    sqlite_path: Path = SQLITE_DB_PATH,
    client: Any | None = None,
    vector_index: VectorIndex | None = None,
    backlog_store: BacklogStore | None = None,
    load_persisted_ingested: bool = True,
) -> FastAPI:
    """Create FastAPI application instance with isolated paths/dependencies."""
    runtime = _RuntimeContainer(
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        ingested_dir=ingested_dir,
        sqlite_path=sqlite_path,
        client=client,
        vector_index=vector_index,
        backlog_store=backlog_store,
        load_persisted_ingested=load_persisted_ingested,
    )

    fastapi_app = FastAPI(
        title="Crema HR & HSE Policy RAG API",
        version="1.0.0",
        description="Verified policy assistant with deterministic citation verification and k-anonymity.",
    )
    fastapi_app.state.runtime = runtime

    @fastapi_app.get("/", response_class=HTMLResponse)
    def serve_ui() -> HTMLResponse:
        if not STATIC_INDEX_PATH.exists():
            raise HTTPException(status_code=404, detail="Không tìm thấy giao diện Web.")
        return HTMLResponse(content=STATIC_INDEX_PATH.read_text(encoding="utf-8"))

    @fastapi_app.get("/health")
    @fastapi_app.get("/v1/health")
    def health_check() -> dict[str, Any]:
        idx, _ = runtime.ensure_retriever()
        doc_ids = sorted({ch.doc_id for ch in idx.chunks})
        return {
            "status": "ok",
            "total_chunks": len(idx.chunks),
            "total_documents": len(doc_ids),
            "documents": doc_ids,
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": GENERATION_MODEL,
            "rewriter_model": REWRITER_MODEL,
            "tau_cosine_reject": TAU_COSINE_REJECT,
            "tau_cosine_floor": TAU_COSINE_FLOOR,
            "db_backend": DB_BACKEND,
            "k_anonymity_min_hits": K_ANONYMITY_MIN_HITS,
        }

    @fastapi_app.post("/v1/chat", response_model=ChatResponse)
    @fastapi_app.post("/v1/ask", response_model=ChatResponse)
    def chat_endpoint(req: ChatRequest) -> ChatResponse:
        _, retriever = runtime.ensure_retriever()
        res = ask_policy(
            query=req.query,
            retriever=retriever,
            history=req.history or None,
            client=runtime.get_client(),
        )
        backlog_recorded = False
        if res.status in {"OUT_OF_SCOPE", "ABSTAINED_VERIFIER"}:
            runtime.get_store().record_unanswered(req.query, status=res.status)
            backlog_recorded = True

        return ChatResponse(**res.model_dump(), backlog_recorded=backlog_recorded)

    @fastapi_app.get("/v1/backlog")
    def get_backlog(
        _role_key: str = Depends(require_staff_api_key),
    ) -> dict[str, Any]:
        items = runtime.get_store().list_public_backlog(min_hits=K_ANONYMITY_MIN_HITS)
        return {
            "k_anonymity_min_hits": K_ANONYMITY_MIN_HITS,
            "count": len(items),
            "items": [item.model_dump() for item in items],
        }

    @fastapi_app.post("/v1/ingest", response_model=IngestResponse)
    async def ingest_document(
        _role_key: str = Depends(require_staff_api_key),
        file: Annotated[UploadFile | None, File()] = None,
        preset: Annotated[str | None, Query()] = None,
    ) -> IngestResponse:
        ingested_dir.mkdir(parents=True, exist_ok=True)
        target_path: Path

        if preset == "extra_05":
            src_preset = EVAL_DIR / "extra_05_so_tay_nha_thau.docx"
            if not src_preset.exists():
                raise HTTPException(status_code=404, detail="Không tìm thấy tệp mẫu extra_05.")
            target_path = ingested_dir / src_preset.name
            shutil.copy2(src_preset, target_path)
        elif file is not None and file.filename:
            safe_name = Path(file.filename).name
            ext = Path(safe_name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Định dạng không hỗ trợ ({ext}). Chỉ nhận: .pdf, .docx, .md, .txt.",
                )
            content = await file.read()
            if not content:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tệp tải lên rỗng.")
            target_path = ingested_dir / safe_name
            target_path.write_bytes(content)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vui lòng gửi tệp (.pdf/.docx/.md/.txt) hoặc preset=extra_05.",
            )

        try:
            return runtime.ingest_file(target_path)
        except ScannedPDFError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return fastapi_app


app = create_app()
