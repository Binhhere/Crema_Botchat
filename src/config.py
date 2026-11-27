"""Configuration constants and runtime settings for Crema HR & HSE Policy RAG."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
EVAL_DIR = PROJECT_ROOT / "eval"

CHUNKER_VERSION = "1.0.1"

DOC_NATURE_LABELS: dict[str, str] = {
    "consolidated_law": "[Văn bản pháp luật hợp nhất]",
    "legal_decree": "[Nghị định pháp luật]",
    "enterprise_reference": "[Tiêu chuẩn doanh nghiệp tham chiếu]",
    "synthetic_policy": "[Văn bản mẫu mô phỏng]",
}

# Gemini model settings (configured here for Stage 2+)
EMBEDDING_MODEL = os.getenv("CREMA_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_OUTPUT_DIM = int(os.getenv("CREMA_EMBEDDING_DIM", "768"))
GENERATION_MODEL = os.getenv("CREMA_GENERATION_MODEL", "gemini-3.6-flash")
REWRITER_MODEL = os.getenv("CREMA_REWRITER_MODEL", "gemini-3.6-flash")
FALLBACK_FLASH_MODELS: tuple[str, ...] = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
)

# Calibrated retrieval gates (to be tuned in Stage 4)
TAU_COSINE_REJECT = float(os.getenv("CREMA_TAU_COSINE_REJECT", "0.65"))
TAU_COSINE_FLOOR = float(os.getenv("CREMA_TAU_COSINE_FLOOR", "0.58"))

# Auth & storage settings (Stage 5)
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")
SQLITE_DB_PATH = PROJECT_ROOT / "data" / "crema_backlog.db"


def get_gemini_api_key() -> str | None:
    """Return GEMINI_API_KEY from environment, reloading .env once if empty."""
    val = os.getenv("GEMINI_API_KEY", "").strip()
    if not val:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        val = os.getenv("GEMINI_API_KEY", "").strip()
    return val or None
