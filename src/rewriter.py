"""Conditional 0ms query rewriter (accent restoration + multi-turn coreference resolution)."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import httpx
from google.genai import errors as genai_errors
from google.genai import types

from src.config import FALLBACK_FLASH_MODELS, REWRITER_MODEL
from src.indexer import call_gemini_with_backoff, get_genai_client
from src.models import RewriteReason, RewriteResult

_ALPHA_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
_COREF_RE = re.compile(
    r"(?<!\w)(?:đó|này|ấy|trên|vừa nêu|như trên|còn|thế còn|thế|vậy|bao nhiêu|mức nào|sao)(?!\w)",
    flags=re.IGNORECASE,
)

_REWRITE_SYSTEM_PROMPT = (
    "Bạn là bộ chuẩn hóa truy vấn tìm kiếm tài liệu Nhân sự & An toàn lao động bằng tiếng Việt. "
    "Nhiệm vụ: chỉ viết lại câu hỏi của người dùng thành MỘT câu hỏi tiếng Việt có dấu đầy đủ, "
    "rõ nghĩa và độc lập ngữ cảnh.\n"
    "Quy tắc bắt buộc:\n"
    "1. Nếu câu hỏi không dấu: khôi phục dấu tiếng Việt chính xác theo ngữ cảnh HR/HSE.\n"
    "2. Nếu có lịch sử hội thoại và câu mới dùng từ quy chiếu/tỉnh lược: ghép chủ đề từ lịch sử vào câu hỏi.\n"
    "3. Giữ nguyên tuyệt đối mọi con số, tỷ lệ %, số Điều/Khoản/Mục, mã văn bản "
    "(như 88/2020, 158/2025, 28/2021, 145/2020, VBHN, LOTO, PTW, PPE).\n"
    "4. KHÔNG trả lời câu hỏi, KHÔNG bịa thêm quy định, chỉ xuất duy nhất 1 dòng câu hỏi đã chuẩn hóa."
)


def has_vietnamese_diacritics(text: str) -> bool:
    """Return True if text contains đ/Đ or any Unicode combining diacritical mark."""
    if "đ" in text or "Đ" in text:
        return True
    decomposed = unicodedata.normalize("NFD", text)
    return any(unicodedata.category(ch) == "Mn" for ch in decomposed)


def needs_query_rewrite(
    query: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[bool, RewriteReason]:
    """0ms pure-Python gate deciding whether query needs LLM restoration/coreference."""
    nfc_query = unicodedata.normalize("NFC", query.strip())
    words = _ALPHA_WORD_RE.findall(nfc_query)

    # Rule (a): unaccented Vietnamese typed quickly (>= 3 words, 0 diacritic chars)
    if len(words) >= 3 and not has_vietnamese_diacritics(nfc_query):
        return True, "unaccented_vietnamese"

    # Rule (b): multi-turn follow-up that is short (< 7 words) or contains coreference markers
    has_history = bool(history and any(str(turn.get("content", "")).strip() for turn in history))
    if has_history and (len(words) < 7 or bool(_COREF_RE.search(nfc_query.lower()))):
        return True, "contextual_followup"

    return False, "standalone_clear"


def _format_history_turns(history: list[dict[str, str]] | None, max_turns: int = 4) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for turn in history[-max_turns:]:
        role = str(turn.get("role", "user")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        speaker = "Người dùng" if role == "user" else "Trợ lý"
        lines.append(f"- {speaker}: {content}")
    return "\n".join(lines)


def rewrite_query(
    query: str,
    history: list[dict[str, str]] | None = None,
    client: Any | None = None,
    model: str = REWRITER_MODEL,
) -> RewriteResult:
    """Conditionally rewrite query via gemini-2.5-flash only when the 0ms gate triggers."""
    cleaned_query = unicodedata.normalize("NFC", query.strip())
    should_rewrite, reason = needs_query_rewrite(cleaned_query, history=history)
    if not should_rewrite:
        return RewriteResult(
            original_query=query,
            rewritten_query=cleaned_query,
            was_rewritten=False,
            reason=reason,
            api_calls=0,
        )

    resolved_client = client if client is not None else get_genai_client()
    history_block = _format_history_turns(history)
    user_prompt = (
        f"Lịch sử hội thoại gần nhất:\n{history_block}\n\nCâu hỏi hiện tại: {cleaned_query}"
        if history_block
        else f"Câu hỏi cần khôi phục dấu: {cleaned_query}"
    )

    cfg = types.GenerateContentConfig(
        system_instruction=_REWRITE_SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=256,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        http_options=types.HttpOptions(
            timeout=20000,
            retry_options=types.HttpRetryOptions(attempts=2),
        ),
    )
    candidate_models = [model, *[m for m in FALLBACK_FLASH_MODELS if m != model]]
    resp = None
    last_exc: Exception | None = None
    for candidate_model in candidate_models:
        try:
            resp = call_gemini_with_backoff(
                lambda m=candidate_model: resolved_client.models.generate_content(
                    model=m,
                    contents=user_prompt,
                    config=cfg,
                ),
                max_retries=1,
                initial_delay=1.5,
            )
            break
        except (genai_errors.APIError, httpx.HTTPError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            if isinstance(exc, genai_errors.APIError):
                code = getattr(exc, "code", None)
                if code not in (404, 429, 500, 503, 504):
                    raise
    if resp is None:
        assert last_exc is not None
        raise last_exc
    raw_out = (resp.text or "").strip().strip('"').strip("'").strip()
    first_line = raw_out.splitlines()[0].strip() if raw_out else ""
    final_query = unicodedata.normalize("NFC", first_line) if first_line else cleaned_query

    return RewriteResult(
        original_query=query,
        rewritten_query=final_query,
        was_rewritten=True,
        reason=reason,
        api_calls=1,
    )
