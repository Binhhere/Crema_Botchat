"""PII/identity-scrubbed FAQ backlog storage with date-only logging and k-anonymity (>= 3)."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import threading
import unicodedata
from datetime import date
from pathlib import Path
from typing import Protocol

import httpx

from src.config import (
    DB_BACKEND,
    K_ANONYMITY_MIN_HITS,
    SQLITE_DB_PATH,
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
)
from src.models import BacklogEntry
from src.retriever import strip_vietnamese_accents

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\w)")
_EMP_CODE_RE = re.compile(r"\b(?:ILD|MNV|NV)[\s\-_]*\d{2,6}\b", flags=re.IGNORECASE)
_CITIZEN_ID_RE = re.compile(r"\b\d{9,12}\b")

_SELF_INTRO_NAME_RE = re.compile(
    r"\b(?:tôi\s+tên\s+là|tôi\s+là|em\s+tên\s+là|em\s+là|mình\s+là|họ\s+và\s+tên\s*:?|họ\s+tên\s*:?)"
    r"\s+([^\.,;!\?\(\)\n]{2,35}?)"
    r"(?=\s+(?:mã|sđt|điện\s+thoại|email|ở|thuộc|làm|tại|xin|muốn|hỏi|cần)|[\.,;!\?\(\)]|$)",
    flags=re.IGNORECASE,
)

_HONORIFIC_NAME_RE = re.compile(
    r"\b(?:anh|chị|ông|bà|cô|chú|bạn|sếp|quản\s+lý|giám\s+sát|"
    r"Anh|Chị|Ông|Bà|Cô|Chú|Bạn|Sếp|Quản\s+lý|Giám\s+sát)\s+"
    r"(?:[A-ZÀ-ỸĐ][a-zà-ỹđ]+(?:\s+[A-ZÀ-ỸĐ][a-zà-ỹđ]+){0,3})"
)

_VN_SURNAME_FULLNAME_RE = re.compile(
    r"\b(?:Nguyễn|Trần|Lê|Phạm|Hoàng|Huỳnh|Phan|Vũ|Võ|Đặng|Bùi|Đỗ|Hồ|Ngô|Dương|Lý|"
    r"Đinh|Trương|Đoàn|Mai|Trịnh|Đào|Cao|Lương|Tạ|Phùng|Tô|"
    r"Nguyen|Tran|Le|Pham|Hoang|Huynh|Phan|Vu|Vo|Dang|Bui|Do|Ho|Ngo|Duong|Ly|Dinh|Truong)"
    r"(?:\s+[A-ZÀ-ỸĐ][a-zà-ỹđA-ZÀ-ỸĐ]*){1,3}\b"
)

_UNIT_ORG_RE = re.compile(
    r"\b(?:phân\s+xưởng|phan\s+xuong|nhà\s+xưởng|nha\s+xuong|xưởng|xuong|"
    r"tổ\s+chuyền|to\s+chuyen|dây\s+chuyền|day\s+chuyen|chuyền|chuyen|"
    r"tổ\s+sản\s+xuất|tổ\s+vận\s+hành|tổ\s+bảo\s+trì|tổ\s+đóng\s+gói|tổ|"
    r"bộ\s+phận|bo\s+phan|phòng\s+ban|phong\s+ban)"
    r"(?:\s+(?:sản\s+xuất|san\s+xuat|vận\s+hành|van\s+hanh|kỹ\s+thuật|ky\s+thuat|"
    r"bảo\s+trì|bao\s+tri|đóng\s+gói|dong\s+goi|sấy\s+lạnh|say\s+lanh|"
    r"sấy\s+phun|say\s+phun|rang\s+xay|cơ\s+điện|co\s+dien|kho\s+vận|kho\s+van|"
    r"kế\s+toán|ke\s+toan|nhân\s+sự|nhan\s+su|qa|qc|hse|it|số|so|[a-z]{0,2}\d{1,3}|[a-d])){1,4}\b",
    flags=re.IGNORECASE,
)

_PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+\]")
_FILLER_TOKENS = {
    "cho",
    "toi",
    "em",
    "minh",
    "hoi",
    "xin",
    "vui",
    "long",
    "giup",
    "ad",
    "oi",
    "a",
    "nha",
    "nhe",
    "da",
    "vang",
    "ve",
    "tai",
    "o",
    "ma",
    "sdt",
    "email",
    "la",
    "ten",
    "thuoc",
    "lam",
    "khong",
    "co",
}


def scrub_pii_and_identity(text: str) -> str:
    """Scrub emails, phone numbers, employee IDs, person names, and team/workshop labels."""
    cleaned = unicodedata.normalize("NFC", text).replace("\r", " ").replace("\n", " ")
    cleaned = _EMAIL_RE.sub("[EMAIL]", cleaned)
    cleaned = _PHONE_RE.sub("[SDT]", cleaned)
    cleaned = _EMP_CODE_RE.sub("[MA_NV]", cleaned)
    cleaned = _CITIZEN_ID_RE.sub("[MA_DINH_DANH]", cleaned)
    cleaned = _SELF_INTRO_NAME_RE.sub(lambda m: f"{m.group(0).split()[0]} là [TEN_NGUOI]", cleaned)
    cleaned = _VN_SURNAME_FULLNAME_RE.sub("[TEN_NGUOI]", cleaned)
    cleaned = _HONORIFIC_NAME_RE.sub("[TEN_NGUOI]", cleaned)
    cleaned = _UNIT_ORG_RE.sub("[BO_PHAN]", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def compute_topic_key(scrubbed_question: str) -> str:
    """Derive stable topic key ignoring PII placeholders and accented/unaccented surface differences."""
    without_tags = _PLACEHOLDER_RE.sub(" ", scrubbed_question)
    unaccented = strip_vietnamese_accents(without_tags)
    tokens = re.findall(r"[a-z0-9]+", unaccented)
    core_tokens = [tok for tok in tokens if tok not in _FILLER_TOKENS]
    normalized_topic = " ".join(core_tokens) if core_tokens else " ".join(tokens)
    return hashlib.sha256(normalized_topic.encode("utf-8")).hexdigest()[:24]


class BacklogStore(Protocol):
    """Protocol implemented by SQLite and Supabase FAQ backlog stores."""

    def record_unanswered(
        self,
        raw_question: str,
        status: str = "OUT_OF_SCOPE",
        asked_date: date | None = None,
    ) -> BacklogEntry: ...

    def list_public_backlog(
        self,
        min_hits: int = K_ANONYMITY_MIN_HITS,
    ) -> list[BacklogEntry]: ...


class SQLiteBacklogStore:
    """Local SQLite backlog store enforcing date-only timestamps and k-anonymity (>= 3)."""

    def __init__(self, db_path: Path = SQLITE_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS crema_faq_backlog (
                    topic_key TEXT PRIMARY KEY,
                    scrubbed_question TEXT NOT NULL,
                    asked_date TEXT NOT NULL CHECK (
                        length(asked_date) = 10
                        AND asked_date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
                    ),
                    hit_count INTEGER NOT NULL DEFAULT 1 CHECK (hit_count >= 1),
                    sample_status TEXT NOT NULL DEFAULT 'OUT_OF_SCOPE'
                )
                """
            )
            conn.commit()

    def record_unanswered(
        self,
        raw_question: str,
        status: str = "OUT_OF_SCOPE",
        asked_date: date | None = None,
    ) -> BacklogEntry:
        """Scrub PII/identity, store strictly YYYY-MM-DD date, and increment topic counter."""
        iso_date = (asked_date or date.today()).isoformat()
        scrubbed = scrub_pii_and_identity(raw_question)
        topic_key = compute_topic_key(scrubbed)

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO crema_faq_backlog (
                    topic_key, scrubbed_question, asked_date, hit_count, sample_status
                ) VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(topic_key) DO UPDATE SET
                    asked_date = excluded.asked_date,
                    hit_count = crema_faq_backlog.hit_count + 1,
                    sample_status = excluded.sample_status
                """,
                (topic_key, scrubbed, iso_date, status),
            )
            row = conn.execute(
                """
                SELECT topic_key, scrubbed_question, asked_date, hit_count, sample_status
                FROM crema_faq_backlog
                WHERE topic_key = ?
                """,
                (topic_key,),
            ).fetchone()
            conn.commit()

        assert row is not None
        return BacklogEntry(
            topic_key=str(row["topic_key"]),
            scrubbed_question=str(row["scrubbed_question"]),
            asked_date=str(row["asked_date"]),
            hit_count=int(row["hit_count"]),
            sample_status=str(row["sample_status"]),
        )

    def list_public_backlog(
        self,
        min_hits: int = K_ANONYMITY_MIN_HITS,
    ) -> list[BacklogEntry]:
        """Return only backlog topics meeting the hard k-anonymity threshold (>= 3 hits)."""
        threshold = max(int(min_hits), K_ANONYMITY_MIN_HITS)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT topic_key, scrubbed_question, asked_date, hit_count, sample_status
                FROM crema_faq_backlog
                WHERE hit_count >= ?
                ORDER BY hit_count DESC, asked_date DESC, topic_key ASC
                """,
                (threshold,),
            ).fetchall()

        return [
            BacklogEntry(
                topic_key=str(r["topic_key"]),
                scrubbed_question=str(r["scrubbed_question"]),
                asked_date=str(r["asked_date"]),
                hit_count=int(r["hit_count"]),
                sample_status=str(r["sample_status"]),
            )
            for r in rows
        ]


class SupabaseBacklogStore:
    """Supabase PostgREST adapter mirroring SQLiteBacklogStore for cloud deployment when configured."""

    def __init__(
        self,
        supabase_url: str = SUPABASE_URL,
        service_key: str = SUPABASE_SERVICE_KEY,
    ) -> None:
        if not supabase_url or not service_key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required for DB_BACKEND=supabase.")
        self.endpoint = f"{supabase_url.rstrip('/')}/rest/v1/crema_faq_backlog"
        self.headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    def record_unanswered(
        self,
        raw_question: str,
        status: str = "OUT_OF_SCOPE",
        asked_date: date | None = None,
    ) -> BacklogEntry:
        iso_date = (asked_date or date.today()).isoformat()
        scrubbed = scrub_pii_and_identity(raw_question)
        topic_key = compute_topic_key(scrubbed)

        with httpx.Client(timeout=10.0) as http:
            existing = http.get(
                self.endpoint,
                headers=self.headers,
                params={"topic_key": f"eq.{topic_key}", "select": "*"},
            )
            existing.raise_for_status()
            rows = existing.json()
            if rows:
                next_hits = int(rows[0]["hit_count"]) + 1
                resp = http.patch(
                    self.endpoint,
                    headers=self.headers,
                    params={"topic_key": f"eq.{topic_key}"},
                    json={"asked_date": iso_date, "hit_count": next_hits, "sample_status": status},
                )
            else:
                resp = http.post(
                    self.endpoint,
                    headers=self.headers,
                    json={
                        "topic_key": topic_key,
                        "scrubbed_question": scrubbed,
                        "asked_date": iso_date,
                        "hit_count": 1,
                        "sample_status": status,
                    },
                )
            resp.raise_for_status()
            row = resp.json()[0]

        return BacklogEntry(
            topic_key=str(row["topic_key"]),
            scrubbed_question=str(row["scrubbed_question"]),
            asked_date=str(row["asked_date"])[:10],
            hit_count=int(row["hit_count"]),
            sample_status=str(row["sample_status"]),
        )

    def list_public_backlog(
        self,
        min_hits: int = K_ANONYMITY_MIN_HITS,
    ) -> list[BacklogEntry]:
        threshold = max(int(min_hits), K_ANONYMITY_MIN_HITS)
        with httpx.Client(timeout=10.0) as http:
            resp = http.get(
                self.endpoint,
                headers=self.headers,
                params={
                    "hit_count": f"gte.{threshold}",
                    "order": "hit_count.desc,asked_date.desc",
                    "select": "*",
                },
            )
            resp.raise_for_status()
            rows = resp.json()

        return [
            BacklogEntry(
                topic_key=str(r["topic_key"]),
                scrubbed_question=str(r["scrubbed_question"]),
                asked_date=str(r["asked_date"])[:10],
                hit_count=int(r["hit_count"]),
                sample_status=str(r["sample_status"]),
            )
            for r in rows
        ]


def create_backlog_store(
    backend: str | None = None,
    sqlite_path: Path | None = None,
) -> BacklogStore:
    """Instantiate configured FAQ backlog store (defaults to local SQLite)."""
    selected = (backend or os.getenv("DB_BACKEND", DB_BACKEND)).strip().lower()
    if selected == "supabase":
        return SupabaseBacklogStore()
    return SQLiteBacklogStore(db_path=sqlite_path or SQLITE_DB_PATH)
