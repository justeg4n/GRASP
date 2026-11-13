"""Phân loại lỗi API theo phạm vi ảnh hưởng (xem spec mục 6).

- TRANSIENT: retry được (429, 5xx, timeout, mất kết nối).
- BAD_REQUEST: lỗi của chính request/câu hỏi đó (400) -> không retry.
- CREDIT / AUTH / CONFIG: lỗi toàn cục -> dừng cả lần chạy ngay.
- UNKNOWN_4XX: 4xx lạ -> không retry; collector đếm chuỗi liên tiếp làm lưới an toàn.
"""

from __future__ import annotations

from enum import StrEnum

CREDIT_KEYWORDS = ("insufficient", "balance", "quota", "credit", "billing", "payment")


class ErrorKind(StrEnum):
    TRANSIENT = "transient"
    BAD_REQUEST = "bad_request"
    CREDIT = "credit"
    AUTH = "auth"
    CONFIG = "config"
    UNKNOWN_4XX = "unknown_4xx"


def classify(status: int | None, message: str = "") -> ErrorKind:
    """status=None nghĩa là lỗi tầng mạng (timeout, connection reset)."""
    msg = (message or "").lower()
    if status is None:
        return ErrorKind.TRANSIENT
    if status == 429 and not _mentions_credit(msg):  # rate limit; 429 kèm "quota" là hết credit
        return ErrorKind.TRANSIENT
    if status == 402 or _mentions_credit(msg):
        return ErrorKind.CREDIT
    if status in (401, 403):
        return ErrorKind.AUTH
    if status == 404:
        return ErrorKind.CONFIG  # sai model / sai base_url
    if status >= 500 or status in (408, 409, 425):
        return ErrorKind.TRANSIENT
    if status in (400, 413, 422):
        return ErrorKind.BAD_REQUEST
    return ErrorKind.UNKNOWN_4XX


def _mentions_credit(msg: str) -> bool:
    return any(k in msg for k in CREDIT_KEYWORDS)


class APIFailure(Exception):
    """Gốc của mọi lỗi API đã phân loại. `source` = teacher | reader | extractor."""

    def __init__(self, kind: ErrorKind, message: str, source: str = "", status: int | None = None):
        super().__init__(f"[{source}] {kind.value} (status={status}): {message}")
        self.kind, self.source, self.status, self.detail = kind, source, status, message


class TransientExhausted(APIFailure):
    """Hết lượt retry cho lỗi tạm thời."""


class FatalAPIError(APIFailure):
    """Hết credit, sai key, sai model: phải dừng toàn bộ."""


class BadRequest(APIFailure):
    """Lỗi do request cụ thể (prompt quá dài, …)."""


class UnknownClientError(APIFailure):
    """4xx không rõ nghĩa."""


class BudgetExceeded(Exception):
    def __init__(self, spent: float, budget: float):
        super().__init__(f"đã chi ${spent:.4f} >= budget ${budget:.2f}")
        self.spent, self.budget = spent, budget


def to_exception(kind: ErrorKind, message: str, source: str, status: int | None) -> APIFailure:
    if kind in (ErrorKind.CREDIT, ErrorKind.AUTH, ErrorKind.CONFIG):
        return FatalAPIError(kind, message, source, status)
    if kind is ErrorKind.BAD_REQUEST:
        return BadRequest(kind, message, source, status)
    if kind is ErrorKind.UNKNOWN_4XX:
        return UnknownClientError(kind, message, source, status)
    return TransientExhausted(kind, message, source, status)
