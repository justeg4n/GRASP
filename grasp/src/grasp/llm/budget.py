"""Đếm token + chi phí, dùng chung cho mọi client gọi API trả phí (teacher, extractor)."""

from __future__ import annotations

from dataclasses import dataclass, field

from grasp.llm.errors import BudgetExceeded


@dataclass
class Price:
    input_per_m: float = 0.0  # USD / 1M token input
    output_per_m: float = 0.0  # USD / 1M token output

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.input_per_m + tokens_out * self.output_per_m) / 1e6


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    calls: int = 0

    def add(self, other: Usage) -> None:
        self.tokens_in += other.tokens_in
        self.tokens_out += other.tokens_out
        self.usd += other.usd
        self.calls += other.calls

    def as_dict(self) -> dict:
        return {
            "in": self.tokens_in,
            "out": self.tokens_out,
            "usd": round(self.usd, 6),
            "calls": self.calls,
        }


@dataclass
class CostMeter:
    """Budget cứng: `check()` raise BudgetExceeded khi tổng chi >= budget_usd.

    `spent_before` là chi phí của các lần chạy trước (đọc từ output đã có) để budget tính trên cả dataset.
    """

    budget_usd: float | None = None
    spent_before: float = 0.0
    total: Usage = field(default_factory=Usage)

    @property
    def spent(self) -> float:
        return self.spent_before + self.total.usd

    def record(self, price: Price, tokens_in: int, tokens_out: int) -> Usage:
        u = Usage(tokens_in, tokens_out, price.cost(tokens_in, tokens_out), 1)
        self.total.add(u)
        return u

    def check(self) -> None:
        if self.budget_usd is not None and self.spent >= self.budget_usd:
            raise BudgetExceeded(self.spent, self.budget_usd)
