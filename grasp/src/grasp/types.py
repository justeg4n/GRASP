"""Kiểu dữ liệu dùng chung cho toàn pipeline. Thay đổi ở đây ảnh hưởng mọi module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["rd", "qf", "ki"]


@dataclass
class Triple:
    head: str
    relation: str
    tail: str
    doc_id: str | None = None  # tài liệu nguồn, dùng để nối vào context

    def text(self) -> str:
        """Dạng chuẩn dùng trong prompt và làm completion SFT: <head; relation; tail>."""
        return f"<{self.head}; {self.relation}; {self.tail}>"

    def plain(self) -> str:
        """Dạng câu phẳng để encode khi rank triple."""
        return f"{self.head} {self.relation} {self.tail}"


@dataclass
class Passage:
    doc_id: str
    text: str
    title: str | None = None
    score: float | None = None


@dataclass
class AgentCall:
    """Một lượt gọi agent: chính là 1 mẫu SFT (prompt -> completion) nếu trajectory đạt."""

    role: Role
    step: int
    prompt: str
    completion: str  # output đã chuẩn hoá về format canonical
    raw: str  # output gốc của teacher
    retried: bool = False  # đã phải retry vì parse lỗi


@dataclass
class Step:
    """Một bước suy luận: reader sinh thought -> RD -> (QF -> retrieve -> KI)."""

    t: int
    thought: str = ""
    retrieve: bool | None = None  # action của RD agent
    query: str | None = None  # output của QF agent
    gap: str | None = None  # knowledge gap QF nêu ra
    chain: list[Triple] = field(default_factory=list)  # KG reasoning chain (QF)
    retrieved: list[Passage] = field(default_factory=list)
    candidates: list[Triple] = field(default_factory=list)
    selected: Triple | None = None  # triple KI agent chọn
    skipped: str | None = None  # lý do bỏ qua retrieval (query None, query trùng, …)
    logprobs: dict[str, float] = field(default_factory=dict)  # theo role: rd/qf/ki


@dataclass
class Trajectory:
    question: str
    gold_answers: list[str]
    qid: str = ""
    dataset: str = ""
    split: str = ""
    steps: list[Step] = field(default_factory=list)
    calls: list[AgentCall] = field(default_factory=list)
    context: list[Passage] = field(default_factory=list)
    reader_output: str = ""
    answer: str | None = None
    em: float | None = None
    f1: float | None = None
    status: str = "ok"  # ok | parse_error | bad_request | abandoned
    error: str | None = None
    reward: float | None = None
    usage: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    parent_id: str | None = None  # phục vụ group rollout (cây nhánh)
    node_id: str | None = None
