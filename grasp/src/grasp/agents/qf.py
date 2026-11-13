"""Query Formulation agent: dựng KG reasoning chain từ thought rồi sinh query có mục tiêu."""

from __future__ import annotations

import re
from dataclasses import dataclass

from grasp.agents.base import Agent, ParseError
from grasp.kg.triples import format_chain, parse_triples
from grasp.types import Triple

GAP_PREFIX = "To continue answering the question, we need to know:"
_SECTION = re.compile(
    r"^\s*(?:\**\s*)?(?:step\s*\d\s*[:.\-]\s*)?(reasoning chain|knowledge gaps?|retrieval query|query)\s*\**\s*:\s*\**\s*(.*)$",
    re.IGNORECASE,
)
_NONE = {"none", "n/a", "na", "null", "no query", "-", ""}


@dataclass
class QFState:
    question: str
    thoughts: list[str]


@dataclass
class QFAction:
    chain: list[Triple]
    gap: str
    query: str | None  # None = không cần thêm thông tin


class QueryFormulationAgent(Agent):
    role = "qf"

    def build_prompt(self, state: QFState) -> str:
        return self.template.format(question=state.question, thoughts="\n".join(state.thoughts))

    def parse(self, output: str, state=None) -> QFAction:
        sections: dict[str, list[str]] = {}
        current = None
        for line in (output or "").splitlines():
            m = _SECTION.match(line)
            if m:
                name = m.group(1).lower()
                current = (
                    "chain"
                    if name.startswith("reasoning")
                    else "gap"
                    if name.startswith("knowledge")
                    else "query"
                )
                sections[current] = [m.group(2)]
            elif current is not None:
                sections[current].append(line)
        if "query" not in sections:
            raise ParseError("thiếu dòng Query")
        chain = parse_triples("\n".join(sections.get("chain", [])))
        gap = " ".join(" ".join(sections.get("gap", [])).split())
        if gap.lower().startswith(GAP_PREFIX.lower()):
            gap = gap[len(GAP_PREFIX) :].strip()
        query = " ".join(" ".join(sections["query"]).split()).strip().strip('"').strip()
        if query.lower().rstrip(".") in _NONE:
            query = None
        return QFAction(chain=chain, gap=gap, query=query)

    def retry_hint(self) -> str:
        return (
            "Use exactly three lines: 'Reasoning Chain: ...', 'Knowledge Gap: To continue answering the "
            "question, we need to know: ...' and 'Query: ...'."
        )

    def canonical(self, parsed: QFAction, state=None) -> str:
        return (
            f"Reasoning Chain: {format_chain(parsed.chain)}\n"
            f"Knowledge Gap: {GAP_PREFIX} {parsed.gap or 'nothing, the reasoning is sufficient'}\n"
            f"Query: {parsed.query or 'None'}"
        )
