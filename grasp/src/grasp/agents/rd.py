"""Retrieval Decision agent: state = (q, context, thoughts) -> Yes / No."""

from __future__ import annotations

import re
from dataclasses import dataclass

from grasp.agents.base import Agent, ParseError

YES = "Yes (retrieval is required)"
NO = "No (retrieval is not required)"
_ANSWER = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)


@dataclass
class RDState:
    question: str
    context: str  # context đã format (rỗng nếu chưa retrieve gì)
    history: list[str]  # các thought trước
    current: str  # thought vừa sinh


class RetrievalDecisionAgent(Agent):
    role = "rd"

    def build_prompt(self, state: RDState) -> str:
        return self.template.format(
            context=state.context or "None",
            question=state.question,
            history="\n".join(state.history) if state.history else "None",
            current=state.current,
        )

    def parse(self, output: str, state=None) -> bool:
        m = _ANSWER.match((output or "").strip())
        if not m:
            raise ParseError("không bắt đầu bằng Yes/No")
        return m.group(1).lower() == "yes"

    def retry_hint(self) -> str:
        return (
            "Answer with exactly 'Yes (retrieval is required)' or 'No (retrieval is not required)'."
        )

    def canonical(self, parsed: bool, state=None) -> str:
        return YES if parsed else NO
