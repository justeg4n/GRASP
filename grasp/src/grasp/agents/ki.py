"""Knowledge Integration agent: chọn 1 triple lấp khoảng trống suy luận hiện tại."""

from __future__ import annotations

import re
from dataclasses import dataclass

from grasp.agents.base import Agent, ParseError
from grasp.kg.triples import format_chain, parse_triples, triple_key
from grasp.types import Triple

_INDEX = re.compile(r"^\W*\[?\s*(\d+)\s*\]")


@dataclass
class KIState:
    question: str
    chain: list[Triple]
    query: str
    candidates: list[Triple]


class KnowledgeIntegrationAgent(Agent):
    role = "ki"

    def build_prompt(self, state: KIState) -> str:
        cands = "\n".join(f"[{i + 1}] {t.text()}" for i, t in enumerate(state.candidates))
        return self.template.format(
            candidates=cands,
            question=state.question,
            chain=format_chain(state.chain),
            query=state.query,
        )

    def retry_hint(self) -> str:
        return (
            "It is not one of the candidate triples. Copy exactly one triple from the Candidate Triples "
            "list, character for character, or give its number such as [3]. Do not rephrase or infer "
            "new triples."
        )

    def parse(self, output: str, state: KIState) -> Triple:
        """Trả về Triple được chọn (tài liệu nguồn của nó sẽ được nối vào context)."""
        text = (output or "").strip()
        found = parse_triples(text)
        if found:
            key = triple_key(found[0])
            for c in state.candidates:
                if triple_key(c) == key:
                    return c
        m = _INDEX.match(text)
        if m and 1 <= int(m.group(1)) <= len(state.candidates):
            return state.candidates[int(m.group(1)) - 1]
        if found and (near := _nearest(found[0], state.candidates)) is not None:
            return near
        raise ParseError("triple không nằm trong danh sách candidate")

    def canonical(self, parsed: Triple, state=None) -> str:
        return parsed.text()


def _stems(s: str) -> set[str]:
    return {w[:5] for w in re.findall(r"\w+", s.lower()) if len(w) > 2}


def _nearest(t: Triple, candidates: list[Triple]) -> Triple | None:
    """Model viết lại triple (vd. 'Irish' thay vì 'Irish film director'): chỉ nhận khi cùng head và
    relation/tail có từ chung, đồng thời phải là ứng viên tốt nhất DUY NHẤT."""
    head = triple_key(Triple(t.head, "", "")).split(" | ")[0]
    want = _stems(f"{t.relation} {t.tail}")
    scored = []
    for c in candidates:
        if triple_key(Triple(c.head, "", "")).split(" | ")[0] != head:
            continue
        overlap = len(want & _stems(f"{c.relation} {c.tail}"))
        if overlap:
            scored.append((overlap, c))
    scored.sort(key=lambda x: -x[0])
    if not scored or (len(scored) > 1 and scored[0][0] == scored[1][0]):
        return None
    return scored[0][1]
