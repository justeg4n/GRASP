"""Parse / so khớp knowledge triple dạng <head; relation; tail>."""

from __future__ import annotations

import re

from grasp.types import Triple

_TRIPLE = re.compile(r"<\s*([^<>;]+?)\s*;\s*([^<>;]+?)\s*;\s*([^<>]+?)\s*>")


def parse_triples(text: str, doc_id: str | None = None) -> list[Triple]:
    return [
        Triple(h.strip(), r.strip(), t.strip(), doc_id) for h, r, t in _TRIPLE.findall(text or "")
    ]


def triple_key(t: Triple) -> str:
    """Khoá so khớp không phân biệt hoa thường / khoảng trắng / dấu câu cuối."""
    return " | ".join(
        " ".join(x.lower().strip(" .\"'").split()) for x in (t.head, t.relation, t.tail)
    )


def format_chain(chain: list[Triple]) -> str:
    return ", ".join(t.text() for t in chain) if chain else "None"
