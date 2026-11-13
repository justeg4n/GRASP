"""Parse and validate the text contracts used by the SFT dataset."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

TRIPLE_RE = re.compile(r"<\s*([^;<>]+?)\s*;\s*([^;<>]+?)\s*;\s*([^<>]+?)\s*>")
QF_RE = re.compile(
    r"^\s*Reasoning\s+Chain\s*:\s*(.*?)\s*\n"
    r"\s*Knowledge\s+Gap\s*:\s*(.*?)\s*\n"
    r"\s*Query\s*:\s*(.*?)\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
GAP_PREFIX_RE = re.compile(
    r"^To\s+continue\s+answering\s+the\s+question,?\s*"
    r"we\s+need\s+to\s+know\s*:\s*",
    flags=re.IGNORECASE,
)


def parse_triples(text: str) -> list[dict[str, str]]:
    return [
        {
            "head": match.group(1).strip(),
            "relation": match.group(2).strip(),
            "tail": match.group(3).strip(),
        }
        for match in TRIPLE_RE.finditer(text)
    ]


def parse_ki_candidates(prompt: str) -> list[dict[str, str]]:
    """Extract only the numbered KI candidate block, excluding instruction examples."""
    marker = "Candidate Triples:"
    if marker not in prompt:
        return []
    section = prompt.split(marker, 1)[1].split("\n\nQuestion:", 1)[0]
    return parse_triples(section)


def _normalized(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def match_candidate(selected: dict[str, Any], candidates: list[dict[str, Any]]) -> int | None:
    """Return the first zero-based candidate index matching a generated triple."""
    wanted = tuple(_normalized(selected.get(key)) for key in ("head", "relation", "tail"))
    for index, candidate in enumerate(candidates):
        actual = tuple(_normalized(candidate.get(key)) for key in ("head", "relation", "tail"))
        if actual == wanted:
            return index
    return None


def parse_output(role: str, text: str) -> tuple[Any, bool, str | None]:
    cleaned = text.strip()
    if role == "rd":
        match = re.fullmatch(
            r"(Yes|No)(?:\s*\([^\r\n()]+\))?\s*[.!]?",
            cleaned,
            flags=re.IGNORECASE,
        )
        if not match:
            return None, False, "RD output must start with Yes or No and contain no explanation"
        return match.group(1).casefold() == "yes", True, None

    if role == "qf":
        match = QF_RE.fullmatch(cleaned)
        if not match:
            return None, False, "QF output must contain Reasoning Chain, Knowledge Gap, and Query"
        chain_text, gap_text, query_text = (part.strip() for part in match.groups())
        chain = [] if chain_text.casefold() == "none" else parse_triples(chain_text)
        if chain_text.casefold() != "none" and not chain:
            return None, False, "QF reasoning chain contains no valid <head; relation; tail> triple"
        gap = GAP_PREFIX_RE.sub("", gap_text).strip()
        query = None if query_text.casefold() == "none" else query_text
        if not gap:
            return None, False, "QF knowledge gap is empty"
        return {"chain": chain, "knowledge_gap": gap, "query": query}, True, None

    if role == "ki":
        triples = parse_triples(cleaned)
        if len(triples) != 1 or TRIPLE_RE.fullmatch(cleaned) is None:
            return None, False, "KI output must be exactly one <head; relation; tail> triple"
        return triples[0], True, None

    raise ValueError(f"Unsupported role: {role}")
