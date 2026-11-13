"""Rank triple bằng E5 theo (question + query), giữ top-k làm candidate cho KI (SPARKLE: top-20)."""

from __future__ import annotations

from grasp.kg.triples import triple_key
from grasp.types import Triple


def rank_triples(
    encoder, question: str, query: str, triples: list[Triple], top_k: int = 20
) -> list[tuple[Triple, float]]:
    uniq, seen = [], set()
    for t in triples:
        k = triple_key(t)
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    if not uniq:
        return []
    q = encoder.encode_queries([f"{question} {query}".strip()])[0]
    p = encoder.encode_passages([t.plain() for t in uniq])
    scores = p @ q
    order = sorted(range(len(uniq)), key=lambda i: -float(scores[i]))[:top_k]
    return [(uniq[i], float(scores[i])) for i in order]
