"""Prompt contract used by the released GRASP SFT JSONL data."""

# The literal lines below intentionally mirror the training data byte-for-byte.
# ruff: noqa: E501

from __future__ import annotations

import json
from typing import Any

ROLES = ("rd", "qf", "ki")
PROMPT_SCHEMA = "grasp-sft-text-v1"

RD_INSTRUCTION = """You are a retrieval decision assistant. A reasoning model is answering a question step by step. Decide whether its latest reasoning step needs external information that must be retrieved before reasoning can safely continue.

Answer "Yes" when the current step needs a specific fact (entity, date, number, location, title, relation) that is not already stated, even in paraphrase, in the Retrieved Context or the Previous Reasoning; when it introduces a new entity whose facts are not yet supported; or when it expresses uncertainty ("I think", "probably", "not sure", "unknown") about knowledge that is missing from the context.

Answer "No" when the needed information is already in the Retrieved Context or can be directly inferred from it; when the step only restates the question, summarises, or draws a conclusion from facts already present; or when it relies only on general knowledge, definitions, or simple arithmetic. Uncertain wording alone is not a reason to retrieve if the fact is already in the context.

Output exactly one of the following and nothing else:
Yes (retrieval is required)
No (retrieval is not required)"""

QF_INSTRUCTION = """You are a query formulation assistant. A reasoning model is answering a question step by step. Help it close the gap between its current reasoning and the final answer by writing one focused retrieval query. Work in three steps.

Step 1 - Reasoning Chain: from the thoughts, extract only the facts that directly help answer the question, as knowledge triples <head; relation; tail>. Use only facts explicitly stated in the thoughts; skip speculative, uncertain or irrelevant content. Write "None" if there are no such facts.
Step 2 - Knowledge Gap: state the single next piece of information needed to move the reasoning toward the answer.
Step 3 - Query: write a concise search query for that information, containing the key entities and the relation. If the reasoning is already sufficient to answer the question, write "None".

Output exactly three lines in this format:
Reasoning Chain: <head1; relation1; tail1>, <head2; relation2; tail2>
Knowledge Gap: To continue answering the question, we need to know: [the missing information]
Query: [the search query, or None]"""

KI_INSTRUCTION = """You are a knowledge integration assistant. A reasoning model is answering a question step by step. From the candidate knowledge triples, select the single triple that is most useful for continuing the reasoning toward the answer.

The selected triple should logically extend the current reasoning chain and help bridge the gap to the final answer. Use the query as a soft hint about the current information need, but it may be noisy; prefer coherence with the reasoning chain and relevance to the question. Ignore unrelated, redundant or off-topic candidates.

Output only the selected triple, copied exactly from the candidate list in the form <head; relation; tail>, and nothing else."""


def _as_lines(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value) or "None"
    text = str(value or "").strip()
    return text or "None"


def _format_context(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "None"
    passages = []
    for item in value:
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("doc_id") or "Untitled")
            text = str(item.get("text") or "")
            passages.append(f"Title: {title}\nText: {text}")
        else:
            passages.append(f"Text: {item}")
    return "\n\n".join(passages)


def format_triple(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return str(value)
    return "<{head}; {relation}; {tail}>".format(
        head=value.get("head", ""),
        relation=value.get("relation", ""),
        tail=value.get("tail", ""),
    )


def build_prompt(role: str, state: dict[str, Any]) -> str:
    """Build the exact user-message contract represented in the SFT dataset."""
    if role not in ROLES:
        raise ValueError(f"Unsupported role: {role}")
    question = str(state.get("question", "")).strip()
    if not question:
        raise ValueError("state.question is required")

    thoughts = [str(item) for item in state.get("thoughts", []) if str(item).strip()]
    if role == "rd":
        current = str(state.get("current_thought") or (thoughts[-1] if thoughts else "")).strip()
        if not current:
            raise ValueError("RD state requires current_thought or a non-empty thoughts list")
        previous_value = state.get("previous_reasoning")
        previous = (
            _as_lines(previous_value)
            if previous_value is not None
            else _as_lines(thoughts[:-1])
        )
        return f"""{RD_INSTRUCTION}

---
Retrieved Context:
{_format_context(state.get("context"))}

Question:
{question}

Previous Reasoning:
{previous}

Current Reasoning Step:
{current}

Output:
"""

    if role == "qf":
        return f"""{QF_INSTRUCTION}

---
Question: {question}

Thoughts:
{_as_lines(thoughts)}
"""

    chain = ", ".join(format_triple(item) for item in state.get("chain", [])) or "None"
    candidates = "\n".join(
        f"[{index}] {format_triple(item)}"
        for index, item in enumerate(state.get("candidates", []), start=1)
    )
    if not candidates:
        raise ValueError("KI state.candidates must contain at least one candidate")
    query = str(state.get("query") or "None")
    return f"""{KI_INSTRUCTION}

---
Candidate Triples:
{candidates}

Question: {question}

Reasoning Chain: {chain}

Query: {query}

Selected Triple:
"""


def prompt_from_record(record: dict[str, Any]) -> str:
    direct = str(record.get("prompt", "")).strip()
    if direct:
        return direct
    state = record.get("state")
    if not isinstance(state, dict):
        raise ValueError("Each record needs either prompt or state")
    return build_prompt(str(record.get("role", "")), state)


def canonical_state(record: dict[str, Any]) -> str:
    """Stable representation included in outputs for auditing."""
    return json.dumps(record.get("state", {}), ensure_ascii=False, sort_keys=True)
