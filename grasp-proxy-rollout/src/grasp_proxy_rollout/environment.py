"""Complete reader -> RD -> QF -> retriever -> KI rollout environment."""
from __future__ import annotations

import re
from typing import Any

from grasp_proxy_rollout import TRAJECTORY_SCHEMA_VERSION
from grasp_proxy_rollout.metrics import trajectory_reward
from grasp_proxy_rollout.parsers import match_candidate, parse_output
from grasp_proxy_rollout.prompts import build_prompt

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
THOUGHT_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
ROLE_MAX_TOKENS = {"rd": 16, "qf": 256, "ki": 128}


def _reader_prompt(
    question: str,
    context: list[dict[str, Any]],
    thoughts: list[str],
    final_only: bool,
) -> str:
    context_text = "\n\n".join(
        f"[{item['doc_id']}] {item.get('title') or ''}\n{item['text']}" for item in context
    ) or "(empty)"
    thought_text = "\n".join(thoughts) or "(none)"
    if final_only:
        instruction = "Return the final short answer now as <answer>...</answer>."
    else:
        instruction = (
            "Return one next thought as <think>...</think>, "
            "or the final answer as <answer>...</answer>."
        )
    return f"""You are the frozen reader in an adaptive RAG system.
Use the supplied context when external evidence is required. {instruction}

Question: {question}
Context:
{context_text}
Reasoning so far:
{thought_text}
"""


def _compact_retrieved(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": item["doc_id"],
            "title": item.get("title"),
            "score": item.get("score"),
        }
        for item in passages
    ]


class RolloutEnvironment:
    def __init__(
        self,
        proxy,
        reader,
        searcher,
        *,
        max_steps: int,
        proxy_temperature: float,
        proxy_top_p: float,
        alpha: float,
        retrieval_cost: float,
        behavior_checkpoint: str,
        initial_retrieval: bool = True,
    ):
        self.proxy = proxy
        self.reader = reader
        self.searcher = searcher
        self.max_steps = max_steps
        self.proxy_temperature = proxy_temperature
        self.proxy_top_p = proxy_top_p
        self.alpha = alpha
        self.retrieval_cost = retrieval_cost
        self.behavior_checkpoint = behavior_checkpoint
        self.initial_retrieval = initial_retrieval

    def _act(self, role: str, state: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(role, state)
        generated = self.proxy.generate_batch(
            [prompt],
            max_new_tokens=ROLE_MAX_TOKENS[role],
            temperature=self.proxy_temperature,
            top_p=self.proxy_top_p,
            samples_per_state=1,
        )[0][0]
        parsed, valid, error = parse_output(role, generated.text)
        return {
            "role": role,
            "observation": state,
            "prompt": prompt,
            "response": generated.text,
            "parsed_action": parsed,
            "valid_format": valid,
            "validation_error": error,
            "old_logprob_sum": float(sum(generated.token_logprobs)),
            "old_logprob_mean": generated.mean_logprob,
            "token_count": generated.token_count,
            "response_token_ids": generated.token_ids,
            "old_token_logprobs": generated.token_logprobs,
            "behavior_policy_checkpoint": self.behavior_checkpoint,
        }

    def run(
        self,
        example: dict[str, Any],
        *,
        group_id: str,
        trajectory_id: str,
        sample_index: int,
    ) -> dict[str, Any]:
        question = str(example["question"])
        gold_answers = [str(value) for value in example["gold_answers"]]
        gold_doc_ids = [str(value) for value in example.get("gold_doc_ids", [])]
        initial_passages = self.searcher.search(question) if self.initial_retrieval else []
        context: list[dict[str, Any]] = []
        seen_doc_ids: set[str] = set()
        for passage in initial_passages:
            doc_id = str(passage["doc_id"])
            if doc_id not in seen_doc_ids:
                context.append(passage)
                seen_doc_ids.add(doc_id)
        initial_context_doc_ids = [item["doc_id"] for item in context]
        thoughts: list[str] = []
        steps = []
        final_answer = ""
        termination = "max_steps"
        final_reader = None

        for step_index in range(self.max_steps):
            reader_prompt = _reader_prompt(question, context, thoughts, final_only=False)
            reader_response = self.reader.generate(reader_prompt)
            answer_match = ANSWER_RE.search(reader_response)
            if answer_match:
                final_answer = answer_match.group(1).strip()
                termination = "reader_answer"
                steps.append(
                    {
                        "step": step_index,
                        "reader_prompt": reader_prompt,
                        "reader_response": reader_response,
                        "reader_thought": None,
                        "context_before_doc_ids": [item["doc_id"] for item in context],
                        "proxy_actions": [],
                        "retrieved": [],
                        "selected_candidate": None,
                        "context_after_doc_ids": [item["doc_id"] for item in context],
                    }
                )
                break

            thought_match = THOUGHT_RE.search(reader_response)
            thought = thought_match.group(1).strip() if thought_match else reader_response.strip()
            thoughts.append(thought)
            context_before = [item["doc_id"] for item in context]
            state = {
                "question": question,
                "context": [
                    {
                        "doc_id": item["doc_id"],
                        "title": item.get("title"),
                        "text": item["text"],
                    }
                    for item in context
                ],
                "thoughts": list(thoughts),
                "current_thought": thought,
            }
            rd_action = self._act("rd", state)
            retrieve = rd_action["parsed_action"]
            if retrieve is None:
                retrieve = True
                rd_action["environment_fallback"] = "invalid RD output; defaulted to retrieval"

            actions = [rd_action]
            retrieved = []
            selected = None
            retrieval_performed = bool(retrieve)
            if retrieve:
                qf_action = self._act("qf", state)
                actions.append(qf_action)
                qf_parsed = qf_action["parsed_action"] if qf_action["valid_format"] else None
                raw_query = qf_parsed.get("query") if qf_parsed else None
                query = str(raw_query).strip() if raw_query else question
                chain = qf_parsed.get("chain", []) if qf_parsed else []
                if not qf_parsed or not raw_query:
                    qf_action["environment_fallback"] = (
                        "invalid or empty QF query; used original question"
                    )
                passages = self.searcher.search(query)
                retrieved = _compact_retrieved(passages)
                candidate_builder = getattr(self.searcher, "candidate_triples", None)
                candidates = (
                    candidate_builder(query, passages) if candidate_builder is not None else []
                )
                if not candidates:
                    for passage in passages:
                        triples = passage.get("triples", []) or [
                            {
                                "head": passage.get("title") or passage["doc_id"],
                                "relation": "contains evidence",
                                "tail": passage["text"][:500],
                            }
                        ]
                        for triple in triples:
                            candidate = dict(triple)
                            candidate["doc_id"] = str(
                                candidate.get("doc_id") or passage["doc_id"]
                            )
                            candidates.append(candidate)

                if candidates:
                    ki_state = {
                        "question": question,
                        "chain": chain,
                        "query": query,
                        "candidates": candidates,
                    }
                    ki_action = self._act("ki", ki_state)
                    actions.append(ki_action)
                    selected_triple = (
                        ki_action["parsed_action"] if ki_action["valid_format"] else None
                    )
                    chosen_index = (
                        match_candidate(selected_triple, candidates)
                        if isinstance(selected_triple, dict)
                        else None
                    )
                    if chosen_index is None:
                        chosen_index = 0
                        ki_action["valid_format"] = False
                        ki_action["validation_error"] = (
                            "KI triple was not copied from the candidate list"
                        )
                        ki_action["environment_fallback"] = "selected candidate 0"
                    ki_action["selected_candidate_index"] = chosen_index
                    selected = candidates[chosen_index]
                    selected_doc = next(
                        (
                            item
                            for item in passages
                            if item["doc_id"] == selected["doc_id"]
                        ),
                        passages[0],
                    )
                    if all(item["doc_id"] != selected_doc["doc_id"] for item in context):
                        context.append(selected_doc)

            steps.append(
                {
                    "step": step_index,
                    "reader_prompt": reader_prompt,
                    "reader_response": reader_response,
                    "reader_thought": thought,
                    "retrieval_performed": retrieval_performed,
                    "context_before_doc_ids": context_before,
                    "proxy_actions": actions,
                    "retrieved": retrieved,
                    "selected_candidate": selected,
                    "context_after_doc_ids": [item["doc_id"] for item in context],
                }
            )

        if not final_answer:
            reader_prompt = _reader_prompt(question, context, thoughts, final_only=True)
            reader_response = self.reader.generate(reader_prompt)
            answer_match = ANSWER_RE.search(reader_response)
            final_answer = (
                answer_match.group(1).strip() if answer_match else reader_response.strip()
            )
            termination = "forced_final_answer"
            final_reader = {"prompt": reader_prompt, "response": reader_response}

        retrieval_count = sum(bool(step.get("retrieval_performed")) for step in steps)
        reward = trajectory_reward(
            final_answer,
            gold_answers,
            [item["doc_id"] for item in context],
            gold_doc_ids,
            retrieval_count,
            self.alpha,
            self.retrieval_cost,
        )
        return {
            "trajectory_schema": TRAJECTORY_SCHEMA_VERSION,
            "trajectory_id": trajectory_id,
            "group_id": group_id,
            "sample_index": sample_index,
            "question_id": str(example["id"]),
            "question": question,
            "gold_answers": gold_answers,
            "gold_doc_ids": gold_doc_ids,
            "initial_retrieval": {
                "enabled": self.initial_retrieval,
                "query": question if self.initial_retrieval else None,
                "retrieved": _compact_retrieved(initial_passages),
                "context_doc_ids": initial_context_doc_ids,
                "counts_toward_retrieval_cost": False,
            },
            "steps": steps,
            "final_context_doc_ids": [item["doc_id"] for item in context],
            "final_answer": final_answer,
            "final_reader": final_reader,
            "termination": termination,
            "reward": reward,
            "grpo_advantage": None,
            "metadata": example.get("metadata", {}),
        }
