"""Reward metrics kept inside the standalone handoff package."""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_f1(prediction: str, gold_answers: list[str]) -> float:
    def score(gold: str) -> float:
        pred_tokens = normalize_answer(prediction).split()
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            return float(pred_tokens == gold_tokens)
        common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if common == 0:
            return 0.0
        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    return max((score(gold) for gold in gold_answers), default=0.0)


def supporting_document_recall(context_ids: list[str], gold_ids: list[str]) -> float | None:
    if not gold_ids:
        return None
    return len(set(context_ids).intersection(gold_ids)) / len(set(gold_ids))


def trajectory_reward(
    prediction: str,
    gold_answers: list[str],
    context_ids: list[str],
    gold_doc_ids: list[str],
    retrievals: int,
    alpha: float,
    retrieval_cost: float,
) -> dict[str, float | int | None]:
    f1 = answer_f1(prediction, gold_answers)
    recall = supporting_document_recall(context_ids, gold_doc_ids)
    # Do not punish datasets that do not provide supporting-document labels.
    combined = f1 if recall is None else alpha * recall + (1.0 - alpha) * f1
    cost = retrieval_cost * retrievals
    return {
        "answer_f1": f1,
        "context_recall": recall,
        "retrieval_count": retrievals,
        "retrieval_cost": cost,
        "total": combined - cost,
    }
