"""Metric chuẩn của literature — giữ nguyên cách normalize để so được với bài báo."""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD style: lower, bỏ dấu câu, bỏ mạo từ, gộp khoảng trắng."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def _f1_single(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    # yes/no/noanswer phải khớp tuyệt đối (giống script eval HotpotQA)
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        return float(p == g)
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pt), same / len(gt)
    return 2 * precision * recall / (precision + recall)


def f1(pred: str, golds: list[str]) -> float:
    return max((_f1_single(pred, g) for g in golds), default=0.0)


def context_recall(context_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    if not gold_doc_ids:
        return 0.0
    ctx = set(context_doc_ids)
    return sum(d in ctx for d in gold_doc_ids) / len(gold_doc_ids)
