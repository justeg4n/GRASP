"""Câu hỏi train (bản jsonl của FlashRAG: {id, question, golden_answers}) và split cố định theo seed.

Split chỉ lưu danh sách qid (`splits/{dataset}.json`, commit vào git) để mọi người dùng đúng cùng tập câu.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SPLIT_ORDER = ("sft_train", "sft_val", "rl_pool")


def load_questions(path: str | Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[str(d["id"])] = {
                    "qid": str(d["id"]),
                    "question": d["question"],
                    "golden_answers": d["golden_answers"],
                }
    return out


def make_splits(qids: list[str], sizes: dict[str, int], seed: int) -> dict[str, list[str]]:
    """Lấy mẫu không lặp, các split không giao nhau, tái lập được theo seed."""
    total = sum(sizes.values())
    pool = sorted(qids)
    if total > len(pool):
        raise ValueError(f"cần {total} câu nhưng dataset chỉ có {len(pool)}")
    picked = random.Random(seed).sample(pool, total)
    out, i = {}, 0
    for name in [s for s in SPLIT_ORDER if s in sizes] + [s for s in sizes if s not in SPLIT_ORDER]:
        out[name] = picked[i : i + sizes[name]]
        i += sizes[name]
    return out


def write_splits(
    path: str | Path, dataset: str, source: str, seed: int, splits: dict[str, list[str]]
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "source": source,
        "seed": seed,
        "sizes": {k: len(v) for k, v in splits.items()},
        "splits": splits,
    }
    Path(path).write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_splits(path: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["splits"]


def select(
    questions: dict[str, dict], splits: dict[str, list[str]], names: list[str], dataset: str
) -> list[dict]:
    out = []
    for name in names:
        for qid in splits[name]:
            out.append({**questions[qid], "dataset": dataset, "split": name})
    return out
