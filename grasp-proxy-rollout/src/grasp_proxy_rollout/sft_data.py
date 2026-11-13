"""Validation and tokenization for the released chat-style SFT JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grasp_proxy_rollout.parsers import match_candidate, parse_ki_candidates, parse_output
from grasp_proxy_rollout.prompts import ROLES


class SFTValidationError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SFTValidationError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise SFTValidationError(f"{path}:{line_number}: each row must be an object")
            yield line_number, row


def validate_sft_row(row: dict[str, Any], *, location: str = "row") -> None:
    role = str(row.get("role", "")).casefold()
    if role not in ROLES:
        raise SFTValidationError(f"{location}: role must be one of {ROLES}")
    if not str(row.get("qid", "")).strip():
        raise SFTValidationError(f"{location}: qid is required")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise SFTValidationError(f"{location}: messages must contain user and assistant turns")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise SFTValidationError(f"{location}: messages[{index}] must be an object")
        if message.get("role") not in {"system", "user", "assistant"}:
            raise SFTValidationError(f"{location}: messages[{index}].role is invalid")
        if not isinstance(message.get("content"), str):
            raise SFTValidationError(f"{location}: messages[{index}].content must be text")
    if messages[-1]["role"] != "assistant":
        raise SFTValidationError(f"{location}: final message must have role=assistant")
    if not any(message["role"] == "user" for message in messages[:-1]):
        raise SFTValidationError(f"{location}: at least one user message is required")
    target = messages[-1]["content"]
    parsed, valid, error = parse_output(role, target)
    if not valid:
        raise SFTValidationError(f"{location}: invalid {role.upper()} target: {error}")
    if role == "rd" and target not in {
        "Yes (retrieval is required)",
        "No (retrieval is not required)",
    }:
        raise SFTValidationError(f"{location}: RD target is not an exact released label")
    if role == "ki":
        candidates = parse_ki_candidates(messages[-2]["content"])
        if not candidates:
            raise SFTValidationError(f"{location}: KI prompt contains no candidate triples")
        if match_candidate(parsed, candidates) is None:
            raise SFTValidationError(
                f"{location}: KI target is not present in the Candidate Triples block"
            )


def scan_sft_file(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    counts: Counter[str] = Counter()
    qids = set()
    duplicate_qid_roles = 0
    seen_examples = set()
    for line_number, row in iter_jsonl(path):
        validate_sft_row(row, location=f"{path}:{line_number}")
        role = str(row["role"]).casefold()
        counts[role] += 1
        qid = str(row["qid"])
        dataset = str(row.get("dataset", ""))
        qids.add((dataset, qid))
        example_key = (dataset, qid, int(row.get("step", 0)), role)
        duplicate_qid_roles += int(example_key in seen_examples)
        seen_examples.add(example_key)
    if not counts:
        raise SFTValidationError(f"{path}: dataset is empty")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": sum(counts.values()),
        "questions": len(qids),
        "roles": {role: counts[role] for role in ROLES},
        "duplicate_qid_step_roles": duplicate_qid_roles,
        "valid": True,
    }


def _token_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def tokenize_sft_row(
    row: dict[str, Any],
    tokenizer,
    *,
    max_length: int,
    prompt_head_tokens: int,
) -> dict[str, Any]:
    """Tokenize one chat and mask all non-assistant-target tokens from the loss."""
    validate_sft_row(row)
    messages = row["messages"]
    prompt_ids = _token_list(
        tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = _token_list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise SFTValidationError(
            "Tokenizer chat template does not make the generation prompt a prefix of the full chat"
        )
    target_ids = full_ids[len(prompt_ids) :]
    if not target_ids:
        raise SFTValidationError("Assistant target produced no tokens")
    if len(target_ids) >= max_length:
        raise SFTValidationError(
            f"Assistant target has {len(target_ids)} tokens, exceeding max_length={max_length}"
        )

    prompt_budget = max_length - len(target_ids)
    truncated = len(prompt_ids) > prompt_budget
    if truncated:
        head = min(prompt_head_tokens, prompt_budget // 2)
        tail = prompt_budget - head
        prompt_ids = prompt_ids[:head] + (prompt_ids[-tail:] if tail else [])

    input_ids = prompt_ids + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
        "length": len(input_ids),
        "was_truncated": truncated,
    }


def load_tokenized_dataset(
    path: str | Path,
    tokenizer,
    *,
    max_length: int,
    prompt_head_tokens: int,
    max_samples: int | None = None,
    shuffle_before_select: bool = False,
    seed: int = 42,
):
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(path), split="train")
    if max_samples is not None:
        if shuffle_before_select:
            dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    original_columns = dataset.column_names
    dataset = dataset.map(
        lambda row: tokenize_sft_row(
            row,
            tokenizer,
            max_length=max_length,
            prompt_head_tokens=prompt_head_tokens,
        ),
        remove_columns=original_columns,
        desc=f"Tokenizing {Path(path).name}",
    )
    return dataset


@dataclass
class AssistantOnlyCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        model_inputs = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_inputs,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        width = int(batch["input_ids"].shape[1])
        labels = [
            feature["labels"] + [-100] * (width - len(feature["labels"]))
            for feature in features
        ]
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Validate GRASP chat-style SFT JSONL")
    parser.add_argument("paths", nargs="+", help="One or more JSONL files")
    parser.add_argument("--output", help="Optional JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {"files": [scan_sft_file(path) for path in args.paths]}
    report["rows"] = sum(item["rows"] for item in report["files"])
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
