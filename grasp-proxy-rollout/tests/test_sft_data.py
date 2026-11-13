import json

import pytest

from grasp_proxy_rollout.sft_data import (
    SFTValidationError,
    scan_sft_file,
    tokenize_sft_row,
    validate_sft_row,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = "<system>helpful</system>"
        for message in messages:
            text += f"<{message['role']}>{message['content']}</{message['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode()) if tokenize else text


def _row(role="rd", target="Yes (retrieval is required)"):
    prompt = "A sufficiently long user prompt."
    if role == "ki":
        prompt = "Candidate Triples:\n[1] <head; relation; tail>\n\nQuestion: test?"
    return {
        "qid": "q1",
        "dataset": "test",
        "split": "train",
        "role": role,
        "step": 0,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
    }


def test_sft_validation_and_scan(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        _row(),
        _row(
            "qf",
            "Reasoning Chain: None\nKnowledge Gap: missing fact\nQuery: target query",
        ),
        _row("ki", "<head; relation; tail>"),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    report = scan_sft_file(path)
    assert report["rows"] == 3
    assert report["roles"] == {"rd": 1, "qf": 1, "ki": 1}
    assert report["valid"] is True


def test_invalid_sft_target_is_rejected():
    with pytest.raises(SFTValidationError, match="invalid RD target"):
        validate_sft_row(_row(target="Maybe"))


def test_ki_target_must_come_from_candidate_block():
    with pytest.raises(SFTValidationError, match="not present"):
        validate_sft_row(_row("ki", "<different; relation; triple>"))


def test_tokenization_masks_prompt_and_preserves_target_when_truncated():
    tokenizer = FakeTokenizer()
    row = _row()
    encoded = tokenize_sft_row(
        row,
        tokenizer,
        max_length=64,
        prompt_head_tokens=12,
    )
    target_ids = list(b"Yes (retrieval is required)</assistant>")
    assert encoded["was_truncated"] is True
    assert encoded["input_ids"][-len(target_ids) :] == target_ids
    assert encoded["labels"][-len(target_ids) :] == target_ids
    assert all(label == -100 for label in encoded["labels"][: -len(target_ids)])
