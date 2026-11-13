"""Smoke-test a trained proxy checkpoint against held-out SFT conversations."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from grasp_proxy_rollout.model import SFTProxy
from grasp_proxy_rollout.parsers import match_candidate, parse_ki_candidates, parse_output
from grasp_proxy_rollout.prompts import ROLES
from grasp_proxy_rollout.sft_data import iter_jsonl, validate_sft_row

ROLE_MAX_TOKENS = {"rd": 16, "qf": 256, "ki": 128}


def _sample_by_role(path: str, per_role: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    samples: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    seen = defaultdict(int)
    for line_number, row in iter_jsonl(path):
        validate_sft_row(row, location=f"{path}:{line_number}")
        role = str(row["role"]).casefold()
        seen[role] += 1
        if len(samples[role]) < per_role:
            samples[role].append(row)
            continue
        replacement = rng.randrange(seen[role])
        if replacement < per_role:
            samples[role][replacement] = row
    missing = [role for role, rows in samples.items() if not rows]
    if missing:
        raise ValueError(f"No validation examples found for roles: {missing}")
    return samples


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def validate_checkpoint(args) -> dict[str, Any]:
    if args.samples_per_role < 1:
        raise ValueError("samples_per_role must be positive")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.max_sequence_length <= max(ROLE_MAX_TOKENS.values()):
        raise ValueError("max_sequence_length is too small for the generated output budget")
    if args.prompt_head_tokens < 0:
        raise ValueError("prompt_head_tokens must be non-negative")
    if not 0 <= args.minimum_format_rate <= 1:
        raise ValueError("minimum_format_rate must be in [0, 1]")
    samples = _sample_by_role(args.data_file, args.samples_per_role, args.seed)
    proxy = SFTProxy.load(args.checkpoint, load_in_4bit=args.load_in_4bit)
    records = []
    for role in ROLES:
        rows = samples[role]
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            prompts = [row["messages"][-2]["content"] for row in batch_rows]
            batches = proxy.generate_batch(
                prompts,
                max_new_tokens=ROLE_MAX_TOKENS[role],
                temperature=0.0,
                top_p=1.0,
                samples_per_state=1,
                max_input_tokens=args.max_sequence_length - ROLE_MAX_TOKENS[role],
                prompt_head_tokens=args.prompt_head_tokens,
            )
            for row, candidates in zip(batch_rows, batches, strict=True):
                generated = candidates[0].text
                parsed, valid, error = parse_output(role, generated)
                if role == "ki" and valid:
                    prompt_candidates = parse_ki_candidates(row["messages"][-2]["content"])
                    if match_candidate(parsed, prompt_candidates) is None:
                        valid = False
                        error = "KI output is not present in Candidate Triples"
                target = row["messages"][-1]["content"]
                records.append(
                    {
                        "qid": str(row["qid"]),
                        "role": role,
                        "step": int(row.get("step", 0)),
                        "target": target,
                        "generated": generated,
                        "valid_format": valid,
                        "validation_error": error,
                        "exact_match": _normalize(generated) == _normalize(target),
                        "token_count": candidates[0].token_count,
                        "mean_logprob": candidates[0].mean_logprob,
                    }
                )

    summary = {}
    for role in ROLES:
        role_records = [record for record in records if record["role"] == role]
        summary[role] = {
            "examples": len(role_records),
            "format_rate": sum(record["valid_format"] for record in role_records)
            / len(role_records),
            "exact_match_rate": sum(record["exact_match"] for record in role_records)
            / len(role_records),
        }
    minimum = min(item["format_rate"] for item in summary.values())
    passed = minimum >= args.minimum_format_rate
    report = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_kind": proxy.checkpoint_kind,
        "base_model": proxy.base_model,
        "validation_file": str(Path(args.data_file).expanduser().resolve()),
        "minimum_format_rate_required": args.minimum_format_rate,
        "minimum_format_rate_observed": minimum,
        "passed": passed,
        "batch_size": args.batch_size,
        "max_sequence_length": args.max_sequence_length,
        "prompt_head_tokens": args.prompt_head_tokens,
        "summary": summary,
        "records": records,
    }
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not passed:
        raise RuntimeError(
            f"Checkpoint format validation failed: minimum role rate {minimum:.3f} "
            f"< required {args.minimum_format_rate:.3f}"
        )
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Validate an SFT proxy checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-file", required=True, help="Held-out SFT JSONL")
    parser.add_argument("--samples-per-role", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=3072)
    parser.add_argument("--prompt-head-tokens", type=int, default=768)
    parser.add_argument("--minimum-format-rate", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/checkpoint_validation.json")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = validate_checkpoint(parse_args())
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
