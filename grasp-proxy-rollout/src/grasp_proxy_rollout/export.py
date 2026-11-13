"""Export deterministic or sampled SFT-proxy actions to role-separated JSON files."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grasp_proxy_rollout import SCHEMA_VERSION
from grasp_proxy_rollout.model import SFTProxy
from grasp_proxy_rollout.parsers import parse_output
from grasp_proxy_rollout.prompts import PROMPT_SCHEMA, ROLES, prompt_from_record

ROLE_MAX_TOKENS = {"rd": 16, "qf": 256, "ki": 128}


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = json.loads(text)
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError("Input must be a JSON array or JSONL objects")
    return records


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(args) -> Path:
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _read_records(input_path)
    prepared = []
    for source_index, record in enumerate(records):
        role = str(record.get("role", "")).lower()
        if role not in ROLES:
            raise ValueError(f"Record {source_index} has invalid role: {role}")
        prepared.append(
            {
                "source_index": source_index,
                "source_id": str(record.get("id", source_index)),
                "role": role,
                "prompt": prompt_from_record(record),
                "metadata": record.get("metadata", {}),
            }
        )

    proxy = SFTProxy.load(args.checkpoint, load_in_4bit=args.load_in_4bit)
    outputs: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    for role in ROLES:
        role_items = [item for item in prepared if item["role"] == role]
        for start in range(0, len(role_items), args.batch_size):
            batch = role_items[start : start + args.batch_size]
            generations = proxy.generate_batch(
                [item["prompt"] for item in batch],
                max_new_tokens=ROLE_MAX_TOKENS[role],
                temperature=args.temperature,
                top_p=args.top_p,
                samples_per_state=args.samples_per_state,
            )
            for item, candidates in zip(batch, generations, strict=True):
                for sample_index, candidate in enumerate(candidates):
                    parsed, valid, error = parse_output(role, candidate.text)
                    outputs[role].append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            **item,
                            "sample_index": sample_index,
                            "raw_output": candidate.text,
                            "parsed_action": parsed,
                            "valid_format": valid,
                            "validation_error": error,
                            "mean_logprob": candidate.mean_logprob,
                            "token_count": candidate.token_count,
                            "token_ids": candidate.token_ids,
                            "token_logprobs": candidate.token_logprobs,
                        }
                    )

    file_names = {role: f"{role}_outputs.json" for role in ROLES}
    for role, file_name in file_names.items():
        _write_json(output_dir / file_name, outputs[role])
    counts = Counter(item["role"] for item in prepared)
    summary = {
        "input_states": len(prepared),
        "generated_actions": sum(len(items) for items in outputs.values()),
        "counts_by_role": dict(counts),
        "valid_by_role": {
            role: sum(bool(item["valid_format"]) for item in items)
            for role, items in outputs.items()
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "prompt_schema": PROMPT_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_kind": proxy.checkpoint_kind,
        "base_model": proxy.base_model,
        "input_file": str(input_path),
        "input_sha256": _fingerprint(input_path),
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "samples_per_state": args.samples_per_state,
        "files": file_names,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="JSON array or JSONL state records")
    parser.add_argument("--output-dir", default="outputs/action_audit")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples-per-state", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    output_dir = export(parse_args())
    print(f"SFT proxy JSON exported to: {output_dir}")


if __name__ == "__main__":
    main()
