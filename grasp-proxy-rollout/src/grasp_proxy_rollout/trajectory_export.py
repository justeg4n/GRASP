"""Export complete grouped SFT-policy trajectories for PPO/GRPO handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from grasp_proxy_rollout import SCHEMA_VERSION, TRAJECTORY_SCHEMA_VERSION
from grasp_proxy_rollout.environment import RolloutEnvironment
from grasp_proxy_rollout.model import SFTProxy
from grasp_proxy_rollout.prompts import PROMPT_SCHEMA
from grasp_proxy_rollout.reader import OpenAIReader
from grasp_proxy_rollout.retriever import DenseSearcher
from grasp_proxy_rollout.schema import TRAJECTORY_SCHEMA


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    records = (
        [json.loads(line) for line in text.splitlines() if line.strip()]
        if path.suffix.lower() == ".jsonl"
        else json.loads(text)
    )
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError("QA input must be a JSON array or JSONL objects")
    normalized = []
    seen = set()
    for index, item in enumerate(records):
        if item.get("status") not in (None, "ok"):
            continue
        question_id = str(item.get("id", item.get("qid", index)))
        if question_id in seen:
            raise ValueError(f"Duplicate question id: {question_id}")
        seen.add(question_id)
        answers = item.get("gold_answers", item.get("answers", []))
        if isinstance(answers, str):
            answers = [answers]
        if not str(item.get("question", "")).strip() or not answers:
            raise ValueError(f"Question and gold_answers are required at row {index}")
        source_metadata = {
            key: item[key]
            for key in ("dataset", "split")
            if item.get(key) is not None
        }
        supplied_metadata = item.get("metadata", item.get("meta", {}))
        normalized.append(
            {
                "id": question_id,
                "question": str(item["question"]),
                "gold_answers": [str(value) for value in answers],
                "gold_doc_ids": [str(value) for value in item.get("gold_doc_ids", [])],
                "metadata": {
                    **source_metadata,
                    **(supplied_metadata if isinstance(supplied_metadata, dict) else {}),
                },
            }
        )
    return normalized


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    managed = [
        path / "manifest.json",
        path / "summary.json",
        path / "groups.json",
        path / "trajectory_schema.json",
        *path.glob("trajectories_*.json"),
    ]
    existing = [file for file in managed if file.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output already contains rollout files: {path}; choose a new directory or "
            "pass --overwrite-output-dir"
        )
    for file in existing:
        file.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assign_group_advantages(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(item["reward"]["total"]) for item in trajectories]
    mean = fmean(rewards)
    std = pstdev(rewards)
    advantages = (
        [0.0] * len(rewards)
        if std < 1e-6
        else [(value - mean) / (std + 1e-6) for value in rewards]
    )
    for trajectory, advantage in zip(trajectories, advantages, strict=True):
        trajectory["grpo_advantage"] = advantage
    return {
        "group_id": trajectories[0]["group_id"],
        "question_id": trajectories[0]["question_id"],
        "trajectory_ids": [item["trajectory_id"] for item in trajectories],
        "rewards": rewards,
        "reward_mean": mean,
        "reward_std": std,
        "grpo_advantages": advantages,
        "zero_variance": std < 1e-6,
    }


def _new_stats() -> dict[str, Any]:
    return {
        "trajectories": 0,
        "actions": 0,
        "valid_actions": 0,
        "reward_sum": 0.0,
        "answer_f1_sum": 0.0,
        "recall_sum": 0.0,
        "recall_count": 0,
    }


def _update_stats(stats: dict[str, Any], trajectories: list[dict[str, Any]]) -> None:
    actions = [
        action
        for trajectory in trajectories
        for step in trajectory["steps"]
        for action in step["proxy_actions"]
    ]
    stats["trajectories"] += len(trajectories)
    stats["actions"] += len(actions)
    stats["valid_actions"] += sum(bool(action["valid_format"]) for action in actions)
    for trajectory in trajectories:
        stats["reward_sum"] += float(trajectory["reward"]["total"])
        stats["answer_f1_sum"] += float(trajectory["reward"]["answer_f1"])
        recall = trajectory["reward"]["context_recall"]
        if recall is not None:
            stats["recall_sum"] += float(recall)
            stats["recall_count"] += 1


def _summary(stats: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
    trajectory_count = int(stats["trajectories"])
    action_count = int(stats["actions"])
    return {
        "questions": len(groups),
        "trajectories": trajectory_count,
        "proxy_actions": action_count,
        "valid_action_rate": (
            stats["valid_actions"] / action_count
            if action_count
            else 0.0
        ),
        "mean_reward": stats["reward_sum"] / trajectory_count,
        "mean_answer_f1": stats["answer_f1_sum"] / trajectory_count,
        "mean_context_recall": (
            stats["recall_sum"] / stats["recall_count"] if stats["recall_count"] else None
        ),
        "zero_variance_groups": sum(bool(group["zero_variance"]) for group in groups),
    }


def export_trajectories(args) -> Path:
    import torch

    if args.group_size > 1 and args.proxy_temperature <= 0:
        raise ValueError("group_size > 1 requires proxy_temperature > 0 for policy diversity")
    if args.group_size < 1 or args.groups_per_shard < 1:
        raise ValueError("group_size and groups_per_shard must be positive")
    if min(args.max_steps, args.top_k, args.max_candidate_triples, args.reader_max_tokens) < 1:
        raise ValueError(
            "max_steps, top_k, max_candidate_triples, and reader_max_tokens must be positive"
        )
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.proxy_temperature < 0 or not 0 < args.proxy_top_p <= 1:
        raise ValueError("proxy_temperature must be non-negative and proxy_top_p in (0, 1]")
    if not 0.0 <= args.alpha <= 1.0 or args.retrieval_cost < 0:
        raise ValueError("alpha must be in [0, 1] and retrieval_cost must be non-negative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    qa_path = Path(args.qa).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _prepare_output_dir(output_dir, overwrite=args.overwrite_output_dir)
    examples = _read_json_records(qa_path)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("QA input is empty")

    proxy = SFTProxy.load(args.checkpoint, load_in_4bit=args.load_in_4bit)
    reader = OpenAIReader(
        args.reader_url,
        args.reader_api_key,
        args.reader_model,
        max_tokens=args.reader_max_tokens,
        temperature=args.reader_temperature,
        seed=args.seed,
    )
    searcher = DenseSearcher(
        str(index_dir),
        args.encoder,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
        top_k=args.top_k,
        max_candidate_triples=args.max_candidate_triples,
        triple_cache=args.triple_cache,
        require_triples=args.require_triples,
    )
    environment = RolloutEnvironment(
        proxy,
        reader,
        searcher,
        max_steps=args.max_steps,
        proxy_temperature=args.proxy_temperature,
        proxy_top_p=args.proxy_top_p,
        alpha=args.alpha,
        retrieval_cost=args.retrieval_cost,
        behavior_checkpoint=str(Path(args.checkpoint).expanduser().resolve()),
        initial_retrieval=args.initial_retrieval,
    )

    stats = _new_stats()
    group_records = []
    shard_files = []
    pending = []
    groups_in_pending = 0
    for question_index, example in enumerate(examples):
        group_id = f"{example['id']}::group-0"
        group = [
            environment.run(
                example,
                group_id=group_id,
                trajectory_id=f"{group_id}::sample-{sample_index}",
                sample_index=sample_index,
            )
            for sample_index in range(args.group_size)
        ]
        group_records.append(_assign_group_advantages(group))
        _update_stats(stats, group)
        pending.extend(group)
        groups_in_pending += 1
        if groups_in_pending >= args.groups_per_shard or question_index == len(examples) - 1:
            shard_name = f"trajectories_{len(shard_files):05d}.json"
            _write_json(output_dir / shard_name, pending)
            shard_files.append(shard_name)
            pending = []
            groups_in_pending = 0

    _write_json(output_dir / "groups.json", group_records)
    _write_json(output_dir / "summary.json", _summary(stats, group_records))
    _write_json(output_dir / "trajectory_schema.json", TRAJECTORY_SCHEMA)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_schema": TRAJECTORY_SCHEMA_VERSION,
        "prompt_schema": PROMPT_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "initial on-policy PPO/GRPO rollout from the SFT behavior policy",
        "on_policy_validity": (
            "These old log-probabilities are valid only until the actor receives its first update. "
            "Collect new trajectories from every updated actor."
        ),
        "behavior_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_kind": proxy.checkpoint_kind,
        "base_model": proxy.base_model,
        "qa_file": str(qa_path),
        "qa_sha256": _sha256(qa_path),
        "index_dir": str(index_dir),
        "index_file": str(searcher.index_path),
        "index_sha256": _sha256(searcher.index_path),
        "passage_store": str(searcher.corpus_path),
        "passage_store_sha256": _sha256(searcher.corpus_path),
        "triple_store": str(searcher.triple_path) if searcher.triple_path else None,
        "triple_store_sha256": (
            _sha256(searcher.triple_path) if searcher.triple_path else None
        ),
        "reader": {
            "base_url": args.reader_url,
            "model": args.reader_model,
            "temperature": args.reader_temperature,
        },
        "retriever": {
            "encoder": args.encoder,
            "top_k": args.top_k,
            "max_candidate_triples": args.max_candidate_triples,
        },
        "rollout": {
            "group_size": args.group_size,
            "max_steps": args.max_steps,
            "proxy_temperature": args.proxy_temperature,
            "proxy_top_p": args.proxy_top_p,
            "seed": args.seed,
        },
        "reward": {"alpha": args.alpha, "retrieval_cost": args.retrieval_cost},
        "initial_retrieval": {
            "enabled": args.initial_retrieval,
            "query": "original question",
            "counts_toward_retrieval_cost": False,
        },
        "trajectory_shards": shard_files,
        "groups_file": "groups.json",
        "summary_file": "summary.json",
        "schema_file": "trajectory_schema.json",
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qa", required=True, help="JSON/JSONL questions with gold answers")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/sft_initial_rollouts")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--reader-url", default="http://localhost:8000/v1")
    parser.add_argument("--reader-api-key", default="EMPTY")
    parser.add_argument(
        "--reader-model",
        required=True,
        help="Model name exposed by the frozen OpenAI-compatible reader endpoint",
    )
    parser.add_argument("--reader-max-tokens", type=int, default=512)
    parser.add_argument("--reader-temperature", type=float, default=0.0)
    parser.add_argument("--encoder", default="intfloat/e5-large-v2")
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--passage-prefix", default="passage: ")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-candidate-triples", type=int, default=20)
    parser.add_argument("--triple-cache")
    parser.add_argument("--require-triples", action="store_true")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--groups-per-shard", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--proxy-temperature", type=float, default=0.7)
    parser.add_argument("--proxy-top-p", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--retrieval-cost", type=float, default=0.0)
    parser.add_argument(
        "--initial-retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Seed reader context with question retrieval before proxy actions; enabled by "
            "default to match the released teacher trajectories"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    output = export_trajectories(parse_args())
    print(f"Complete RL handoff written to: {output}")


if __name__ == "__main__":
    main()
