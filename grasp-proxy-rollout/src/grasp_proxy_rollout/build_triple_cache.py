"""Build a reusable SQLite triple cache from teacher trajectory JSONL files."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grasp_proxy_rollout.sft_data import iter_jsonl, sha256_file


def _candidate_rows(trajectory: dict[str, Any]):
    for step in trajectory.get("steps", []):
        for candidate in step.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            values = tuple(
                str(candidate.get(key) or "").strip()
                for key in ("doc_id", "head", "relation", "tail")
            )
            if all(values):
                yield values


def build_triple_cache(
    trajectory_paths: list[str],
    output_path: str,
    *,
    include_parse_errors: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    auxiliary = [output.with_name(f"{output.name}{suffix}") for suffix in ("-wal", "-shm")]
    if output.exists() or any(path.exists() for path in auxiliary):
        if not overwrite:
            raise FileExistsError(
                f"Triple cache already exists: {output}; use --overwrite to rebuild it"
            )
        for path in [output, *auxiliary]:
            if path.exists():
                path.unlink()
    connection = sqlite3.connect(output)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS triples (
            doc_id TEXT NOT NULL,
            head TEXT NOT NULL,
            relation TEXT NOT NULL,
            tail TEXT NOT NULL,
            PRIMARY KEY (doc_id, head, relation, tail)
        ) WITHOUT ROWID"""
    )
    connection.execute("CREATE INDEX IF NOT EXISTS triples_doc_id ON triples(doc_id)")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS sources (
            path TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL,
            processed_at_utc TEXT NOT NULL
        )"""
    )

    files = []
    trajectories = 0
    skipped = 0
    candidates_seen = 0
    pending = []
    try:
        for source in trajectory_paths:
            path = Path(source).expanduser().resolve()
            digest = sha256_file(path)
            for _, trajectory in iter_jsonl(path):
                if not include_parse_errors and trajectory.get("status") not in (None, "ok"):
                    skipped += 1
                    continue
                trajectories += 1
                rows = list(_candidate_rows(trajectory))
                candidates_seen += len(rows)
                pending.extend(rows)
                if len(pending) >= 10_000:
                    connection.executemany(
                        "INSERT OR IGNORE INTO triples VALUES (?, ?, ?, ?)", pending
                    )
                    connection.commit()
                    pending.clear()
            files.append({"path": str(path), "sha256": digest})
            connection.execute(
                "INSERT OR REPLACE INTO sources VALUES (?, ?, ?)",
                (str(path), digest, datetime.now(UTC).isoformat()),
            )
        if pending:
            connection.executemany(
                "INSERT OR IGNORE INTO triples VALUES (?, ?, ?, ?)", pending
            )
        connection.commit()
        unique_triples = connection.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        unique_documents = connection.execute(
            "SELECT COUNT(DISTINCT doc_id) FROM triples"
        ).fetchone()[0]
    finally:
        connection.close()
    return {
        "output": str(output),
        "sources": files,
        "trajectories": trajectories,
        "skipped_trajectories": skipped,
        "candidate_rows_seen": candidates_seen,
        "unique_triples": int(unique_triples),
        "unique_documents": int(unique_documents),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build triples.sqlite from trajectories")
    parser.add_argument("--trajectories", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-parse-errors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_triple_cache(
        args.trajectories,
        args.output,
        include_parse_errors=args.include_parse_errors,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
