"""Lọc trajectory đạt -> cặp (prompt, completion) theo role. Lọc lại không cần gọi API."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from grasp.sft.collector import read_records

SPLIT_TO_FILE = {"sft_train": "train", "sft_val": "val"}


def passes(rec: dict, em_min: float = 1.0, f1_min: float = 0.8) -> bool:
    if rec.get("status") != "ok":
        return False
    return (rec.get("em") or 0.0) >= em_min or (rec.get("f1") or 0.0) >= f1_min


def to_rows(rec: dict) -> list[dict]:
    rows = []
    for c in rec.get("calls", []):
        rows.append(
            {
                "qid": rec["qid"],
                "dataset": rec["dataset"],
                "split": rec["split"],
                "role": c["role"],
                "step": c["step"],
                "messages": [
                    {"role": "user", "content": c["prompt"]},
                    {"role": "assistant", "content": c["completion"]},
                ],
            }
        )
    return rows


def export(
    traj_paths: list[str | Path], out_dir: str | Path, em_min: float = 1.0, f1_min: float = 0.8
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        name: open(out_dir / f"{name}.jsonl", "w", encoding="utf-8")  # noqa: SIM115
        for name in set(SPLIT_TO_FILE.values())
    }
    stats = {
        "trajectories": Counter(),
        "kept": Counter(),
        "rows": Counter(),
        "roles": Counter(),
        "status": Counter(),
    }
    try:
        for p in traj_paths:
            for rec in read_records(Path(p)).values():
                ds, split = rec.get("dataset", ""), rec.get("split", "")
                stats["status"][rec.get("status")] += 1
                if split not in SPLIT_TO_FILE:
                    continue
                stats["trajectories"][f"{ds}/{split}"] += 1
                if not passes(rec, em_min, f1_min):
                    continue
                stats["kept"][f"{ds}/{split}"] += 1
                for row in to_rows(rec):
                    files[SPLIT_TO_FILE[split]].write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["rows"][SPLIT_TO_FILE[split]] += 1
                    stats["roles"][row["role"]] += 1
    finally:
        for f in files.values():
            f.close()
    stats = {k: dict(v) for k, v in stats.items()}
    stats["filter"] = {"em_min": em_min, "f1_min": f1_min}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def push_to_hub(out_dir: str | Path, repo_id: str, private: bool = True) -> None:
    from datasets import load_dataset

    out_dir = Path(out_dir)
    ds = load_dataset("json", data_files={k: str(out_dir / f"{k}.jsonl") for k in ("train", "val")})
    ds.push_to_hub(repo_id, private=private)
