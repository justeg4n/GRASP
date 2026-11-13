"""Tiện ích nhỏ dùng chung cho các script."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def script_init() -> None:
    """Gọi đầu mỗi script: console Windows mặc định cp1252 không in được tiếng Việt; nạp .env."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()


def load_dotenv(path: str | Path = REPO_ROOT / ".env") -> None:
    """Đọc KEY=VALUE từ .env vào os.environ (không ghi đè biến đã có)."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        if v:
            os.environ.setdefault(k.strip(), v)


def git_commit() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = (
            subprocess.call(["git", "diff", "--quiet"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
            != 0
        )
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
