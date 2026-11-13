"""Kho passage trên đĩa (SQLite). Không nạp corpus vào RAM — wiki-18 ~21M passage.

`row` (0..N-1) là id nội bộ, trùng với id vector trong FAISS và vị trí trong embedding shard.
`doc_id` là id ổn định theo nguồn (id wiki-18, curid Wikipedia, hash nội dung 2Wiki).
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path

from grasp.types import Passage

_SCHEMA = "CREATE TABLE IF NOT EXISTS passages (row INTEGER PRIMARY KEY, doc_id TEXT, title TEXT, text TEXT)"


class PassageStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"chưa có passage store: {self.path} (chạy scripts/build_index.py)"
            )
        self._db = sqlite3.connect(
            f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False
        )
        self._lock = threading.Lock()
        self._n = self._db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]

    def __len__(self) -> int:
        return self._n

    def get(self, rows: list[int]) -> list[Passage]:
        """Giữ đúng thứ tự `rows`; row không tồn tại bị bỏ qua."""
        if not rows:
            return []
        q = f"SELECT row, doc_id, title, text FROM passages WHERE row IN ({','.join('?' * len(rows))})"
        with self._lock:
            found = {
                r: Passage(doc_id=d, title=t, text=x) for r, d, t, x in self._db.execute(q, rows)
            }
        return [found[r] for r in rows if r in found]

    def iter_range(
        self, start: int, stop: int, batch: int = 10_000
    ) -> Iterator[tuple[int, str, str]]:
        """(row, title, text) theo thứ tự row — dùng khi encode corpus."""
        for lo in range(start, stop, batch):
            hi = min(stop, lo + batch)
            with self._lock:
                rows = self._db.execute(
                    "SELECT row, title, text FROM passages WHERE row >= ? AND row < ? ORDER BY row",
                    (lo, hi),
                ).fetchall()
            yield from rows

    def close(self) -> None:
        self._db.close()

    @staticmethod
    def build(
        path: str | Path, passages: Iterable[tuple[str, str, str]], commit_every: int = 200_000
    ) -> int:
        """Ghi (doc_id, title, text) -> SQLite. Ghi ra file tạm rồi rename để không để lại store dở."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
        db = sqlite3.connect(tmp)
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute(_SCHEMA)
        n, buf = 0, []
        for doc_id, title, text in passages:
            buf.append((n, doc_id, title, text))
            n += 1
            if len(buf) >= commit_every:
                db.executemany("INSERT INTO passages VALUES (?,?,?,?)", buf)
                db.commit()
                buf.clear()
        if buf:
            db.executemany("INSERT INTO passages VALUES (?,?,?,?)", buf)
        db.commit()
        db.close()
        tmp.replace(path)
        return n
