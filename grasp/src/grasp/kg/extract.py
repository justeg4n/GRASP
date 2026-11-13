"""Trích knowledge triple LAZY: chỉ doc nào được retrieve mới trích, kết quả cache vĩnh viễn trên đĩa.

Trích offline toàn corpus (26M passage) qua API là không khả thi; số doc thực sự được retrieve khi sinh SFT
chỉ cỡ vài trăm nghìn. Cache khoá theo (doc_id, extractor_model) nên đổi model extractor không dùng lẫn cache.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path

from grasp.agents.base import load_template
from grasp.kg.triples import parse_triples
from grasp.llm.budget import Usage
from grasp.llm.errors import BadRequest
from grasp.types import Passage, Triple


class TripleCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS triples (doc_id TEXT, model TEXT, triples TEXT, PRIMARY KEY (doc_id, model))"
        )
        self._db.commit()
        self._lock = threading.Lock()

    def get(self, doc_id: str, model: str) -> list[Triple] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT triples FROM triples WHERE doc_id = ? AND model = ?", (doc_id, model)
            ).fetchone()
        if row is None:
            return None
        return [Triple(h, r, t, doc_id) for h, r, t in json.loads(row[0])]

    def put(self, doc_id: str, model: str, triples: list[Triple]) -> None:
        payload = json.dumps([[t.head, t.relation, t.tail] for t in triples], ensure_ascii=False)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO triples VALUES (?, ?, ?)", (doc_id, model, payload)
            )
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM triples").fetchone()[0]

    def close(self) -> None:
        self._db.close()


class TripleExtractor:
    def __init__(
        self, client, cache: TripleCache, max_tokens: int | None = None, temperature: float = 0.0
    ):
        self.client, self.cache = client, cache
        self.model = client.model
        self.max_tokens, self.temperature = max_tokens, temperature
        self._inflight: dict[
            str, asyncio.Future
        ] = {}  # nhiều episode cùng cần 1 doc -> chỉ gọi API 1 lần
        self.misses = 0

    async def triples_for(
        self, passages: list[Passage], usage: Usage | None = None
    ) -> list[Triple]:
        results = await asyncio.gather(*(self._one(p, usage) for p in passages))
        return [t for ts in results for t in ts]

    async def _one(self, p: Passage, usage: Usage | None) -> list[Triple]:
        cached = self.cache.get(p.doc_id, self.model)
        if cached is not None:
            return cached
        if p.doc_id in self._inflight:
            return await asyncio.shield(self._inflight[p.doc_id])
        fut = asyncio.get_running_loop().create_future()
        self._inflight[p.doc_id] = fut
        try:
            triples = await self._extract(p, usage)
            self.cache.put(p.doc_id, self.model, triples)
            fut.set_result(triples)
            return triples
        except asyncio.CancelledError:
            fut.cancel()  # chỉ xảy ra khi dừng toàn cục: mọi episode đều đang bị huỷ
            raise
        except Exception as e:
            fut.set_exception(e)
            fut.exception()  # đánh dấu đã đọc để asyncio không cảnh báo
            raise
        finally:
            self._inflight.pop(p.doc_id, None)

    async def _extract(self, p: Passage, usage: Usage | None) -> list[Triple]:
        self.misses += 1
        prompt = load_template("extract").format(title=p.title or "", text=p.text)
        try:
            gen = await self.client.chat(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens, usage=usage
            )
        except BadRequest:
            return []  # doc lỗi (vd. quá dài): cache rỗng, không làm hỏng cả trajectory
        return parse_triples(gen[0].text, doc_id=p.doc_id)
