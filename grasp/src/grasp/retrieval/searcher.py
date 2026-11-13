"""Tìm kiếm: encode query (prefix E5) -> FAISS lấy `candidates` -> re-score bằng embedding gốc -> top_k."""

from __future__ import annotations

import threading
from pathlib import Path

from grasp.retrieval.index import EmbeddingStore
from grasp.retrieval.store import PassageStore
from grasp.types import Passage


class Searcher:
    def __init__(
        self,
        index,
        store: PassageStore,
        encoder,
        emb: EmbeddingStore | None = None,
        top_k: int = 10,
        candidates: int = 100,
        nprobe: int = 64,
        rescore: bool = True,
    ):
        self.index, self.store, self.encoder, self.emb = index, store, encoder, emb
        self.top_k, self.candidates, self.rescore = top_k, candidates, rescore and emb is not None
        if hasattr(index, "nprobe"):
            index.nprobe = nprobe
        self._lock = threading.Lock()

    @classmethod
    def load(cls, index_dir: str | Path, passages_path: str | Path, cfg, encoder) -> Searcher:
        import faiss

        index_dir = Path(index_dir)
        index = faiss.read_index(str(index_dir / "index.faiss"))
        emb = (
            EmbeddingStore(index_dir / "emb", cfg.shard_size) if cfg.get("rescore", True) else None
        )
        return cls(
            index,
            PassageStore(passages_path),
            encoder,
            emb,
            top_k=cfg.top_k,
            candidates=cfg.candidates,
            nprobe=cfg.nprobe,
            rescore=cfg.get("rescore", True),
        )

    def search(self, query: str, top_k: int | None = None) -> list[Passage]:
        k = top_k or self.top_k
        q = self.encoder.encode_queries([query])
        n_cand = max(k, self.candidates) if self.rescore else k
        with self._lock:
            scores, rows = self.index.search(q, n_cand)
        pairs = [(int(r), float(s)) for r, s in zip(rows[0], scores[0], strict=True) if r >= 0]
        if self.rescore and pairs:
            vecs = self.emb.get([r for r, _ in pairs])
            exact = vecs @ q[0]
            pairs = [(r, float(s)) for (r, _), s in zip(pairs, exact, strict=True)]
        pairs.sort(key=lambda p: -p[1])
        pairs = pairs[:k]
        passages = self.store.get([r for r, _ in pairs])
        for p, (_, s) in zip(passages, pairs, strict=True):
            p.score = s
        return passages
