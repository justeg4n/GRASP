"""Bộ encode E5 (sentence-transformers). Import torch lazy để unit test chạy không cần torch."""

from __future__ import annotations

import threading

import numpy as np


def passage_text(title: str | None, text: str) -> str:
    return f"{title}. {text}" if title else text


class E5Encoder:
    def __init__(
        self,
        name: str,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        device: str | None = None,
        fp16: bool = True,
        max_seq_length: int = 512,
        batch_size: int = 128,
        **_ignored,
    ):
        from sentence_transformers import SentenceTransformer

        self.query_prefix, self.passage_prefix, self.batch_size = (
            query_prefix,
            passage_prefix,
            batch_size,
        )
        self.model = SentenceTransformer(name, device=device)
        self.model.max_seq_length = max_seq_length
        if fp16 and self.model.device.type == "cuda":
            self.model.half()
        self.dim = self.model.get_sentence_embedding_dimension()
        self._lock = threading.Lock()  # GPU dùng chung giữa các luồng

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        with self._lock:
            emb = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )
        return emb.astype(np.float32, copy=False)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def encode_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts], show_progress)
