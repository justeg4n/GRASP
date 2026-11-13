"""FAISS search over JSON or row-aligned SQLite passage stores."""

from __future__ import annotations

import json
import sqlite3
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any


def _model_leaf(model_name: str) -> str:
    return model_name.rstrip("/\\").rsplit("/", 1)[-1]


def resolve_retrieval_files(
    index_dir: str | Path,
    encoder_name: str,
) -> tuple[Path, Path]:
    """Resolve both flat indexes and the released corpus/encoder directory layout."""
    root = Path(index_dir).expanduser().resolve()
    index_candidates = [
        root / "index.faiss",
        root / _model_leaf(encoder_name) / "index.faiss",
    ]
    index_path = next((path for path in index_candidates if path.is_file()), None)
    if index_path is None:
        searched = ", ".join(str(path) for path in index_candidates)
        raise FileNotFoundError(f"FAISS index not found; searched: {searched}")

    passage_roots = [root, index_path.parent, index_path.parent.parent]
    passage_candidates = [
        directory / filename
        for directory in passage_roots
        for filename in ("passages.sqlite", "corpus.json", "corpus.jsonl")
    ]
    passage_path = next((path for path in passage_candidates if path.is_file()), None)
    if passage_path is None:
        searched = ", ".join(str(path) for path in passage_candidates)
        raise FileNotFoundError(f"Passage store not found; searched: {searched}")
    return index_path, passage_path


def read_corpus(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Corpus must be a JSON array or JSONL objects")
    normalized = []
    seen = set()
    for index, record in enumerate(records):
        doc_id = str(record.get("doc_id", index))
        if doc_id in seen:
            continue
        seen.add(doc_id)
        normalized.append(
            {
                "doc_id": doc_id,
                "title": record.get("title"),
                "text": str(record["text"]),
                "triples": record.get("triples", []),
            }
        )
    return normalized


def build_index(
    corpus_path: str,
    output_dir: str,
    encoder_name: str,
    passage_prefix: str,
    batch_size: int,
) -> Path:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = read_corpus(corpus_path)
    encoder = SentenceTransformer(encoder_name)
    texts = [f"{passage_prefix}{item.get('title') or ''} {item['text']}" for item in records]
    vectors = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(output / "index.faiss"))
    with (output / "corpus.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)
    return output


class _JsonPassageStore:
    def __init__(self, path: Path):
        self.path = path
        self.records = read_corpus(path)

    def __len__(self) -> int:
        return len(self.records)

    def fetch(self, rows: list[int]) -> list[dict[str, Any]]:
        return [dict(self.records[row]) for row in rows]


class _SQLitePassageStore:
    def __init__(self, path: Path):
        self.path = path
        uri = f"file:{path.as_posix()}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(passages)")
        }
        required = {"row", "doc_id", "title", "text"}
        if not required.issubset(columns):
            missing = ", ".join(sorted(required - columns))
            raise ValueError(f"passages.sqlite is missing columns: {missing}")
        count, minimum, maximum = self.connection.execute(
            'SELECT COUNT(*), MIN("row"), MAX("row") FROM passages'
        ).fetchone()
        self.count = int(count)
        if self.count and (minimum != 0 or maximum != self.count - 1):
            raise ValueError("passages.row must be contiguous and zero-based for FAISS alignment")

    def __len__(self) -> int:
        return self.count

    def fetch(self, rows: list[int]) -> list[dict[str, Any]]:
        if not rows:
            return []
        placeholders = ",".join("?" for _ in rows)
        result = self.connection.execute(
            f'SELECT "row", doc_id, title, text FROM passages '
            f'WHERE "row" IN ({placeholders})',
            rows,
        )
        by_row = {
            int(row): {
                "doc_id": str(doc_id),
                "title": title,
                "text": str(text),
                "triples": [],
            }
            for row, doc_id, title, text in result
        }
        missing = [row for row in rows if row not in by_row]
        if missing:
            raise ValueError(f"Passage rows missing from SQLite: {missing[:5]}")
        return [by_row[row] for row in rows]

    def close(self) -> None:
        self.connection.close()


class _SQLiteTripleStore:
    """Optional cache with columns doc_id, head, relation, and tail."""

    def __init__(self, path: Path):
        self.path = path
        uri = f"file:{path.as_posix()}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(triples)")}
        required = {"doc_id", "head", "relation", "tail"}
        if not required.issubset(columns):
            missing = ", ".join(sorted(required - columns))
            raise ValueError(f"triples.sqlite is missing columns: {missing}")

    def fetch(self, doc_ids: list[str]) -> dict[str, list[dict[str, str]]]:
        if not doc_ids:
            return {}
        placeholders = ",".join("?" for _ in doc_ids)
        rows = self.connection.execute(
            f"SELECT doc_id, head, relation, tail FROM triples "
            f"WHERE doc_id IN ({placeholders})",
            doc_ids,
        )
        output = {doc_id: [] for doc_id in doc_ids}
        for doc_id, head, relation, tail in rows:
            output.setdefault(str(doc_id), []).append(
                {
                    "head": str(head),
                    "relation": str(relation),
                    "tail": str(tail),
                    "doc_id": str(doc_id),
                }
            )
        return output

    def close(self) -> None:
        self.connection.close()


class DenseSearcher:
    def __init__(
        self,
        index_dir: str,
        encoder_name: str,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        top_k: int = 10,
        max_candidate_triples: int = 20,
        triple_cache: str | None = None,
        require_triples: bool = False,
    ):
        import faiss
        from sentence_transformers import SentenceTransformer

        root = Path(index_dir).expanduser().resolve()
        index_path, passage_path = resolve_retrieval_files(root, encoder_name)
        self.index_path = index_path
        self.index = faiss.read_index(str(index_path))
        if passage_path.suffix.casefold() == ".sqlite":
            self.store = _SQLitePassageStore(passage_path)
        else:
            self.store = _JsonPassageStore(passage_path)
        if self.index.ntotal != len(self.store):
            raise ValueError(
                f"FAISS rows ({self.index.ntotal}) do not match corpus rows ({len(self.store)})"
            )
        triple_path = (
            Path(triple_cache).expanduser().resolve()
            if triple_cache
            else passage_path.parent / "triples.sqlite"
        )
        self.triple_store = _SQLiteTripleStore(triple_path) if triple_path.is_file() else None
        if require_triples and self.triple_store is None:
            raise FileNotFoundError(
                "A triple cache is required but was not found; pass --triple-cache"
            )
        self.corpus_path = self.store.path
        self.triple_path = triple_path if triple_path.is_file() else None
        self.encoder = SentenceTransformer(encoder_name)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.top_k = top_k
        self.max_candidate_triples = max_candidate_triples
        self.require_triples = require_triples
        self._query_cache: OrderedDict[str, Any] = OrderedDict()
        self._search_cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._query_cache_size = 1024
        self._search_cache_size = 256

    def _encode_query(self, query: str):
        import numpy as np

        cached = self._query_cache.get(query)
        if cached is not None:
            self._query_cache.move_to_end(query)
            return cached
        vector = self.encoder.encode(
            [f"{self.query_prefix}{query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        if vector.shape[1] != self.index.d:
            raise ValueError(
                f"Encoder dimension {vector.shape[1]} does not match FAISS dimension {self.index.d}"
            )
        self._query_cache[query] = vector
        self._query_cache.move_to_end(query)
        if len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)
        return vector

    def search(self, query: str) -> list[dict[str, Any]]:
        cached = self._search_cache.get(query)
        if cached is not None:
            self._search_cache.move_to_end(query)
            return deepcopy(cached)
        vector = self._encode_query(query)
        scores, indices = self.index.search(vector, self.top_k)
        valid_pairs = [
            (float(score), int(index))
            for score, index in zip(scores[0], indices[0], strict=True)
            if index >= 0
        ]
        passages = self.store.fetch([index for _, index in valid_pairs])
        if self.triple_store is not None:
            triples = self.triple_store.fetch([item["doc_id"] for item in passages])
            for item in passages:
                item["triples"] = triples.get(item["doc_id"], [])
        for item, (score, _) in zip(passages, valid_pairs, strict=True):
            item["score"] = score
        self._search_cache[query] = deepcopy(passages)
        self._search_cache.move_to_end(query)
        if len(self._search_cache) > self._search_cache_size:
            self._search_cache.popitem(last=False)
        return passages

    def candidate_triples(
        self,
        query: str,
        passages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        import numpy as np

        candidates = [
            {**triple, "doc_id": str(triple.get("doc_id") or passage["doc_id"])}
            for passage in passages
            for triple in passage.get("triples", [])
        ]
        if not candidates:
            if self.require_triples:
                raise RuntimeError(
                    "No cached triples were found for the retrieved documents; "
                    "expand the triple cache before strict rollouts"
                )
            return []
        texts = [
            f"{self.passage_prefix}{item['head']} {item['relation']} {item['tail']}"
            for item in candidates
        ]
        vectors = self.encoder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        query_vector = self._encode_query(query)[0]
        scores = vectors @ query_vector
        order = np.argsort(-scores)[: self.max_candidate_triples]
        return [candidates[int(index)] for index in order]

    def close(self) -> None:
        self._query_cache.clear()
        self._search_cache.clear()
        close = getattr(self.store, "close", None)
        if close is not None:
            close()
        if self.triple_store is not None:
            self.triple_store.close()
